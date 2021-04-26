import torch
import torch.nn as nn
import torch.nn.functional as F

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


class EqualizedLRConv2d(nn.Module):
    """
    To achieve equalized learning rate it is essential that layers learn at a similar speed.
    We scale the weights of a layer according to how many weights that layer has.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channels * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # initialize conv layer from random normal distribution
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    """
    Instead of using Batch Normalization we use Pixel Normalization. This layer has no trainable weights.
    It normalizes the feature vector in each pixel to unit length, end applied after the convolutional layers
    in the Generator.
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    """
    Standard convolution block
    """
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super().__init__()
        self.use_pixel_norm = use_pixel_norm
        self.conv1 = EqualizedLRConv2d(in_channels, out_channels)
        self.conv2 = EqualizedLRConv2d(out_channels, out_channels)
        self.act1 = nn.LeakyReLU(0.2)
        self.pn1 = PixelNorm()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.pn1(x) if self.use_pixel_norm else x
        x = self.act1(self.conv2(x))
        x = self.pn1(x) if self.use_pixel_norm else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels):
        """
        :param z_dim: ``int``, latent space dimension
        :param in_channels: ``int``, number of input channels
        :param img_channels: ``int``, number of output channels
        """
        super().__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),   # 1x1 to 4x4
            nn.LeakyReLU(0.2),
            EqualizedLRConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = EqualizedLRConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i+1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels))
            self.rgb_layers.append(EqualizedLRConv2d(conv_out_channels, img_channels,
                                                     kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, up_scaled, generated):
        """
        Interpolation
        :param alpha: ``int``,  constant
        :param up_scaled:
        :param generated:
        :return: value between [-1, 1]
        """
        return torch.tanh(alpha * generated + (1 - alpha) * up_scaled)

    def forward(self, x, alpha, steps):
        """
        :param x:
        :param alpha:
        :param steps: if steps=0 then 4x4, else if steps=1 then 8x8, ...
        :return:
        """
        out = self.initial(x)   # 4x4

        if steps == 0:
            return self.initial_rgb(out)

        for step in range(steps):
            up_scaled = F.interpolate(out, scale_factor=2, mode='nearest')
            out = self.prog_blocks[step](up_scaled)

        final_up_scaled = self.rgb_layers[steps - 1](up_scaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_up_scaled, final_out)


class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels):
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.act1 = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_channels = int(in_channels * factors[i])
            conv_out_channels = int(in_channels * factors[i-1])
            self.prog_blocks.append(ConvBlock(conv_in_channels, conv_out_channels, use_pixel_norm=False))
            self.rgb_layers.append(EqualizedLRConv2d(img_channels, conv_in_channels,
                                                     kernel_size=1, stride=1, padding=0))

        # for 4x4 res
        self.initial_rgb = EqualizedLRConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)    # 4x4
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            EqualizedLRConv2d(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            EqualizedLRConv2d(in_channels, in_channels, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # TODO create Linear EqualizedLRLayer
            EqualizedLRConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def fade_in(self, alpha, out, down_scaled):
        """
        :param alpha:
        :param out: out from conv layer
        :param down_scaled: out from average pooling
        :return:
        """
        return alpha * out + (1 - alpha) * down_scaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.act1(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        down_scaled = self.act1(self.rgb_layers[cur_step+1](self.avg_pool1(x)))
        out = self.avg_pool1(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, out, down_scaled)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool1(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
