import unittest
import torch
from models.model import Generator, Discriminator
from math import log2


class TestModels(unittest.TestCase):

    def setUp(self):
        self.Z_DIM = 50
        self.IN_CHANNELS = 256

    def test_shape(self):
        gen = Generator(self.Z_DIM, self.IN_CHANNELS, img_channels=3)
        critic = Discriminator(self.IN_CHANNELS, img_channels=3)

        for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            num_steps = int(log2(img_size / 4))
            x = torch.randn((1, self.Z_DIM, 1, 1))
            z = gen(x, 0.5, steps=num_steps)
            self.assertEqual(z.shape, (1, 3, img_size, img_size))
            out = critic(z, alpha=0.5, steps=num_steps)
            self.assertEqual(out.shape, (1, 1))
            print(f"Done! Img_size:{img_size}")


if __name__ == '__main__':
    unittest.main()
