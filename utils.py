import time
import torch
import logging
import os
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from math import log2
from config import cfg
from data.dataset import AnimeFacesDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def print_epoch_time(f):
    """Calculate time of each epoch and print it"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("epoch time: %2.1f min" % ((te-ts)/60))
        return result
    return timed


def get_train_dataloader(data_path, img_size):
    """
    We need different size of batches for different image size
    :param data_path: path to dataset folder
    :param img_size: img size
    :return: dataset, dataloader
    """
    batch_size = cfg.BATCH_SIZE[int(log2(img_size / 4))]
    dataset = AnimeFacesDataset(data_path)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dataset, train_dataloader


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    logger.info(f"Saving model ...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_model_path = os.path.join(cfg.OUT_DIR, filename)
    torch.save(checkpoint, save_model_path)
    logger.info(f"Success saved to {save_model_path}")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    logger.info(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logger.info(f"Success loaded model from {checkpoint_file}")


def load_gen(gen, filename, device):
    print("=> Load generator...")
    cp = torch.load(filename, map_location=device)
    gen.load_state_dict(cp['state_dict'])
    print(f"=> Generator model loaded from {filename}")


def save_fixed_noise(fixed_noise, filename="fixed_noise.pth.tar"):
    logger.info(f"Saving fixed noise...")
    fixed_noise = {'fixed_noise': fixed_noise}
    save_fixed_noise_path = os.path.join(cfg.OUT_DIR, filename)
    torch.save(fixed_noise, save_fixed_noise_path)
    logger.info(f"Save fixed noise to {save_fixed_noise_path}")


def load_fixed_noise(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    logger.info(f"Success loaded fixed noise from {checkpoint_file}")
    return checkpoint['fixed_noise']


def get_random_noise(size, dim, device):
    """
     Get random noise from normal distribution
    :param size: ``int``, number of samples (batch)
    :param dim: ``int``, dimension
    :param device: cuda or cpu device
    :return: Tensor([size, dim, 1, 1])
    """
    return torch.randn(size, dim, 1, 1).to(device)


def latent_space_interpolation_sequence(latent_seq, step_interpolation=5):
    """
    Interpolation between noises
    :param latent_seq: Tensor([N, z_dim, 1, 1])
    :param step_interpolation: ``int``: number of steps between each images
    :return: List([samples, z_dim, 1, 1]
    """
    vector = []
    alpha_values = np.linspace(0, 1, step_interpolation)

    start_idxs = [i for i in range(0, len(latent_seq))]
    end_idxs = [i for i in range(1, len(latent_seq))]

    for start_idx, end_idx in zip(start_idxs, end_idxs):
        latent_start = latent_seq[start_idx].unsqueeze(0)
        latent_end = latent_seq[end_idx].unsqueeze(0)
        for alpha in alpha_values:
            vector.append(alpha*latent_end + (1.0 - alpha)*latent_start)
    return torch.cat(vector, dim=0)


def show_batch(batch, save, num_samples=36, figsize=(10, 10), normalize=True):
    """
    Show image
    :param batch: ``Tensor([N, channels, size, size])`` batch
    :param num_samples: ``int``: number of sumples
    :param figsize: ``Tuple(n, n)``: size of image
    :param normalize: if need denormalization
    """
    images = batch[:num_samples, ...]
    nrows = int(np.sqrt(num_samples))
    grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
    fig = plt.figure(figsize=figsize)
    plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))
    plt.axis('off')

    if save:
        save_path = os.path.join(save, 'grid_result.png')
        plt.savefig(save_path)
