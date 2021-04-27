import time
import torch
import logging
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
    batch_size = cfg.BATCH_SIZES[int(log2(img_size / 4))]
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
    logger.info(f"Saving {filename} ...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, filename)
    logger.info(f"Success saved to {filename}")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    logger.info(f"Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logger.info(f"Success loaded model from {checkpoint_file}")
