import os
import numpy as np
import wandb
import torchvision
import errno
import torch
from matplotlib import pyplot as plt
from config import cfg


class MetricLogger:
    """Metric class"""
    def __init__(self, project_version_name, wab=True, show_accuracy=True, resume_id=False):
        """
        :param project_version_name: name of current version of project
        :param wab: good realtime metric, you can register free account in https://wandb.ai/
        :param show_accuracy: if True: show accuracy on real and fake data
        """
        self.project_version_name = project_version_name
        self.show_acc = show_accuracy
        self.data_subdir = f"{os.path.join(cfg.OUT_DIR, self.project_version_name)}/fixed_images"

        if wab:
            if resume_id:
                wandb_id = cfg.RESUME_ID
            else:
                wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id, project='ProGAN', name=project_version_name, resume=True)
            wandb.config.update({
                'learning_rate': cfg.LEARNING_RATE,
                'z_dimension': cfg.Z_DIMENSION,
                'model_depth': cfg.IN_CHANNELS,
                'critic_iter_count': cfg.CRITIC_ITERATIONS,
                'lambda gp': cfg.LAMBDA_GP,
                'prog_epochs': cfg.PROGRESSIVE_EPOCHS,
                'batch_sizes': cfg.BATCH_SIZE,
            })

    def log(self, crt_loss, gen_loss, acc_real=None, acc_fake=None):
        """
        Logging values
        :param crt_loss: ``torch.autograd.Variable``, critical loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, D(x) predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, D(G(z)) paramredicted on fake data
        """
        if isinstance(crt_loss, torch.autograd.Variable):
            crt_loss = crt_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if self.show_acc and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if self.show_acc and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        wandb.log({'d_loss': crt_loss, 'g_loss': gen_loss, 'D(x)': acc_real, 'D(G(z))': acc_fake})

    def log_image(self, images, num_samples, epoch, stage, normalize=True):
        """
        Create image grid and save it
        :param images: ``Tor    ch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param normalize: if True normalize images
        :param epoch: ``int``, current epoch
        """
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        self.save_torch_images(grid, epoch, stage)
        wandb.log({'fixed_noise': [wandb.Image(np.moveaxis(grid.detach().cpu().numpy(), 0, -1))]})

    def save_torch_images(self, grid, epoch, stage):
        """
        Display and save image grid
        :param grid: ``ndarray``, grid image
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        """
        out_dir = self.data_subdir
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        MetricLogger._save_images(fig, out_dir, epoch, stage)
        plt.close()

    @staticmethod
    def _save_images(fig, out_dir, epoch, stage):
        """
        Saves image on drive
        :param fig: pls.figure object
        :param out_dir: path to output dir
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        """
        MetricLogger._make_dir(out_dir)
        image_name = f"epoch{epoch}_{stage}.jpg"
        fig.savefig('{}/{}'.format(out_dir, image_name))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
