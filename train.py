import argparse
import time
import os
import logging
import torch
#import torch_xla.core.xla_model as xm
import torch.optim as optim
from math import log2
from tqdm import tqdm
from utils import (print_epoch_time, get_train_dataloader, load_checkpoint, save_checkpoint, gradient_penalty,
                   load_fixed_noise, save_fixed_noise)
from config import cfg
from models.model import Generator, Critic
from metriclogger import MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description='ProGAN')
    parser.add_argument('--data_path', dest='data_path', help='path to dataset folder',
                        default=None, type=str)
    parser.add_argument('--checkpoint_gen', dest='checkpoint_path_gen', help='path to gen.pth.tar',
                        default=None, type=str)
    parser.add_argument('--checkpoint_crt', dest='checkpoint_path_crt', help='path to crt.pth.tar',
                        default=None, type=str)
    parser.add_argument('--fixed_noise', dest='fixed_noise', help='path to fixed_noise.pth.tar',
                        default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='wand metric id for resume',
                        default=None, type=str)
    parser.add_argument('--device', dest='device', help='use device: gpu, tpu. Default use gpu if available',
                        default=None, type=str)
    parser.add_argument('--out_dir', dest='out_dir', help='path where to save checkpoints, fixed images etc',
                        default=None, type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='start from current epoch for resume training',
                        default=None, type=int)
    parser.add_argument('--api', dest='api', help='wandb_api',
                        default=None, type=str)
    parser.print_help()
    return parser.parse_args()


@print_epoch_time
def train_one_epoch(gen, critic, opt_gen, opt_crt, scaler_gen, scaler_crt,
                    dataloader, metric_logger, dataset, step, alpha, device,
                    fixed_noise, epoch, stage):

    loop = tqdm(dataloader, leave=True)
    for batch_idx, real in enumerate(loop):
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train critic: maximize(E[critic(real)] - E[critic(fake)]) or (-1 * maximize(E[critic(real)]) + E[critic(fake)]
        noise = torch.randn(cur_batch_size, cfg.Z_DIMENSION, 1, 1).to(device)
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=device)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + cfg.LAMBDA_GP * gp + (0.001 * torch.mean(critic_real ** 2))
            )
        opt_crt.zero_grad()
        scaler_crt.scale(loss_critic).backward()
        scaler_crt.step(opt_crt)
        scaler_crt.update()

        # Train generator maximize(E[critic(gen_fake)] <-> min -E[critic(gen_fake)])
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # update alpha
        alpha += cur_batch_size / (cfg.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        alpha = min(alpha, 1)

        if batch_idx % cfg.FREQ == 0:
            with torch.no_grad():
                fixed_fakes = gen(fixed_noise, alpha, step) * 0.5 + 0.5
                metric_logger.log(loss_critic, loss_gen)
                metric_logger.log_image(fixed_fakes, cfg.NUM_SAMPLES, epoch, stage)

        loop.set_postfix(
            gp=gp.item(),
            loss_critic=loss_critic.item(),
        )

    return alpha


if __name__ == '__main__':
    os.environ["WANDB_API_KEY"] = args.api
    logger = logging.getLogger('train')
    args = parse_args()

    assert args.data_path, 'data path not specified'
    if args.wandb_id:
        cfg.RESUME_ID = args.resume_id

    if args.out_dir:
        cfg.OUT_DIR = args.out_dir

    if args.start_epoch:
        start_epoch = args.start_epoch
    else:
        start_epoch = 0

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')
    # elif args.device == 'tpu':
    #     device = xm.xla_device()
    elif args.device is None and not torch.cuda.is_available():
        logger.error(f"device:{args.device}", exc_info=True)
        raise ValueError('Device not specified and gpu is not available')
    logger.info(f'Using device:{args.device}')
    torch.backends.cudnn.benchmarks = True

    # models
    gen = Generator(cfg.Z_DIMENSION, cfg.IN_CHANNELS, cfg.CHANNELS_IMG).to(device)
    crt = Critic(cfg.IN_CHANNELS, cfg.CHANNELS_IMG).to(device)
    # optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    opt_crt = optim.Adam(crt.parameters(), lr=cfg.LEARNING_RATE, betas=(0.0, 0.99))
    # scalers
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_crt = torch.cuda.amp.GradScaler()
    # load models if resume
    if args.checkpoint_path_gen and args.checkpoint_path_crt:
        load_checkpoint(args.checkpoint_path_gen, gen, opt_gen, cfg.LEARNING_RATE)  # load generator
        load_checkpoint(args.checkpoint_path_crt, crt, opt_crt, cfg.LEARNING_RATE)  # load critic
        if args.fixed_noise:
            fixed_noise = load_fixed_noise(args.fixed_noise)
            fixed_noise.to(device)
        else:
            raise ValueError("fixed noise not specified")
        metric_logger = MetricLogger(project_version_name=cfg.PROJECT_VERSION_NAME, resume_id=True)
    else:
        # defining standard params
        metric_logger = MetricLogger(project_version_name=cfg.PROJECT_VERSION_NAME)
        fixed_noise = torch.randn(4, cfg.Z_DIMENSION, 1, 1).to(device)

    gen.train()
    crt.train()

    # train start at step that corresponds to img size that we set in cfg.START_TRAIN_IMG_SIZE
    step = int(log2(cfg.START_TRAIN_IMG_SIZE / 4))
    for num_epochs in cfg.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1
        cfg.IMG_SIZE = 4 * 2 ** step
        dataset, train_dataloader = get_train_dataloader(args.data_path, cfg.IMG_SIZE)
        logger.info(f"Current image size: {cfg.IMG_SIZE}")

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}]")
            alpha = train_one_epoch(gen, crt, opt_gen, opt_crt, scaler_gen, scaler_crt, train_dataloader, metric_logger,
                                    dataset, step, alpha, device, fixed_noise, epoch, stage=4*2**step)
            if cfg.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=f"gen_img{4 * 2 ** step}_epoch{epoch}.pth.tar")
                save_checkpoint(crt, opt_crt, filename=f"crt_img{4 * 2 ** step}_epoch{epoch}.pth.tar")
                save_fixed_noise(fixed_noise, filename=f"fixed_noise{4 * 2 ** step}_epoch{epoch}.pth.tar")
        # do progress to the next image size
        step += 1
        start_epoch = 0
