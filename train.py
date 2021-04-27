import argparse
import time
import logging
import os
import torch
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
from utils import print_epoch_time
from data.dataset import AnimeFacesDataset
from config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='ProGAN')
    parser.add_argument('--data_path', dest='data_path', help='path to dataset folder',
                        default=None, type=str)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='path to checkpoint.pth.tar',
                        default=None, type=str)
    parser.add_argument('--wandb_id', dest='wandb_id', help='wand metric id for resume',
                        default=None, type=str)
    parser.add_argument('--device', dest='device', help='use device: gpu, tpu. Default use gpu if available',
                        default=None, type=str)
    parser.print_help()
    return parser.parse_args()


@print_epoch_time
def train_one_epoch():
    pass


if __name__ == '__main__':
    logger = logging.getLogger('train')
    args = parse_args()

    assert args.data_path, 'data path not specified'

    logger.info(f'Start {__name__} at {time.ctime()}')
    logger.info(f'Called with args: {args.__dict__}')
    logger.info(f'Config params: {cfg.__dict__}')

    if args.device == 'gpu':
        device = torch.device('cuda')
    elif args.device == 'tpu':
        device = xm.xla_device()
    elif args.device is None and not torch.cuda.is_available():
        logger.error(f"device:{args.device}", exc_info=True)
        raise ValueError('Device not specified and gpu is not available')

    logger.info(f'Using device:{args.device}')

    # TODO step batch
    dataset = AnimeFacesDataset(args.data_path)
    # train_dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
    #                               num_workers=2, drop_last=True, pin_memory=True)
    # TODO define models, optimizers and load checkpoint
