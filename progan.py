import argparse
import torch
import os
import imageio
import torch.nn.functional as F
from models.model import Generator
from config import cfg
from math import log2
from utils import load_gen, get_random_noise, show_batch, latent_space_interpolation_sequence


def parse_args():
    parser = argparse.ArgumentParser(description='AnimeFace-ProGAN')
    parser.add_argument('--path_ckpt', dest='path_ckpt', help='Path to checkpoint of generator', default=None, type=str)
    parser.add_argument('--num_samples', dest='num_samples', help='Number of samples', default=1, type=int)
    parser.add_argument('--steps', dest='steps', help='Number of step interpolation', default=5, type=int)
    parser.add_argument('--device', dest='device', help='cpu or gpu', default=None, type=str)
    parser.add_argument('--out_path', dest='out_path', help='Path to output folder, default=save to project folder',
                        default=None, type=str)
    parser.add_argument('--gif', dest='gif', help='Create gif', default=None, type=bool)
    parser.add_argument('--grid', dest='grid', help='Draw grid of images', default=None, type=bool)
    parser.add_argument('--z_size', dest='z_size', help='The size of latent space, default=256', default=256, type=int)
    parser.add_argument('--img_size', dest='img_size', help='Size of output image', default=6, type=int)
    parser.add_argument('--resize', dest='resize', help='if you want to resize images', default=None, type=int)
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.path_ckpt, 'Path to checkpoint not specified'

    if args.gif and args.num_samples < 2:
        raise ValueError('for GIF num_samples must be greater than 1')

    if not args.out_path:
        out_path = 'ProGAN-Anime-Faces'
    else:
        out_path = args.out_path

    if args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    gen = Generator(cfg.Z_DIMENSION, cfg.IN_CHANNELS, cfg.CHANNELS_IMG).to(device)
    load_gen(gen, args.path_ckpt, device)
    gen.eval()
    alpha = 1
    step = int(log2(cfg.START_TRAIN_IMG_SIZE / 4))

    if args.grid:
        noise = get_random_noise(args.num_samples, args.z_size, device)
        print("==> Generate IMAGE GRID...")
        output = gen(noise, alpha, step)
        show_batch(output, out_path, num_samples=args.num_samples, figsize=(args.img_size, args.img_size))
    elif args.gif:
        noise = get_random_noise(args.num_samples, args.z_size, device)
        print("==> Generate GIF...")
        images = latent_space_interpolation_sequence(noise, step_interpolation=args.steps)
        output = gen(images, alpha, step)
        if args.resize and isinstance(args.resize, int):
            print(f"==> Resize images to {args.resize}px")
            output = F.interpolate(output, size=args.resize)

        images = []
        for img in output:
            img = img.detach().permute(1, 2, 0)
            images.append(img.numpy())
        save_img_name = 'result.gif'
        save_path = os.path.join(out_path, save_img_name)
        imageio.mimsave(save_path, images, fps=8)
        print(f'GIF save to {save_path}')
