## Progressive growing of GAN

This is a simple implementation of paper: [progressive growing of GAN](https://arxiv.org/abs/1710.10196)

## Dataset
Original Dataset: https://www.kaggle.com/scribbless/another-anime-face-dataset

## Results 128x128
Training time =~ 30hours (still underfitting)

![res](https://raw.githubusercontent.com/ErrorInever/ProGAN/master/data/results/grid.png)

Interpolation between latent space

![int](https://raw.githubusercontent.com/ErrorInever/ProGAN/master/data/results/interpolation.gif)


## ARGS

    optional arguments:
      -h, --help            show this help message and exit
      --data_path DATA_PATH
                            path to dataset folder
      --checkpoint_gen CHECKPOINT_PATH_GEN
                            path to gen.pth.tar
      --checkpoint_crt CHECKPOINT_PATH_CRT
                            path to crt.pth.tar
      --fixed_noise FIXED_NOISE
                            path to fixed_noise.pth.tar
      --wandb_id WANDB_ID   wand metric id for resume
      --device DEVICE       use device: gpu, tpu. Default use gpu if available
      --out_dir OUT_DIR     path where to save checkpoints, fixed images etc
      --start_epoch START_EPOCH
                            start from current epoch for resume training
      --api API             wandb_api
      
Other params set up in config.py, don't forget change __C.START_TRAIN_IMG_SIZE = 128, for training from scratch need set up to 4
