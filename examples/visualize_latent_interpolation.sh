#!/bin/bash

IMG_DIR="../../Datasets/CelebA/img_align_celeba/"
VAE="../trained_models/vae123.pt"

IMG_LEFT=$(ls $IMG_DIR | shuf -n 1)
IMG_RIGHT=$(ls $IMG_DIR | shuf -n 1)

python3 ../visualize_latent_interpolation.py --vae $VAE --img_left $IMG_DIR$IMG_LEFT --img_right $IMG_DIR$IMG_RIGHT