#!/bin/bash

ROOT_DIR="../../Datasets/CelebA/"
LOG_DIR="../tensorboard/pvae"
MODEL_DIR="../trained_models"
VAE_OUT="pvae"
N_WORKERS=8

LOSS_L2=0
LOSS_123=1
LOSS_345=2

python3 ../train_vae.py --rootdir $ROOT_DIR --logdir $LOG_DIR --modeldir $MODEL_DIR --loss $LOSS_L2 --workers $N_WORKERS -o $VAE_OUT