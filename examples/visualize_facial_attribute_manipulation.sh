#!/bin/bash

IMG_DIR="../../Datasets/CelebA/img_align_celeba/"
LIST_ATTR="../../Datasets/CelebA/Anno/list_attr_celeba.csv"
VAE="../trained_models/vae123.pt"

ATTR="Eyeglasses"
TARGET_IMG=$(ls $IMG_DIR | shuf -n 1)

python3 ../visualize_facial_attribute_manipulation.py --vae $VAE --attr $ATTR --list_attr $LIST_ATTR --imgdir $IMG_DIR --img $IMG_DIR$TARGET_IMG