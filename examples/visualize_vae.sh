#!/bin/bash

IMG_DIR="../../Datasets/CelebA/img_align_celeba/"
VAE="../trained_models/vae.pt"

IMGS=$(ls $IMG_DIR | shuf -n 10)

IMG_TMP="./img_tmp/"
mkdir $IMG_TMP

for IMG in $IMGS
do
    cp $IMG_DIR$IMG $IMG_TMP
done

python3 ../visualize_vae.py --vae $VAE --imgdir $IMG_TMP

rm -r $IMG_TMP