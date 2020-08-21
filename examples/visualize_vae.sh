#!/bin/bash

IMG_DIR="../../Datasets/CelebA/img_align_celeba/"
PVAE="../trained_models/pvae.pt"
VAE123="../trained_models/vae123.pt"
VAE345="../trained_models/vae345.pt"

IMGS=$(ls $IMG_DIR | shuf -n 10)

IMG_TMP="./img_tmp/"
mkdir $IMG_TMP

for IMG in $IMGS
do
    cp $IMG_DIR$IMG $IMG_TMP
done

python3 ../visualize_vae.py --vae $PVAE $VAE123 $VAE345 --imgdir $IMG_TMP

rm -r $IMG_TMP