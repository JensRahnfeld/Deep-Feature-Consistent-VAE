import argparse
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils.hyperparameters import DIM_LATENT, NORMALIZE_MEAN, NORMALIZE_STDEV
from utils.img_transforms import transform
from models.vae import VAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, required=True, help="path to vae")
    parser.add_argument('--imgdir', type=str, required=True,\
        help="path to image folder")
    
    args = parser.parse_args()

    img_names = os.listdir(args.imgdir)
    n_imgs = len(img_names)

    vae = VAE(DIM_LATENT)
    vae.load_state_dict(torch.load(args.vae))

    fig = plt.figure()

    with torch.no_grad():
        for i in range(n_imgs):
            path = os.path.join(args.imgdir, img_names[i])
            img = Image.open(path)
            
            x_true = transform(img)
            x_true = x_true.unsqueeze(0)
            x_true = x_true.view(1, 3, 64, 64)
            x_rec, mu, logvar = vae(x_true)

            img_true = x_true.squeeze(0).view(64, 64, 3).numpy()
            img_rec = x_rec.squeeze(0).view(64, 64, 3).numpy()

            # denormalize
            img_true = (img_true * NORMALIZE_STDEV) + NORMALIZE_MEAN
            img_rec = (img_rec * NORMALIZE_STDEV) + NORMALIZE_MEAN

            fig.add_subplot(2, n_imgs, i+1)
            plt.imshow(img_true)
            plt.axis('off')

            fig.add_subplot(2, n_imgs, n_imgs+i+1)
            plt.imshow(img_rec)
            plt.axis('off')

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()