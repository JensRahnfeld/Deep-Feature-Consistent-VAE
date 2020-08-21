import argparse
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils.hyperparameters import DIM_LATENT, NORMALIZE_MEAN, NORMALIZE_STDEV
from utils.img_transforms import transform, transform_back
from utils.plots import grid_add_img
from models.vae import VAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, nargs='+', required=True,\
        help="path to vae")
    parser.add_argument('--imgdir', type=str, required=True,\
        help="path to image folder")
    
    args = parser.parse_args()

    img_names = os.listdir(args.imgdir)
    n_imgs = len(img_names)

    rows = len(args.vae) + 1

    fig = plt.figure()

    with torch.no_grad():
        for k in range(len(args.vae)):
            vae = VAE(DIM_LATENT)
            vae.load_state_dict(torch.load(args.vae[k]))
            for i in range(n_imgs):
                path = os.path.join(args.imgdir, img_names[i])
                img = Image.open(path)
                
                x_true = transform(img)
                x_true = x_true.unsqueeze(0)
                x_true = x_true.view(1, 3, 64, 64)
                x_rec, mu, logvar = vae(x_true)

                x_true = x_true.squeeze(0)
                x_rec = x_rec.squeeze(0)

                img_true = transform_back(x_true)
                img_rec = transform_back(x_rec)

                if k == 0: grid_add_img(img_true, fig, rows, n_imgs, i+1)
                grid_add_img(img_rec, fig, rows, n_imgs, (k+1)*n_imgs+i+1)

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()