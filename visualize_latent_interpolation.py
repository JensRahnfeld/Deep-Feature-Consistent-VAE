import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.vae import VAE
from PIL import Image
from utils.hyperparameters import DIM_LATENT, RESIZE_HEIGHT, RESIZE_WIDTH,\
    NORMALIZE_MEAN, NORMALIZE_STDEV
from utils.interpolation import linear_interpolation
from utils.img_transforms import transform, transform_back
from utils.plots import grid_add_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, required=True, help="path to vae")
    parser.add_argument('--img_left', type=str, required=True,\
        help="path to left interpolated image")
    parser.add_argument('--img_right', type=str, required=True,\
        help="path to right interpolated image")
    parser.add_argument('--step_size', type=float, default=0.1,\
        help="step size of linear interpolation")

    args = parser.parse_args()

    vae = VAE(DIM_LATENT)
    vae.load_state_dict(torch.load(args.vae))

    # load images and sample latent state
    img_left = Image.open(args.img_left)
    x_left = transform(img_left)
    x_left = x_left.unsqueeze(0)
    x_left = x_left.view(1, 3, 64, 64)
    img_right = Image.open(args.img_right)
    x_right = transform(img_right)
    x_right = x_right.unsqueeze(0)
    x_right = x_right.view(1, 3, 64, 64)

    with torch.no_grad():
        mu_left, logvar_left = vae.encode(x_left)
        stdev_left = torch.exp(logvar_left/2.0)
        latent_left = vae.sample(mu_left, stdev_left)

        mu_right, logvar_right = vae.encode(x_right)
        stdev_right = torch.exp(logvar_right/2.0)
        latent_right = vae.sample(mu_right, stdev_right)

        interpolation = linear_interpolation(latent_left, latent_right)

        alphas = np.arange(0, 1+args.step_size, args.step_size)
        n_alphas = len(alphas)
        
        fig = plt.figure()

        grid_add_img(transform_back(x_left.squeeze(0)), fig, 1, n_alphas+2, 1)

        for i in range(n_alphas):
            alpha = alphas[i]
            latent = interpolation(alpha)
            x_rec = vae.decode(latent)
            x_rec = x_rec.squeeze(0)
            img_rec = transform_back(x_rec)

            grid_add_img(img_rec, fig, 1, n_alphas+2, i+1+1)
        
        grid_add_img(transform_back(x_right.squeeze(0)), fig, 1, n_alphas+2, n_alphas+2)

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()