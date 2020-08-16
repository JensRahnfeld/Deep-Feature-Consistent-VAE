import argparse
import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils.hyperparameters import DIM_LATENT, CROP_LEFT, CROP_RIGHT, CROP_UPPER,\
    CROP_LOWER, RESIZE_HEIGHT, RESIZE_WIDTH
from utils.img_transforms import transform_crop, transform_resize, transform_scale,\
    transform_to_np, transform_to_tensor
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

    transform = transforms.Compose([
        transform_crop(CROP_LEFT, CROP_UPPER, CROP_RIGHT, CROP_LOWER),
        transform_resize(RESIZE_HEIGHT, RESIZE_WIDTH),
        transform_to_np(),
        transform_scale(255.0),
        transform_to_tensor()
    ])

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

            fig.add_subplot(2, n_imgs, i+1)
            plt.imshow(img_true)
            plt.axis('off')

            fig.add_subplot(2, n_imgs, n_imgs+i+1)
            plt.imshow(img_rec)
            plt.axis('off')
    
    plt.show()