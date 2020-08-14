import torch
import torchvision.transforms as transforms

from utils.hyperparameters import *
from utils.img_transforms import *
from models.vae import *

if __name__ == '__main__':
    print_hyperparameters()
    vae = VAE(DIM_LATENT)
    vae = vae.cuda()

    """
    transform = transforms.Compose([

    ]) """