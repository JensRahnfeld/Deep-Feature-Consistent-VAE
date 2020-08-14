import argparse
import os
import sys
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from utils.hyperparameters import *
from utils.loss_functions import *
from utils.img_transforms import *
from models.vae import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', type=str, required=True,\
        help="path to image folder")
    parser.add_argument('--logdir', type=str, default="./tensorboard",\
        help="path to tensorboard logdir")
    parser.add_argument('--workers', type=int, default=8, help="number of workers")
    
    args = parser.parse_args()

    print_hyperparameters()
    
    vae = VAE(DIM_LATENT)
    vae = vae.cuda()


    transform = transforms.Compose([
        transform_crop(CROP_LEFT, CROP_UPPER, CROP_RIGHT, CROP_LOWER),
        transform_resize(RESIZE_HEIGHT, RESIZE_WIDTH),
        transform_to_np(),
        transform_scale(255.0),
        transform_to_tensor()
    ])

    dataset = ImageFolder(args.imgdir, transform=transform)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=args.workers,\
        drop_last=True)

    if not os.path.exists(args.logdir): os.mkdir(args.logdir)

    writer = SummaryWriter(args.logdir)

    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    def lr_decay(n): return WEIGHT_DECAY**n
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_decay)

    for n in range(EPOCHS):
        print("epoch: {0}".format(n))
        
        t = 0
        for batch in loader:
            sys.stdout.write("\r")
            sys.stdout.write("progress: {0}/{1}".format(t, len(loader)))
            sys.stdout.flush()
            
            x_train, _ = batch
            x_train = x_train.cuda()
            x_train = x_train.view(-1, 3, 64, 64)

            x_rec, mu, logvar = vae(x_train)

            dist_loss = kl_loss(mu, logvar, BETA)
            loss = dist_loss

            loss.backward()
            optimizer.step()

            t += 1

        scheduler.step()

        sys.stdout.write("\n")
        sys.stdout.flush()