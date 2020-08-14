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
from utils.loss_functions import kl_loss, vgg123_loss, vgg345_loss
from utils.img_transforms import transform_crop, transform_resize, transform_scale,\
    transform_to_np, transform_to_tensor
from models.vae import VAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgdir', type=str, required=True,\
        help="path to image folder")
    parser.add_argument('--logdir', type=str, default="./tensorboard",\
        help="path to tensorboard logdir")
    parser.add_argument('--workers', type=int, default=8, help="number of workers")
    parser.add_argument('--modeldir', type=str, default="./trained_models",\
        help="path to folder saving trained model")
    parser.add_argument('-o', type=str, default="vae", help="name of model")
    
    args = parser.parse_args()

    print_hyperparameters()
    
    vae = VAE(DIM_LATENT)
    if torch.cuda.is_available(): vae = vae.cuda()


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
    if not os.path.exists(args.modeldir): os.mkdir(args.modeldir)

    writer = SummaryWriter(args.logdir)

    optimizer = optim.Adam(vae.parameters(), lr=LEARNING_RATE)
    def lr_decay(n): return WEIGHT_DECAY**n
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_decay)

    t = 0

    for n in range(EPOCHS):
        print("epoch: {0}".format(n))
        
        for batch in loader:
            sys.stdout.write("\r")
            sys.stdout.write("progress: {0}/{1}".format(t % len(loader), len(loader)))
            sys.stdout.flush()
            
            x_train, _ = batch
            x_train = x_train.cuda()
            x_train = x_train.view(-1, 3, 64, 64)

            x_rec, mu, logvar = vae(x_train)

            dist_loss = kl_loss(mu, logvar, BETA)
            rec_loss = vgg123_loss(x_rec, x_train, ALPHA)
            loss = dist_loss + rec_loss

            writer.add_scalar("train / kl loss", dist_loss, t)
            writer.add_scalar("train / rec loss", rec_loss, t)
            writer.add_scalar("train / total loss", loss, t)

            loss.backward()
            optimizer.step()

            t += 1

        scheduler.step()

        file_name = os.path.join(args.modeldir, args.o + str(n) + ".tmp" ".pt")
        torch.save(vae.state_dict(), file_name)

        sys.stdout.write("\n")
        sys.stdout.flush()
    
    file_name = os.path.join(args.modeldir, args.o + ".pt")
    torch.save(vae.state_dict(), file_name)

    writer.close()