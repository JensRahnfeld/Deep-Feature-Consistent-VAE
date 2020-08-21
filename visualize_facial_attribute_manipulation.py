import argparse
import csv
import numpy as np
import os
import torch

from PIL import Image
from models.vae import VAE
from utils.hyperparameters import DIM_LATENT
from utils.img_transforms import transform, transform_back


attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',\
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',\
    'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',\
    'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',\
    'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes',\
    'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',\
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',\
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',\
    'Wearing_Necktie', 'Young']


def mean_latent(vae, file_names, imgdir):
    latent_res = torch.zeros(DIM_LATENT)
    n = len(file_names)

    with torch.no_grad():
        for file_name in file_names:
            path = os.path.join(args.imgdir, file_name)
            img = Image.open(path)
            x = transform(img)
            x = x.unsqueeze(0)
            mean, logvar = vae.encode(x)
            latent = vae.sample(mean, torch.exp(logvar/2.0))
            latent = latent.squeeze(0)

            latent_res += latent
    
    latent_res /= n

    return latent_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae', type=str, required=True, help="path to vae")
    parser.add_argument('--list_attr', type=str, required=True,\
        help="path to list_attr_celeba.csv")
    parser.add_argument('--attr', type=str, required=True, choices=attributes,\
        help="attribute used")
    parser.add_argument('--imgdir', type=str, required=True,\
        help="path to image folder")
    parser.add_argument('--size', type=int, default=1000, help="number of images")

    args = parser.parse_args()

    vae = VAE(DIM_LATENT)
    vae.load_state_dict(torch.load(args.vae))

    files_pos = []
    files_neg = []

    csvfile = open(args.list_attr, 'r')
    reader = csv.DictReader(csvfile)

    for row in reader:
        if row[args.attr] == '1':
            files_pos.append(row['image_id'])
        elif row[args.attr] == '-1':
            files_neg.append(row['image_id'])
        else: raise ValueError("found non-matching label")
    
    files_pos = np.random.choice(files_pos, args.size, replace=False)
    files_neg = np.random.choice(files_neg, args.size, replace=False)

    latent_pos = mean_latent(vae, files_pos, args.imgdir)
    latent_neg = mean_latent(vae, files_neg, args.imgdir)