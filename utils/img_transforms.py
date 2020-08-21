import numpy as np
import torch
import torchvision.transforms as transforms

from utils.hyperparameters import *


def transform_crop(left, upper, right, lower):
    def _crop(img):
        img = img.crop((left, upper, right, lower))
        return img
    
    return _crop

def transform_resize(height, width):
    def _resize(img):
        img = img.resize((height, width))
        return img

    return _resize

def transform_denormalize(mean, stdev):
    def _denormalize(tensor):
        dtype = tensor.dtype
        mean_t = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        mean_t = mean_t[:, None, None]
        stdev_t = torch.as_tensor(stdev, dtype=dtype, device=tensor.device)
        stdev_t = stdev_t[:, None, None]
        tensor = (tensor * stdev_t) + mean_t
        
        return tensor

    return _denormalize

transform = transforms.Compose([
        transform_crop(CROP_LEFT, CROP_UPPER, CROP_RIGHT, CROP_LOWER),
        transform_resize(RESIZE_HEIGHT, RESIZE_WIDTH),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STDEV)
    ])

transform_back = transforms.Compose([
        transform_denormalize(NORMALIZE_MEAN, NORMALIZE_STDEV),
        transforms.ToPILImage()
    ])