import numpy as np
import torch

from PIL import Image


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

def transform_scale(scale=255.0):
    def _scale(x):
        x = (x / scale)
        return x
    
    return _scale

def transform_to_np():
    def _to_np(x):
        x = np.array(x).astype('float32')
        return x
    
    return _to_np

def transform_to_tensor():
    def _to_tensor(x):
        x = torch.tensor(x)
        return x
    
    return _to_tensor