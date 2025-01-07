# dataloaders/transforms.py

import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as tr_F

class RandomMirror(object):
    """
    Randomly flip the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = label
        return sample

class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()

        sample['image'] = img
        sample['label'] = label
        return sample

from PIL import ImageFilter

class DilateScribble(object):
    """
    Dilation of the mask (scribble annotations).
    Args:
        size: the size by which the mask will be dilated
    """

    def __init__(self, size=1):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        # Assuming 'label' is a binary mask or a list of masks.
        if isinstance(label, dict):
            for catId, mask in label.items():
                # Perform dilation on each mask (binary mask)
                label[catId] = mask.filter(ImageFilter.MaxFilter(self.size))
        else:
            label = label.filter(ImageFilter.MaxFilter(self.size))

        sample['image'] = img
        sample['label'] = label
        return sample
