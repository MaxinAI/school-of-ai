"""
Created on Apr 23, 2020

Network model for handwritten character recognition

@author: Levan Tsinadze
"""

import numpy as np
from collections import OrderedDict
from PIL.Image import Image
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import (transforms, datasets)
from fastai.vision import *
from path_utils import data_path

tfms = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])

path = data_path()
image_path = path / 'geomnist_dataset'


def img_loader(img_path: str):
    with open(img_path, mode='rb') as fl:
        with PIL.Image.open(fl) as img:
            return img.convert('L')


train_dataset = datasets.ImageFolder(str(image_path / 'train_geo'), loader=img_loader, transform=tfms)
valid_dataset = datasets.ImageFolder(str(image_path / 'val_geo'), loader=img_loader, transform=tfms)
test_dataset = datasets.ImageFolder(str(image_path / 'test_geo'), loader=img_loader, transform=tfms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

data = DataBunch(train_loader, valid_loader, test_dl=test_loader)


class FlattenLayer(nn.Module):
    """Flatten layer"""

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.flatten(x, 1)


def conv2(ni: int, nf: int):
    return conv_layer(ni, nf, stride=2)


model = nn.Sequential(
    conv2(1, 8),  # 14
    conv2(8, 16),  # 7
    conv2(16, 32),  # 4
    conv2(32, 16),  # 2
    conv2(16, 33),  # 1
    nn.AdaptiveAvgPool2d((1, 1)),
    FlattenLayer()  # remove (1,1) grid
)

sz = 32
x_test = torch.randn(4, 1, sz, sz)

model(x_test)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.1)
