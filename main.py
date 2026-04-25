import os
import sys

import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt



# Here is a starting code for you to load the image dataset

train_transform = torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize([0.229, 0.224, 0.225],  [0.485, 0.456, 0.406])
                                            ])
test_transform = torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.229, 0.224, 0.225],  [0.485, 0.456, 0.406])
                                            ])

train_dataset = torchvision.datasets.ImageFolder(root='./data/image/train', transform=train_transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                               shuffle=True, num_workers=2)

dev_dataset = torchvision.datasets.ImageFolder(root='./data/image/test',
                                               transform=test_transform)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=128,
                                             shuffle=False, num_workers=2)

for batch_num, (feats, labels) in enumerate(train_dataloader):
    print(feats.shape)
    print(labels)
    break
