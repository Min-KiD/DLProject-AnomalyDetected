import numpy as np
import pandas as pd
import cv2
import os
import random
from glob import glob

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

from pytorchvideo.data import LabeledVideoDataset, make_clip_sampler, labeled_video_dataset
from pytorch_lightning import LightningDataModule
import torch

from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
    UniformTemporalSubsample,
    Permute
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)

import math

from sklearn.metrics import RocCurveDisplay, roc_curve, auc

def plot_roc(fpr, tpr):
    plt.figure()
    lw = 2 # linewidth
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='pink', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")

    print('ROC curve (area = %0.2f)' % roc_auc)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        vid_mod = torch.hub.load('facebookresearch/pytorchvideo', model='efficient_x3d_xs', pretrained=True)
        self.vid_model = vid_mod
        self.relu = nn.ReLU()
        self.linear = nn.Linear(400, 1)
        #parameters
        self.lr=1e-3
        self.batch_size = 2
        #loss function
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, x):
        x = self.vid_model(x)
        x = self.relu(x)
        x = self.linear(x)
        return x
        
    def training_step(self, batch, batch_idx):
        y_hat = self(batch['video'])
        loss = self.loss(y_hat, batch['label'])
        self.log("train_loss", loss, sync_dist=True)
                return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch['video'])
        loss = self.loss(y_hat, batch['label'])
        self.log("val_loss", loss, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        return {'optimizer':opt, 'lr_scheduler':scheduler}

class MyDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.BATCH_SIZE = 1
        self.NUM_WORKERS = 0
    
    def train_dataloader(self):
        train_dataset = labeled_video_dataset(
            '/kaggle/working/train.txt',
            clip_sampler=make_clip_sampler('random', 25.6),
            transform=video_transform,
            decode_audio=False
        )
        train_loader = DataLoader(train_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, shuffle=False)
        return train_loader
    def val_dataloader(self):
        val_dataset = labeled_video_dataset(
            '/kaggle/working/val.txt',
            clip_sampler=make_clip_sampler('random', 25.6),
            transform=video_transform,
            decode_audio=False
        )
        val_loader = DataLoader(val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS, shuffle=False)
        return val_loader
    
    def test_dataloader(self):
        test_dataset = labeled_video_dataset(
            '/kaggle/working/val.txt',
            clip_sampler=make_clip_sampler('random', 25.6),
            transform=video_transform,
            decode_audio=False
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        return test_loader

def main():
    random_seed = 13
    np.random.seed(random_seed)
    random.seed(random_seed)

    video_model = torch.hub.load('facebookresearch/pytorchvideo', model='efficient_x3d_xs', pretrained=True)

    video_transform=Compose([
        ApplyTransformToKey(
            key='video',
            transform = Compose([
            UniformTemporalSubsample(64),
            Lambda(lambda x: x/255),
            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
            ShortSideScale(60),
            RandomHorizontalFlip(p=0.5)
            ])
        ),
        ApplyTransformToKey(
            key='label',
            transform = Lambda(lambda x: torch.tensor([float(x)]))
        )
    ])

    dm = iter(MyDataModule().val_dataloader())
    result = []
    label = []

    while True:
        batch = next(dm, False)
        if(batch == False):
            break
        x = batch['video']
        label.append(batch['label'].detach().numpy()[0][0])
        result.append(model(x).detach().numpy()[0][0])

    sigmoid_v = np.vectorize(sigmoid)

    result = np.array(result)
    pred = sigmoid_v(result)
    lab = np.array(label)
    x = pd.DataFrame(zip(pred, lab), columns=['pred', 'label'])

    fpr, tpr, thresholds = roc_curve(lab, pred)
    plot_roc(fpr, tpr)
    