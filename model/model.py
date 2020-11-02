# library imports
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np
import pytorch_lightning as pl
import random
from argparse import Namespace, ArgumentParser

# utility functions
from model.config import default_options
from .nerf import NeRF

# for debugging purpose; remove later
import matplotlib.pyplot as plt


class NeRFModel(pl.LightningModule):
    def __init__(self, 
                 conf: Namespace,
                 *args, **kwargs
    ):
        super(NeRFModel, self).__init__()
        self.save_hyperparameters(conf)


        # trainables variables
        self.coarse = NeRF(hparams)
        self.fine = NeRF(hparams)
                
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            parser.add_argument('--{}'.format(name), **args)
        return parser

    def __repr__(self):
        return repr(self.hparams)

    # -- meta learning setups -- 
    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam
        elif(self.hparams.optimizer == 'Ranger'):
            opt = Ranger
        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                                   list(range(100, 100+self.hparams.scheduler_last_epoch)),
                                                   self.hparams.scheduler_rate)

        return [optimizer], [scheduler]

    def forward(self, points, view):
        '''
        Args:
            points: Bx3 shaped tensor representing positions
            View: Bx2 shaped tensor representing viewing angles
                Viewing angles are theta and rho
        Returns:
            Colors: Bx3 shaped tensor representing RGB
            Density: Bx1 shaped tensor representing occupancy
        '''
    def training_step(self, batch, batch_idx):
        '''
        Performes training. 

        1. sample points from images based on ranges, intrinsic, extrinsics
          - this gives direction and positions
        2. perform NeRF pass based on points and directions
        3. compute loss based on projected points

        Args:
            - Batch: Batch contains images, intrinsics, extrinsics, depths, ranges
              images: NxCxHxW images
              intrinsics: Nx3x3 camera intrinsics
              extrinsics: Nx4x4 camera extrinsics
              depths: Nx1xHxW depths
              ranges: [min_d, max_d]
        '''
        return

    def validation_step(self, batch, batch_idx):
        '''
        Batch contains images, intrinsics, extrinsics, depths
        '''
        return

    def test_step(self, batch, batch_idx):
        return