# library imports
import torch
import torch.nn as nn
import torch.nn.functional as NF
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np
import pytorch_lightning as pl
import math
import random
from argparse import Namespace, ArgumentParser

# utility functions
from model.config import default_options
from .nerf import NeRF
from utils.camera import Camera
from .positional_encoding import PositionalEncoding
from utils.visualizations import to_depth_image

# for debugging purpose; remove later
import matplotlib.pyplot as plt


class NeRFModel(pl.LightningModule):
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(NeRFModel, self).__init__()
        self.save_hyperparameters(hparams)


        # trainables variables
        self.point_encode = PositionalEncoding(6)
        self.dir_encode = PositionalEncoding(4)

        self.nerf = NeRF(self.hparams)
                
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
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
        I = batch['images'][:1]
        K = batch['intrinsics'][0]
        E = batch['extrinsics'][0]
        D = batch['depths'][:1]
        IS = NF.interpolate(I, size=D.shape[-2:])[0].permute(1, 2, 0)
        ranges = batch['ranges']
        camera = Camera(K, E, I.shape[-2:])
        camera.resize(D.shape[-2:])
        N = self.hparams.nerf_num_depths
        pts, depths, dirs = camera.sample(N, ranges)
        pts_encoded = self.point_encode(pts)
        dirs_encoded = self.dir_encode(dirs).repeat(len(pts), 1, 1, 1)

        # 
        PN, PH, PW, PC = pts_encoded.shape
        DN, DH, DW, DC = dirs_encoded.shape

        flat_pts_encoded = pts_encoded.view(-1, PC)
        flat_dirs_encoded = dirs_encoded.view(-1, DC)
        flat_rgb, flat_alpha = self.nerf(flat_pts_encoded, flat_dirs_encoded)
        rgbs = flat_rgb.view(PN, PH, PW, -1)
        alphas = flat_alpha.view(DN, DH, DW, -1)

        inf_rgb, inf_d, inf_acc = camera.render(rgbs, alphas, depths)

        mse = NF.mse_loss(inf_rgb, IS)
        psnr = -10.0 * math.log10(mse.clamp_min(1e-5))
        loss = mse
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)

        # add figure 
        if((batch_idx % 5) == 0):
            gt_d_image = to_depth_image(D[0, 0].unsqueeze(-1))
            inf_d_image = to_depth_image(inf_d)
            inf_acc_image = to_depth_image(inf_acc)
            self.logger.experiment.add_image('train/gt_image', IS.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('train/inf_image', inf_rgb.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('train/inf_d', inf_d_image, self.global_step)
            self.logger.experiment.add_image('train/gt_d', gt_d_image, self.global_step)
            self.logger.experiment.add_image('train/inf_acc', inf_acc_image, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Performes validation. 

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
        I = batch['images'][:1]
        K = batch['intrinsics'][0]
        E = batch['extrinsics'][0]
        D = batch['depths'][:1]

        IS = NF.interpolate(I, size=D.shape[-2:])[0].permute(1, 2, 0)
        ranges = batch['ranges']
        camera = Camera(K, E, I.shape[-2:])
        camera.resize(D.shape[-2:])
        N = self.hparams.nerf_num_depths
        pts, depths, dirs = camera.sample(N, ranges)
        pts_encoded = self.point_encode(pts)
        dirs_encoded = self.dir_encode(dirs).repeat(len(pts), 1, 1, 1)

        # 
        PN, PH, PW, PC = pts_encoded.shape
        DN, DH, DW, DC = dirs_encoded.shape

        flat_pts_encoded = pts_encoded.view(-1, PC)
        flat_dirs_encoded = dirs_encoded.view(-1, DC)
        flat_rgb, flat_alpha = self.nerf(flat_pts_encoded, flat_dirs_encoded)
        rgbs = flat_rgb.view(PN, PH, PW, -1)
        alphas = flat_alpha.view(DN, DH, DW, -1)

        inf_rgb, inf_d, inf_acc = camera.render(rgbs, alphas, depths)

        mse = NF.mse_loss(inf_rgb, IS)
        psnr = -10.0 * math.log10(mse.clamp_min(1e-5))
        loss = mse
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)

        # add figure 
        if((batch_idx % 5) == 0):
            gt_d_image = to_depth_image(D[0, 0].unsqueeze(-1))
            inf_d_image = to_depth_image(inf_d)
            inf_acc_image = to_depth_image(inf_acc)
            self.logger.experiment.add_image('val/gt_image', IS.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('val/inf_image', inf_rgb.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('val/inf_d', inf_d_image, self.global_step)
            self.logger.experiment.add_image('val/gt_d', gt_d_image, self.global_step)
            self.logger.experiment.add_image('val/inf_acc', inf_acc_image, self.global_step)
        return

    def test_step(self, batch, batch_idx):
        return
        