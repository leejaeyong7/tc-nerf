# library imports
from model.propagator import Propagator
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
import time


class NeRFModel(pl.LightningModule):
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(NeRFModel, self).__init__()
        self.save_hyperparameters(hparams)


        # trainables variables
        self.point_encode = PositionalEncoding(hparams.nerf_point_encode)
        self.dir_encode = PositionalEncoding(hparams.nerf_dir_encode)

        self.nerf = NeRF(self.hparams)

        # non-trainable variables
        self.propagator = Propagator(hparams)
                
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
        return optimizer

    def sample(self, N, batch):
        '''
        Given min/max ranges of depths, return sampled depths based on uniform distrb.
        '''

        return pts, nhwd, dirs 

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

        # Bx(3+3+3+2) = Bx(color(rgb), dir(xyz), pos(xyz), range(min_d, max_d))
        dev = batch.device
        B = batch.shape
        rgb = batch[:3]
        dirs = batch[3:6]
        poses = batch[6:9]
        max_d = batch[9:10]
        min_d = batch[10:]

        # Bx1
        sampled_depths = torch.rand(N, dtype=torch.float, device=dev) * (max_d - min_d) + min_d
        pts = poses + sampled_depths * dirs


        # I = batch['images'][:1]
        # K = batch['intrinsics'][0]
        # E = batch['extrinsics'][0]
        # D = batch['depths'][:1].permute(0, 2, 3, 1)
        # IS = NF.interpolate(I, size=D.shape[1:3])[0].permute(1, 2, 0)
        # ranges = batch['ranges']
        # camera = Camera(K, E, I.shape[-2:])
        # camera.resize(D.shape[1:3])
        # N = self.hparams.nerf_num_depths
        # pts, depths, dirs = camera.sample(N, ranges)
        # pts_encoded = self.point_encode(pts)
        # dirs_encoded = self.dir_encode(dirs).repeat(len(pts), 1, 1, 1)

        # # 
        # PN, PH, PW, PC = pts_encoded.shape
        # DN, DH, DW, DC = dirs_encoded.shape

        # flat_pts_encoded = pts_encoded.view(-1, PC)
        # flat_dirs_encoded = dirs_encoded.view(-1, DC)
        # flat_rgb, flat_sigma = self.nerf(flat_pts_encoded, flat_dirs_encoded)
        # rgbs = flat_rgb.view(PN, PH, PW, -1)
        # sigmas = flat_sigma.view(DN, DH, DW, -1)

        # inf_rgb, inf_d, inf_w = camera.render(rgbs, sigmas, depths, True)

        # # NxHxWx1
        # d_sigma = (ranges[1] - ranges[0]) / 256
        # mask = (D > 0)[0]

        # # occ_loss = (1 - (-(inf_d - D[0]).abs() / d_sigma).exp())[mask].mean()
        # # occ_loss = (inf_d - D[0]).abs()[D[0] > 0].mean()
        # # optionally, add ground truth depth for accelerating training
        # gt_occ = NF.one_hot(((D < depths).sum(0)).clamp(0, DN - 1), PN).permute(3, 0, 1, 2).float()

        # # # 1xHxWx1

        # w = inf_w.clamp(1e-5, 1-1e-5)
        # pos_occ = 1 - (w * gt_occ).sum(0)
        # neg_occ = (w * (1- gt_occ)).sum(0)
        # pos_occ_loss = pos_occ[mask].sum() / gt_occ.sum(0)[mask].sum()
        # neg_occ_loss = neg_occ[mask].sum() / (1 - gt_occ).sum(0)[mask].sum()
        # occ_loss = pos_occ_loss + neg_occ_loss
        # # occ_loss = -(sigmas.softmax(0).log() * gt_occ).sum(0)[mask].mean()
        # # occ_loss = NF.binary_cross_entropy(inf_w, gt_occ.float(), reduction='none')[mask.repeat(PN, 1, 1, 1)].mean()

        # mse = NF.mse_loss(inf_rgb, IS)
        # psnr = -10.0 * math.log10(mse.clamp_min(1e-5))
        # loss = mse
        # self.log('train/loss', loss)
        # self.log('train/occ_loss', occ_loss)
        # self.log('train/alpha_max', sigmas.max())
        # self.log('train/alpha_min', sigmas.min())
        # # self.log('train/pos_occ_loss', pos_occ_loss)
        # # self.log('train/neg_occ_loss', neg_occ_loss)
        # self.log('train/mse_loss', mse)
        # self.log('train/max_w', inf_w.max())
        # self.log('train/psnr', psnr)

        # # add figure 
        # if((batch_idx % 5) == 0):
        #     gt_d_image = to_depth_image(D[0])
        #     inf_d_image = to_depth_image(inf_d)
        #     inf_acc_image = to_depth_image(inf_w.sum(0))
        #     gt_acc_image = to_depth_image((inf_w * gt_occ).sum(0))
        #     self.logger.experiment.add_image('train/gt_image', IS.permute(2, 0, 1), self.global_step)
        #     self.logger.experiment.add_image('train/inf_image', inf_rgb.permute(2, 0, 1), self.global_step)
        #     self.logger.experiment.add_image('train/inf_d', inf_d_image, self.global_step)
        #     self.logger.experiment.add_image('train/gt_d', gt_d_image, self.global_step)
        #     self.logger.experiment.add_image('train/inf_acc', inf_acc_image, self.global_step)
        #     self.logger.experiment.add_image('train/gt_acc', gt_acc_image, self.global_step)
        # return loss

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
        self.propagator.kernel = self.propagator.kernel.to(self._device)
        I = batch['images'][:1]
        K = batch['intrinsics'][0]
        E = batch['extrinsics'][0]
        D = batch['depths'][:1]

        IS = NF.interpolate(I, size=D.shape[-2:])[0].permute(1, 2, 0)
        ranges = batch['ranges']
        camera = Camera(K, E, I.shape[-2:])
        camera.resize(D.shape[-2:])
        start = time.time()
        inf_rgb, inf_d, inf_acc =  self.exhaustive_render(camera, ranges)
        end = time.time()
        pm_start = time.time()
        pm_rgb, pm_d, pm_acc =  self.patch_match_render(camera, ranges)
        pm_end = time.time()

        mse = NF.mse_loss(inf_rgb, IS)
        pmmse = NF.mse_loss(pm_rgb, IS)
        psnr = -10.0 * math.log10(mse.clamp_min(1e-5))
        pmpsnr = -10.0 * math.log10(pmmse.clamp_min(1e-5))
        loss = mse
        self.log('val/time_elapsed', end - start)
        self.log('val/pm_time_elapsed', pm_end - pm_start)
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        self.log('val/pm_psnr', pmpsnr)
        self.log('val/pm_loss', pmmse)

        # add figure 
        if((batch_idx % 5) == 0):
            gt_d_image = to_depth_image(D[0, 0].unsqueeze(-1))
            inf_d_image = to_depth_image(inf_d)
            inf_acc_image = to_depth_image(inf_acc.sum(0))
            pm_d_image = to_depth_image(pm_d)
            pm_acc_image = to_depth_image(pm_acc.sum(0))
            self.logger.experiment.add_image('val/gt_image', IS.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('val/gt_d', gt_d_image, self.global_step)
            self.logger.experiment.add_image('val/inf_image', inf_rgb.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('val/inf_d', inf_d_image, self.global_step)
            self.logger.experiment.add_image('val/inf_acc', inf_acc_image, self.global_step)
            self.logger.experiment.add_image('val/pm_image', pm_rgb.permute(2, 0, 1), self.global_step)
            self.logger.experiment.add_image('val/pm_d', pm_d_image, self.global_step)
            self.logger.experiment.add_image('val/pm_acc', pm_acc_image, self.global_step)
        return

    def exhaustive_render(self, camera, ranges):
        N = self.hparams.nerf_num_depths
        # render using exhastive search
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
        sigmas = flat_alpha.view(DN, DH, DW, -1)

        inf_rgb, inf_d, inf_acc = camera.render(rgbs, sigmas, depths)
        return inf_rgb, inf_d, inf_acc

    def patch_match_render(self, camera, ranges):
        # N = self.hparams.nerf_num_depths
        N = 1

        depths, dirs = camera.init(1, ranges)
        # pts = NxHxWx3
        # depths = NxHxWx1
        # dirs = 1xHxWx3

        # encoded dirs are shared all the time
        dirs_encoded = self.dir_encode(dirs)
        for i in range(8):
            # propagate current depths = Nx1xHxWx1 => NPxHxWx1
            prop_d = self.propagator(depths)

            # convert them to points = NPxHxWx1 => NPxHxWx3
            prop_pts = camera.back_project(prop_d)
            prop_pts_encoded = self.point_encode(prop_pts)
            PNP, PH, PW, PC = prop_pts_encoded.shape

            # test out hypothesis
            flat_prop_pts = prop_pts_encoded.view(-1, PC)
            flat_alpha = self.nerf.forward_points(flat_prop_pts)
            alphas = flat_alpha.view(PNP, PH, PW, -1)

            # replace depths
            top_alphas = alphas.topk(dim=0, k=N).indices
            depths = prop_d.gather(0, top_alphas)


        # 
        pts = camera.back_project(depths)
        pts_encoded = self.point_encode(pts)
        PN, PH, PW, PC = pts_encoded.shape
        DN, DH, DW, DC = dirs_encoded.shape

        flat_pts_encoded = pts_encoded.view(-1, PC)
        flat_dirs_encoded = dirs_encoded.view(-1, DC)
        flat_rgb, flat_alpha = self.nerf(flat_pts_encoded, flat_dirs_encoded)
        rgbs = flat_rgb.view(PN, PH, PW, -1)
        sigmas = flat_alpha.view(DN, DH, DW, -1)

        inf_rgb, inf_d, inf_acc = camera.render(rgbs, sigmas, depths)
        return inf_rgb, inf_d, inf_acc


    def test_step(self, batch, batch_idx):
        return
        