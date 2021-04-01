# library imports
from utils.transforms import to_bchw, to_bhwc
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
        B = batch.shape[0]
        rgb = batch[:, :3].view(B, 3)
        dirs = batch[:, 3:6].view(B, 1, 3)
        poses = batch[:, 6:9].view(B, 1, 3)
        min_d = batch[:, 9:10].view(B, 1, 1)
        max_d = batch[:, 10:].view(B, 1, 1)

        # Bx1
        FN = self.hparams.nerf_num_fine_depths
        CN = self.hparams.nerf_num_coarse_depths
        #################
        # coarse nerf
        c_steps = torch.linspace(0, 1, CN+1, device=dev).view(1, -1, 1)
        c_depths = 1 / (1 / min_d * (1 - c_steps) + 1 / max_d *  c_steps)
        min_ds = c_depths[:, :-1]
        max_ds = c_depths[:, 1:]
        c_ds = torch.rand_like(min_ds) * (max_ds - min_ds) + min_ds

        c_pts = poses + c_ds * dirs

        # BxPC, BxDC
        c_pts_encoded = self.point_encode(c_pts)
        dirs_encoded = self.dir_encode(dirs)
        PC = c_pts_encoded.shape[-1]
        DC = dirs_encoded.shape[-1]

        flat_c_pts_encoded = c_pts_encoded.view(-1, PC)
        flat_c_dirs_encoded = dirs_encoded.repeat(1, CN, 1).view(-1, DC)
        c_cand_rgb, c_cand_alpha = self.nerf(flat_c_pts_encoded, flat_c_dirs_encoded)
        c_cand_rgb = c_cand_rgb.view(-1, CN, 3)
        c_cand_alpha = c_cand_alpha.view(-1, CN, 1)

        # BxNxC => BxC
        c_inf_rgb, c_inf_d, c_inf_w = self.render(c_cand_rgb, c_cand_alpha, c_ds , True)

        ##################
        # fine nerf
        mid_depths = (c_ds[:, 1:] + c_ds[:, :-1]) * 0.5
        f_ds = self.sample_pdf(mid_depths, c_inf_w[:, 1:-1].detach(), FN, det=False)
        f_pts = poses + f_ds * dirs

        f_pts_encoded = self.point_encode(f_pts)
        flat_f_pts_encoded = f_pts_encoded.view(-1, PC)
        flat_f_dirs_encoded = dirs_encoded.repeat(1, FN, 1).view(-1, DC)
        f_cand_rgb, f_cand_alpha = self.nerf(flat_f_pts_encoded, flat_f_dirs_encoded)
        f_cand_rgb = f_cand_rgb.view(-1, FN, 3)
        f_cand_alpha = f_cand_alpha.view(-1, FN, 1)
        f_inf_rgb, f_inf_d, c_inf_w = self.render(f_cand_rgb, f_cand_alpha, f_ds , True)

        # NxHxWx1
        c_mse = NF.mse_loss(c_inf_rgb, rgb)
        f_mse = NF.mse_loss(f_inf_rgb, rgb)
        c_psnr = -10.0 * math.log10(c_mse)
        f_psnr = -10.0 * math.log10(f_mse)
        loss = c_mse + f_mse
        self.log('train/loss', loss)
        self.log('train/coarse_loss', c_mse)
        self.log('train/fine_loss', f_mse)
        self.log('train/coarse_psnr', c_psnr)
        self.log('train/fine_psnr', f_psnr)
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
        I = batch['images']
        rgb = I.permute(1, 2, 0)
        K = batch['intrinsics']
        E = batch['extrinsics']
        H, W = I.shape[-2:]

        ranges = batch['ranges']
        camera = Camera(K, E, (H, W))
        torch.cuda.synchronize()
        start = time.time()
        inf_rgb, inf_d, inf_op =  self.exhaustive_render(camera, ranges)
        torch.cuda.synchronize()
        end = time.time()

        torch.cuda.empty_cache()

        pm_start = time.time()
        pm_rgb, pm_d, pm_op = self.patch_match_render(camera, ranges)
        torch.cuda.synchronize()
        pm_end = time.time()

        mse = NF.mse_loss(inf_rgb, rgb)
        pmmse = NF.mse_loss(pm_rgb, rgb)
        psnr = -10.0 * math.log10(mse)
        pmpsnr = -10.0 * math.log10(pmmse)
        loss = mse
        self.log('val/time_elapsed', end - start)
        self.log('val/pm_time_elapsed', pm_end - pm_start)
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        self.log('val/pm_psnr', pmpsnr)
        self.log('val/pm_loss', pmmse)

        # add figure 
        inf_d_image = to_depth_image(inf_d)
        inf_acc_image = to_depth_image(inf_op)
        pm_d_image = to_depth_image(pm_d)
        pm_acc_image = to_depth_image(pm_op)
        self.logger.experiment.add_image('val/gt_image', I, self.global_step)
        self.logger.experiment.add_image('val/inf_image', inf_rgb.permute(2, 0, 1), self.global_step)
        self.logger.experiment.add_image('val/inf_d', inf_d_image, self.global_step)
        self.logger.experiment.add_image('val/inf_op', inf_acc_image, self.global_step)
        self.logger.experiment.add_image('val/pm_image', pm_rgb.permute(2, 0, 1), self.global_step)
        self.logger.experiment.add_image('val/pm_d', pm_d_image, self.global_step)
        self.logger.experiment.add_image('val/pm_acc', pm_acc_image, self.global_step)
        return

    def exhaustive_render(self, camera, ranges):
        CN = self.hparams.nerf_num_coarse_depths
        FN = self.hparams.nerf_num_fine_depths
        H, W = camera.H, camera.W
        # render using exhastive search
        c_pts, c_depths, dirs = camera.sample(CN, ranges)
        dirs_encoded = self.dir_encode(dirs).view(H*W, -1)

        c_sigmas = []
        for i in range(CN):
            c_pts_encoded = self.point_encode(c_pts[i]).view(H*W, -1)
            flat_c_alpha = self.nerf.forward_points(c_pts_encoded)
            c_sigmas.append(flat_c_alpha)

        c_sigmas = torch.stack(c_sigmas, 1)
        c_ds = c_depths.view(CN, -1, 1).transpose(0, 1)
        mid_depths = (c_ds[:, 1:] + c_ds[:, :-1]) * 0.5

        # compute weights for coarse system
        c_inf_w = self.compute_weights(c_sigmas, c_ds)

        c_sigmas = None
        c_pts = None
        c_depths = None
        c_ds = None
        torch.cuda.empty_cache()

        # ()xFN
        f_depths = []
        chunks = 1024
        for b in range(0, len(mid_depths), chunks):
            f_d_chunk= self.sample_pdf(mid_depths[b:b+chunks], c_inf_w[b:b+chunks, 1:-1].detach(), FN, det=False)
            f_depths.append(f_d_chunk)
        c_inf_w = None

        f_depths = torch.cat(f_depths).transpose(0, 1).view(FN, H, W, 1)
        f_pts = camera.back_project(f_depths)

        f_sigmas = []
        f_colors = []
        for i in range(FN):
            f_pts_encoded = self.point_encode(f_pts[i]).view(H*W, -1)
            flat_f_color, flat_f_alpha = self.nerf(f_pts_encoded, dirs_encoded)
            f_colors.append(flat_f_color)
            f_sigmas.append(flat_f_alpha)
        f_sigmas = torch.stack(f_sigmas, 1)
        f_colors = torch.stack(f_colors, 1)
        f_ds = f_depths.view(FN, -1, 1).transpose(0, 1)
        f_rgb, f_d, f_w = self.render(f_colors, f_sigmas, f_ds)
        f_op = f_w.sum(1)

        return f_rgb.view(H, W, 3), f_d.view(H, W, 1), f_op.view(H, W, 1)

    def patch_match_render(self, camera, ranges):
        # N = self.hparams.nerf_num_depths
        N = 1

        # depths, dirs = camera.init(1, ranges)
        # pts = NxHxWx3
        # depths = NxHxWx1
        # dirs = 1xHxWx3

        # encoded dirs are shared all the time
        iters = {
            3: 8,
            2: 2,
            1: 2,
            0: 1
        }
        H, W = camera.H, camera.W
        self.propagator.kernel = self.propagator.kernel.to(self._device)
        max_octave = max(iters.keys())
        min_octave = min(iters.keys())
        min_d, max_d = ranges
        for octave in reversed(sorted(iters.keys())):
            iter_count = iters[octave]
            scale = 0.5 ** octave
            camera.resize((int(H * scale), int(W * scale)))
            if(octave == max_octave):
                depths, dirs = camera.init(N, ranges)

            dirs_encoded = self.dir_encode(dirs)
            for i in range(iter_count):
                up_d = []
                # propagate current depths = NxHxWx1 => PxNxHxWx1
                prop_ds = self.propagator(depths, ranges)
                for prop_d in prop_ds:
                    alphas_list = []
                    for d in prop_d.unsqueeze(1):
                        # convert them to points = NPxHxWx1 => NPxHxWx3
                        prop_pts = camera.back_project(d)
                        prop_pts_encoded = self.point_encode(prop_pts)
                        PNP, PH, PW, PC = prop_pts_encoded.shape

                        # test out hypothesis
                        flat_prop_pts = prop_pts_encoded.view(-1, PC)
                        flat_alpha = self.nerf.forward_points(flat_prop_pts)
                        alpha = flat_alpha.view(PNP, PH, PW, -1)
                        alphas_list.append(alpha)
                    alphas = torch.cat(alphas_list)

                    # replace depths
                    top_alphas = alphas.topk(dim=0, k=1).indices
                    valid_alphas = (alphas > 5)
                    top_valid_alphas = valid_alphas.float().topk(dim=0, k=1).indices
                    good_updates = prop_d.gather(0, top_valid_alphas)
                    any_valid_alphas = valid_alphas.any(0, keepdim=True)
                    updated = prop_d.gather(0, top_alphas)
                    updated[any_valid_alphas] = good_updates[any_valid_alphas]
                    up_d.append(updated)

                depths = torch.cat(up_d, 0)
                

            # upsample depth / dirs
            if(octave != min_octave):
                depths = to_bhwc(NF.interpolate(to_bchw(depths), scale_factor=2))
                dirs = to_bhwc(NF.interpolate(to_bchw(dirs), scale_factor=2))

        # 
        depths = depths.sort(0).values
        pts = camera.back_project(depths)
        rgbs = []
        sigmas = []
        for i in range(N):
            pts_encoded = self.point_encode(pts[i]).view(H*W, -1)
            flat_rgb, flat_alpha = self.nerf(pts_encoded, dirs_encoded.view(H*W, -1))
            rgbs.append(flat_rgb)
            sigmas.append(flat_alpha)
        rgbs = torch.stack(rgbs, 1)
        sigmas= torch.stack(sigmas, 1)
        ds = depths.view(N, -1, 1).transpose(0, 1)

        inf_rgb, inf_d, inf_w = self.render(rgbs, sigmas, ds)
        inf_op = inf_w.sum(1)
        return inf_rgb.view(H, W, 3), inf_d.view(H, W, 1), inf_op.view(H, W, 1)

        
    def compute_weights(self, sigma, depths, add_noise=False):
        noise_std = 1
        if(add_noise):
            noise = torch.randn_like(sigma) * noise_std
        else:
            noise = 0.0
        sig = NF.relu(sigma + noise)
        one_e_10 = torch.ones_like(depths[:, :1]) * 1e10
        dists = torch.cat((depths[:, 1:] - depths[:, :-1], one_e_10), 1)
        alpha = 1 - (-sig * dists).exp()
        alphas_shifted = torch.cat((torch.ones_like(alpha[:, :1]), 1-alpha+1e-10), 1)
        w = alpha * torch.cumprod(alphas_shifted, 1)[:, :-1]
        return w

    def render(self, rgb, sigma, depths, add_noise=False):
        # sigma = NxHxWx1
        # depths = NxHxWx1
        # rgb = NxHxWx3
        w = self.compute_weights(sigma, depths, add_noise)
        rgb_map = (w * rgb).sum(1)
        depth_map = (w * depths).sum(1)
        return rgb_map, depth_map, w

    def test_step(self, batch, batch_idx):
        return
        
    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.

        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero

        Outputs:
            samples: the sampled samples
        """
        bins = bins.squeeze(-1)
        weights = weights.squeeze(-1)
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, 1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], 1)  # (N_rays, N_samples_+1) 
                                                                # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                            # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples.unsqueeze(-1).sort(1).values
