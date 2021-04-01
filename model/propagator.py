import torch
import torch.nn as nn
import time
import torch.nn.functional as NF

from utils.transforms import *

class Propagator(nn.Module):
    def __init__(self, hparams=None):
        super(Propagator, self).__init__()
        self.hparams = hparams

        prop_shape = self.define_shape()

        # register buffers
        self.pad = self.create_pad(prop_shape)
        # self.register_buffer('kernel', self.create_kernel(prop_shape, 1))
        self.kernel = self.create_kernel(prop_shape, 1)
        # self.register_buffer('kernel', self.create_kernel(prop_shape, 1))

    def define_shape(self):
        return torch.FloatTensor([
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0
        ]).view(11, 11)

    def create_pad(self, prop_shape):
        '''
        Given prop shape of KxK pixels, creates padding that makes the propagated
        shapes to be identical to original size.
        '''
        prop_size = prop_shape.shape[-1]
        return nn.ReflectionPad2d(prop_size // 2)

    def create_kernel(self, prop_shape, C=1):
        '''
        Given prop shape of KxK pixels, 
        creates kernel of shape Nx1xKxK that is used as convolutional filter
        '''
        KH, KW = prop_shape.shape
        valid_cells = prop_shape.nonzero()
        S = len(valid_cells)
        kernel = torch.zeros((S*C, C, KH, KW)).float()
        for i, valid_cell in enumerate(valid_cells):
            for d in range(C):
                kernel[C*i + d, d, valid_cell[1], valid_cell[0]] = 1
        return kernel


    def forward(self, propagatable, ranges, num_perts=3):
        '''
        Given 1xHxWxK propagatable geometry, 
        propagate them to SxHxWxK neighbors
        NxHxWx1 => (NxP)xHxWx1 => (NxA)xHxWx1
        '''
        p = to_bchw(propagatable)
        dev = propagatable.device
        N, H, W = propagatable.shape[:3]
        # padded plane => 1x4xWxH => 1x1x4xHxW
        padded = self.pad(p)

        # returns 1xS4xHxW => Sx4xHxW
        propagated = NF.conv2d(padded, self.kernel).view(N, -1, H, W, 1)

        min_d, max_d = ranges
        # d_ranges= (max_d - min_d) / torch.arange(N+1, device=dev).view(-1, 1, 1, 1, 1).float()

        pert_scale = (max_d - min_d) / 256
        d_pert_scale = pert_scale * 3 * 2 ** -torch.arange(num_perts, device=dev).view(1, -1, 1, 1, 1).float()
        d_perts = d_pert_scale * (torch.rand((N, num_perts, H, W, 1), device=dev) * 2 - 1)
        pert_d = (propagatable.unsqueeze(1) + d_perts).clamp(min_d, max_d)

        # ensure good views for all propagated values
        # NxPxHxWx1
        return torch.cat((propagated, pert_d), 1).sort(1).values
