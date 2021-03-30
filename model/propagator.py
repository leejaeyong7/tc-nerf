import torch
import torch.nn as nn
import torch.nn.functional as NF

def from_homogeneous(points):
    return points[:, :, :, :-1] / points[:, :, :, -1:]

def to_homogeneous(points):
    dims = list(points.shape)
    dims[3] = 1
    ones = torch.ones(tuple(dims), dtype=points.dtype, device=points.device)
    return torch.cat((points, ones), 3)

def to_vector(points):
    return points.squeeze(-1)

def from_vector(points):
    return points.unsqueeze(-1)

def to_bchw(bhwc):
    return bhwc.permute(0, 3, 1, 2)

def to_bhwc(bchw):
    return bchw.permute(0, 2, 3, 1)

def resize_bhwc(tensor, size, mode='nearest'):
    return to_bhwc(NF.interpolate(to_bchw(tensor), size=size, mode=mode))

class Propagator(nn.Module):
    def __init__(self, hparams=None):
        super(Propagator, self).__init__()
        self.hparams = hparams

        prop_shape = self.define_shape()

        # register buffers
        self.pad = self.create_pad(prop_shape)
        self.kernel = self.create_kernel(prop_shape, 1)

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


    def forward(self, propagatable):
        '''
        Given 1xHxWxK propagatable geometry, 
        propagate them to SxHxWxK neighbors
        '''
        p = to_bchw(propagatable)
        H, W = p.shape[-2:]
        # padded plane => 1x4xWxH => 1x1x4xHxW
        padded = self.pad(p)
        C = 1

        # returns 1xS4xHxW => Sx4xHxW
        propagated = to_bhwc(NF.conv2d(padded, self.kernel).view(-1, C, H, W))

        # ensure good views for all propagated values
        return propagated 
