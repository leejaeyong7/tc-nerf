from utils.io import *
from utils.io import read_pfm, read_camera
from utils.transforms import *
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as NF
from torchvision.transforms.functional import to_tensor

class RayLoader(Dataset):
    def __init__(self, samples, options):
        super(RayLoader, self).__init__()
        self.options = options
        self.samples = self.generate_rays_from_samples(samples)

    def __len__(self) -> int:
        '''
        Returns number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, index):
        # 1x8
        return self.samples[index]

    def generate_rays_from_samples(self, samples):
        data = []
        for sample in samples:
            image_path = sample['image']
            camera_path = sample['camera']
            image = self.load_image(image_path)
            npK, npE, ref_ranges = self.load_camera(camera_path)
            ranges = self.get_depth_range(ref_ranges)

            # Nx3x3
            if(self.options.get('resize')):
                size = self.options.get('resize')
            else:
                size = None

            K = torch.from_numpy(npK)
            if not (size is None):
                K = self.resize_intrinsic(K, image, *size)

            # Nx4x4
            E = torch.from_numpy(npE)

            if not (size is None):
                image = self.resize_image(image, *size)

            H, W = image.shape[-2:]

            # rescale so that the depth becomes clamped
            scale = ranges[0] * 0.75
            ranges[0] /= scale 
            ranges[1] /= scale 
            E[:3, 3] /= scale 
            R = E[:3, :3]
            t = E[:3, 3:]

            # 3xHW
            cam_rays = self.camera_rays(K, H, W)
            world_rays = R.t() @ cam_rays
            world_rays = world_rays / world_rays.norm(p=2, dim=0, keepdim=True)
            world_pos = (-R.t() @ t)

            # obtain rays based on the K, E = 3xHxW
            # obtain colors = 3xHxW
            # obtain ranges = 
            colors = image.view(3, -1)
            dirs = world_rays.view(3, -1)
            pos = world_pos.view(3, 1).repeat(1, H*W)
            rs = torch.FloatTensor(ranges).view(2, 1).repeat(1, H*W)

            # (3+3+3+2)xH*W
            dat = torch.cat((colors, dirs, pos, rs), dim=0)
            data.append(dat)

        # (N*H*W)x(3+3+3+2)
        return torch.cat(data,  dim=-1).t()

    def get_depth_range(self, ranges):
        '''
        Arguments:
            depth_start(float): starting value of depth interpolation
            depth_interval(float): interval distance between depth planes
            num_intervals(int): number of planes to inerpolate

        Returns:
            torch.Tensor: Px4 plane equations in hessian normal form.
        '''
        if(len(ranges) == 2):
            depth_start = 425.0
            depth_end = 900.0
            return [depth_start, depth_end]
        else:
            depth_start = ranges[0]
            depth_intv = ranges[1]
            depth_num = int(ranges[2])
            return [depth_start, depth_start + depth_intv * depth_num]

    def load_depth(self, depth_full_path:Path) -> np.ndarray:
        ''' Loads depths given full path '''
        return read_pfm(depth_full_path)

    def load_image(self, image_full_path:Path)-> Image:
        ''' Loads image given full path '''
        return to_tensor(Image.open(image_full_path))

    def load_camera(self, camera_full_path:Path)-> tuple:
        ''' Loads camera given full path to file '''
        k, e, r = read_camera(camera_full_path)
        if(self.options.get('scaled_camera')):
            k *= 4
            k[2, 2] = 1
        return k, e, r

    def resize_image(self, image, height, width):
        return NF.interpolate(image.unsqueeze(0), size=(height, width), mode='bilinear')[0]

    def resize_intrinsic(self, K, image, height, width):
        KC = K.clone()
        KC[0] *= float(width) / float(image.shape[-1])
        KC[1] *= float(height) / float(image.shape[-2])
        return KC

    def pixel_points(self, H, W, offset=0.5):
        '''
        Given width and height, creates a mesh grid, and returns homogeneous 
        coordinates
        of image in a 3 x W*H Tensor

        Arguments:
            width {Number} -- Number representing width of pixel grid image
            height {Number} -- Number representing height of pixel grid image

        Returns:
            torch.Tensor -- 1x2xHxW, oriented in x, y order
        '''
        O = offset
        x_coords = torch.linspace(O, W - 1 + O, W)
        y_coords = torch.linspace(O, H - 1 + O, H)

        # HxW grids
        y_grid_coords, x_grid_coords = torch.meshgrid([y_coords, x_coords])

        # HxWx2 grids => 1xHxWx2 grids
        return torch.stack([ x_grid_coords, y_grid_coords ], 2).unsqueeze(0)

    def camera_rays(self, K, H, W):

        # x = HWx3
        x = from_vector(to_homogeneous(self.pixel_points(H, W))).view(-1, 3)
        # 3x3
        Ki = K.inverse()

        # 3xHW
        return Ki @ x.t()