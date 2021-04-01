from utils.io import *
from utils.io import read_pfm, read_camera
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as NF
from torchvision.transforms.functional import to_tensor

class SampleLoader(Dataset):
    def __init__(self, samples, options):
        super(SampleLoader, self).__init__()
        self.samples = samples
        self.options = options

    def __len__(self) -> int:
        '''
        Returns number of samples in the dataset.
        '''
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        '''
        Actually fetches dataset.
        '''
        sample = self.samples[index]
        set_name = sample['set_name']
        ref_id = sample['ref_id']

        image_path = sample['image']
        camera_path = sample['camera']

        # load from paths
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
            torch_intrinsics = self.resize_intrinsic(K, image, *size)

        # Nx4x4
        E = torch.from_numpy(npE)

        if not (size is None):
            image = self.resize_images(image, *size)


        # rescale so that the depth becomes clamped
        scale = ranges[0] * 0.75
        ranges[0] /= scale 
        ranges[1] /= scale 
        E[:3, 3] /= scale 

        # if self.get_depth:
        # on training, additionally load in depths
        data = {
            'images': image,
            'intrinsics': K,
            'extrinsics': E,
            'ranges': ranges
        }

        return data

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