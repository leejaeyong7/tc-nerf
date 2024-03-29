import random
from os import path
from pathlib import Path
from .base_data_module import BaseDataModule
import torch


class DTU(BaseDataModule):
    train_sets = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42,
                  44, 45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 
                  65, 68, 69, 70, 71, 72, 74, 76, 83, 84, 85, 87, 88, 89, 90, 
                  91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 
                  105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120, 121, 
                  122, 123, 124, 125, 126, 127, 128]
    val_sets = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86,
                106, 117]
    test_sets = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49,
                 62, 75, 77, 110, 114, 118]
    train_sets = [1]
    val_sets = [1]

    def __init__(self, *args, **kwargs):
        super(DTU, self).__init__(*args, **kwargs)

    def get_train_sets(self)->list:
        self.mode = 'train'
        return [str(i) for i in DTU.train_sets]
    def get_val_sets(self)->list:
        self.mode = 'val'
        return [str(i) for i in DTU.val_sets]
    def get_test_sets(self)->list:
        self.mode = 'test'
        return [str(i) for i in DTU.test_sets]

    def get_image_path(self, image_folder: Path, image_id: int) -> Path:
        ''' returns full path to image file '''
        light_id = 3
        image_name = '{:08d}_{}.png'.format(image_id, light_id)
        return image_folder / image_name

    def get_depth_range(self, ranges):
        '''
        Arguments:
            depth_start(float): starting value of depth interpolation
            depth_interval(float): interval distance between depth planes
            num_intervals(int): number of planes to inerpolate
        Returns:
            list containing min, max depth
        '''
        depth_start = 425.0
        depth_end = 900.0
        return [depth_start, depth_end]

class DTULR(DTU):
    def __init__(self, dataset_dir, options={}, *args, **kwargs):
        super(DTULR, self).__init__(dataset_dir / 'dtu_mvsnet',  options={**options, **{'scaled_camera': True, 'scaled_depth': True}}, *args, **kwargs)

    def parse_camera(self, camera_file_path: str) -> tuple:
        '''
        Loads camera from path
        Return:
            Extrinsic: 4x4 numpy array containing R | t
            Intrinsic: 3x3 numpy array containing intrinic matrix
            Range: min, interval, num_interval, max
        '''
        K, E, tokens = super.parse_camera(camera_file_path)
        K *= 4
        K[2, 2] = 1
        return K, E, tokens

class DTUHR(DTU):
    def __init__(self, dataset_dir, *args, **kwargs):
        super(DTUHR, self).__init__(dataset_dir / 'dtu_mine',  *args, **kwargs)
