import random
import cv2
import torch
import os
from os import path
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import re
import math
from utils.io import read_pfm, read_camera

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from pathlib import Path
from .sample_loader import SampleLoader
from .ray_loader import RayLoader

class BaseDataModule(pl.LightningDataModule):
    """
    folder structure for datasets

    root/
        set_name
            cameras/
                00000000.txt
            images/
                00000000.jpg
            depths/
                00000000.pfm
            pair.txt:
    """
    def __init__(self, dataset_dir, num_views=10, batch_size=1, options={}):
        '''
        reads from path to generate file paths for samples
        '''

        # setup variables shared by all datasets
        self.dataset_dir = Path(dataset_dir)
        self.num_views = num_views
        self.batch_size = 1
        self.options = options

    def image_folder_name(self):
        return 'images'

    def camera_folder_name(self):
        return 'cameras'

    def depth_folder_name(self):
        return 'depths'

    def pair_file(self):
        return 'pair.txt'

    def get_samples_from_sets(self, target_sets, num_views, train=False):
        # iterate over sets
        samples = []
        for target_set in target_sets:
            set_dir  = self.dataset_dir / target_set

            image_dir = set_dir / self.image_folder_name()
            camera_dir = set_dir / self.camera_folder_name()
            depth_dir = set_dir / self.depth_folder_name()
            pair_file = set_dir / self.pair_file()

            # pairs = {
            #   image_id: [neighbor_ids] 
            # }
            pairs = self.parse_pairs(pair_file)
            all_image_ids = list(pairs.keys())
            for ref_image_id, neighbor_list in pairs.items():
                if(train):
                    if (ref_image_id % 12) == 0:
                        continue
                else:
                    if (ref_image_id % 12) != 0:
                        continue


                neigh_image_ids = [ref_image_id] + neighbor_list

                neigh_image_hash = {}
                for n_id in neigh_image_ids:
                    neigh_image_hash[n_id] = n_id

                remaining_hash = {}
                for image_id in all_image_ids:
                    if not (image_id in neigh_image_hash):
                        remaining_hash[image_id] = image_id
                remainig_ids = list(remaining_hash.keys())
                selected_image_ids = neigh_image_ids[:num_views]


                sample = {
                    'set_name': target_set,
                    'ref_id': ref_image_id,
                    'images': [],
                    'cameras': [],
                    'depths': []
                }

                for image_id in selected_image_ids:
                    image_path = self.get_image_path(image_dir, image_id)
                    camera_path = self.get_camera_path(camera_dir, image_id)

                    sample['images'].append(image_path)
                    sample['cameras'].append(camera_path)
                    if not(self.options.get('skip_depth')):
                        depth_path = self.get_depth_path(depth_dir, image_id)
                        sample['depths'].append(depth_path)
                samples.append(sample)
        return samples

    def setup(self, stage=None):
        if (stage == 'fit') or (stage == None):
            try:
                train_sets = self.get_train_sets()
                num_views = self.num_views if not self.options.get('num_train_views') else self.options['num_train_views']
                train_samples = self.get_samples_from_sets(train_sets, num_views, True)
                self.train_dataset = RayLoader(train_samples, self.options)
            except Exception as e:
                print(repr(e))
                raise Exception("Training data must be present!")

            try:
                val_sets = self.get_val_sets()
                val_samples = self.get_samples_from_sets(val_sets, num_views)
                self.val_dataset = SampleLoader(val_samples, self.options)
            except:
                self.val_dataset = None

        if (stage == 'test') or (stage == None):
            try:
                test_sets = self.get_test_sets()
                test_samples = self.get_samples_from_sets(test_sets)
                self.test_dataset = SampleLoader(test_samples, self.options)
            except Exception as e:
                print(repr(e))
                raise Exception('Test data must be present!')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, shuffle=True)

    def val_dataloader(self):
        if(self.val_dataset):
            return DataLoader(self.val_dataset, batch_size=None)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=None)

    def batch_images(self, images):
        return torch.stack(images)

    def parse_pairs(self, pair_file_path: Path)-> dict:
        '''
        Parses pair.txt and returns dictionary of neighbors in sorted list
        Pair.txt file format: 
            NUM_IMAGES
            IMAGE_ID_0
            NUM_NEIGHBORS NEIGHBOR_ID_0 SCORE_0 NEIGHBOR_ID_1 SCORE_1 ...
            IMAGE_ID_1
            ...
        '''
        pair_dict = {}
        with open(pair_file_path, 'r') as f:
            lines = f.readlines()
            num_images = int(lines[0])
            for i in range(num_images):
                image_id = int(lines[i * 2 + 1])
                pair_dict[image_id] = []
                image_neighbor_lines = lines[i * 2 + 2]
                tokens = image_neighbor_lines.split()
                num_neighbors = int(tokens[0])
                for n in range(num_neighbors):
                    neighbor_id = int(tokens[n * 2 + 1])
                    pair_dict[image_id].append(neighbor_id)
        return pair_dict

    def get_depth_path(self, depth_folder: Path, image_id: int) -> Path:
        ''' returns full path to depth file '''
        depth_name = '{:08d}.pfm'.format(image_id)
        return depth_folder / depth_name

    def get_image_path(self, image_folder: Path, image_id: int)-> Path:
        ''' returns full path to image file '''
        image_name = '{:08d}.jpg'.format(image_id)
        return image_folder / image_name

    def get_camera_path(self, camera_folder: Path, image_id: int)-> Path:
        ''' returns full path to camera file '''
        camera_name = '{:08d}_cam.txt'.format(image_id)
        return camera_folder / camera_name

    # overrideables 
    def get_train_sets(self):
        raise NotImplementedError

    def get_val_sets(self):
        raise NotImplementedError

    def get_test_sets(self):
        raise NotImplementedError
