import random
from os import path
from .base_data_module import BaseDataModule

class BlendedMVS(BaseDataModule):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(BlendedMVS, self).__init__(dataset_dir, num_views, options=options)

    def get_train_sets(self):
        set_list_file = self.dataset_dir / 'training_list.txt'
        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def get_val_sets(self):
        set_list_file = self.dataset_dir / 'validation_list.txt'
        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def get_test_sets(self):
        raise Exception('No test sets found for Blended MVS Dataset')

class BlendedMVSLR(BlendedMVS):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(BlendedMVSLR, self).__init__(dataset_dir / 'blended_lr', num_views, options={})

    def get_train_sets(self):
        set_list_file = self.dataset_dir / 'BlendedMVS_training.txt'

        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def depth_folder_name(self):
        return 'rendered_depth_maps'

    def image_folder_name(self):
        return 'blended_images'

    def camera_folder_name(self):
        return 'cams'

    def pair_file(self):
        return 'cams/pair.txt'

class BlendedMVSHR(BlendedMVS):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(BlendedMVSHR, self).__init__(dataset_dir / 'blended_hr', num_views, options={**options, **{'unscaled_depth': True}})

    def get_train_sets(self):
        set_list_file = self.dataset_dir / 'BlendedMVS_training.txt'

        with open(set_list_file, 'r') as f:
            set_lines = f.readlines()
        return [set_line.strip() for set_line in set_lines]

    def depth_folder_name(self):
        return 'rendered_depth_maps'

    def image_folder_name(self):
        return 'blended_images'

    def camera_folder_name(self):
        return 'cams'

    def pair_file(self):
        return 'cams/pair.txt'

class Debug(BlendedMVS):
    def __init__(self, dataset_dir, num_views=5, options={}):
        super(Debug, self).__init__(dataset_dir / 'debug', num_views, options={})

    def get_train_sets(self):
        return ['5a3ca9cb270f0e3f14d0eddb']

    def get_val_sets(self):
        return ['5a3ca9cb270f0e3f14d0eddb']
        # raise Exception("Validation Data do not exist")