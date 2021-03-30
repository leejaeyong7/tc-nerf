default_options = {
    'learning_rate': {
        'type': float,
        'default': 1e-3
    },
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'scheduler_last_epoch':{
        'type': int,
        'default': 5
    },
    'scheduler_rate':{
        'type': float,
        'default': 0.7
    },
    'nerf_num_depths':{
        'type': int,
        'default': 32
    },
    'nerf_network_depth':{
        'type': int,
        'default': 8
    },
    'nerf_channels':{
        'type': int,
        'default': 256
    },
    'nerf_skips':{
        'type': int,
        'nargs':'*',
        'default': [4]
    },
    'nerf_point_encode':{
        'type': int,
        'default': 10
    },
    'nerf_dir_encode':{
        'type': int,
        'default': 4
    }
}
