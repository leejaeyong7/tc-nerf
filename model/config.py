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
}
