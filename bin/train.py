import sys
sys.path.append('.')
sys.path.append('..')
import os

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch.utils.data import DataLoader

# setup locaml
from nerf import NeRF
import data_modules as DataModules

def main(args, hparams):
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    data_path = Path(args.dataset_path)
    log_path = Path(args.log_path)

    checkpoint_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, monitor='val/loss', save_top_k=-1, save_last=True)
    logger = TensorBoardLogger(log_path, name=experiment_name)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)

    # setup model / trainer
    model = NeRF(hparams)
    trainer = Trainer.from_argparse_args(args,
                                         resume_from_checkpoint=last_ckpt,
                                         logger=logger, 
                                         flush_logs_every_n_steps=1,
                                         log_every_n_steps=1,
                                         checkpoint_callback=checkpoint_callback)

    # setup data module
    data_module = getattr(DataModules, args.dataset)(data_path, options=vars(args))
    data_module.prepare_data()
    data_module.setup('fit')

    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = NeRF.add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str, choices=DataModules.__all__, required=True)
    parser.add_argument('--resume', dest='resume', action='store_true')

    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(resume=False)

    args = parser.parse_args()

    main(args, hparams)
