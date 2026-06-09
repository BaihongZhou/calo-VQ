import torch
from torch.utils.data import (
        Dataset, DataLoader,
        RandomSampler, SequentialSampler, BatchSampler
        )
import pytorch_lightning as pl

import os
from functools import partial
import h5py

from calo_ldm.util import instantiate_from_config


class CaloDarkSHINE(Dataset):
    """DarkSHINE ECAL dataset (export.h5).

    Keys: `condition` (N,1) incident energy [MeV], `energy` (N,43,43,11) deposited
    energy per cell, `label` (N,43,43,11) per-event hit flag (unused by the model;
    the geometry mask is a fixed, sample-independent buffer loaded separately).

    The data is channels-LAST (N, x=43, y=43, depth=11). We permute it to
    channels-FIRST (N, depth=11, x=43, y=43) so depth is the conv channel axis and
    (x, y) is the 2D image plane -- matching the (N, C, H, W) convention used by the
    encoder/decoder/discriminator. No periodic/rotation augmentation is applied:
    the staggered geometry mask alternates by layer parity, so an x/y flip would
    misalign the data with the mask.
    """
    def __init__(self, file_path, load_partial=None, key_energy='energy',
                 key_condition='condition'):
        super().__init__()
        print(f"Load DarkSHINE from {file_path}", os.path.exists(file_path))
        if load_partial:
            print("WARNING: load partial dataset (only for debug!)")
        with h5py.File(file_path, 'r') as h5_file:
            E = torch.from_numpy(h5_file[key_energy][:load_partial]).float()      # (N,43,43,11)
            self._e_inc = torch.from_numpy(h5_file[key_condition][:load_partial]).float()  # (N,1)
        # channels-last (N, x, y, depth) -> channels-first (N, depth, x, y)
        self._data = E.permute(0, 3, 1, 2).contiguous()
        self.dataset_len = len(self._e_inc)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        return {
            'pixels_E': self._data[index].contiguous(),   # (.., 11, 43, 43)
            'E_inc': self._e_inc[index],                  # (.., 1)
        }


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False,
                 shuffle_val_dataloader=False, pin_memory=False, batched_indices=True):
        super().__init__()
        self.batch_size = batch_size
        self.batched_indices = batched_indices
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if self.num_workers > 1:
            print("NOTE: multiple dataloader, watch out your memory!!")
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        # if test is not None:
        #     self.dataset_configs["test"] = test
        #     self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        self.pin_memory = pin_memory


    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        if self.batched_indices:
            sampler = BatchSampler(RandomSampler(self.datasets['train']),
                        batch_size=self.batch_size,
                        drop_last=True) # drop small batch in case we are using batchnorm statistics
            return DataLoader(self.datasets['train'], batch_size=None,
                    num_workers=self.num_workers, pin_memory=self.pin_memory,
                    sampler=sampler)
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory)

    def _val_dataloader(self, shuffle=False):
        assert not shuffle # we need fixed dataset for metric eval.
        if self.batched_indices:
            sampler = BatchSampler(SequentialSampler(self.datasets['validation']),
                        batch_size=self.batch_size,
                        drop_last=True)
            return DataLoader(self.datasets['validation'], batch_size=None,
                    num_workers=self.num_workers, pin_memory=self.pin_memory,
                    sampler=sampler)

        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=shuffle, pin_memory=self.pin_memory)
