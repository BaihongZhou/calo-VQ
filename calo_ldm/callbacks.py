import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
# PL2: rank-zero helpers moved out of utilities.distributed.
from pytorch_lightning.utilities.rank_zero import rank_zero_info

import os
import psutil
import time

from omegaconf import OmegaConf

### callbacks taken from the stable-diffusion repo (DarkSHINE physics metrics are
### logged directly from the LightningModules in calo_ldm/metrics.py, not here).

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_exception(self, trainer, pl_module, exception):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    # PL2: `on_pretrain_routine_start` was removed; `setup` runs before fit/validate.
    def setup(self, trainer, pl_module, stage=None):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class CUDACallback(Callback):
    def getMem(self):
        pid = os.getpid()
        process = psutil.Process()
        children = psutil.Process(pid).children(recursive=True)
        total_memory = process.memory_info().rss
        for child in children:
            total_memory += child.memory_info().rss
        return total_memory/1024/1024/1024

    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    # PL2: `trainer.root_gpu` was removed and `on_train_epoch_end` no longer takes
    # `outputs`. Made CPU-safe so it works on machines without CUDA (e.g. macOS).
    def on_train_epoch_start(self, trainer, pl_module):
        self.start_time = time.time()
        device = pl_module.device
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)