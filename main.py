import argparse, os, sys, datetime, glob
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from calo_ldm.util import instantiate_from_config

# NOTE: This file was migrated from pytorch-lightning 1.6.5 to PL 2.x.
# The old `Trainer.add_argparse_args` / `from_argparse_args` machinery and the
# `--gpus` accelerator flag were removed upstream, so trainer configuration is
# now driven entirely through the `lightning.trainer` config section plus a few
# explicit CLI flags. Everything else (config merging, callbacks, logging) is
# kept behaviourally equivalent.


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n", "--name", type=str, const=True, default="", nargs="?",
                        help="postfix for logdir")
    parser.add_argument("-r", "--resume", type=str, const=True, default="", nargs="?",
                        help="resume from logdir or checkpoint in logdir")
    parser.add_argument("-b", "--base", nargs="*", metavar="base_config.yaml",
                        help="paths to base configs. Loaded from left-to-right. "
                             "Parameters can be overwritten or added with command-line "
                             "options of the form `--key value`.",
                        default=list())
    parser.add_argument("-t", "--train", type=str2bool, const=True, default=False, nargs="?",
                        help="train")
    parser.add_argument("--no-test", type=str2bool, const=True, default=False, nargs="?",
                        help="disable test")
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument("-d", "--debug", type=str2bool, nargs="?", const=True, default=False,
                        help="enable post-mortem debugging")
    parser.add_argument("-s", "--seed", type=int, default=23, help="seed for seed_everything")
    parser.add_argument("-f", "--postfix", type=str, default="", help="post-postfix for default name")
    parser.add_argument("-l", "--logdir", type=str, default="logs", help="directory for logging")
    parser.add_argument("--scale_lr", type=str2bool, nargs="?", const=True, default=True,
                        help="scale base-lr by ngpu * batch_size * n_accumulate")
    # PL2: `--gpus` no longer exists on Trainer. Keep it for CLI compatibility and map
    # it onto accelerator/devices ourselves. None/0 -> CPU.
    parser.add_argument("--gpus", type=str, default=None,
                        help="comma-separated GPU ids or a count. Omit / 0 for CPU.")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="convenience override for lightning.trainer.max_epochs")
    return parser


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd so `main.DataModuleFromConfig` etc. resolve when running as `python main.py`
    sys.path.append(os.getcwd())

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    print("-->opt configs", opt)
    print("-->unk configs", unknown)
    for c in unknown:
        assert " " not in c  # check the splitting of config is correct!
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint")

    resume_ckpt = None
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        resume_ckpt = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[-1])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    trainer = None
    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        trainer_config = lightning_config.get("trainer", OmegaConf.create())

        # ---- accelerator / devices (PL2) ----
        if opt.gpus is None or str(opt.gpus).strip() in ("", "0", "-0"):
            print("CPU mode.", file=sys.stderr)
            trainer_config["accelerator"] = "cpu"
            trainer_config["devices"] = 1
            cpu = True
            ngpu = 1
        else:
            gpu_ids = [g for g in str(opt.gpus).split(",") if g != ""]
            trainer_config["accelerator"] = "gpu"
            if len(gpu_ids) == 1 and gpu_ids[0].isdigit() and gpu_ids[0] != "-1":
                # a single integer is interpreted as a device *count*
                trainer_config["devices"] = int(gpu_ids[0])
                ngpu = int(gpu_ids[0])
            else:
                trainer_config["devices"] = [int(g) for g in gpu_ids]
                ngpu = len(gpu_ids)
            print(f"GPU mode, devices={trainer_config['devices']}", file=sys.stderr)
            cpu = False

        if opt.max_epochs is not None:
            trainer_config["max_epochs"] = opt.max_epochs
        if "max_epochs" not in trainer_config:
            trainer_config["max_epochs"] = 10000

        accumulate_grad_batches = trainer_config.get("accumulate_grad_batches", 1)
        lightning_config.trainer = trainer_config

        # ---- model ----
        model = instantiate_from_config(config.model)

        # ---- logger ----
        trainer_kwargs = dict()
        # Default to Weights & Biases in OFFLINE mode (runs anywhere, no login;
        # data under {logdir}/wandb -- `wandb sync` later, or set mode=online +
        # entity/project in the config's lightning.logger to stream live).
        # Native wandb.Histogram / wandb.Image are used by calo_ldm/metrics.py.
        default_logger_cfg = {
            "target": "pytorch_lightning.loggers.WandbLogger",
            "params": {
                "save_dir": logdir,
                "project": "calo-vq-darkshine",
                "name": nowname,
                "mode": "offline",
            },
        }
        logger_cfg = lightning_config.get("logger", OmegaConf.create())
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # ---- model checkpoint ----
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "epoch={epoch:06}",
                "auto_insert_metric_name": False,
                "verbose": True,
                "save_last": True,
            },
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3
        modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        # ---- callbacks ----
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "calo_ldm.callbacks.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                },
            },
            "cuda_callback": {"target": "calo_ldm.callbacks.CUDACallback"},
            "checkpoint_callback": modelckpt_cfg,
        }
        # LearningRateMonitor raises if there is no scheduler, so only add it when one
        # is configured on the model.
        if config.model.get("params", {}).get("scheduler_config", None) is not None:
            default_callbacks_cfg["learning_rate_logger"] = {
                "target": "pytorch_lightning.callbacks.LearningRateMonitor",
                "params": {"logging_interval": "step"},
            }

        callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        # ---- trainer ----
        trainer_kwargs.update(OmegaConf.to_container(trainer_config, resolve=True))
        trainer = Trainer(**trainer_kwargs)

        # ---- data ----
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # ---- learning rate ----
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * "
                  "{} (batchsize) * {:.2e} (base_lr)".format(
                      model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                print("Training mode...")
                trainer.fit(model, data, ckpt_path=resume_ckpt)
            except Exception:
                melk()
                raise
        else:
            print("Validation only mode...")
            trainer.validate(model, data, ckpt_path=resume_ckpt)

    except Exception:
        if trainer and opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        if trainer and opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
