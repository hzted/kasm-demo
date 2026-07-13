import os

os.environ.setdefault("NUMEXPR_MAX_THREADS", "24")

import hydra
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.rank_zero import _get_rank
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar

from datamodules import build_datamodule
from models import build_module
from utils.hydra import fix_relative_path


def _trainer_devices(conf):
    if conf.gpu is None or not torch.cuda.is_available():
        return "cpu", 1
    gpu = [conf.gpu] if isinstance(conf.gpu, int) else list(conf.gpu)
    return "gpu", gpu


@hydra.main(config_path="./configs", config_name="cfg", version_base="1.3")
def main(conf):
    fix_relative_path(conf)
    os.makedirs(conf.output_dir, exist_ok=True)
    os.makedirs(conf.cache_dir, exist_ok=True)

    print(conf)
    print(f"################ rank={_get_rank()} cwd={os.getcwd()} output={conf.output_dir}")
    pl.seed_everything(conf.seed)

    module = build_module(conf=conf)
    datamodule = build_datamodule(conf=conf)

    if conf.debug:
        print("################ Debug Mode ################")
        conf.train.num_workers = 0
        conf.train.prefetch_factor = None
        conf.train.persistent_workers = False
        conf.dev.num_workers = 0
        conf.dev.prefetch_factor = None
        conf.dev.persistent_workers = False
        conf.test.num_workers = 0
        conf.test.prefetch_factor = None
        conf.test.persistent_workers = False

    decision_mode = "min" if conf.test.monitor == "val/loss" else "max"
    ckpt_cb = ModelCheckpoint(
        dirpath=conf.output_dir,
        monitor=conf.test.monitor,
        mode=decision_mode,
        save_top_k=1,
        save_last=False,
        save_weights_only=False,
    )
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        ckpt_cb,
    ]

    accelerator, devices = _trainer_devices(conf)
    trainer = pl.Trainer(
        max_epochs=conf.train.num_epoch,
        callbacks=callbacks,
        gradient_clip_val=conf.train.clip_grad,
        accelerator=accelerator,
        devices=devices,
        precision="32",
        overfit_batches=conf.overfit_batches,
        val_check_interval=conf.val_check_interval,
        num_sanity_val_steps=conf.num_sanity_val_steps,
        deterministic=conf.deterministic,
        strategy="ddp_find_unused_parameters_true"
        if accelerator == "gpu" and isinstance(devices, list) and len(devices) > 1
        else "auto",
    )

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=conf.ckpt_path)
    trainer.test(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
