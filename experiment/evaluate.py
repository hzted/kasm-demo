import os

os.environ.setdefault("NUMEXPR_MAX_THREADS", "24")

import hydra
import pytorch_lightning as pl
import torch

from datamodules import build_datamodule
from models.modelmodules.kasm_module import KASMModule
from utils.hydra import fix_relative_path


def _trainer_devices(conf):
    if conf.gpu is None or not torch.cuda.is_available():
        return "cpu", 1
    gpu = [conf.gpu] if isinstance(conf.gpu, int) else list(conf.gpu)
    return "gpu", gpu


@hydra.main(config_path="./configs", config_name="cfg", version_base="1.3")
def main(conf):
    fix_relative_path(conf)
    if not conf.ckpt_path:
        raise ValueError("Set ckpt_path=/path/to/checkpoint.ckpt for evaluation.")

    os.makedirs(conf.output_dir, exist_ok=True)
    os.makedirs(conf.cache_dir, exist_ok=True)

    datamodule = build_datamodule(conf=conf)
    module = KASMModule.load_from_checkpoint(
        checkpoint_path=conf.ckpt_path,
        conf=conf,
        strict=False,
    )
    accelerator, devices = _trainer_devices(conf)
    trainer = pl.Trainer(accelerator=accelerator, devices=devices, precision="32")
    trainer.test(model=module, datamodule=datamodule)


if __name__ == "__main__":
    main()
