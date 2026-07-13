import json
import os

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, MaxMetric, MeanMetric

from models import build_model
from .registry import register_module
from utils.logger import get_logger

log = get_logger(__name__)


@register_module("KASM")
class KASMModule(pl.LightningModule):
    def __init__(self, conf, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.conf = conf
        self.net = build_model(conf=conf)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.conf.model.num_class)

        self.train_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=self.conf.model.num_class, average="macro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.test_acc_best = MaxMetric()
        self.test_f1_best = MaxMetric()

        self.test_preds = []
        self.test_targets = []
        self.test_aspects = []

    def on_train_start(self):
        self.val_acc_best.reset()
        self.val_f1_best.reset()

    def _shared_loss(self, batch):
        output = self.net(**batch)
        logits, targets = output["logits"], batch["label_ids"]
        loss_doc = self.criterion(logits, targets)
        loss_seq = output["loss_seq"] if output["loss_seq"] is not None else 0.0
        loss = loss_doc + self.conf.model.lambda_seq * loss_seq
        preds = torch.argmax(logits, dim=1)
        return loss, preds, targets

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_loss(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self) -> None:
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        avg_loss = self.train_loss.compute().item()
        self.train_loss.reset()
        log.info(f"epoch: {self.current_epoch}, lr: {current_lr:.6f}, train/loss: {avg_loss:.4f}")
        self.log("train/loss_epoch", avg_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, preds, targets = self._shared_loss(batch)
        if dataloader_idx == 0:
            self.val_loss(loss)
            self.val_acc(preds, targets)
            self.val_f1(preds, targets)
            self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            self.test_loss(loss)
            self.test_acc(preds, targets)
            self.test_f1(preds, targets)
            self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        self.val_acc_best(val_acc)
        self.val_f1_best(val_f1)
        self.log("val/loss_epoch", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/acc_epoch", val_acc, prog_bar=True, sync_dist=True)
        self.log("val/f1_epoch", val_f1, prog_bar=True, sync_dist=True)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True, sync_dist=True)
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_f1.reset()

        test_loss = self.test_loss.compute()
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()
        self.test_acc_best(test_acc)
        self.test_f1_best(test_f1)
        self.log("test/loss_epoch", test_loss, prog_bar=True, sync_dist=True)
        self.log("test/acc_epoch", test_acc, prog_bar=True, sync_dist=True)
        self.log("test/f1_epoch", test_f1, prog_bar=True, sync_dist=True)
        self.log("test/acc_best", self.test_acc_best.compute(), prog_bar=True, sync_dist=True)
        self.log("test/f1_best", self.test_f1_best.compute(), prog_bar=True, sync_dist=True)
        self.test_loss.reset()
        self.test_acc.reset()
        self.test_f1.reset()

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_loss(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_preds.extend(preds.cpu().numpy().tolist())
        self.test_targets.extend(targets.cpu().numpy().tolist())
        self.test_aspects.extend(batch["aspect_ids"].cpu().numpy().tolist())
        return {"loss": loss, "preds": preds, "targets": targets, "aspect_ids": batch["aspect_ids"]}

    def on_test_epoch_end(self):
        log.info("")
        log.info(f"test/acc: {self.test_acc.compute().item():.4f}")
        log.info(f"test/f1: {self.test_f1.compute().item():.4f}")

        os.makedirs(self.conf.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.conf.output_dir,
            f"preds-{self.conf.model.arch}-{self.conf.data.name}.json",
        )
        with open(output_path, "w") as fw:
            json.dump(
                {
                    "preds": self.test_preds,
                    "label_ids": self.test_targets,
                    "aspect_ids": self.test_aspects,
                },
                fw,
            )

    def configure_optimizers(self):
        train_conf = self.conf.train
        optimizer = AdamW(self.net.parameters(), lr=train_conf.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=train_conf.scheduler_factor,
            patience=train_conf.scheduler_patience,
        )
        return (
            [optimizer],
            {
                "scheduler": scheduler,
                "monitor": "val/loss/dataloader_idx_0",
                "interval": "epoch",
                "frequency": 1,
            },
        )
