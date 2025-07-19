from collections import OrderedDict

import torch
from lightning import LightningModule
from timm import scheduler as timm_scheduler
from torch import nn, optim

from src.schema import PairResult

losses = {
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "BCELoss": nn.BCELoss,
    "L1Loss": nn.L1Loss,
    "NLLLoss": nn.NLLLoss,
    "KLDivLoss": nn.KLDivLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
}


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = []
        self.cum_results = {
            "train": PairResult(),
            "val": PairResult(),
            "test": PairResult(),
        }

        self.loss_fn = losses[self.hparams.loss["name"]](**self.hparams.loss["params"])

        self.optimizer = getattr(optim, self.hparams.optimizer["name"])
        if self.hparams.last_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif self.hparams.last_activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif self.hparams.last_activation == "identity":
            self.activation = nn.Identity()
        else:
            raise ValueError("last activation not defined")

    def add_metrics(self, metric):
        self.metrics.append(metric)

    def step(self, batch, phase):
        x, y = self._extract_batch(batch)
        y_pred = self(x)
        loss = self.loss_fn(y_pred.float(), y.float())

        y_pred = self.activation(y_pred)
        for metric in self.metrics:
            metric(phase, self.log, y_pred, y)

        self.log(f"{phase}_loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), self.hparams.lr["init_value"])
        schedulers = []
        if self.hparams.lr["scheduler"] is not None:
            scheduler_class = getattr(
                timm_scheduler, self.hparams.lr["scheduler"]["name"]
            )
            scheduler = scheduler_class(
                optimizer,
                t_initial=self.hparams.epochs,
                **self.hparams.lr["scheduler"]["params"],
            )
            schedulers.append({"scheduler": scheduler, "interval": "epoch"})
        return [optimizer], schedulers

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _extract_batch(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return batch[0], batch[1]
        else:
            return batch["input"], batch["target"]

    def load_pretrained(self, path: str, head_key: str = None):
        checkpoint = torch.load(path)

        if "num_classes" in self.hparams.keys():
            if "state_dict" not in checkpoint.keys():
                raise KeyError("State dict not found !")

            state_dict = OrderedDict()
            if (
                checkpoint["hyper_parameters"]["num_classes"]
                == self.hparams.num_classes
            ):
                head_key = None

            for k, v in checkpoint["state_dict"].items():
                if head_key in k:
                    continue
                state_dict[k.replace("model.", "")] = v

            if hasattr(self, "model"):
                self.model.load_state_dict(state_dict, strict=False)

            del state_dict
        else:
            self.load_state_dict(checkpoint["state_dict"])
        del checkpoint
