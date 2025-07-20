import torch
from lightning import LightningDataModule
from lightning import Trainer as LightningTrainer
from lightning.pytorch import seed_everything
from torch.nn import Module

from ml_framework.base.callback import Callback
from ml_framework.base.logger import LoggerFactory
from ml_framework.base.schema import Config, TrainerConfig

class Trainer:
    def __init__(self, config: Config) -> None:
        seed_everything(config.seed)
        torch.set_float32_matmul_precision('medium')
        trainer_cfg: TrainerConfig = config.trainer
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1
        if trainer_cfg.gpus == 0:
            accelerator = "cpu"
        else:
            devices = trainer_cfg.gpus
        self.logger = LoggerFactory.create(config)

        self.callbacks = Callback.parse(trainer_cfg.callbacks, {
            "logger": self.logger
        })

        self.trainer = LightningTrainer(
            max_epochs=trainer_cfg.epochs,
            devices=devices,
            log_every_n_steps=trainer_cfg.log_every,
            check_val_every_n_epoch=trainer_cfg.val_every,
            callbacks=self.callbacks,
            precision=trainer_cfg.precision,
            accelerator=accelerator,
            logger=self.logger
        )

    def train(self, model: Module, data_loader: LightningDataModule) -> None:
        self.trainer.fit(
            model=model,
            datamodule=data_loader
        )

    def test(self, model: Module, data_loader: LightningDataModule) -> None:
        self.trainer.test(
            model=model,
            datamodule=data_loader
        )
