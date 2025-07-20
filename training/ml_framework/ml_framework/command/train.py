from pathlib import Path
from typing import List

import click
import numpy as np
from pydantic import BaseModel, ValidationError
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import os 
from ml_framework.base.data.crawler import Crawlers
from ml_framework.base.data.data_module import DataModule
from ml_framework.base.data.metadata import DataComposer
from ml_framework.base.loader import Loader
from ml_framework.base.schema import Config, DatasetConfig, ProcessConfig, ResumeCkpt
from ml_framework.base.trainer import Trainer
from ml_framework.base.model import LoadModelPath



def setup_dataloader(config: Config):
    print("Setup Data Loader...")
    data_composer = DataComposer(verbose=True)
    for dataset_name, dataset_config in config.data.dataset.items():
        if isinstance(dataset_config, DatasetConfig):
            crawler = getattr(Crawlers, dataset_config.crawler.name)(
                folder_path=config.directory.data / dataset_config.path,
                dataset_name=dataset_name
            )
            data_composer.compose(dataset_name, dataset_config, crawler)

    data_composer.show()

    data_module = DataModule(
        composer=data_composer,
        config=config
    )
    return data_module, data_composer


def setup_metrics(config: Config):
    print("Setup Metrics...")
    metrics = []
    for metric_config in config.metric:
        print(f"Load {metric_config.name}...")
        metric = Loader.get_metric(metric_config.name)(
            name=metric_config.name,
            alias=metric_config.alias,
            on_bar=metric_config.on_bar,
            phase=metric_config.phase,
            device="cuda" if config.trainer.gpus > 0 else "cpu",
            **metric_config.params
        )
        metrics.append(metric)
    return metrics


@click.command()
@click.option('--config', '-c', "config_path", type=click.Path(exists=True), help='Path to the config file')
@click.option('--dry-run',  is_flag=True, show_default=True, default=False, help='Run training without saving artifacts (automatically run 1 epoch only)')
def train(config_path, dry_run):

    print("Load Configs...")
    config = Loader.load_config(Path(config_path))
    config.experiment_name = Path(config_path).stem
    print(config)
    np.random.seed(config.seed)
    seed_everything(config.seed)

    Loader.load_models(config)

    if dry_run:
        config.trainer.epochs = 1

    data_module, data_composer = setup_dataloader(config)
    metrics = setup_metrics(config)

    try:
        print("Training...")
        if "name" not in config.model:
            raise ValueError("Model name property is not found in config")

        model_class = Loader.get_model(config.model["name"])
        ModelConfig = Loader.get_model_config(config.model["name"])
        config.model = ModelConfig(**config.model)
        resume_ckpt_path = LoadModelPath.load(config)

        if resume_ckpt_path is None:
            model = model_class(**config.model.model_dump(exclude={"name"}),
                                **config.trainer.model_dump(exclude={"name"}))
        else:
            model = model_class.load_from_checkpoint(
                resume_ckpt_path,
                **config.model.model_dump(exclude={"name"}),
                **config.trainer.model_dump(exclude={"name"}))

        model.metrics = metrics
        trainer = Trainer(config)
        trainer.train(model, data_module)
        
        print("Training done")
    except ValidationError as e:
        print(e)
        exit()

    if len(data_composer.test()) > 0:
        print("Testing...")
        for callback in trainer.callbacks:
            if type(callback).__name__ == "ModelCheckpoint":
                print("Test using model:", Path(callback.best_model_path).name)
                model = model_class.load_from_checkpoint(callback.best_model_path)
                model.metrics = metrics
        trainer.test(model, data_module)
        print("Testing done")
