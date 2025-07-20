from pathlib import Path
from typing import List, Optional, Union

import pydantic


class PipelineConfig(pydantic.BaseModel):
    name: str
    params: dict


class ProcessConfig(PipelineConfig):
    target: List[str]
    phase: List[str]


class DatasetConfig(pydantic.BaseModel):
    path: Path
    driver: str
    crawler: PipelineConfig = PipelineConfig(name="file", params={})
    pipeline: list[PipelineConfig]


class DataLoaderConfig(pydantic.BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool


class DirectoryConfig(pydantic.BaseModel):
    log: str
    weights: str
    data: str
    model: str


class DataConfig(pydantic.BaseModel):
    dataset: dict[str, DatasetConfig]
    preprocess: List[ProcessConfig]
    batch_preprocess: List[ProcessConfig]
    postprocess: List[ProcessConfig]

class ResumeCkpt(pydantic.BaseModel):
    registry: str 
    run_name: str
    file_name: str

class ModelConfig(pydantic.BaseModel):
    name: str
    resume_ckpt: Optional[ResumeCkpt] = None
    loss: PipelineConfig
    last_activation: str = "softmax"


class ClassificationConfig(ModelConfig):
    num_classes: int
    class_weight: Optional[list] = None

class SegmentationConfig(ModelConfig):
    in_channels:int
    out_channels:int 
    depth:int = 5
    features:int = 32
    pretrained:Optional[str] = None

class LRConfig(pydantic.BaseModel):
    init_value: float
    scheduler: Optional[PipelineConfig] = None

class SamplerConfig(pydantic.BaseModel):
    name: str
    params: dict


class TrainerConfig(pydantic.BaseModel):
    name: str
    epochs: int
    callbacks: List[PipelineConfig]
    log_every: int
    val_every: int
    gpus: int
    precision: Union[int, str]
    lr: LRConfig
    optimizer: PipelineConfig
    loader: DataLoaderConfig
    sampler: Optional[SamplerConfig] = None


class MetricConfig(pydantic.BaseModel):
    name: str
    alias: str = None
    on_bar: bool
    params: dict = {}
    phase: Union[str, List[str]] = ["train", "val", "test"]


class LoggerConfig(pydantic.BaseModel):
    name: str


class Config(pydantic.BaseModel):
    project_name: str
    experiment_name: Optional[str] = None
    seed: int
    directory: Union[Path, DirectoryConfig]
    model: Union[Path, dict]
    data: Union[Path, DataConfig]
    trainer: Union[Path, TrainerConfig]
    metric: Union[Path, List[MetricConfig]]
    logger: Union[Path, LoggerConfig]


class PairResult(pydantic.BaseModel):
    y_true: list = []
    y_pred: list = []

    def extend(self, y_true, y_pred):
        self.y_true.extend(y_true)
        self.y_pred.extend(y_pred)

    def reset(self):
        self.y_true = []
        self.y_pred = []
