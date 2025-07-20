import importlib.util
import os
from glob import glob
from pathlib import Path

import yaml

from ml_framework.base.metric import Metric, SklearnMetric, TorchMetric
from ml_framework.base.model import Model
from ml_framework.base.process import (AlbumentationWrapper, ArgMax, CustomBiasField,
                                       LabelEncoder, OneHotEncoder,
                                       ProcessPipeline, ToGray, 
                                       TorchVisionWrapper, LoadSegmentationLabel, LesionAug)
from ml_framework.base.schema import ClassificationConfig, Config, ModelConfig


class Loader:
    _config: Config = None
    _models: dict[str, Model] = {}
    _model_configs: dict[str, ModelConfig] = {}
    _processes: dict[str, ProcessPipeline] = {
        "alb": AlbumentationWrapper,
        "tv": TorchVisionWrapper,
        "label_encoding": LabelEncoder,
        "argmax": ArgMax,
        "one_hot_encoding": OneHotEncoder,
        "load_segmentation_label":LoadSegmentationLabel,
        "to_gray": ToGray,
        "lesion_aug": LesionAug,
        "custom_bias_field": CustomBiasField
    }
    _metrics: dict[str, Metric] = {
        "sklearn": SklearnMetric,
        "torchmetrics": TorchMetric
    }

    @classmethod
    def _load_file(cls, config: str):
        config_dict = {}
        if Path(config).is_file() and Path(config).suffix == ".yaml":
            with open(config, "r", encoding="UTF-8") as f:
                config_dict = yaml.safe_load(f)

            for key, value in config_dict.items():
                if isinstance(value, str):
                    if Path(value).suffix == ".yaml":
                        config_dict[key] = cls._load_file(
                            Path(config).parent / Path(value))
            return config_dict
        return config

    @classmethod
    def load_config(cls, config_path: Path):
        if cls._config is None:
            cls._config = Config(**cls._load_file(config_path))
        return cls._config

    @classmethod
    def get_config(cls):
        return cls._config

    @classmethod
    def get_process(cls, name: str) -> ProcessPipeline:
        for key, process_class in cls._processes.items():
            if key in name:
                return process_class
        return None

    @classmethod
    def get_metric(cls, name: str) -> Metric:
        if name not in cls._processes:
            for key, metric_class in cls._metrics.items():
                if key in name:
                    return metric_class
        return None

    @classmethod
    def load_models(cls, config:Config):
        print("\n\nModel Loadings...\n")
        for model_path in glob(str(Path(config.directory.model) /"*.py")):
            print("Load model from", model_path)
            module_name = Path(model_path).stem
            spec = importlib.util.spec_from_file_location(
                module_name, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for class_name, blueprint in module.__dict__.items():
                if class_name.startswith("__"):
                    continue
                if isinstance(blueprint, type):
                    if issubclass(blueprint, ModelConfig) and (
                        blueprint != ModelConfig and
                        blueprint != ClassificationConfig
                    ):
                        cls._model_configs[module_name] = blueprint

                    if issubclass(blueprint, Model) and blueprint != Model:
                        cls._models[module_name] = blueprint
        print("\n")

    @classmethod
    def get_model(cls, name: str):
        return cls._models[name]

    @classmethod
    def get_model_config(cls, name: str):
        return cls._model_configs[name]
