from abc import ABCMeta, abstractmethod
from typing import Any, Literal

import sklearn.metrics as sklearn_metric
import torch
import torchmetrics


class Metric(metaclass=ABCMeta):

    def __init__(self, name, on_bar, phase, alias, ml_type) -> None:
        self.name = name
        self.on_bar = on_bar
        self.phase = phase
        self.alias = alias
        self.ml_type = ml_type

    @abstractmethod
    def __call__(self,
                 phase: Literal["train", "test", "val"],
                 log: callable,
                 y_true: torch.Tensor,
                 y_pred: torch.Tensor) -> Any:
        pass


class SklearnMetric(Metric):
    def __init__(self, name, on_bar, alias, phase,ml_type, **params):
        name = name.replace("sklearn.", "")
        super().__init__(name=name, on_bar=on_bar, alias=alias, phase=phase,ml_type=ml_type)
        self.metric = getattr(sklearn_metric, name)(**params)

    def __call__(self, phase, log, y_pred, y):
        score = self.metric(y_pred, y)
        log(phase + "_"+self.name, score, sync_dist=True, prog_bar=self.on_bar)
        return score


class TorchMetric(Metric):
    def __init__(self, name, on_bar, alias, phase, ml_type, device, **params):
        name = name.replace("torchmetrics.", "")
        super().__init__(name=name, on_bar=on_bar, alias=alias, phase=phase, ml_type=ml_type)
        self.alias = alias or self.name
        self.metric = getattr(torchmetrics, name)(**params).to(device)

    def __call__(self, phase, log, y_pred, y):
        if self.ml_type == "classification":
            score = self.metric(y_pred.argmax(dim=1), y.argmax(dim=1))
        elif self.ml_type == "segmentation":
            score = self.metric((y_pred > 0.5), y)
        else:
            raise ValueError("Task is not supported")
        log(phase + "_"+self.alias, score, sync_dist=True, prog_bar=self.on_bar)
        return score
