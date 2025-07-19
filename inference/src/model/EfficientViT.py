from typing import Union

from timm.models import efficientvit_msra

from src.model.base import Model
from src.schema import ClassificationConfig


class EfficientVitMSRAConfig(ClassificationConfig):
    size: str
    in_channels: int = 3
    freeze: bool = True
    pretrained: Union[bool, str] = False


class EfficientVitMSRAWrapper(Model):
    def __init__(self, **kwargs):
        super().__init__()

        efficientvit_msras = {
            "m0": efficientvit_msra.efficientvit_m0,
            "m1": efficientvit_msra.efficientvit_m1,
            "m2": efficientvit_msra.efficientvit_m2,
            "m3": efficientvit_msra.efficientvit_m3,
            "m4": efficientvit_msra.efficientvit_m4,
            "m5": efficientvit_msra.efficientvit_m5,
        }
        pretrained = self.hparams.pretrained
        if isinstance(pretrained, str):
            pretrained = False

        self.model = efficientvit_msras[self.hparams.size](
            pretrained=pretrained,
            in_chans=self.hparams.in_channels,
            num_classes=self.hparams.num_classes,
        )
