
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import mlflow
import torch
from lightning import LightningModule
from timm import scheduler as timm_scheduler
from torch import nn, optim

from ml_framework.base.schema import Config, PairResult, ResumeCkpt


class DuckNetDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DuckNetDiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # Flatten predictions and targets
        targets = (targets > 0.5).float()
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice Loss
        return 1 - dice
    

# class dice_loss(nn.Module):
#     def __init__(self):
#         super(dice_loss, self).__init__()
#         self.eps=1e-7

#     def forward(self, x, target):
#         target = target.type(x.type())
#         dims = (0,) + tuple(range(2, target.ndimension()))
#         intersection = torch.sum(x * target, dims)
#         cardinality = torch.sum(x + target, dims)
#         dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
#         return (1 - dice_loss)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def __call__(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class dice_loss_worst(nn.Module):
    def __init__(self):
        super(dice_loss_worst, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps))
        min_dice_val = torch.min(dice_loss)
        return (1 - min_dice_val)

class dice_loss_percentile(nn.Module):
    def __init__(self, percentile_val):
        super(dice_loss_percentile, self).__init__()
        self.eps=1e-7
        self.percentile_val = percentile_val
    def forward(self, x, target):

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_sorted = sorted(dice_score)
        index = x.size()[0] - int(x.size()[0] * self.percentile_val/100)
        dice_sorted = dice_sorted[index]
        return 1. - dice_sorted

class dice_loss_mean_percentile(nn.Module):
    def __init__(self, percentile_val):
        super(dice_loss_mean_percentile, self).__init__()
        self.eps=1e-7
        self.percentile_val = percentile_val
    def forward(self, x, target):

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_sorted = sorted(dice_score)
        index = x.size()[0] - int(x.size()[0] * self.percentile_val/100)
        dice_sorted = dice_sorted[index]
        return (1. - dice_score.mean()) + (1. - dice_sorted)


class dice_loss_both2(nn.Module):
    def __init__(self):
        super(dice_loss_both2, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_sorted = sorted(dice_score)
        return (1. - dice_score.mean()) + (1 - dice_sorted[0])

class dice_loss_worst2(nn.Module):
    def __init__(self):
        super(dice_loss_worst2, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_sorted = sorted(dice_score)
        dice_sorted = dice_sorted[0]
        return 1. - dice_sorted


class dice_loss_normal2(nn.Module):
    def __init__(self):
        super(dice_loss_normal2, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):
        #input_soft = F.softmax(x, dim=1)

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        return torch.mean(1. - dice_score)

class dice_loss_both(nn.Module):
    def __init__(self):
        super(dice_loss_both, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps))
        min_dice_val = torch.min(dice_loss)
        dice_loss_mean = (2. * intersection / (cardinality + self.eps)).mean()
        total = (1 - min_dice_val) + (1 - dice_loss_mean)
        return total


class dice_loss_both_weighted(nn.Module):
    def __init__(self):
        super(dice_loss_both_weighted, self).__init__()
        self.eps=1e-7
    def forward(self, x, target, weight):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps))
        min_dice_val = torch.min(dice_loss)
        dice_loss_mean = (2. * intersection / (cardinality + self.eps)).mean()
        total = (1. -weight) * (1 - min_dice_val) + (weight) * (1 - dice_loss_mean)
        return total


class dice_loss_half(nn.Module):
    def __init__(self):
        super(dice_loss_half, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        x = x.half()
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)

class dice_loss_weighted(nn.Module):
    def __init__(self):
        super(dice_loss_weighted, self).__init__()
        self.eps=1e-7
    def forward(self, x, target):

        # compute the actual dice score
        dims = (1, 2, 3, 4)
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        dice_score = 1 - dice_score
        max_val = torch.max(dice_score)
        weights = torch.div(dice_score, max_val)
        weighted_vals = max_val * weights
        return weighted_vals.mean()

class dice_loss_DANN(nn.Module):
    def __init__(self):
        super(dice_loss_DANN, self).__init__()
        self.eps=1e-7

    def forward(self, x, target):
        label_pred = x
        label_true = target[0]
        domains = target[1]

        _, domains = torch.max(domains, dim=1)
        bool_0 = torch.eq(domains, 0)
        bool_0 = bool_0.type(torch.LongTensor).cuda()

        indexs = torch.linspace(1, len(label_true), len(label_true))
        indexs = indexs.type(torch.LongTensor).cuda()

        msked_indexs = torch.mul(indexs, bool_0)
        msked_indexs = msked_indexs.type(torch.LongTensor).cuda()

        msked_indexs = msked_indexs[msked_indexs != 0]
        msked_indexs = msked_indexs - 1

        label_pred_msk_0 = label_pred[msked_indexs]
        label_true_msk_0 = label_true[msked_indexs]

        loss_0 = self._dice(label_pred_msk_0, label_true_msk_0)

        bool_1 = torch.eq(domains, 1)
        bool_1 = bool_1.type(torch.LongTensor).cuda()

        indexs = torch.linspace(1, len(label_true), len(label_true))
        indexs = indexs.type(torch.LongTensor).cuda()

        msked_indexs = torch.mul(indexs, bool_1)
        msked_indexs = msked_indexs.type(torch.LongTensor).cuda()

        msked_indexs = msked_indexs[msked_indexs != 0]
        msked_indexs = msked_indexs - 1


        label_pred_msk_1 = label_pred[msked_indexs]
        label_true_msk_1 = label_true[msked_indexs]

        loss_1 = self._dice(label_pred_msk_1, label_true_msk_1)

        loss = loss_0 + loss_1
        return loss, [loss_0, loss_1]

    def _dice(self, x, target):
        num_classes = target.shape[1]   # Channels first
        target = target.type(x.type())
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)
        dice_loss = (2. * intersection / (cardinality + self.eps)).mean()
        return (1 - dice_loss)



losses = {
    'MSELoss': nn.MSELoss,
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'BCELoss': nn.BCELoss,
    'L1Loss': nn.L1Loss,
    'NLLLoss': nn.NLLLoss,
    'KLDivLoss': nn.KLDivLoss,
    'SmoothL1Loss': nn.SmoothL1Loss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "DiceLoss": DiceLoss,
    "DuckNetDiceLoss": DuckNetDiceLoss
}


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.metrics = []
        self.cum_results = {
            "train": PairResult(),
            "val": PairResult(),
            "test": PairResult()
        }

        self.loss_fn = losses[self.hparams.loss["name"]](
            **self.hparams.loss["params"]
        )
        
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
        optimizer = self.optimizer(
            self.parameters(), self.hparams.lr["init_value"])
        schedulers = []
        if self.hparams.lr["scheduler"] is not None:
            scheduler_class = getattr(timm_scheduler, self.hparams.lr["scheduler"]["name"])
            scheduler = scheduler_class(optimizer,
                                        t_initial=self.hparams.epochs,
                                        **self.hparams.lr["scheduler"]["params"])
            schedulers.append({
                "scheduler": scheduler,
                "interval": "epoch"
            })
        return [optimizer], schedulers

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def _extract_batch(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            return batch[0], batch[1]
        else:
            return batch["input"], batch["target"]
        
    def load_pretrained(self, path:str, head_key:str = None):
        checkpoint = torch.load(path)

        if "num_classes" in self.hparams.keys():
            if "state_dict" not in checkpoint.keys():
                raise KeyError("State dict not found !")
            
            state_dict = OrderedDict()
            if checkpoint["hyper_parameters"]["num_classes"] == self.hparams.num_classes:
                head_key = None 

            for k,v in checkpoint["state_dict"].items():
                if head_key in k:
                    continue
                state_dict[k.replace("model.","")] = v
        
            if hasattr(self, "model"):
                self.model.load_state_dict(state_dict, strict=False)
            
            del state_dict
        else:
            self.load_state_dict(checkpoint["state_dict"])
        del checkpoint


class LoadModelPath:
    @staticmethod
    def load_from_mlflow(config:Config):
        tracking_uri = f"file:{str(os.path.join(config.directory.log,'mlflow'))}"
        mlflow.set_tracking_uri(tracking_uri)
        ckpt_path = Path(config.directory.log).joinpath('mlflow')

        resume_ckpt = config.model.resume_ckpt 

        runs = mlflow.search_runs(filter_string=f'attributes.run_name = "{resume_ckpt.run_name}"',
                                  experiment_names=[config.project_name]
                )
        if len(runs) == 0:
            raise ValueError("Run not found")
  
        run = runs.iloc[0]
        ckpt_path = ckpt_path.joinpath(run["experiment_id"],run["run_id"],"checkpoints",resume_ckpt.file_name)
        return ckpt_path
    

    @staticmethod
    def load(config: Config):
        if config.model.resume_ckpt is None:
            return None 
        if not isinstance(config.model.resume_ckpt, ResumeCkpt):
            return None
        
        if config.model.resume_ckpt.registry == "mlflow":
            return LoadModelPath.load_from_mlflow(config)
        
        else:
            raise ValueError("Load method not supported")

