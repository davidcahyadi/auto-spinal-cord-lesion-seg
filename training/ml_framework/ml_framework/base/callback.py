
import lightning.pytorch.callbacks as pl_callbacks
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger
from ml_framework.base.schema import PipelineConfig
from pathlib import Path 
import numpy as np
import wandb
import torch



class SegmentationImageSampling(pl_callbacks.Callback):
    def __init__(
        self,
        image_amount=5,
        class_labels={},
    ):
        self.image_amount = image_amount
        self.class_labels = {}
        for key in class_labels.keys():
            self.class_labels[int(key)] = class_labels[key]

    def _get_image(self,data, index):
        d = data[index]
        if d.shape[0] == 3:
            d = np.transpose(d, (1, 2, 0))
        return d

    def _create_segmentation_image(self, images, targets, preds):
        imgs = []
        for i in range(images.shape[0]):
            image = self._get_image(images, i)
            target = self._get_image(targets, i)
            pred = self._get_image(preds, i)


            img = wandb.Image(
                image,
                masks={
                    "ground truth": {
                        "mask_data": target.squeeze(),
                        "class_labels": self.class_labels,
                    },
                    "prediction": {
                        "mask_data": pred.squeeze(),
                        "class_labels": self.class_labels,
                    },
                },
                caption=f"Image #{i + 1}",
            )
            imgs.append(img)
        return imgs

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        pl_module.in_step_mode = False
        with torch.inference_mode():
            images, targets = next(iter(trainer.val_dataloaders))
            images = images[:self.image_amount]
            targets = targets[:self.image_amount]
            if pl_module.hparams.last_activation == "sigmoid":
                preds = torch.sigmoid(pl_module(images.to(pl_module.device)))
            else:
                preds = pl_module(images.to(pl_module.device))
            preds = (preds > 0.5).int()
            preds = preds.cpu().detach().numpy()
            imgs = self._create_segmentation_image(
                images.cpu().numpy(), targets.cpu().numpy(), preds
            )
            wandb.log({"Segmentation Sample": imgs})
        pl_module.train()
        pl_module.in_step_mode = True



class PruningCallback(pl_callbacks.Callback):
    """
    Callback to handle pruning and fine-tuning cycles during training.
    """
    def __init__(
        self,
        prune_every_n_epochs: int = 5,
        prune_percentage: float = 10,
        pruning_mode: str = 'Taylor',
        total_pruning_iterations: int = 5,
        recovery_epochs: int = 10
    ):
        super().__init__()
        self.prune_every_n_epochs = prune_every_n_epochs
        self.prune_percentage = prune_percentage
        self.pruning_mode = pruning_mode
        self.total_pruning_iterations = total_pruning_iterations
        self.recovery_epochs = recovery_epochs
        self.current_pruning_iteration = 0
        self.fine_tuning_epoch = 0
        self.is_fine_tuning = False
        self.dropout_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        return self.on_epoch_end(trainer, pl_module)
    
    def on_epoch_end(self, trainer, pl_module):
        """Handle pruning and fine-tuning at the end of each epoch"""
        # Check if we need to start a new pruning cycle
        if (
            not self.is_fine_tuning and 
            (trainer.current_epoch + 1) % self.prune_every_n_epochs == 0 and
            self.current_pruning_iteration < self.total_pruning_iterations
        ):
            # Perform pruning
            print(f"\nStarting pruning iteration {self.current_pruning_iteration + 1}/{self.total_pruning_iterations}")
            
            # Get a batch of training data for pruning analysis
            train_dataloader = trainer.train_dataloader
            
            # Prune the model
            new_drop = pl_module.prune_model(train_dataloader)
            
            # Log the new dropout rates
            print(f"New Dropout Values = {new_drop}")
            self.dropout_history.append(new_drop)
            
            # Save the pruned model
            pruned_path = f"pruned_model_iter_{self.current_pruning_iteration}.pt"
            trainer.save_checkpoint(pruned_path)
            
            # Start fine-tuning phase
            self.is_fine_tuning = True
            self.fine_tuning_epoch = 0
            self.current_pruning_iteration += 1
        
        # Fine-tuning phase after pruning
        if self.is_fine_tuning:
            self.fine_tuning_epoch += 1
            
            # Log fine-tuning progress
            print(f"Fine-tuning Epoch {self.fine_tuning_epoch}/{self.recovery_epochs} after pruning iteration {self.current_pruning_iteration}")
            
            # Save fine-tuned model
            if self.fine_tuning_epoch % 5 == 0 or self.fine_tuning_epoch == self.recovery_epochs:
                finetuned_path = f"finetuned_model_iter_{self.current_pruning_iteration-1}_epoch_{self.fine_tuning_epoch}.pt"
                trainer.save_checkpoint(finetuned_path)
            
            # End fine-tuning phase if we've completed the recovery epochs
            if self.fine_tuning_epoch >= self.recovery_epochs:
                self.is_fine_tuning = False




class CustomCallbacks:
    SegmentationSampler = SegmentationImageSampling
    PruningCallback = PruningCallback

class Callback:
    _library= {
        "custom": CustomCallbacks,
        "pl": pl_callbacks
    }

    @staticmethod
    def parse(config: list[PipelineConfig], additional_props:dict) -> None:
        callbacks = []
        for cb in config:
            cb = Callback.update_config(cb, additional_props)
            for callback_class in Callback._library.values():
                if cb.name in dir(callback_class):
                    print("Assign callback :",cb.name)
                    callback = getattr(callback_class, cb.name)(**cb.params)
                    callbacks.append(callback)
        return callbacks
    

    @staticmethod
    def update_config(config: PipelineConfig, additional_props):
        if config.name == "ModelCheckpoint":
            if additional_props["logger"] and isinstance(additional_props["logger"], WandbLogger):
                print("Found Model Checkpoint and WandbLogger")
                config.params["dirpath"] = str(Path(additional_props["logger"].save_dir).joinpath(
                        "wandb",
                        additional_props["logger"].experiment._settings.sync_dir,
                        "checkpoints"
                    )
                )
            if additional_props["logger"] and isinstance(additional_props["logger"], MLFlowLogger):
                config.params["pathdir"] = None
                # TODO: Create setting for mlflow
        return config

