from pathlib import Path
from time import time
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


class ImageSegmenter:
    """Orchestrates hierarchical segmentation of medical images using deep learning models."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """
        Initialize the segmenter with necessary components.

        Args:
            model: Model to use for inference
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for inference
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def segment(
        self,
        dataset: torch.utils.data.Dataset,
        output_dir: Path,
    ) -> tuple[List[str], float]:
        """
        Processes a series of images and saves segmentation masks.

        Returns:
            Tuple of (segmented paths, total processing time)
        """
        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False
        )

        segmented_paths = []
        total_images = 0
        start_time = time()

        for batch, paths in dataloader:
            batch_masks = self._get_segmentation(self.model, batch)

            for i, path in enumerate(paths):
                # Preserve the folder structure
                rel_path = Path(path)

                # Create output path with 'predict_' prefix
                output_path = (
                    Path(output_dir) / rel_path.parent / f"predict_{rel_path.name}"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save the segmentation mask as PNG
                mask = batch_masks[i].cpu().numpy().astype(np.uint8)
                img = Image.fromarray(mask.squeeze())
                output_path = output_path.with_suffix(".png")  # Ensure .png extension
                img.save(output_path)
                segmented_paths.append(str(output_path))

            total_images += len(paths)

        end_time = time()
        total_time = end_time - start_time

        return total_time

    def _get_segmentation(
        self, model: torch.nn.Module, batch: torch.Tensor
    ) -> torch.Tensor:
        """Gets model predictions for a batch of images."""
        with torch.inference_mode():
            preds = model(batch.to(self.device))
            preds = torch.sigmoid(preds)
            return (preds > 0.5).float() * 255
