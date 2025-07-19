from collections import Counter
from pathlib import Path
from time import time
from typing import List

import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm


class ImageClassifier:
    """Orchestrates hierarchical filtering of medical images using deep learning models."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        batch_size: int = 8,
        aggregate_by: str = "majority",
    ):
        """
        Initialize the filter with necessary components.

        Args:
            model: Model to use for inference
            dataset: Dataset to use for inference
            device: Device to run inference on ('cuda' or 'cpu')
            batch_size: Batch size for inference
            aggregate_by: Method to aggregate predictions ('majority' or 'single')
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.aggregate_by = aggregate_by

    def classify(
        self,
        dataset: torch.utils.data.Dataset,
        selected_label: int | list[int],
        group_level: int = 1,
        min_series_length: int = 4,
    ) -> tuple[List[str], float]:
        """
        Processes a single series of images.

        Returns:
            Tuple of (filtered paths, average processing time per image)
        """
        if isinstance(selected_label, int):
            selected_label = [selected_label]

        dataloader = DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=False
        )

        grouped_paths = {}
        total_images = 0
        start_time = time()

        for batch, paths in tqdm(dataloader, desc="Classifying images"):
            batch_predictions = self._get_predictions(self.model, batch)

            for i, path in enumerate(paths):
                prediction = batch_predictions[i].item()
                group_name = Path(path).parts[-(group_level + 1)]

                if group_name not in grouped_paths:
                    grouped_paths[group_name] = []
                grouped_paths[group_name].append((path, prediction))
            total_images += len(paths)

        end_time = time()
        total_time = end_time - start_time

        # Analyze predictions by group
        filtered_paths = []
        for group_name, path_predictions in grouped_paths.items():
            predictions = [pred for _, pred in path_predictions]
            pred_counter = Counter(predictions)
            majority_label, _ = pred_counter.most_common(1)[0]

            if (
                majority_label in selected_label
                and len(predictions) >= min_series_length
            ):
                filtered_paths.extend([path for path, _ in path_predictions])

        return filtered_paths, total_time

    def _get_predictions(
        self, model: torch.nn.Module, batch: torch.Tensor
    ) -> torch.Tensor:
        """Gets model predictions for a batch of images."""
        with torch.inference_mode():
            preds = model(batch.to(self.device))
            preds = softmax(preds, dim=1)
            return preds.argmax(dim=1)
