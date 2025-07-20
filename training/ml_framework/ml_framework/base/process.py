
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import albumentations as A
import albumentations
import albumentations.pytorch
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from typing import Any, Dict, List, Optional
from scipy.ndimage import gaussian_filter
from ml_framework.custom.ms_lesion_augmentor import MSLesionAugmentor

from ml_framework.constants.app import MetadataHeader, ProcessTarget


class ProcessPipeline(metaclass=ABCMeta):
    def __init__(self, name: str,
                 phases: list[str],
                 target: list[str],
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        self.name = name
        self.phases = phases
        self.target = target
        self.full_metadata = full_metadata

    @abstractmethod
    def transform(self, metadata: dict, data: Any, label: Any):
        pass


class AlbumentationWrapper(ProcessPipeline):
    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None
                 ) -> None:
        name = name.replace("alb.", "")
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)
        if self.name == "ToTensorV2":
            self.transformer = getattr(albumentations.pytorch,self.name)(**params)
        else:
            self.transformer = getattr(A, self.name)(**params)

    def transform(self, metadata: dict, data: Any, label: Any):
        process = {}
        is_data = ProcessTarget.DATA in self.target
        is_label = ProcessTarget.LABEL in self.target
        if is_data:
            process["image"] = data
        if is_label:
            process["mask"] = label

        result = self.transformer(**process)
        if is_data:
            data = result["image"]
        if is_label:
            label = result["mask"]

        return data, label


class TorchVisionWrapper(ProcessPipeline):

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        name = name.replace("tv.", "")
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)
        if params.get("dtype") is not None and isinstance(params["dtype"], str):
            params["dtype"] = getattr(torch, params["dtype"])

        self.transformer = getattr(v2, self.name)(**params)

    def transform(self, metadata: dict, data: Any, label: Any):
        if ProcessTarget.DATA in self.target:
            data = self.transformer(data)
        if ProcessTarget.LABEL in self.target:
            label = self.transformer(label)
        return data, label

class ToGray(ProcessPipeline):
  
    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)



    def transform(self, metadata: dict, data: Any, label: Any):
        if ProcessTarget.DATA in self.target:
            if isinstance(data, np.ndarray):
                if data.shape[-1] == 3:
                    gray_img = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
                    data = np.expand_dims(gray_img, axis=2)

            else:
                raise ValueError("Data is not in numpy type!")
        return data, label
  

class LabelEncoder(ProcessPipeline):

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)
        self.ordered = params["ordered"]
        self.encoding = {}
        self._defines_encoding()

    def _defines_encoding(self):
        if self.full_metadata is None:
            raise ValueError("Full metadata is required for label encoding")

        unique_dataset = self.full_metadata[MetadataHeader.DATASET].unique().tolist()

        for dataset_name in unique_dataset:
            unique_labels = self.full_metadata[self.full_metadata[MetadataHeader.DATASET]
                                            == dataset_name].label.unique()
            unique_labels = set(self.encoding.keys()).union(unique_labels)
            if self.ordered:
                unique_labels = sorted(unique_labels)
            
            self.encoding = {
                label: i for i, label in enumerate(unique_labels)
            }
        

    def transform(self, metadata: dict, data: Any, label: Any):
        if ProcessTarget.LABEL in self.target:
            label = self.encoding[label]
        return data, label


class LoadSegmentationLabel(ProcessPipeline):

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)

    def transform(self, metadata: dict, data: Any, label: Any):
        if ProcessTarget.LABEL in self.target:
            arr = np.array(Image.open(metadata[MetadataHeader.LABEL]))
            label = np.expand_dims((arr > 0).astype(np.uint8), axis=-1)

        return data, label

class OneHotEncoder(ProcessPipeline):

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)
        self.total_label = {}
        self._defines_encoding()

    def _defines_encoding(self):
        if self.full_metadata is None:
            raise ValueError("Full metadata is required for label encoding")

        unique_labels = self.full_metadata.label.unique()
        self.total_label = len(unique_labels)

    def transform(self, metadata: dict, data: Any, label: Any):
        if not isinstance(label, int):
            raise ValueError("Label must be integer for one hot encoding")
        if ProcessTarget.LABEL in self.target:
            label = np.eye(self.total_label)[label]
        return data, label


class ArgMax(ProcessPipeline):

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)

    def transform(self, metadata: dict, data: Any, label: Any):
        if ProcessTarget.PREDICTION in self.target:
            if isinstance(data, torch.Tensor):
                data = data.argmax(dim=1)
        return data, label


class LesionAug(ProcessPipeline):
    def __init__(self, name, phases, target,
                 params,full_metadata = None):
        super().__init__(name, phases, target, full_metadata)
        self.augmentor = MSLesionAugmentor(
            min_lesions=params["min_lesions"],
            max_lesions=params["max_lesions"],
            min_size=params["min_size"],
            max_size=params["max_size"],
        )

    def transform(self, metadata, data, label):
        data, label = self.augmentor.augment(data, label)
        return data, label
    



class CustomBiasField(ProcessPipeline):
    """
    Simulates MRI bias field by applying a smooth multiplicative intensity non-uniformity.
    This transform is specifically designed for MRI data augmentation.
    """

    def __init__(self,
                 name: str,
                 target: list[str],
                 phases: List[str],
                 params: Dict,
                 full_metadata: Optional[pd.DataFrame] = None) -> None:
        """
        Initialize the CustomBiasField augmentation.
        
        Args:
            name: Name of the transformation
            target: List of targets to apply the transform to (data, label, prediction)
            phases: List of phases when to apply the transform (train, val, test)
            params: Dictionary containing transform parameters:
                - coefficient_range: Range of coefficient values (default: [0.3, 0.7])
                - order: Order of the bias field polynomial (default: 3)
                - p: Probability of applying the augmentation (default: 0.5)
            full_metadata: Optional dataframe containing full dataset metadata
        """
        super().__init__(name=name,
                         target=target,
                         phases=phases,
                         full_metadata=full_metadata)
        
        # Extract parameters with defaults
        self.coefficient_range = params.get('coefficient_range', [0.3, 0.7])
        self.order = params.get('order', 3)
        self.p = params.get('p', 0.5)

    def transform(self, metadata: dict, data: Any, label: Any):
        """
        Apply bias field augmentation to the input data.
        
        Args:
            metadata: Dictionary containing metadata for the current sample
            data: Input image data
            label: Segmentation label data
            
        Returns:
            Tuple of (transformed_data, label)
        """
        if np.random.random() > self.p:
            return data, label
            
        # Process each target independently
        if ProcessTarget.DATA in self.target and data is not None:
            data = self._apply_bias_field(data)
            
        # Note: We don't apply bias field to labels or predictions as it's an intensity transform
            
        return data, label
    
    def _apply_bias_field(self, image):
        """
        Internal method to apply the bias field transformation.
        
        Args:
            image: Input image (numpy array or torch tensor)
            
        Returns:
            Transformed image with same type as input
        """
        # Convert torch tensor to numpy if needed
        is_torch = isinstance(image, torch.Tensor)
        original_device = None
        
        if is_torch:
            original_device = image.device
            original_dtype = image.dtype
            image = image.cpu().numpy()
        
        # Store original shape and data range
        original_shape = image.shape
        is_uint8 = image.dtype == np.uint8
        min_val = 0
        max_val = 255 if is_uint8 else np.max(image)
        
        # Handle multi-channel images
        if len(original_shape) > 2:
            # For 3D volumes or multi-channel 2D
            if len(original_shape) == 4:  # 3D volume with channels
                h, w, d = original_shape[1:4]
                channels = original_shape[0]
                reshaped = image.reshape(channels, -1)
            elif len(original_shape) == 3:  # 2D with channels
                h, w = original_shape[1:3]
                channels = original_shape[0]
                reshaped = image.reshape(channels, h, w)
        else:
            # Single channel 2D
            h, w = original_shape
            channels = 1
            reshaped = image.reshape(1, h, w)
        
        # Create coordinate grid
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        
        # Normalize coordinates to [-1, 1]
        y_grid = 2 * (y_grid / h) - 1
        x_grid = 2 * (x_grid / w) - 1
        
        # Generate random coefficients for polynomial basis
        num_coeffs = (self.order + 1) * (self.order + 2) // 2
        coefficients = np.random.uniform(
            low=self.coefficient_range[0],
            high=self.coefficient_range[1],
            size=num_coeffs
        )
        
        # Create bias field using polynomial basis functions
        bias_field = np.zeros((h, w))
        idx = 0
        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                bias_field += coefficients[idx] * (x_grid ** i) * (y_grid ** j)
                idx += 1
        
        # Apply Gaussian smoothing to make it more realistic
        bias_field = gaussian_filter(bias_field, sigma=min(h, w) / 30)
        
        # Normalize bias field to [0.8, 1.2] range (±20% intensity variation)
        bias_field = 0.8 + 0.4 * (bias_field - np.min(bias_field)) / (np.max(bias_field) - np.min(bias_field) + 1e-10)
        
        # Apply bias field multiplicatively to each channel
        result = np.zeros_like(reshaped)
        for c in range(channels):
            if len(original_shape) == 4:  # 3D volume
                for d_idx in range(d):
                    result[c, :, :, d_idx] = reshaped[c, :, :, d_idx] * bias_field
            else:  # 2D image
                result[c] = reshaped[c] * bias_field
        
        # Reshape back to original shape
        result = result.reshape(original_shape)
        
        # Clip values to maintain original range
        if is_uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Convert back to torch tensor if needed
        if is_torch:
            result = torch.from_numpy(result).to(device=original_device, dtype=original_dtype)
            
        return result