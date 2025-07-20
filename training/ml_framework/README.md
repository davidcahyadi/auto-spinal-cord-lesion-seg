# ML Framework

A custom machine learning framework designed specifically for medical imaging tasks, particularly spinal cord lesion segmentation. This framework provides a modular, configuration-driven approach to training deep learning models on medical images.

## 🌟 Features

-   **Medical Image Focused**: Specialized for DICOM, NIfTI, and medical image formats
-   **Configuration-Driven**: YAML-based experiment management
-   **Modular Architecture**: Easy to extend and customize
-   **PyTorch Lightning Integration**: Distributed training and best practices
-   **Experiment Tracking**: Built-in Weights & Biases integration
-   **Data Pipeline**: Efficient medical image loading and preprocessing
-   **Auto Mixed Precision**: Memory-efficient training
-   **Model Zoo**: Pre-implemented medical imaging models

## 🚀 Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# Or using Poetry
poetry install
```

### Basic Usage

```bash
# Train a model
ml_framework train --config experiment_config.yaml

# Get help
ml_framework --help
```

## 📋 Requirements

-   Python 3.10+
-   PyTorch 2.0+
-   PyTorch Lightning 2.0+
-   See `pyproject.toml` for complete dependencies

## 🏗️ Architecture

### Core Components

```
ml_framework/
├── base/              # Abstract base classes
│   ├── __init__.py
│   ├── model.py       # BaseModel class
│   ├── dataset.py     # BaseDataset class
│   ├── trainer.py     # BaseTrainer class
│   └── loss.py        # BaseLoss class
├── command/           # CLI commands
│   ├── __init__.py
│   └── train.py       # Training command
├── constants/         # Framework constants
├── custom/           # Custom implementations
│   ├── models/       # Medical imaging models
│   ├── datasets/     # Medical dataset loaders
│   ├── losses/       # Medical imaging losses
│   └── transforms/   # Medical image transforms
├── utils/            # Utility functions
└── main.py           # CLI entry point
```

### Base Classes

#### BaseModel

Abstract base for all models with medical imaging utilities:

```python
from ml_framework.base.model import BaseModel

class MyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Model implementation

    def forward(self, x):
        # Forward pass
        return output
```

#### BaseDataset

Medical image dataset handling:

```python
from ml_framework.base.dataset import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)
        # Dataset implementation

    def __getitem__(self, idx):
        # Load and preprocess medical image
        return data, label
```

## ⚙️ Configuration

### Experiment Configuration

Complete experiment setup via YAML:

```yaml
project_name: "Spinal Cord Lesion Segmentation"
seed: 420
directory: "../../_directory_ssd.yaml"

model:
    name: UNET
    in_channels: 1
    out_channels: 1
    depth: 4
    features: 16
    out_activation: null
    last_activation: sigmoid
    use_batch_norm: True
    loss:
        name: DiceLoss
        params: {}

trainer:
    name: "pytorch_lightning"
    epochs: 200
    log_every: 1
    val_every: 1
    gpus: 1
    precision: 32
    lr:
        init_value: 1e-3
    optimizer:
        name: Adam
        params: {}
    loader:
        batch_size: 64
        num_workers: 4
        pin_memory: true
        persistent_workers: true

callbacks:
    - name: LearningRateMonitor
      params:
          logging_interval: epoch
    - name: ModelCheckpoint
      params:
          filename: "{epoch:03d}-{val_Dice:.6f}"
          save_top_k: 5
          save_last: true
          mode: max
          monitor: val_Dice
```

### Dataset Configuration

Separate dataset configuration files:

```yaml
dataset:
    lesion_segmentation:
        path: "lesion_segmentation"
        crawler:
            name: segmentation_file
            params: {}
        driver: pillow
        pipeline:
            - name: shuffle
              params: {}

preprocess:
    - name: load_segmentation_label
      params: {}
      target: [label]
      phase: [train, val, test]
    - name: alb.Resize
      params:
          width: 320
          height: 320
      target: [data, label]
      phase: [train, val, test]
    - name: to_gray
      params: {}
      target: [data]
      phase: [train, val, test]
    - name: alb.RandomBrightnessContrast
      params:
          brightness_limit: 0.2
          contrast_limit: 0.2
      target: [data]
      phase: [train]
```

## 🧠 Supported Models

### Segmentation Models

-   **U-Net**: Standard implementation with customizable depth
-   **Attention U-Net**: U-Net with attention gates
-   **U-Net++**: Nested U-Net architecture
-   **DeepLab**: Dilated convolutions for segmentation

### Classification Models

-   **ResNet**: Medical imaging adapted ResNet variants
-   **EfficientNet**: Efficient convolutional networks
-   **Vision Transformer**: Transformer-based classification

### Custom Models

Easy to add custom models by extending `BaseModel`:

```python
from ml_framework.base.model import BaseModel
import torch.nn as nn

class CustomUNet(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super().__init__()
        # Custom implementation
        self.encoder = self._build_encoder(in_channels)
        self.decoder = self._build_decoder(out_channels)

    def forward(self, x):
        # Custom forward pass
        features = self.encoder(x)
        output = self.decoder(features)
        return output
```

## 📊 Data Pipeline

### Medical Image Support

-   **DICOM**: Direct DICOM file reading
-   **NIfTI**: Neuroimaging format support
-   **Standard Formats**: PNG, JPEG, TIFF support

### Preprocessing Pipeline

```python
# Example preprocessing configuration
preprocess:
    - name: load_segmentation_label  # Load ground truth masks
    - name: alb.Resize              # Resize to target size
      params: {width: 320, height: 320}
    - name: to_gray                 # Convert to grayscale
    - name: normalize_intensity     # Medical image normalization
    - name: alb.RandomBrightnessContrast  # Augmentation
      phase: [train]                # Only during training
```

### Data Crawlers

Specialized data loaders for medical imaging:

-   `segmentation_file`: Paired image-mask loading
-   `classification_folder`: Folder-based classification
-   `dicom_series`: DICOM series loading
-   `nifti_volume`: NIfTI volume loading

## 🔧 Training Features

### PyTorch Lightning Integration

-   Automatic distributed training
-   Mixed precision training
-   Gradient accumulation
-   Advanced callbacks

### Callbacks

```yaml
callbacks:
    - name: ModelCheckpoint
      params:
          monitor: val_Dice
          mode: max
          save_top_k: 5

    - name: EarlyStopping
      params:
          monitor: val_loss
          patience: 20

    - name: LearningRateMonitor
      params:
          logging_interval: epoch

    - name: SegmentationSampler # Custom medical imaging callback
      params:
          image_amount: 3
          log_frequency: 10
```

### Loss Functions

Medical imaging specific losses:

-   **DiceLoss**: For segmentation tasks
-   **FocalLoss**: For imbalanced datasets
-   **CombinedLoss**: Multiple loss combination
-   **WeightedCrossEntropy**: Class-weighted loss

```yaml
loss:
    name: CombinedLoss
    params:
        losses:
            - name: DiceLoss
              weight: 0.5
            - name: FocalLoss
              weight: 0.5
```

## 📈 Experiment Tracking

### Weights & Biases Integration

Automatic logging of:

-   Training/validation metrics
-   Model parameters and gradients
-   Sample predictions
-   System resources
-   Hyperparameters

### Custom Metrics

Medical imaging specific metrics:

-   Dice coefficient
-   Intersection over Union (IoU)
-   Sensitivity/Specificity
-   Hausdorff distance

## 🛠️ Advanced Usage

### Custom Loss Implementation

```python
from ml_framework.base.loss import BaseLoss
import torch.nn as nn

class TverskyLoss(BaseLoss):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        # Tversky loss implementation
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = tp / (tp + self.alpha * fp + self.beta * fn)
        return 1 - tversky
```

### Custom Data Augmentation

```python
from ml_framework.base.preprocess import BasePreprocess
import albumentations as alb

class MedicalAugmentation(BasePreprocess):
    def __init__(self, intensity_range=0.1):
        self.transform = alb.Compose([
            alb.RandomBrightnessContrast(
                brightness_limit=intensity_range,
                contrast_limit=intensity_range
            ),
            alb.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50
            )
        ])

    def __call__(self, data):
        return self.transform(image=data['image'], mask=data['mask'])
```

### Multi-GPU Training

```yaml
trainer:
    strategy: "ddp" # Distributed Data Parallel
    devices: [0, 1, 2, 3] # Use GPUs 0-3
    precision: 16 # Mixed precision
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**

    ```yaml
    loader:
        batch_size: 32 # Reduce batch size
    trainer:
        precision: 16 # Use mixed precision
    ```

2. **Data Loading Errors**

    - Check file paths in dataset configuration
    - Verify image formats are supported
    - Ensure proper file permissions

3. **Training Instability**
    - Reduce learning rate
    - Add gradient clipping
    - Use learning rate scheduling

### Performance Optimization

```yaml
loader:
    num_workers: 8 # Parallel data loading
    pin_memory: true # GPU memory optimization
    persistent_workers: true # Reuse workers

trainer:
    precision: 16 # Mixed precision
    accumulate_grad_batches: 4 # Gradient accumulation
```

## 📚 Examples

### Basic Segmentation Training

```bash
ml_framework train --config configs/unet_segmentation.yaml
```

### Classification with Custom Dataset

```bash
ml_framework train \
    --config configs/resnet_classification.yaml \
    --dataset configs/custom_dataset.yaml
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ml_framework train \
    --config configs/distributed_training.yaml
```

## 🔗 Dependencies

Core dependencies managed in `pyproject.toml`:

-   `torch >= 2.0.0`: Deep learning framework
-   `lightning >= 2.2.0`: Training utilities
-   `albumentations >= 1.4.0`: Data augmentation
-   `wandb >= 0.17.3`: Experiment tracking
-   `timm >= 1.0.9`: Model architectures
-   `pandas >= 2.2.0`: Data manipulation
-   `scikit-learn >= 1.4.0`: Machine learning utilities

## 🚀 Future Enhancements

-   [ ] Support for 3D medical images
-   [ ] Advanced medical image preprocessing
-   [ ] Integration with medical imaging standards (PACS)
-   [ ] Model interpretability tools
-   [ ] Automated hyperparameter optimization
-   [ ] Model compression utilities

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This framework is part of the spinal cord lesion segmentation research project. See the main project LICENSE for details.

## 📞 Support

For framework-specific issues:

-   Check configuration syntax
-   Review model compatibility
-   Verify data pipeline setup
-   Monitor resource usage

Contact: liu.david.chd@gmail.com
