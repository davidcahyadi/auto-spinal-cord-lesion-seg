# Training Framework

This directory contains the complete training infrastructure for the spinal cord lesion segmentation project, including a custom machine learning framework, experiment configurations, and dataset definitions.

## 📁 Directory Structure

```
training/
├── ml_framework/           # Custom ML training framework
│   ├── ml_framework/      # Core framework code
│   ├── pyproject.toml     # Package configuration
│   └── requirements.txt   # Dependencies
├── experiment/            # Model configurations and experiments
│   ├── foundation_training/
│   ├── axial_basic_classification/
│   ├── cervical_spine_sequence/
│   ├── spine_location_view/
│   ├── spinal_cord_classification/
│   ├── spinal_cord_segmentation_pretrained/
│   ├── lesion_segmentation/
│   └── lesion_segmentation_pretrained/
├── dataset/              # Dataset configuration files
└── _directory_colab.yaml # Directory configuration for Colab
```

## 🚀 Quick Start

### Installation

1. **Navigate to the ML framework directory:**

```bash
cd training/ml_framework
```

2. **Install the framework:**

```bash
pip install -e .
# or using Poetry
poetry install
```

3. **Verify installation:**

```bash
ml_framework --help
```

### Running Training

**Basic training command:**

```bash
ml_framework train --config ../experiment/lesion_segmentation/lesion_segmentation_latest.yaml
```

**With custom dataset configuration:**

```bash
ml_framework train \
    --config ../experiment/lesion_segmentation/lesion_segmentation_latest.yaml \
    --dataset ../dataset/lesion_segmentation_s1_1_320.yaml
```

## 🧠 ML Framework

The custom ML framework (`ml_framework/`) is designed specifically for medical imaging tasks with the following features:

### Core Features

-   **Modular Architecture**: Easy to extend with new models and datasets
-   **Configuration-driven**: YAML-based experiment configuration
-   **Experiment Tracking**: Integration with Weights & Biases
-   **Medical Image Support**: Specialized loaders for DICOM, NIfTI formats
-   **Data Augmentation**: Medical image-specific augmentations
-   **Multi-GPU Training**: Distributed training support
-   **Automatic Mixed Precision**: Memory-efficient training

### Framework Components

```
ml_framework/
├── base/           # Base classes for models, datasets, trainers
├── command/        # CLI commands (train, evaluate, etc.)
├── constants/      # Framework constants and configurations
├── custom/         # Custom implementations for medical imaging
├── utils/          # Utility functions and helpers
└── main.py         # CLI entry point
```

### Supported Models

-   **U-Net**: Standard and enhanced variants for segmentation
-   **ResNet**: Classification backbone with medical imaging adaptations
-   **Attention U-Net**: Segmentation with attention mechanisms
-   **Multi-task Networks**: Joint classification and segmentation

### Data Pipeline

The framework provides specialized data handling for medical images:

```python
# Example dataset configuration
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
```

## 🔧 Experiment Configuration

Each experiment is defined by a YAML configuration file containing:

### Model Configuration

```yaml
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
```

### Training Configuration

```yaml
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
```

### Callbacks and Monitoring

```yaml
callbacks:
    - name: LearningRateMonitor
      params:
          logging_interval: epoch
    - name: ModelCheckpoint
      params:
          filename: "{epoch:03d}-{val_Dice:.6f}"
          save_top_k: 5
          mode: max
          monitor: val_Dice
    - name: SegmentationSampler
      params:
          image_amount: 3
```

## 📊 Experiments

### 1. Foundation Training

Pre-training on RadImageNet for medical imaging features.

**Location**: `experiment/foundation_training/`
**Purpose**: Transfer learning base for all downstream tasks
**Models**: ResNet variants with medical imaging adaptations

### 2. Anatomy Classification

Classifies anatomical regions in MRI scans.

**Location**: `experiment/axial_basic_classification/`
**Purpose**: First stage of hierarchical classification
**Input**: Raw MRI slices
**Output**: Anatomical region labels

### 3. Spine Location & View Classification

Determines spinal location and imaging view orientation.

**Location**: `experiment/spine_location_view/`
**Purpose**: Spatial orientation classification
**Input**: Anatomically filtered images
**Output**: Location (cervical, thoracic, lumbar) and view (axial, sagittal, coronal)

### 4. Cervical Spine Sequence Classification

Identifies MRI sequence types for cervical spine images.

**Location**: `experiment/cervical_spine_sequence/`
**Purpose**: Sequence-specific processing
**Input**: Cervical spine images
**Output**: Sequence type (T1w, T2w, T2\*w, etc.)

### 5. Spinal Cord Segmentation

Segments the spinal cord structure within MRI images.

**Location**: `experiment/spinal_cord_segmentation_pretrained/`
**Purpose**: Anatomical structure segmentation
**Architecture**: U-Net with pre-trained encoder
**Input**: Classified spine images
**Output**: Spinal cord masks

### 6. Lesion Segmentation

Final precise lesion segmentation within the spinal cord.

**Location**: `experiment/lesion_segmentation/` and `lesion_segmentation_pretrained/`
**Purpose**: Lesion detection and segmentation
**Architecture**: Enhanced U-Net with attention
**Input**: Spinal cord regions
**Output**: Lesion segmentation masks

## 📈 Dataset Configurations

Dataset configurations are stored in the `dataset/` directory with naming convention:
`{task}_{split}_{augmentation_factor}_{image_size}.yaml`

### Available Datasets

-   **Foundation Training**: `foundation_radimagenet_*.yaml`
-   **Cervical Sequence**: `cervical_spine_sequence_*.yaml`
-   **Lesion Segmentation**: `lesion_segmentation_*.yaml`
-   **Spinal Cord Classification**: `sagittal_sc_classification_*.yaml`
-   **Spinal Cord Segmentation**: `sagittal_sc_segmentation_*.yaml`
-   **Spine Location**: `spine_location_view_*.yaml`

### Dataset Configuration Format

```yaml
dataset:
    task_name:
        path: "data_directory"
        crawler:
            name: crawler_type
            params: {}
        driver: pillow
        pipeline:
            - name: shuffle
              params: {}

preprocess:
    - name: load_segmentation_label
      target: [label]
      phase: [train, val, test]
    - name: alb.Resize
      params:
          width: 320
          height: 320
      target: [data, label]
      phase: [train, val, test]
    - name: to_gray
      target: [data]
      phase: [train, val, test]
```

## 🔄 Training Pipeline

### 1. Data Preparation

```bash
# Organize your data according to the dataset configuration
# Ensure proper directory structure for crawlers
```

### 2. Configuration Setup

```bash
# Copy and modify experiment configuration
cp experiment/lesion_segmentation/lesion_segmentation_template.yaml my_experiment.yaml
# Edit configuration as needed
```

### 3. Training Execution

```bash
# Start training
ml_framework train --config my_experiment.yaml

# Monitor with Weights & Biases
# Visit wandb dashboard for real-time metrics
```

### 4. Model Evaluation

```bash
# Framework automatically saves best models
# Check logs/ directory for checkpoints
```

## 🛠️ Advanced Usage

### Custom Model Implementation

1. **Create model class in `ml_framework/custom/models/`:**

```python
from ml_framework.base.model import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        # Model implementation

    def forward(self, x):
        # Forward pass
        return output
```

2. **Register model in configuration:**

```yaml
model:
    name: MyCustomModel
    custom_param: value
```

### Custom Loss Functions

```python
from ml_framework.base.loss import BaseLoss

class MyCustomLoss(BaseLoss):
    def forward(self, pred, target):
        # Loss calculation
        return loss
```

### Custom Data Augmentation

```python
from ml_framework.base.preprocess import BasePreprocess

class MyCustomAugmentation(BasePreprocess):
    def __call__(self, data):
        # Augmentation logic
        return augmented_data
```

## 📊 Monitoring and Logging

### Weights & Biases Integration

The framework automatically logs:

-   Training/validation metrics
-   Learning rate schedules
-   Model parameters
-   Sample predictions
-   System resources

### Local Logging

Logs are saved to the configured log directory:

```
logs/
├── wandb/          # W&B local files
├── checkpoints/    # Model checkpoints
└── tensorboard/    # TensorBoard logs (optional)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

    - Reduce batch size in configuration
    - Enable gradient checkpointing
    - Use automatic mixed precision

2. **Data Loading Errors**

    - Check dataset path configuration
    - Verify file permissions
    - Ensure proper data format

3. **Configuration Errors**
    - Validate YAML syntax
    - Check parameter names
    - Verify model compatibility

### Performance Optimization

1. **Data Loading**

    - Increase `num_workers` for faster data loading
    - Enable `pin_memory` for GPU training
    - Use `persistent_workers` for efficiency

2. **Training Speed**
    - Enable automatic mixed precision
    - Use gradient accumulation for large batch sizes
    - Optimize image preprocessing pipeline

## 📝 Development Guidelines

### Adding New Experiments

1. Create experiment configuration file
2. Define corresponding dataset configuration
3. Test with small subset of data
4. Document experiment purpose and results

### Code Style

-   Follow PEP 8 conventions
-   Use type hints where applicable
-   Document complex functions
-   Write unit tests for new components

## 🔗 Related Documentation

-   [Main Project README](../README.md)
-   [Inference Documentation](../inference/README.md)
-   [PyTorch Lightning Documentation](https://lightning.ai/docs/)
-   [Weights & Biases Documentation](https://docs.wandb.ai/)

## 📞 Support

For training-specific questions:

-   Check experiment logs for error details
-   Review configuration file syntax
-   Ensure data paths are correct
-   Monitor resource usage during training

For technical support, contact: liu.david.chd@gmail.com
