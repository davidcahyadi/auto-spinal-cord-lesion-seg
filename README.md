# auto-spinal-cord-lesion-seg
Inference source code, sample data and model weights for the journal article "Optimizing Spinal Cord Lesion Segmentation Using Hier-archical Classification and U-NET Based Segmentation Model"

## Prerequisites
- Python 3.10 or higher
- Git

## Installation
1. Clone the repository
```bash
git clone https://github.com/your-repo/auto-spinal-cord-lesion-seg.git
```
2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install the dependencies
```bash
pip install -r requirements.txt
```
## Model Weights
You'll need to download the pre-trained model weights and place them in a `weights` directory:
- `anatomy.ckpt`
- `spine_location_view.ckpt`
- `spine_sequence.ckpt`
- `spinal_cord.ckpt`
- `lesion_segmentation.ckpt`

Notes: To download the weights, you can download from this [link](https://drive.google.com/drive/folders/1xFOiLHuXXMFXSrvEGC4iggeEZIxP3kpA?usp=drive_link).

## Data Sample
To see the sample data, you can download from this [link](https://drive.google.com/drive/folders/1oklPwOPRiev0fWvoQ7D4yd6RjENhuOfu?usp=drive_link).

The data it self originated from the [Spine Generic Dataset](https://github.com/spine-generic/data-multi-subject) with some modifications to add synthetic lesions.

## Usage

To see the usage of the CLI tool, run:
```bash
python src/cli.py --help
```

To run the inference, you can use the following command:
```bash
python src/cli.py inference -i <path/to/input/directory> -o <path/to/output/directory>
```
