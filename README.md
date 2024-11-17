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
You'll need to download the pre-trained model weights and place them in a weights directory:
- `anatomy.ckpt`
- `spine_location_view.ckpt`
- `spine_sequence.ckpt`
- `spinal_cord.ckpt`
- `lesion_segmentation.ckpt`

Notes: To download the weights, you can download from this link. (We will upload the weights to the cloud soon.)

## Data Sample
To see the sample data, you can download from this link. (We will upload the data to the cloud soon.)

## Usage

To see the usage of the CLI tool, run:
```bash
python src/cli.py --help
```

To run the inference, you can use the following command:
```bash
python src/cli.py inference -i <path/to/input/directory> -o <path/to/output/directory>
```
