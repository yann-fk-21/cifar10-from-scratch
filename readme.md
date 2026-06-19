# CIFAR-10 from Scratch

This project implements a custom **TinyVGG** convolutional neural network to classify images from the **CIFAR-10** dataset using PyTorch.

## Features
- Data downloading and preprocessing using `kagglehub`.
- Custom training loop using PyTorch.
- Training and validation metrics tracking.
- Saving trained models for future use.

## Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- kagglehub
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yann-fk-21/cifar10-from-scratch.git
   cd cifar10-from-scratch
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To start training the model, run `src/train.py` directly:
```bash
python src/train.py --batch_size 32 --epochs 5 --lr 0.001 --hidden_units 10
```

## Project Structure
- `src/`: Contains the core implementation files.
  - `model_builder.py`: Defines the `TinyVGG` architecture.
  - `data_setup.py`: Handles data downloading, preprocessing, and DataLoader creation.
  - `train.py`: The main entry point for the training process.
  - `engine.py`: Contains helper functions for training and validation loops.
  - `predict.py`: Contains utilities for model inference.
  - `utils.py`: Contains various helper functions, such as saving models.
- `notebooks/`: Contains Jupyter notebooks for exploration and demonstration.
- `requirements.txt`: Python package dependencies.
