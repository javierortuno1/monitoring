# Simple MNIST with TensorBoard Monitoring

This project demonstrates how to train a simple convolutional neural network on MNIST dataset with TensorBoard monitoring.

## Project Structure
```
simple_mnist/
├── model.py        # Contains the CNN model architecture
├── train.py        # Training script with TensorBoard logging
└── README.md
```

## Installation

### Environment Setup
1. Create a new conda environment with Python 3.8:
```bash
conda create --name tensorboard-test python=3.8
```

2. Activate the environment:
```bash
conda activate tensorboard-test
```

3. Install required packages:
```bash
# Install PyTorch with CPU support
conda install pytorch torchvision cpuonly -c pytorch

# Install TensorBoard and dependencies
conda install tensorboard six
```

## Training the Model

1. Run the training script:
```bash
python train.py
```

This will:
- Download the MNIST dataset (if not present)
- Train the model for 5 epochs
- Save logs to the 'runs' directory

## Viewing Training Progress

1. Start TensorBoard (in a separate terminal):
```bash
tensorboard --logdir=runs
```

2. Open your web browser and go to:
```
http://localhost:6006
```

## What You'll See in TensorBoard

- Training and validation loss curves
- Accuracy metrics
- Model architecture graph
- Parameter histograms
- Weight distributions

## Troubleshooting

If you encounter TensorBoard issues:
```bash
conda remove tensorboard
conda install tensorboard six
```

Then restart TensorBoard:
```bash
tensorboard --logdir=runs
```

## Model Architecture

The model is a simple CNN with:
- 2 convolutional layers
- Max pooling
- Dropout for regularization
- 2 fully connected layers
- ReLU activation functions

## Dataset

Using MNIST dataset:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes (digits 0-9)