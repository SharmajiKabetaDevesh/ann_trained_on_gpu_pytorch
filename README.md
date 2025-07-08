

# Comparing Training on GPU vs CPU using PyTorch

This repository demonstrates how training a simple neural network model using PyTorch differs in terms of performance when run on a **CPU** versus a **GPU**. The goal is to showcase how GPU acceleration can significantly reduce training time for deep learning models, even for relatively simple architectures.

## Table of Contents

* [Overview](#overview)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Setup](#training-setup)
* [How to Run](#how-to-run)

  * [Running on CPU](#running-on-cpu)
  * [Running on GPU](#running-on-gpu)
* [Performance Comparison](#performance-comparison)
* [Conclusion](#conclusion)


---

## Overview

The notebook `AnnusingGPUSpytorch.ipynb` trains a simple feedforward neural network on the popular MNIST dataset. It compares training time and model performance on both CPU and GPU, illustrating the computational efficiency of GPU-based training.

---

## Requirements

Install the required packages using:

```bash
pip install torch torchvision matplotlib
```

For GPU support, ensure:

* CUDA-compatible GPU is available
* Proper NVIDIA drivers and CUDA Toolkit are installed
* PyTorch is installed with CUDA support

Check your CUDA-enabled PyTorch installation with:

```python
import torch
torch.cuda.is_available()  # should return True for GPU support
```

---

## Dataset

The notebook uses the **MNIST** dataset, which consists of 28x28 grayscale images of handwritten digits (0–9). It is automatically downloaded using `torchvision.datasets.MNIST`.

---

## Model Architecture

The model is a simple feedforward Artificial Neural Network (ANN) with:

* Input layer: 784 neurons (flattened 28x28 image)
* Hidden layer: 128 neurons with ReLU activation
* Output layer: 10 neurons (for digits 0–9) with LogSoftmax activation

Loss function: Negative Log Likelihood (`nn.NLLLoss`)
Optimizer: Stochastic Gradient Descent (`torch.optim.SGD`)

---

## Training Setup

The code supports training on either CPU or GPU. You can control the device using:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The training loop moves both the model and input tensors to the selected device, allowing seamless switching between CPU and GPU execution.

---

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. Open the notebook:

   ```bash
   jupyter notebook AnnusingGPUSpytorch.ipynb
   ```

3. Execute the notebook cells in sequence.

### Running on CPU

If you don’t have a GPU, or wish to force CPU usage:

```python
device = torch.device("cpu")
```

### Running on GPU

To utilize a GPU (if available):

```python
device = torch.device("cuda")
```

This will speed up tensor operations and significantly reduce training time.

---

## Performance Comparison

| Device | Epochs | Total Training Time | Accuracy |
| ------ | ------ | ------------------- | -------- |
| CPU    | 5      | \~X minutes         | \~Y%     |
| GPU    | 5      | \~Z seconds         | \~Y%     |

> Replace X, Y, Z with actual values after running the notebook.

Generally, GPU training results in **faster execution** with **identical accuracy**, assuming the same training configuration.

---

## Conclusion

This project highlights the advantage of GPU acceleration for training deep learning models using PyTorch. Even simple models can benefit significantly from GPU usage in terms of speed, making iterative experimentation and prototyping more efficient.

For more complex models or larger datasets, the performance gain from using GPUs becomes even more substantial.


