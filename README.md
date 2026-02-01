# CIFAR-10 Optimizer Comparison Experiments

This repository contains a set of experiments comparing different optimization algorithms for training a convolutional neural network (CNN) on the CIFAR-10 dataset using PyTorch.

The core objective of this project is to understand how different optimizers affect:
- Convergence speed
- Training stability
- Final validation accuracy

All experiments are implemented and documented in a single Jupyter Notebook.

---

## 📘 Project Overview

CIFAR-10 is a standard benchmark dataset in computer vision, consisting of 60,000 32×32 RGB images across 10 classes.  
This project evaluates and compares the following optimizers:

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- Adam
- RMSprop

Each optimizer is trained under similar conditions to allow for a fair comparison.

---

## 📂 Repository Structure

```
cifar10-optimizer-comparison-experiments/
├── optimizer_comparison_cifar10.ipynb   # Main experiment notebook
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- Jupyter Notebook

Install dependencies manually:

```bash
pip install torch torchvision numpy matplotlib
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/omobabello/cifar10-optimizer-comparison-experiments.git
cd cifar10-optimizer-comparison-experiments
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open and run:

```
optimizer_comparison_cifar10.ipynb
```

Run all cells to reproduce the experiments and plots.

---

## 📊 Experiments & Evaluation

The notebook includes:

- CIFAR-10 data loading and preprocessing
- A simple CNN architecture
- Training loops for multiple optimizers
- Accuracy and loss plots for comparison
- Observations on optimizer behavior

Metrics tracked:
- Training loss
- Validation loss
- Validation accuracy per epoch

---

## 📈 Results

Results may vary depending on random seeds and hardware, but generally:

- SGD converges slower but is stable
- SGD with momentum improves convergence speed
- Adam converges faster early but may plateau
- RMSprop shows adaptive behavior similar to Adam

The notebook visualizes these differences clearly using plots.

---

## 🔧 Extensions

You can extend this project by:
- Adding optimizers such as AdamW or Adagrad
- Testing deeper CNN architectures
- Applying data augmentation
- Logging experiments with TensorBoard or Weights & Biases

---

## 📜 License

This project is open-source and available for educational and experimental use.

---

## 👤 Author

**Bello Opeyemi**  
GitHub: https://github.com/omobabello
