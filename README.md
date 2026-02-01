# CIFAR-10 Optimizer Comparison Experiments

This repository presents a set of controlled experiments comparing the performance of various optimization algorithms for training a convolutional neural network (CNN) on the CIFAR-10 dataset. The work is intended to serve as a reproducible benchmark for researchers and practitioners evaluating optimizer behavior in deep learning.

---

## 📚 Abstract

The study investigates how optimizer choice impacts **training convergence, generalization performance, and stability**. CIFAR-10, a widely used benchmark dataset in computer vision, provides a standardized framework for this comparison.

Optimizers evaluated include:

- Stochastic Gradient Descent (SGD)  
- SGD with Momentum  
- Adam  
- RMSprop  

Performance is analyzed through **quantitative metrics** (loss, accuracy) and **qualitative visualizations** (training curves, convergence plots).

---

## 🔬 Methodology

1. **Data Preprocessing:**  
   - CIFAR-10 dataset: 50,000 training and 10,000 test images  
   - Normalization and optional augmentation applied

2. **Model Architecture:**  
   - A standard convolutional neural network with multiple convolutional and fully connected layers  
   - Designed for comparability across different optimizer configurations

3. **Training Procedure:**  
   - Consistent training hyperparameters across optimizers (epochs, batch size, learning rate)  
   - Evaluation of training and validation metrics per epoch

4. **Analysis:**  
   - Compare convergence speed and final validation accuracy  
   - Visualize optimizer performance with training and validation curves  
   - Assess stability across multiple runs

---

## 🛠️ Reproducibility

### Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  
- Jupyter Notebook

Install dependencies:

```bash
pip install torch torchvision numpy matplotlib
```

### Running the Experiments

1. Clone the repository:

```bash
git clone https://github.com/omobabello/cifar10-optimizer-comparison-experiments.git
cd cifar10-optimizer-comparison-experiments
```

2. Launch Jupyter Notebook:

```bash
jupyter notebook
```

3. Open `optimizer_comparison_cifar10.ipynb` and execute all cells to reproduce experiments.

---

## 📊 Experimental Results

The notebook includes detailed plots of:

- Training and validation loss curves  
- Training and validation accuracy curves  
- Comparative convergence analysis  

Typical observations:

- **SGD:** Stable convergence, slower early training  
- **SGD + Momentum:** Faster convergence than vanilla SGD  
- **Adam:** Rapid initial convergence, may plateau  
- **RMSprop:** Adaptive behavior similar to Adam, sensitive to learning rate  

> All results are reproducible with consistent random seeds and hyperparameter settings.

---

## ⚡ Potential Extensions

- Include additional optimizers (AdamW, Adagrad, Adadelta)  
- Experiment with deeper or alternative CNN architectures  
- Apply data augmentation strategies and regularization techniques  
- Integrate experiment tracking tools (TensorBoard, Weights & Biases)

---

## 📜 License

This repository is released under an open-source license for academic and experimental purposes.

---

## 👤 Author

**Bello Opeyemi**  
GitHub: [https://github.com/omobabello](https://github.com/omobabello)
