## Geometry-Preserving Dimensionality Reduction for Visualizing Loss of Neural Networks

### Setup

To install the same set of packages used to implement this project, run

```bash
conda env create -f environment.yml
```

This project also uses torch 2.8.0+cu129 and torchvision 0.23.0+cu129. These must be installed separately.

### Baseline Models

Baseline model files trained for the pytorch ResNet-101 and Densenet-121 models can be found at https://drive.google.com/drive/folders/1IXGw31f3xzLHAfiXX7nR3R5EmkNtDbEy since they are too large to store on GitHub. Please download these files to a directory named `models/` in the root directory for everything to work as intended.