# AdaEC-PINN

Title: "An adaptive error correction method for training physics-informed neural networks to solve inverse problems with noisy observational data"

### Publication:
[An adaptive error correction method for training physics-informed neural networks to solve inverse problems with noisy observational data](https://www.sciencedirect.com/science/article/pii/S0021999126007801)

*Journal of Computational Physics*. Available online 29 September 2026.

DOI: [10.1016/j.jcp.2026.115431](https://doi.org/10.1016/j.jcp.2026.115431)

## Code

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Liwgreat/AdaEC-PINN)

### Requirements:

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![PyDOE Version](https://img.shields.io/badge/PyDOE-0.3.8-blue.svg)
![PyTorch Version](https://img.shields.io/badge/pytorch-1.10.0-brightgreen.svg)


### Folder tree:
```plaintext  
Noise Type/PDE Name/
├── O-PINN # The original method for training PINN_inv
├── EC-PINN # The EC method for training PINN_inv
├── AdaEC-PINN # The AdaEC method for training PINN_inv
└── utils_training # The utils of the network structure
```
