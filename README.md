# AdaEC

### Title: 
"An Adaptive Error Correction Method for Training PINNs to Solve Inverse Problems with Noisy Observational Data"

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5325078


### Abstract: 
Physics-informed neural networks (PINNs) provide a data-driven framework for solving partial differential equation (PDE) inverse problems, whose unknown parameters are estimated by utilizing observational data. In many practical problems, since the available observational data often contain noise and even are of a limited size, the common training strategy of inverse problems could not provide reasonable numerical solutions and even leads to the training failure. In this study, we first present an error correction (EC) method that uses the error correction factors to mitigate the negative effect of the random noise in observational data. However, such an EC method could cause the extra bias to the obtained training results after the noise is nearly eliminated. To address this issue, we further propose an adaptive error correction (AdaEC) method, which dynamically adjusts the learning rate of the EC factors by monitoring the relationship between the PDE residual loss calculated on the observational data and the one calculated on the collocation points during the training process. Since the AdaEC method does not either introduce any additional module into PINNs or require any prior knowledge on the noise distribution, it has a high applicability in practice. The theoretical analysis and numerical experiments support the effectiveness of the proposed AdaEC method. Finally, we also illustrate the practical engineering application of AdaEC in Timoshenko beam inverse problem.


## Usage for Academic Purposes

Currently, the paper is under review, and the use of this code is restricted to academic review. 


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
