# RLSP - Efficient Video Super-Resolution through Recurrent Latent Space Propagation

Official repository containing code and other material from the paper "Efficient Video Super-Resolution through Recurrent Latent Space Propagation" (https://arxiv.org/abs/1909.08080).

![Alt text](docs/rlsp_model.jpg?raw=true "Title")

# Abstract
With the recent trend for ultra high definition displays, the demand for high quality and efficient video super-resolution (VSR) has become more important than ever. Previous methods adopt complex motion compensation strategies to exploit temporal information when estimating the missing high frequency details. However, as the motion estimation problem is a highly challenging problem, inaccurate motion compensation may affect the performance of VSR algorithms. Furthermore, the complex motion compensation module may also introduce a heavy computational burden, which limits the application of these methods in real systems. In this paper, we propose an efficient recurrent latent space propagation (RLSP) algorithm for fast VSR. RLSP introduces high-dimensional latent states to propagate temporal information between frames in an implicit manner. Our experimental results show that RLSP is a highly efficient and effective method to deal with the VSR problem. We outperform current state-of-the-art method DUF with over 70x speed-up.

# Code
The original RLSP model implementation in TensorFlow from the paper is available in "RLSP/tensorflow". The provided training script is not complete, data loading and training details need to be added. The original model checkpoints of RLSP 7-48, RLSP 7-64 and RLSP 7-128 from the paper are available in the folder "RLSP/evaluate", along with a script to evaluate your own videos.

A complete implementation in Pytorch is available in the folder "RLSP/pytorch". Please note, this version implements a model with RGB output, as opposed to Y in the paper. For adaptation to Y output, please refer to code in "RLSP/evaluate/super_resolve.py" for color space transformations.
Please specify the train and model parameters in "RLSP/pytorch/parameters.py". The model can be trained by running the script "RLSP/pytorch/train.py":
```
python train.py
```
In order to evaluate your trained model, import the system class and load the latest checkpoint (default) or a specific checkpoint:
```
# load
import torch
from system import System
system = System()
system.load_checkpoint("checkpoint_name.pt")  # optional

# evaluate
with torch.no_grad():
  y = system.rlsp(x)
```
# Citation
```
@inproceedings{fuoli_rlsp_2019,
title={Efficient Video Super-Resolution through Recurrent Latent Space Propagation},
author={Dario Fuoli and Shuhang Gu and Radu Timofte},
booktitle={ICCV Workshops},
year={2019},
}
```
