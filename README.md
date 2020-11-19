# RLSP - Efficient Video Super-Resolution through Recurrent Latent Space Propagation

This is the official repository containing code and other material from the paper "Efficient Video Super-Resolution through Recurrent Latent Space Propagation" (https://arxiv.org/abs/1909.08080).

# Abstract
With the recent trend for ultra high definition displays, the demand for high quality and efficient video super-resolution (VSR) has become more important than ever. Previous methods adopt complex motion compensation strategies to exploit temporal information when estimating the missing high frequency details. However, as the motion estimation problem is a highly challenging problem, inaccurate motion compensation may affect the performance of VSR algorithms. Furthermore, the complex motion compensation module may also introduce a heavy computational burden, which limits the application of these methods in real systems. In this paper, we propose an efficient recurrent latent space propagation (RLSP) algorithm for fast VSR. RLSP introduces high-dimensional latent states to propagate temporal information between frames in an implicit manner. Our experimental results show that RLSP is a highly efficient and effective method to deal with the VSR problem. We outperform current state-of-the-art method DUF with over 70x speed-up.

Citation:
```
@inproceedings{fuoli_rlsp_2019,
title={Efficient Video Super-Resolution through Recurrent Latent Space Propagation},
author={Dario Fuoli and Shuhang Gu and Radu Timofte},
booktitle={ICCV Workshops},
year={2019},
}
```
