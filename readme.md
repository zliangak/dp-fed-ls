# Exploring Private Federated Learning with Laplacian Smoothing

This repository contains the source code of the paper 

>Liang, Z., Wang, B., Gu, Q., Osher, S. and Yao, Y., 2020. Exploring private federated learning with laplacian smoothing. arXiv preprint arXiv:2005.00218

It includes training differentially-private federated logistic regression over MNIST, CNN over SVHN and LSTM over [Shakespeare dataset](https://github.com/TalwalkarLab/leaf), 
with uniform or Poisson subsampling.

## Dependency
The code is run with `Python==3.8.8`, `torch==1.8.0`, `torchvison==0.9.0`

## Dataset
For [Shakespeare dataset](https://github.com/TalwalkarLab/leaf), we went through the preprocessing steps and placed 
the result into a [json file](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliangak_connect_ust_hk/El4Fs_Za4nZMuzVcoB4ndoYBKphObrekRQJCYQs_smwRfw?e=OJfKRf).
It is a python dict object with keys `['users', 'user_data', 'num_samples']`, which represent roles' names, their corresponding samples (lines) and sample numbers.


## Moment Accountants
Moment Accountant for uniform subsampling in `dp/autodp` is borrowed from [this repo](https://github.com/yuxiangw/autodp). And the one for
poisson subsampling in `dp/google_accountant` is borrowed from [this repo](https://github.com/tensorflow/privacy/tree/master/tensorflow_privacy/privacy).




## Experiment in the paper
For the experiments mentioned in the paper, one can find the bash script in bash/bash.sh.
