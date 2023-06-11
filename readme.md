# Exploring Private Federated Learning with Laplacian Smoothing

This repository contains the source code of the paper Exploring Private Federated Learning with Laplacian Smoothing.
It includes training differentially-private federated logistic regression over MNIST, CNN over SVHN and LSTM over [Shakespeare dataset](https://github.com/TalwalkarLab/leaf), 
with uniform or Poisson subsampling.


## Dataset
For [Shakespeare dataset](https://github.com/TalwalkarLab/leaf), we went through the preprocessing steps and placed 
all data into a [json file](https://hkustconnect-my.sharepoint.com/:f:/g/personal/zliangak_connect_ust_hk/El4Fs_Za4nZMuzVcoB4ndoYBKphObrekRQJCYQs_smwRfw?e=OJfKRf).
It is a python dict object with keys ['users', 'user_data', 'num_samples'], which represent roles' names, their corresponding samples (lines) and sample numbers.

## Experiment in the paper
For all the experiments mentioned in the paper, one can find the bash script in bash/bash.sh.
