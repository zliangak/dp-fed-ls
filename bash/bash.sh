#!/bin/bash



#### For logistic regression over MNIST
#### for eps=6, v=0.8087
# uniform subsampling
python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type 'uniform' --G 0.3 --v 0.8087 --epochs 30 --sigma_ls 0
python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type 'uniform' --G 0.3 --v 0.8087 --epochs 30 --sigma_ls 1
# poisson subsampling
python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type 'poisson' --G 0.3 --v 0.8087 --epochs 30 --sigma_ls 0
python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type 'poisson' --G 0.3 --v 0.8087 --epochs 30 --sigma_ls 1




#### For CNN over SVHN
# uniform subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'uniform' --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.0
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'uniform' --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.5
# poisson subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'poisson' --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.0
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'poisson' --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.5




#### For LSTM over Shakespeare
# uniform subsampling
python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type 'uniform' --G 5 --z 1.6 --epochs 100 --sigma_ls 0.0
python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type 'uniform' --G 5 --z 1.6 --epochs 100 --sigma_ls 1.0
# poisson subsampling
python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type 'poisson' --G 5 --z 1.6 --epochs 100 --sigma_ls 0.0
python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type 'poisson' --G 5 --z 1.6 --epochs 100 --sigma_ls 1.0




#### Extreme noise for CNN over SVHN
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.125 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type 'poisson' --G 0.3 --z 3.0 --epochs 200 --sigma_ls 0.0 --comments extreme
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.125 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type 'poisson' --G 0.3 --z 3.0 --epochs 200 --sigma_ls 1.0 --comments extreme




##### AUC evolution curve for CNN over SVHN
# for uniform subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type 'uniform' --epochs 200 --mi --comments mi_auc --train_size 64000
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type 'uniform' --G 0.3 --z 1.0 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type 'uniform' --G 0.3 --z 1.0 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000
# for poisson subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type 'poisson' --epochs 200 --mi --comments mi_auc --train_size 64000
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type 'poisson' --G 0.3 --z 1.0 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type 'poisson' --G 0.3 --z 1.0 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000




## CNN over SVHN with James-stein and soft-thresholding estimator
# for uniform subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'uniform' --G 0.7 --z 1.5 --epochs 200 --estimator js --comments other-estimators
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'uniform' --G 0.7 --z 1.5 --epochs 200 --estimator th --comments other-estimators
# for poisson subsampling
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'poisson' --G 0.7 --z 1.5 --epochs 200 --estimator js --comments other-estimators
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type 'poisson' --G 0.7 --z 1.5 --epochs 200 --estimator th --comments other-estimators




##### Gradient frequency for CNN over SVHN
python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain  --iid --sampling_type 'poisson' --epochs 200 --sigma_ls 0.0 --sampling_freq 1000 --freq --comments freq


