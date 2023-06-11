#!/bin/bash

#### For logistic regression over MNIST
for sampling_type in 'uniform'
    do
    for v in 0.8087 0.6641 0.5824 0.5260 ## for eps 6 7 8 9 respectively
        do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
        done
    done


for sampling_type in 'poisson'
    do
    for v in 0.3409 0.2916 0.2599 0.2366 ## for eps 6 7 8 9 respectively
        do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
        done
    done
wait




#### For CNN over SVHN
for sampling_type in 'uniform' 'poisson'
    do
    for z in 1.5 1.3 1.1 1.0
        do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 0.0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 0.5 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 1.0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 1.5 &
     done
    done
wait






#### For LSTM over Shakespeare
for sampling_type in 'uniform' 'poisson'
    do
    for z in 1.6 1.4 1.2 1.0
        do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G 5 --z $z --epochs 100 --sigma_ls 0.0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G 5 --z $z --epochs 100 --sigma_ls 0.5 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G 5 --z $z --epochs 100 --sigma_ls 1.0 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G 5 --z $z --epochs 100 --sigma_ls 1.5 &
     done
    done
wait






#### Extreme noise for CNN over SVHN
for sampling_type in 'poisson'
    do
    for z in 2.0 2.5 3.0
        do
          for lr_inner in 0.025 0.05 0.1 0.125
          do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr_inner --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 0.0 --comments extreme &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr_inner --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 1.0 --comments extreme &
          done
     done
    done
wait






##### AUC evolution curve for CNN over SVHN
for sampling_type in 'uniform' 'poisson'
    do
    for z in 1.0 1.3
      do
       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type $sampling_type --epochs 200 --mi --comments mi_auc --train_size 64000 &
       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
     done
  done
wait






## CNN over SVHN with James-stein and soft-thresholding estimator
for sampling_type in 'uniform' 'poisson'
    do
    for z in 1.5 1.3 1.1 1.0
        do
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --estimator js --comments other-estimators &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --estimator th --comments other-estimators &
     done
    done
wait





##### Gradient frequency for CNN over SVHN
for sampling_type in 'poisson'
    do
    for z in 1.0
     do
         srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain      --iid --sampling_type $sampling_type --epochs 200 --sigma_ls 0.0 --sampling_freq 1000 --freq --comments freq &
         srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sampling_freq 1000 --sigma_ls 0.0 --freq --comments freq &
         srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G 0.7 --z $z --epochs 200 --sampling_freq 1000 --sigma_ls 1.0 --freq --comments freq &
     done
    done
wait

