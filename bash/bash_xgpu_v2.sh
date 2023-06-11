#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.


#SBATCH -J DP-Fed-LS #Slurm job name

# Set the maximum runtime, uncomment if you need it
#SBATCH -t 120:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
#SBATCH --mail-user=zliangak@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) "gpu" or "gpu-share"
#SBATCH -p x-gpu

# To use 1 cpu cores and 8 gpu devices in a node
#SBATCH -N 1 -n 1 --gres=gpu:10

# Setup runtime environment if necessary
# or you can source ~/.bashrc or ~/.bash_profile

#SBATCH --ntasks=10
# Go to the job submission directory and run your application
cd $HOME/DP-FED-LS-CODE/dp-fed-ls
# Execute applications in parallel



###### For CNN over SVHN
#for sampling_type in 'poisson'
#    do
#    for z in 1.5
#        do
#            for G in 0.3 0.5
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait


#for sampling_type in 'poisson'
#    do
#    for z in 1.5
#        do
#            for G in 0.7 0.9
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait


#for sampling_type in 'poisson'
#    do
#    for z in 1.5
#        do
#            for G in 0.3 0.5
#              do
#                for lr in 0.125
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait
#
#
#for sampling_type in 'poisson'
#    do
#    for z in 1.5
#        do
#            for G in 0.7 0.9
#              do
#                for lr in 0.125
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait

#for sampling_type in 'poisson'
#    do
#    for z in 1.5
#        do
#            for G in 0.1
#              do
#                for lr in 0.125 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait

## have not run yet
# Epoch = 10, batch size=32
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.0 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.25 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.5 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 1.0 &
# epoch = 5, batch size=64
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.0 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.25 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.5 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 1.0 &
## epoch = 5, batch size=32
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.0 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.25 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 0.5 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 5 --local_bs 32 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.5 --epochs 200 --sigma_ls 1.0 &


#for sampling_type in 'poisson'
#    do
#    for z in 1.5 1.3 1.1 1.0
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.25 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                    done
#              done
#        done
#    done
#wait


#### For LSTM over Shakespeare
#for sampling_type in 'poisson'
#    do
#    for z in 1.6
#        do
#          for G in 3 7
#            do
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
#            done
#      done
#    done
#wait


# batch 1
#for sampling_type in 'poisson'
#    do
#    for z in 1.7 1.5 1.3 1.1 1.0
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait

#for sampling_type in 'uniform'
#    do
#    for z in 1.7 1.5 1.3 1.1 1.0
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait



#for sampling_type in 'uniform'
#    do
#    for z in 1.6 1.8 2.0 2.2 2.4
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait

#for sampling_type in 'poisson'
#    do
#    for z in 1.6 1.8 2.0 2.2 2.4
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait


#### Extreme noise for CNN over SVHN
#for sampling_type in 'poisson'
#    do
#    for z in 3.0 3.5 4.0
#        do
#          for lr_inner in 0.025 0.05 0.1 0.125
#          do
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr_inner --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type $sampling_type --G 0.3 --z $z --epochs 200 --sigma_ls 0.0 --comments extreme &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr_inner --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --dp --iid --sampling_type $sampling_type --G 0.3 --z $z --epochs 200 --sigma_ls 1.0 --comments extreme &
#          done
#     done
#    done
#wait

###### AUC evolution curve for CNN over SVHN
#for sampling_type in 'uniform' 'poisson'
#    do
#    for z in 1.8 #2.2
#      do
#       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type $sampling_type --epochs 200 --mi --comments mi_auc --train_size 64000 &
#       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type $sampling_type --G 0.3 --z $z --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#       srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type $sampling_type --G 0.3 --z $z --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
#     done
#  done
#wait

# uniform
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type uniform --epochs 200 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type uniform --epochs 200 --mi --comments mi_auc --train_size 64000 &
## uniform z=1.8
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 1.8 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 1.8 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 1.8 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
## uniform z=2.2
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 2.2 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 2.2 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type uniform --G 0.3 --z 2.2 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
#
## poisson
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type poisson --epochs 200 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --iid --sampling_type poisson --epochs 200 --mi --comments mi_auc --train_size 64000 &
## poisson z=1.8
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.8 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.8 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 1.8 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
## poisson z=2.2
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 2.2 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 2.2 --epochs 200 --sigma_ls 0.0 --mi --comments mi_auc --train_size 64000 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.2 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --pretrain --dp --iid --sampling_type poisson --G 0.3 --z 2.2 --epochs 200 --sigma_ls 1.0 --mi --comments mi_auc --train_size 64000 &
#
#wait



## For LSTM over Shakespeare
#for sampling_type in 'poisson'
#    do
#    for z in 1.2 1.4
#        do
#          for G in 3 5 7 9 11
#            do
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.5 &
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
#            done
#      done
#    done
#wait

# have not run
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type poisson --G 7 --z 1.6 --epochs 100 --sigma_ls 0.5 &


# order
#for sampling_type in 'poisson'
#    do
#    for z in 2.0
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      for order in "bchw" "bwhc" "bhwc"
#                        do
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 0.5 --comments order &
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 1.0 --comments order &
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 1.5 --comments order &
#                        done
#                    done
#              done
#        done
#    done
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain      --iid --sampling_type poisson --epochs 200 --sigma_ls 0.0 --comments order &
#wait


## order
#for sampling_type in 'uniform'
#    do
#    for z in 2.0
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      for order in "bchw" "bwhc" "bhwc"
#                        do
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 0.5 --comments order &
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 1.0 --comments order &
#                          srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --order $order --sigma_ls 1.5 --comments order &
#                        done
#                    done
#              done
#        done
#    done
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain      --iid --sampling_type uniform --epochs 200 --sigma_ls 0.0 --comments order &
#wait


## learning rate
## For LSTM over Shakespeare
#for sampling_type in 'poisson'
#    do
#    for z in 1.6
#        do
#          for G in 7
#            do
#              for lr in 1.07 1.27 1.67 1.87
#                do
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.25 &
#                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.5 &
#                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
#                done
#          done
#      done
#    done
#
##
##srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type poisson --G 7 --z 1.6 --epochs 100 --sigma_ls 0.25 &
#srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type poisson --G 7 --z 1.6 --epochs 100 --sigma_ls 0.5 &
#wait


# learning rate
# For LSTM over Shakespeare
#for sampling_type in 'poisson'
#    do
#    for z in 1.6
#        do
#          for G in 5
#            do
#              for lr in 1.07 1.27 1.67 1.87
#                do
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.25 &
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.5 &
##                  srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
#                done
#          done
#      done
#    done
#wait

#for sampling_type in 'uniform'
#    do
#    for z in 1.6 1.8 2.0 2.2 2.4
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait

#for sampling_type in 'poisson'
#    do
#    for z in 1.6 1.8 2.0 2.2 2.4
#        do
#            for G in 0.3
#              do
#                for lr in 0.1
#                    do
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.0 &
##                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 0.5 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.0 &
#                      srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name cnn_svhn --data_root $HOME/data/SVHN --frac 0.05 --local_ep 10 --local_bs 64 --lr_outer 1 --lr_inner $lr --gamma 0.99 --wd 4e-5 --optim sgd --num_users 2000 --pretrain --dp --iid --sampling_type $sampling_type --G $G --z $z --epochs 200 --sigma_ls 1.5 &
#                    done
#              done
#        done
#    done
#wait


## For LSTM over Shakespeare
#for sampling_type in 'uniform'
#    do
#    for z in 0.8 1.0 1.2 1.4 1.6
#        do
#          for G in 5
#            do
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.5 &
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.5 &
#            done
#      done
#    done
#wait

### For LSTM over Shakespeare
#for sampling_type in 'poisson'
#    do
#    for z in 0.8 1.0 1.2 1.4 1.6
#        do
#          for G in 5
#            do
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.0 &
#              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 0.5 &
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.0 &
##              srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name lstm --data_root $HOME/data/Shakespeare --frac 0.2 --local_ep 5 --local_bs 50 --lr_outer 1 --lr_inner 1.47 --gamma 0.99 --wd 4e-5 --optim sgd --pretrain --dp --sampling_type $sampling_type --G $G --z $z --epochs 100 --sigma_ls 1.5 &
#            done
#      done
#    done
#wait


# original

##### For logistic regression over MNIST
##for sampling_type in 'uniform'
##    do
##    for v in 0.8087 0.6641 0.5824 0.5260 ## for eps 6 7 8 9 respectively
##        do
###            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
###            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
##        done
##    done
##wait
#
#for sampling_type in 'poisson'
#    do
#    for v in 0.3409 0.2916 0.2599 0.2366 ## for eps 6 7 8 9 respectively
#        do
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait


### local ep = 10

#### For logistic regression over MNIST
#for sampling_type in 'uniform'
#    do
#    for v in 0.8087 0.6641 0.5824 0.5260 ## for eps 6 7 8 9 respectively
#        do
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait

#for sampling_type in 'poisson'
#    do
#    for v in 0.3409 0.2916 0.2599 0.2366 ## for eps 6 7 8 9 respectively
#        do
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 0 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 1 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 2 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 10 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.3 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait


### clip = 0.4

#### For logistic regression over MNIST
#for sampling_type in 'uniform'
#    do
#    for v in 0.8087 0.6641 0.5824 0.5260 ## for eps 6 7 8 9 respectively
#        do
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 0 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 1 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 2 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait

for sampling_type in 'poisson'
    do
    for v in 0.3409 0.2916 0.2599 0.2366 ## for eps 6 7 8 9 respectively
        do
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 0 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 1 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 2 &
            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.4 --v $v --epochs 30 --sigma_ls 3 &
        done
    done
wait



### clip = 0.5

#### For logistic regression over MNIST
#for sampling_type in 'uniform'
#    do
#    for v in 0.8087 0.6641 0.5824 0.5260 ## for eps 6 7 8 9 respectively
#        do
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 0 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 1 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 2 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 1000 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait

#for sampling_type in 'poisson'
#    do
#    for v in 0.3409 0.2916 0.2599 0.2366 ## for eps 6 7 8 9 respectively
#        do
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 0 &
#            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 1 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 2 &
##            srun -n1 --gres=gpu:1 --exclusive /home/zliangak/anaconda3/envs/dl38/bin/python main.py --model_name logistic --data_root $HOME/data/ --frac 0.05 --local_ep 5 --local_bs 10 --lr_outer 1 --lr_inner 0.1 --gamma 0.99 --wd 4e-5 --optim sgd --num_users 500 --dp --iid --sampling_type $sampling_type --G 0.5 --v $v --epochs 30 --sigma_ls 3 &
#        done
#    done
#wait