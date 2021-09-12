import argparse
import pathlib


def parser_args():
    parser = argparse.ArgumentParser(description='DP-FL-LS Param')

    # ========================= federated learning parameters ========================
    parser.add_argument('--num_users', type=int, default=200,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.01,
                        help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: B")
    parser.add_argument('--lr_outer', type=float, default=1,
                        help="learning rate")
    parser.add_argument('--lr_inner', type=float, default=0.1,
                        help="learning rate for inner update")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="exponential weight decay")
    parser.add_argument('--iid', action='store_true',
                        help='dataset is split iid or not')
    parser.add_argument('--wd', type=float, default=4e-5,
                        help='weight decay')
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer: [sgd, adam]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='communication round')
    parser.add_argument('--data_root', type=pathlib.Path, default='$HOME/data/',
                        help='root for data')

    # =============================== DP argument ====================================
    parser.add_argument('--dp', action='store_true',
                        help='whether we use differential privacy or not')
    parser.add_argument('--sampling_type', choices=['poisson', 'uniform'],
                        default='uniform', type=str,
                        help='which kind of DP sampling we use')
    parser.add_argument('--G', type=float, default=1,
                        help='clipping parameter (sensitivity)')
    parser.add_argument('--z', default=0, type=float,
                        help='noise multiplier')
    parser.add_argument('--v', default=None, type=float,
                        help='given noise')

    # ============================ Model arguments ===================================
    parser.add_argument('--model_name', type=str, default='cnn_svhn',
                        help='dataset name')
    parser.add_argument('--pretrain', action='store_true',
                        help='use pretrain model or not')

    # ========================= Laplacian Smoothing parameters =======================
    parser.add_argument('--estimator', default='ls', type=str)
    parser.add_argument('--sigma_ls', default=0, type=float,
                        help='parameter for laplacian smoothing')
    parser.add_argument('--order', default='bcwh', type=str,
                        help='the order of gradient slicing')

    # =========================== Other parameters ===================================
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--log_interval', default=1, type=int,
                        help='interval for evaluating loss and accuracy')
    parser.add_argument('--comments', default=None, type=str,
                        help='any comment')
    parser.add_argument('--model_log', action='store_true',
                        help='whether store model weights for all communication round')
    parser.add_argument('--train_size', default=None, type=int,
                        help='size of training set for MNIST and SVHN')
    parser.add_argument('--train_frac', default=None, type=float,
                        help='fraction of training set for LSTM')
    parser.add_argument('--mi', action='store_true',
                        help='whether we implement membership inference attack during the training')
    parser.add_argument('--freq', action='store_true',
                        help='whether we calculate the frequency of gradient across epochs')
    parser.add_argument('--sampling_freq', type=int, default=None)
    parser.add_argument('--hist', action='store_true',
                        help='whether we record the histogram of the gradient across epochs')
    parser.add_argument('--num_bins', default=100, type=int,
                        help='number of bins of our histogram')
    parser.add_argument('--debug', action='store_true',
                        help='in debug mode, we would not store accuracy and model weights')

    args = parser.parse_args()

    return args
