import os

from utils.args import parser_args
from utils.help import *
from utils.datasets import *

from fed import *
import torch

def main(args):

    if not args.debug:
        logs = {'fl': None,
                'arguments': {
                    'frac': args.frac,
                    'local_ep': args.local_ep,
                    'local_bs': args.local_bs,
                    'lr_outer': args.lr_outer,
                    'lr_inner': args.lr_inner,
                    'gamma': args.gamma,
                    'iid': args.iid,
                    'wd': args.wd,
                    'optim': args.optim,
                    'dp': args.dp,
                    'G': args.G,
                    'z': args.z,
                    'model_name': args.model_name,
                    'pretrain': args.pretrain,
                    'log_interval': args.log_interval,
                    'num_classes': args.num_classes,
                    'sigma_ls': args.sigma_ls,
                    'sampling_type': args.sampling_type,
                    'epochs': args.epochs,
                    'v': args.v,
                    'comment': args.comments,
                    'order': args.order,
                    'freq': args.freq,
                    'sampling_freq': args.sampling_freq
                    }
                }
        exp_name = '{}_dp_{}_sampling_{}_epoch_{}_z_{}_G_{}_E_{}_ls_{}_tau_{}_pretrain_{}_v_{}_'
        exp_name = exp_name.format(args.model_name, args.dp, args.sampling_type, args.epochs,
                                   args.z, args.G, args.local_ep, args.sigma_ls, args.frac, args.pretrain, args.v)
        import wandb
        wandb.init(
            project='DP-Fed-LS',
            name=exp_name,
            config=args
        )
    else:
        wandb = None

    print('==> Preparing data...')
    train_set, val_set, test_set, dict_users = get_data(model_name=args.model_name,
                                                        data_root=args.data_root,
                                                        iid=args.iid,
                                                        num_users=args.num_users,
                                                        train_size=args.train_size,
                                                        train_frac=args.train_frac
                                                        )
    if not args.debug:
        logs['arguments']['num_users'] = len(dict_users)
        wandb.config.update({'num_users': len(dict_users)}, allow_val_change=True)
    args.num_users = len(dict_users)

    if args.dp:
        if not args.v:
            eps = get_privacy(frac=args.frac,
                              z=args.z,
                              delta=1/args.num_users**1.1,
                              T=args.epochs,
                              sampling_type=args.sampling_type
                              )
        else:
            eps = ('thm1' if args.sampling_type == 'uniform' else 'thm2')
            print("given v: {}, refer to Theorem 1 or Theorem 2 for privacy guarantee.".format(args.v))
    else:
        eps = None

    if not args.debug:
        wandb.config.epsilon = eps
        logs['epsilon'] = eps
    print('epsilon:', eps)

    model = get_model(args.model_name, args.pretrain)

    fl = DpFederatedLearning(model=model,
                             dict_users=dict_users,
                             frac=args.frac,
                             dp=args.dp,
                             G=args.G,
                             z=args.z,
                             sampling_type=args.sampling_type,
                             v=args.v,
                             model_log=args.model_log,
                             model_name=args.model_name,
                             wandb=wandb,
                             mi=args.mi,
                             freq=args.freq,
                             sampling_freq=args.sampling_freq,
                             data_root=args.data_root,
                             hist=args.hist,
                             num_bins=args.num_bins,
                             )

    val_acc, test_acc = fl.train(train_set=train_set,
                                 val_set=val_set,
                                 test_set=test_set,
                                 epochs=args.epochs,
                                 lr_inner=args.lr_inner,
                                 lr_outer=args.lr_outer,
                                 local_ep=args.local_ep,
                                 local_bs=args.local_bs,
                                 optim=args.optim,
                                 wd=args.wd,
                                 interval=args.log_interval,
                                 gamma=args.gamma,
                                 estimator=args.estimator,
                                 order=args.order,
                                 sigma_ls=args.sigma_ls
                                 )

    if not args.debug:
        fl.wandb = None # we could not save logs if it contains a module
        logs['fl'] = fl
        logs['val_acc'] = val_acc
        logs['test_acc'] = test_acc
        wandb.config.val_acc = val_acc
        wandb.config.test_acc = test_acc

        torch.save(logs, '../weights/{}_val_{:.4f}_test_{:.4f}.pkl'.format(exp_name, val_acc, test_acc))


if __name__ == '__main__':

    args = parser_args()
    print(args)

    if not args.debug:
        # input your wandb id
        os.system('wandb login xxxx')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(args)
