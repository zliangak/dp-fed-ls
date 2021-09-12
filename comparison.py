import argparse
import numpy as np
from dp.autodp import rdp_bank
from dp.autodp import rdp_acct
from dp.google_accountant.google_accountant import *
from scipy import optimize
import sys
import os
import wandb


def get_v(eps, tau, delta, T, G):
    lbds = np.concatenate((np.linspace(1e-3,2e-3,int(1e6)), np.linspace(2e-3,1-1e-3,int(1e6))))

    v = tau * G * (1/eps) * np.sqrt(14 * T * (1/lbds) * (np.log(1/delta)/(1-lbds)+eps))
    alpha = np.log(1/delta)/((1-lbds)*eps)+1

    cond1 = v**2 / (4*G**2)
    cond2 = v**2 * (1/(6*G**2)) * np.log(1/ ( tau*alpha* (1+v**2/(4*G**2)) ))+1

    array_cond1 = np.where(cond1 > 0.67)[0]
    array_cond2 = np.where(cond2 > alpha)[0]

    ls = list(set(array_cond1).intersection(set(array_cond2)))
    if ls:
        vs = v[ls]
        lbd = lbds[ls]
        idx = np.argmin(vs)
        return vs[idx]#, lbd[idx]
    else:
        return None

def get_privacy(frac, z, T, delta, sampling_type):
    '''calculate the epsilon given number of training step T
    '''
    if sampling_type == 'poisson':
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(q=frac,
                          noise_multiplier=z,
                          steps=T,
                          orders=orders
                          )
        eps, delta, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    elif sampling_type == 'uniform':
        func = lambda x: rdp_bank.RDP_gaussian({'sigma': z}, x)
        DPobject = rdp_acct.anaRDPacct()
        for t in range(T):
            DPobject.compose_subsampled_mechanism(func, frac)
        eps = DPobject.get_eps(delta)
    return eps


def get_ac_v(frac, eps, T, delta, sampling_type):
    def f(x):
        return eps - get_privacy(frac, x, T, delta, sampling_type)
    v = optimize.root_scalar(f, bracket=[1e-1, 5])
    return v.root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comparison')
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--T', type=int, default=1000)
    parser.add_argument('--G', type=int, default=1)
    parser.add_argument('--sampling_type', default='uniform')
    parser.add_argument('--upperbound', type=float, default=14.5)
    args = parser.parse_args()

    # input your wandb id
    os.system('wandb login xxxx')

    name = 'K_{}_tau_{}_T_{}_G_{}_type_{}'
    name = name.format(args.K, args.tau, args.T, args.G, args.sampling_type)
    wandb.init(
        project='thm-acc-comparison',
        name=name,
        config=args
    )

    print('finding vss ...')
    delta = 1/args.K**1.1
    epss = np.linspace(0.01, args.upperbound, 1000)
    vss = []
    for eps in epss:
        vss.append(get_v(eps, args.tau, delta, args.T, args.G))
    idx = [i for i in range(len(vss)) if vss[i] is not None]
    if idx == []:
        sys.exit()

    print('finding vac ...')
    v_ac = []
    for i, eps in enumerate(epss):
        if vss[i] is None:
            continue
        else:
            v_ac.append(get_ac_v(args.tau, eps, args.T, delta, args.sampling_type))
    idx = [i for i in range(len(vss))if vss[i] is None][-1]

    np_epss = np.array(epss[idx+1:])
    np_vss = np.array(vss[idx+1:])
    np_v_ac = np.array(v_ac)


    A = np.vstack([np.log(np_epss), np.ones(np_epss.shape[0])]).T
    m_thm, c_thm = np.linalg.lstsq(A, np.log(np_vss), rcond=None)[0]
    m_ac, c_ac = np.linalg.lstsq(A, np.log(np_v_ac), rcond=None)[0]

    wandb.config.m_thm = m_thm
    wandb.config.c_thm = c_thm
    wandb.config.m_ac = m_ac
    wandb.config.c_ac = c_ac
    wandb.config.diff = m_thm - m_ac


