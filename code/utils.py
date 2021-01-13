import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import weight_norm as wn
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def _compute_enery_ising(sample, ham="fm", lattice="s", boundary="periodic"):
    interaction = lambda slice_1,slice_2: slice_1*slice_2
    
    term = interaction(sample[:, :, 1:, :], sample[:, :, :-1, :])
    term = term.sum(dim=(1, 2, 3))
    output = term
    term = interaction(sample[:, :, :, 1:], sample[:, :, :, :-1])
    term = term.sum(dim=(1, 2, 3))
    output += term
    if lattice == 'tri':
        term = interaction(sample[:, :, 1:, 1:], sample[:, :, :-1, :-1])
        term = term.sum(dim=(1, 2, 3))
        output += term
    if boundary == 'periodic':
        term = interaction(sample[:, :, 0, :], sample[:, :, -1, :])
        term = term.sum(dim=(1, 2))
        output += term
        term = interaction(sample[:, :, :, 0], sample[:, :, :, -1])
        term = term.sum(dim=(1, 2))
        output += term
        if lattice == 'tri':
            term = interaction(sample[:, :, 0, 1:], sample[:, :, -1, :-1])
            term = term.sum(dim=(1, 2))
            output += term
            term = interaction(sample[:, :, 1:, 0], sample[:, :, :-1, -1])
            term = term.sum(dim=(1, 2))
            output += term
            term = interaction(sample[:, :, 0, 0], sample[:, :, -1, -1])
            term = term.sum(dim=1)
            output += term
    if ham == 'fm':
        output *= -1
    return output

def _compute_energy_qh_chain(sample, beta, k0,kp,km, penalty_78, horizontal_trotter):
    L_trotter = sample.size()[2]
    L_real = sample.size()[3]
    
    plaquettes = torch.tensor([[[[1,1,1,1],
                            [-1,-1,-1,-1],
                            [1,-1,1,-1],
                            [-1,1,-1,1],
                            [-1,1,1,-1],
                            [1,-1,-1,1],
                            [-1,-1,1,1],
                            [1,1,-1,-1]]]], device=sample.device)
    if(horizontal_trotter):
        horizontal_trotter_permute = [0,1,6,7,5,4,3,2]
        plaquettes = plaquettes[:,:,horizontal_trotter_permute,:]

    if(km==0.0):
        plaquettes = plaquettes[:,:,:6]

    weights = torch.tensor([-(k0+np.log(np.cosh(km)))/beta, -(k0+np.log(np.cosh(km)))/beta,
                    (k0-np.log(np.cosh(kp)))/beta, (k0-np.log(np.cosh(kp)))/beta,
                    (k0-np.log(np.sinh(kp)))/beta, (k0-np.log(np.sinh(kp)))/beta,
                    -(k0+np.log(np.sinh(km)))/beta, -(k0+np.log(np.sinh(km)))/beta], device=sample.device)
    
    penalty = penalty_78* torch.ones(sample.size()[0],device=sample.device)

    E = torch.zeros(sample.size()[0], device=sample.device)
    for row in range(0,L_trotter):
        start = (row%2)==1
        for col in range(start,L_real,2):
            sub_sample = sample[:,:,[row-1,row-1,row,row],[col-1,col,col-1,col]]
            valid_plaquette,plaquette_i = (sub_sample==plaquettes).sum(dim=-1).squeeze().max(dim=-1)
            valid_plaquette = valid_plaquette==4
            # add weight if sub_sample is valid plaquette, otherwise add penalty term
            E += weights[plaquette_i]*valid_plaquette + penalty*~valid_plaquette
    return E


def compute_energy(sample, ham="fm", lattice="s", boundary="periodic", spin_model="ising", j=None, beta=None, penalty_78=1000.0, horizontal_trotter=False):
    if spin_model=="ising":
        return _compute_enery_ising(sample, ham=ham, lattice=lattice, boundary=boundary)
    elif spin_model=="qh_chain":
        m = sample.size()[2]/2 if not horizontal_trotter else sample.size()[3]/2 
        k0,kp,km = heisenberg_coupling(j,beta,m)
        return _compute_energy_qh_chain(sample, beta, k0,kp,km, penalty_78, horizontal_trotter)

def compute_q_qh_chain(sample, obs, beta, k0,kp,km,horizontal_trotter):
    L_trotter = sample.size()[2]
    m = L_trotter/2
    L_real = sample.size()[3]

    plaquettes = torch.tensor([[[[1,1,1,1],
                                [-1,-1,-1,-1],
                                [1,-1,1,-1],
                                [-1,1,-1,1],
                                [-1,1,1,-1],
                                [1,-1,-1,1],
                                [-1,-1,1,1],
                                [1,1,-1,-1]]]], device=sample.device)
    if(horizontal_trotter):
        horizontal_trotter_permute = [0,1,6,7,5,4,3,2]
        plaquettes = plaquettes[:,:,horizontal_trotter_permute,:]

    if(km==0.0):
        plaquettes = plaquettes[:,:,:6]
    """
    F:
    To evaluate the thermal average of the energy:
    Due to the temperature dependent interactions, the thermal average of the energy is the ensemble average
    of the function f = d (beta*E(beta)) / d beta .
    G:
    to evaluate thermal average of specific heat: C*T**2 = <F**2 - G> - <F>**2
    M:
    to evaluate the parallel susceptibility: Chi = beta*(<M**2>-<M>**2) 
    """
    qh_chain_quantities = {"F": torch.tensor([-(k0+km*np.tanh(km))/beta, -(k0+km*np.tanh(km))/beta,
                                                (k0-kp*np.tanh(kp))/beta, (k0-kp*np.tanh(kp))/beta,
                                                (k0- kp*np.cosh(kp)/np.sinh(kp) ) /beta,  (k0- kp*np.cosh(kp)/np.sinh(kp)) /beta,
                                                -(k0+ km*np.cosh(km)/np.sinh(km)) /beta, -(k0+ km*np.cosh(km)/np.sinh(km)) /beta], device=sample.device),
                           "G": torch.tensor([-(km /beta /np.cosh(km))**2, -(km /beta /np.cosh(km))**2,
                                               -(kp /beta /np.cosh(kp))**2, -(kp /beta /np.cosh(kp))**2,
                                               (kp /beta /np.sinh(kp))**2, (kp /beta /np.sinh(kp))**2,
                                               (km /beta /np.sinh(km))**2, (km /beta /np.sinh(km))**2,], device=sample.device),
                            "M": torch.tensor([1/(2*m),-1/(2*m),
                                                0.0,0.0,
                                                0.0,0.0,
                                                0.0,0.0], device=sample.device),
                            "PD": torch.tensor([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],
                                                [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],
                                                [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],
                                                [0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]], device=sample.device)                
                            }
    q = {o: torch.zeros(sample.size()[0], device=sample.device) for o in obs}
    if "PD" in obs:
        q["PD"] = torch.zeros((sample.size()[0],8), device=sample.device)
    illegal = torch.zeros(sample.size()[0], device=sample.device)
    for row in range(0,L_trotter):
        start = (row%2)==1
        for col in range(start,L_real,2):
            sub_sample = sample[:,:,[row-1,row-1,row,row],[col-1,col,col-1,col]]
            plaquette_i = (sub_sample==plaquettes).sum(dim=-1).squeeze().max(dim=-1)
            q = {o: q[o] + qh_chain_quantities[o][plaquette_i[1]]  for o in q}
            #to mark samples in batch with illegal plaquettes
            illegal[plaquette_i[0]<4] += 1
    q["M"] = sample.view(-1,L_real*L_trotter).sum(dim=-1) / (2*m)
    return q, illegal


def heisenberg_coupling(j, beta, m):
    # K_0 = beta*J_z / (4*m)   (m length of Trotter direction)
    k0 = beta*j[0] / (4 * m)
    # K_p/m = beta/ (4*m) (J_x+-J_y)
    kp = beta / (4 * m) * (j[1]+j[2])
    km = beta / (4 * m) * (j[1]-j[2])
    return k0,kp,km


def args_to_str(args):
    str = []
    for k,v in vars(args).items():
        if k != "last_step":
            str.append("{}{}".format(k.replace("_", ""), v))
    str = "_".join(str)
    return str


def show_setups_mcmc(args, logging):
    logging.info("\n\nLattice= {0}\nBeta= {1}\nModel_beta= {5}\nCuda= {2}\nEqui_steps= {3}\nNum_steps= {4}\nAlg= {6}\n".format(
        args.lattice_size, args.sample_beta, args.cuda,args.equi_steps, args.num_steps, args.model_beta, args.alg))


def show_setups_q(args, lattice_size, logging):
    logging.info(
        "\n\nLattice= {0}\nBeta= {1}\nCuda= {2}\nNum_configs= {3}\nNum_iter= {4}\n".format(
            lattice_size, args.sample_beta, args.cuda, args.num_configs, args.n_iter))


def histograms(data, filename, lattice_size=16):
    num_obs = 3
    L = lattice_size**2
    obs = ['E', 'M', '|M|']
    fig, axs = plt.subplots(1, num_obs, figsize=(15, 6))
    # fig.subplots_adjust(hspace=.5, wspace=.001)
    axs = axs.ravel()
    plots = list(range(num_obs))
    color_list = ['royalblue', 'mediumturquoise', 'limegreen']
    for k, i in zip(obs, plots):
        if k == 'M':
            axs[i].hist(data[k].history.cpu()/L, bins=15, range=(-1.0,1.0), density=0, label=k, facecolor=color_list[i],
                        alpha=0.75, edgecolor='black')
        else:
            axs[i].hist(data[k].history.cpu() / L, bins=15, density=0, label=k, facecolor=color_list[i], alpha=0.75,
                        edgecolor='black')
        axs[i].grid(axis='y', alpha=0.75)
        axs[i].set_xlabel(k)
        axs[i].set_ylabel('Counts')
        axs[i].legend()
    fig.savefig(filename+".svg")


def plot_histograms(obs, filename):
    path = os.path.dirname(filename)
    os.makedirs(os.path.abspath(path), exist_ok=True)
    histograms(obs.data, filename)


def normalize_weights(log_weight_hist, include_factor_N=True):
    """
    Calculates normalized importance weights from logarithm of unnormalized weights.
    If include_factor_N is set, then the result will be multiplied by the number of weights, s.t. the expectation is
    simply the sum of weights and history. Otherwise, the average instead of the sum has to be taken.
    """
    # use exp-norm trick:
    # https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick
    log_weight_hist_max, _ = log_weight_hist.max(dim=0)
    log_weigth_hist_norm = log_weight_hist - log_weight_hist_max
    weight_hist = torch.exp(log_weigth_hist_norm)
    if include_factor_N:
        weight_hist = weight_hist / weight_hist.sum(dim=0)
    else:
        weight_hist = weight_hist / weight_hist.mean(dim=0)

    return weight_hist

######## Markov chain ###########

def plaquette_cross_check(path):
    """
    Function to check whether zig zag patterns follow plaquette rules of the checkerboard decomposition.
    For Landau&Cullen Monte Carlo.
    """
    left_allowed = False
    for step in path:
        if step==-1: 
            if left_allowed:
                pass
            else:
                return False
        elif step==0:
            if left_allowed:
                left_allowed = False
            else:
                left_allowed = True
        elif step==1:
        # left allowed means right forbidden, return false
            if left_allowed:
                return False
            else:
                pass
    return True

