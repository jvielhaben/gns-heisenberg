import os
import torch
import math

import numpy as np

from collections import UserDict

import utils as utils


class q_ObservableMeter(UserDict):
    def __init__(self, observables, beta, j, m, horizontal_trotter, device, tag=""):
        self.beta = beta
        self.k0,self.kp,self.km = utils.heisenberg_coupling(j, beta, m)
        self.horizontal_trotter = horizontal_trotter
       
        self.data = observables
        self.weight = None
        
        self.req_obs_functions = ["F","G","M","PD"]
        self.obs_function = {o: Observable() for o in self.req_obs_functions}
        self.illegal_count = 0
        self.illegal_cache = torch.empty(0, device=device)
    
        self.tag = tag

    def update(self, config):
        obs, illegal = utils.compute_q_qh_chain(config, self.req_obs_functions, self.beta, self.k0, self.kp, self.km, self.horizontal_trotter)
        self.illegal_cache = torch.cat((self.illegal_cache, illegal))
        self.illegal_count += (illegal!=0).sum()
        for k in self.obs_function:
            self.obs_function[k]._append(obs[k][illegal==0])

    def reset(self):
        for k in self.obs_function:
            self.obs_function[k].reset()
        self.illegal_count = 0

    def save(self, filename, sample_spec=None):
        path = os.path.dirname(filename)
        os.makedirs(os.path.abspath(path), exist_ok=True)
        if sample_spec is None:
            torch.save(({k: self.obs_function[k].history for k in self.obs_function}, self.aggregate()), filename)
        else:
            torch.save(({k: self.obs_function[k].history for k in self.obs_function}, self.aggregate(), sample_spec), filename)

    def aggregate(self):
        out = {k: self.data[k]._aggregate(self.obs_function, self.weight) for k in self.data}
        out["n_illegal"] = self.illegal_count
        return out

    def set_weight(self, weight):
        weight = weight[self.illegal_cache==0]
        weight = weight / weight.sum()
        self.weight = weight
        if "weights" in self.data:
            self.obs_function["weights"] = Observable()
            self.obs_function["weights"]._append(weight)

    def __str__(self):
        str = ""
        for k in self.data:
            r = self.data[k]._aggregate(self.obs_function, self.weight)
            str += " {}: {}".format(k,r)
        str += " illegal_count: {}".format(self.illegal_count)
        return str


class Observable:
    def __init__(self):
        self.history = None

    def reset(self):
        self.history = None

    def _append(self, tensor):
        if self.history is None:
            self.history = tensor
        else:
            self.history = torch.cat((self.history, tensor))

    def prepare(self):
        return self.history

class plaquette_dist():
    def __init__(self):
        pass
    
    def _aggregate(self, obs_function_dict, weight):
        PD = obs_function_dict["PD"].prepare()
        return PD.sum(dim=0)

class q_Energy():
    def __init__(self):
        pass
        
    def _aggregate(self, obs_function_dict, weight):
        with torch.no_grad():
            F = obs_function_dict["F"].prepare()
            # Cullen&Landau, MC Studies of 1D quantum Heisenberg..., eq. 21
            if weight is None:
                count = F.shape[0]
                mean = F.mean()
                sq_mean = (F**2).mean()
                std = torch.sqrt(abs(sq_mean - mean**2))
                err = std / math.sqrt(count)
                return mean, err 
            else:
                mean = (weight*F).sum()  
                sq_mean = (weight * F**2).sum()
                std = torch.sqrt(abs(sq_mean - mean**2))
                ess = 1 / (weight**2).sum()
                err = std / math.sqrt(ess)
                return mean, err

class q_SpecificHeat():
    def __init__(self, beta):
        self.beta = beta

    def _aggregate(self, obs_function_dict, weight):
        with torch.no_grad():
            F = obs_function_dict["F"].prepare()
            G = obs_function_dict["G"].prepare()
            # Cullen&Landau, MC Studies of 1D quantum Heisenberg..., eq. 22
            if weight is None:
                t1_mean = (F**2-G).mean()
                t2_mean = F.mean()
                
                count = F.shape[0]
                t1_sq_mean = (F**2-G).mean()
                t2_sq_mean = F.mean()
                t1_std = torch.sqrt(abs(t1_sq_mean - t1_mean**2)) / math.sqrt(count)
                t2_std = torch.sqrt(abs(t2_sq_mean - t2_mean**2)) / math.sqrt(count)

                err = math.sqrt(t1_std**2 + (2*t2_mean*t2_std)**2)
                
                return t1_mean-t2_mean**2, err 
            else:
                t1_mean = (weight*(F**2-G)).sum()
                t2_mean = (weight*F).sum()
                
                t1_sq_mean = (weight * (F**2-G)**2).sum()
                t2_sq_mean = (weight * F**2).sum()

                ess = 1 / (weight**2).sum()
                t1_std = torch.sqrt(abs(t1_sq_mean - t1_mean**2)) / math.sqrt(ess)
                t2_std = torch.sqrt(abs(t2_sq_mean - t2_mean**2)) / math.sqrt(ess)

                err = math.sqrt(t1_std**2 + (2*t2_mean*t2_std)**2)

                return t1_mean-t2_mean**2, err


class q_Susceptibility():
    def __init__(self, beta):
        self.beta = beta

    def _aggregate(self, obs_function_dict, weight):
        M = obs_function_dict["M"].prepare()
        # Cullen&Landau, MC Studies of 1D quantum Heisenberg..., eq. 27
        if weight is None:
            t1_mean = (M**2).mean()
            t2_mean = M.mean()
            
            count = M.shape[0]
            t1_sq_mean = (M**2).mean()
            t2_sq_mean = M.mean()
            t1_std = torch.sqrt(abs(t1_sq_mean - t1_mean**2)) / math.sqrt(count)
            t2_std = torch.sqrt(abs(t2_sq_mean - t2_mean**2)) / math.sqrt(count)

            err = math.sqrt(t1_std**2 + (2*t2_mean*t2_std)**2)
            
            return self.beta * (t1_mean-t2_mean**2), self.beta * err
        else:
            t1_mean = (weight*(M**2)).sum()
            t2_mean = (weight*M).sum()
            
            t1_sq_mean = (weight * (M**2)**2).sum()
            t2_sq_mean = (weight * M**2).sum()

            ess = 1 / (weight**2).sum()
            t1_std = torch.sqrt(abs(t1_sq_mean - t1_mean**2)) / math.sqrt(ess)
            t2_std = torch.sqrt(abs(t2_sq_mean - t2_mean**2)) / math.sqrt(ess)

            err = math.sqrt(t1_std**2 + (2*t2_mean*t2_std)**2)

            return self.beta * (t1_mean-t2_mean**2), self.beta * err

class q_Weight():
    def __init__(self):
        pass

    def _aggregate(self, dummy, weight):
        # ess
        return 1 / (weight**2).sum()
