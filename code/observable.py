"""
based on code by Kim Nicoli

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304
"""
import os
import torch
import math

import numpy as np

from collections import UserDict

import utils as utils


def get_iid_statistics(obs):
    with torch.no_grad():
        history = obs.prepare()
        count = history.shape[0]
        mean = history.mean()
        sq_mean = (history**2).mean()

        std = torch.sqrt(abs(sq_mean - mean**2))
        err = std / math.sqrt(count)

        return mean, err

def get_impsamp_statistics(obs):
    with torch.no_grad():
        history = obs.prepare()
        weights = obs.weight / obs.weight.sum()

        mean = (weights * history).sum()
        sq_mean = (weights * history**2).sum()
        std = torch.sqrt(abs(sq_mean - mean**2))

        ess = 1 / (weights**2).sum()
        err = std / math.sqrt(ess)

        return mean, err


class ObservableMeter(UserDict):
    def __init__(self, observables, tag="", stat_func=get_iid_statistics):
        self.data = observables
        self.tag = tag
        self.weight = None
        self.stat_func = stat_func

    def update(self, config):
        for k in self.data:
            self.data[k].update(config)

    def reset(self):
        for k in self.data:
            self.data[k].reset()

    def save(self, filename, sample_spec=None):
        path = os.path.dirname(filename)
        os.makedirs(os.path.abspath(path), exist_ok=True)

        if sample_spec is None:
            torch.save(({k: self.data[k].history for k in self.data}, self.aggregate()), filename)
        else:
            torch.save(({k: self.data[k].history for k in self.data}, self.aggregate(), sample_spec), filename)

    def aggregate(self):
        return {k: self.stat_func(self.data[k]) for k in self.data}

    def set_weight(self, weight):
        self.weight = weight
        self.data["weights"] = Observable()
        self.data ["weights"]._append(weight)

        for k in self.data:
            self.data[k].weight = weight

    def __str__(self):
        res = self.aggregate()
        print(res)
        str = ""
        for k in res:
            r = res[k]
            str += "\n{}: {}+-{}".format(k, r[0], r[1])

        return str


class Observable:
    def __init__(self):
        self.history = None
        self.weight = None

    def update(self, config):
        pass

    def reset(self):
        self.history = None

    def _append(self, tensor):
        if self.history is None:
            self.history = tensor
        else:
            self.history = torch.cat((self.history, tensor))

    def prepare(self):
        return self.history.float()


class Energy(Observable):
    def __init__(self, lattice="s", ham="fm", spin_model="ising"):
        super().__init__()
        self.lattice = lattice
        self.ham = ham
        self.spin_model = spin_model

    def update(self, config):
        if len(config.shape) == 2:
            config = config[None, None, :, :]

        lattice_size_1 = config.shape[-2]
        lattice_size_2 = config.shape[-1]

        self._append(utils.compute_energy(config, ham=self.ham, lattice=self.lattice, spin_model=self.spin_model).double() / (lattice_size_1*lattice_size_2) )


class Magnetization(Observable):
    def __init__(self, spin_model="ising", q=None):
        super().__init__()
        self.lattice_size = None
        self.spin_model = spin_model
        self.q = q

    def update(self, config):
        if len(config.shape) == 2:
            config = config[None, None, :, :]
        
        if self.spin_model=="potts":
            config = 1.0*(config==0)
            self._append( (self.q*config.sum(-1).sum(-1).squeeze(-1) - 1) / (self.q - 1))
        else:
            self._append(config.sum(-1).sum(-1).squeeze(-1))


class AbsMagnetization(Observable):
    def __init__(self):
        super().__init__()
        self.lattice_size = None

    def update(self, config):
        if len(config.shape) == 2:
            config = config[None, None, :, :]

        self._append(torch.abs(config.sum(-1).sum(-1).squeeze(-1)))


class Susceptibility(Magnetization):
    def __init__(self, beta, spin_model="ising",q=None):
        super().__init__()
        self.beta = beta

    def prepare(self):
        mean = self.history.float().mean()
        sq_hist = (self.history**2)

        return self.beta * (sq_hist - mean**2)


class AbsSusceptibility(AbsMagnetization):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def prepare(self):
        mean = self.history.float().mean()
        sq_hist = (self.history**2)

        return self.beta * (sq_hist - mean**2)


class SpecificHeat(Energy):
    def __init__(self, beta, lattice_size, spin_model="ising",q=None):
        super().__init__()
        self.beta = beta
        self.lattice_size = lattice_size

    def prepare(self):
        mean = self.history.float().mean()
        sq_hist = (self.history**2)

        return self.beta**2 * (sq_hist - mean**2) * self.lattice_size[0]*self.lattice_size[1]


class VariationalFreeEnergy(Observable):
    def __init__(self, beta, q, ham="fm", lattice="s", spin_model="ising"):
        super().__init__()
        self.beta = beta
        self.q = q
        self.ham = ham
        self.lattice = lattice
        self.spin_model = spin_model

    def update(self, config):
        if len(config.shape) == 2:
            config = config[None, None, :, :]

        lattice_size_1 = config.shape[-2]
        lattice_size_2 = config.shape[-1]
        energy = utils.compute_energy(config, ham=self.ham, lattice=self.lattice, spin_model=self.spin_model)
        log_prob = self.q.log_prob(config)

        free_energy = log_prob + self.beta * energy
        self._append(free_energy / (self.beta*(lattice_size_1*lattice_size_2)))


class VariationalEntropy(Observable):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def update(self, config):
        if len(config.shape) == 2:
            config = config[None, None, :, :]

        lattice_size_1 = config.shape[-2]
        lattice_size_2 = config.shape[-1]
        log_prob = self.q.log_prob(config)

        self._append(-log_prob / (lattice_size_1*lattice_size_2))
