"""
Neural importance sampling
author: Kim Nicoli

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304
"""

import torch
import math

from tqdm import tqdm

from observable import Observable
from utils import normalize_weights


class ImportanceSampler:
    def __init__(self, model, obs_meter, beta, lattice_size, batch_size=2000, lattice='s', ham="fm"):
        self.obs_meter = obs_meter
        self.batch_size = batch_size
        self.model = model
        self.beta = beta
        self.lattice_size = lattice_size
        self.lattice = lattice
        self.ham = ham

    def run(self, n_iter):
        with torch.no_grad():
            log_weight_hist = []
            log_prob_hist = []
            true_exps = []
            for _ in tqdm(range(n_iter)):
                    samples, _ = self.model.sample(self.batch_size)
                    log_prob = self.model.log_prob(samples)
                    true_exp = (-1.0)*self.beta*self.model.compute_energy(samples, ham=self.ham, lattice=self.lattice)

                    log_weight_hist.append(true_exp-log_prob)
                    log_prob_hist.append(log_prob)
                    true_exps.append(true_exp)

                    self.obs_meter.update(samples)

            log_weight_hist = torch.cat(log_weight_hist)
            log_prob_hist = torch.cat(log_prob_hist)
            true_exps = torch.cat(true_exps)

            weight_hist = normalize_weights(log_weight_hist)
            self.obs_meter.set_weight(weight_hist)
            obs = self.obs_meter.aggregate()

            # add special impsamp observables
            obs.update({"F": (self._free_energy(log_weight_hist), self._std_free_energy(log_weight_hist))})
            obs.update({"S": (self._entropy(log_weight_hist, log_prob_hist, weight_hist), self._std_entropy(true_exps, weight_hist))})

            return obs

    def _free_energy(self, unnorm_log_weights):
        n = unnorm_log_weights.shape[0]
        return -(unnorm_log_weights.logsumexp(dim=0) - math.log(n)) / (self.beta * self.lattice_size[0]*self.lattice_size[1])

    def _std_free_energy(self, unnorm_log_weights):
        n = unnorm_log_weights.shape[0]
        w = normalize_weights(unnorm_log_weights, include_factor_N=False)

        var = 1/n*((w**2).mean()-1)

        return torch.sqrt(var).item() / (self.beta * self.lattice_size[0]*self.lattice_size[1])


    def _entropy(self, unnorm_log_weights, log_probs, weight_hist):
        n = unnorm_log_weights.shape[0]
        s = unnorm_log_weights - torch.logsumexp(unnorm_log_weights, dim=0) + log_probs

        return -((weight_hist * s).sum() + math.log(n)) / (self.lattice_size[0]*self.lattice_size[1])

    def _std_entropy(self, true_exps, weight_hist):
        """
        -1 + Eg2w2 - 2 Egw2 (-1 + g) + Ew2 (1 - 2 g + g^2) where g = beta E and w are normalized impweights
        """

        n = weight_hist.shape[0]
        g = (-true_exps*weight_hist).sum()
        w = n * weight_hist  # normalized weights w/o any additional n factor

        Eg2w2 = ((w*true_exps)**2).mean()
        Egw2 = (-true_exps*w**2).mean()
        Ew2 = (w**2).mean()

        var = (-1 + Eg2w2 - 2*Egw2*(-1+g) + Ew2*(1 - 2*g + g**2))/n

        return torch.sqrt(var).item() / (self.lattice_size[0]*self.lattice_size[1])

