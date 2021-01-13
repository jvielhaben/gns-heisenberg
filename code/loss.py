"""
based on code by Kim Nicoli

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304
"""

import torch

import os
import sys

from utils import compute_energy, normalize_weights

class KullbackLeiblerLoss(torch.nn.Module):
    def __init__(self, model, beta, lattice_size, batch_size, stabilize=True, penalty_78=1000):
        super().__init__()
        self.model = model
        self.beta = beta
        self.lattice_size = lattice_size
        self.batch_size = batch_size
        self.stabilize = stabilize
        self.penalty_78 = penalty_78

    def forward(self):

        with torch.no_grad():
            samples = self.model.sample(self.batch_size)[0]

        log_prob = self.model.log_prob(samples)

        with torch.no_grad():
            actions = self.beta * compute_energy(samples,spin_model=self.model.spin_model, 
                                                j=self.model.j, beta=self.beta, penalty_78=self.model.penalty_78, horizontal_trotter=self.model.horizontal_trotter)
            loss = log_prob + actions
        
        if self.stabilize:
            loss_reinforce = (loss - loss.mean()) * log_prob / 1000
        else:
            loss_reinforce = loss * log_prob           

        with torch.no_grad():
            log_weight_hist = []
            log_prob_hist = []
            true_exp = (-1.0)*self.beta*self.model.compute_energy(samples, ham="fm", lattice="s")

            #estimate the ESS per batch
            log_weight_hist.append(true_exp - log_prob)
            log_prob_hist.append(log_prob)

            log_weight_hist = torch.cat(log_weight_hist)
            log_prob_hist = torch.cat(log_prob_hist)

            weight_hist = normalize_weights(log_weight_hist)

        return loss_reinforce, loss, actions, weight_hist, log_prob_hist, samples
