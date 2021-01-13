"""
PixelCNN model
based on code by Kim Nicoli and Wu et al.

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304

Dian Wu, Lei Wang, and Pan Zhang
Phys. Rev. Lett. 122, 080602
"""

import torch

from numpy import log
from torch import nn
import numpy as np

import utils

class ResBlock(nn.Module):
    def __init__(self, block):
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.exclusive = kwargs.pop('exclusive')
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer('mask', torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + (not self.exclusive):] = 0
        self.mask[kh // 2 + 1:] = 0
        self.weight.data *= self.mask

        # correction to xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.conv2d(x, self.mask * self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups)

    def extra_repr(self):
        return (super(MaskedConv2d, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))


class PixelCNN(nn.Module):
    def __init__(self, **kwargs):
        super(PixelCNN, self).__init__()

        if type(kwargs['L']) is int:
            self.L1 = kwargs['L']
            self.L2 = kwargs['L']
        elif type(kwargs['L']) is tuple or type(kwargs['L']) is list:
            self.L1 = kwargs['L'][0]
            self.L2 = kwargs['L'][1]
        self.spin_model = kwargs["spin_model"]
        self.q = 1 if self.spin_model!="potts" else kwargs["q"]
        self.j = None if self.spin_model!="qh_chain" else kwargs["j"]
        self.beta = None if self.spin_model!="qh_chain" else kwargs["beta"]
        self.sample_method = None if self.spin_model!="qh_chain" else kwargs["sample_method"]
        self.penalty_78 = None if self.spin_model!="qh_chain" else kwargs["penalty_78"]
        self.horizontal_trotter = None if self.spin_model!="qh_chain" else kwargs["horizontal_trotter"]
        self.net_depth = kwargs['net_depth']
        self.net_width = kwargs['net_width']
        self.half_kernel_size = kwargs['half_kernel_size']
        self.bias = kwargs['bias']
        self.sym_p = kwargs['sym_p']
        self.sym_s = kwargs['sym_s']
        self.res_block = kwargs['res_block']
        self.x_hat_clip = kwargs['x_hat_clip']
        self.final_conv = kwargs['final_conv']
        self.epsilon = kwargs['epsilon']
        self.device = kwargs['device']

        # Force the first x_hat to be 0.5
        if self.bias and not (self.sym_p or self.sym_s):
            self.register_buffer('x_hat_mask', torch.ones([self.L1,self.L2]))
            self.x_hat_mask[0, 0] = 0
            self.register_buffer('x_hat_bias', torch.zeros([self.L1,self.L2]))
            self.x_hat_bias[0, 0] = 0.5

        layers = []
        layers.append(
            MaskedConv2d(
                1,
                1 if self.net_depth == 1 else self.net_width,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=True))
        for count in range(self.net_depth - 2):
            if self.res_block:
                layers.append(
                    self._build_res_block(self.net_width, self.net_width))
            else:
                layers.append(
                    self._build_simple_block(self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.net_width, self.net_width if self.final_conv else self.q))
        if self.final_conv:
            layers.append(nn.PReLU(self.net_width, init=0.5))
            layers.append(nn.Conv2d(self.net_width, self.q, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        ### for sampling
        if self.spin_model=="ising" or self.spin_model=="qh_chain":
            self.sampler = lambda x_hat: torch.bernoulli(x_hat) * 2 - 1

        ### for heisenberg chain log prob cbd
        if self.spin_model=="qh_chain":
            self.prob_mask = torch.ones([self.L1, self.L2], device=self.device, requires_grad=False)
            self.prob_mask[1::2,1::2] = 0 
            self.prob_mask[2::2,2::2] = 0
            self.prob_mask[1:,-1] = 0
            self.prob_mask[-1,1:] = 0
            self.prob_mask = self.prob_mask==1

    def _build_simple_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_res_block(self, in_channels, out_channels):
        layers = []
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.bias))
        layers.append(nn.PReLU(in_channels, init=0.5))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.half_kernel_size * 2 + 1,
                padding=self.half_kernel_size,
                bias=self.bias,
                exclusive=False))
        block = ResBlock(nn.Sequential(*layers))
        return block

    def forward(self, x):
        x_hat = self.net(x)

        if self.x_hat_clip:
            # Clip value and preserve gradient
            with torch.no_grad():
                delta_x_hat = torch.clamp(x_hat, self.x_hat_clip,
                                          1 - self.x_hat_clip) - x_hat 
            assert not delta_x_hat.requires_grad
            x_hat = x_hat + delta_x_hat

        # Force the first x_hat to be 0.5
        if self.bias and not (self.sym_p or self.sym_s):
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias

        return x_hat

    # sample = +/-1, +1 = up = white, -1 = down = black
    # x_hat = p(x_{i, j} == +1 | x_{0, 0}, ..., x_{i, j - 1})
    # 0 < x_hat < 1
    # x_hat will not be flipped by z2
    def _sample(self, batch_size):
        sample = torch.zeros([batch_size, 1, self.L1, self.L2], device=self.device)
        for i in range(self.L1):
            for j in range(self.L2):
                x_hat = self.forward(sample)
                sample[:, :, i, j] = self.sampler(x_hat[:, :, i, j])

        return sample


    def _xyz_sample(self, batch_size):
        sample = torch.zeros([batch_size, 1, self.L1, self.L2], device=self.device)
        for row in range(self.L1-1):        
            for col in range(self.L2-1):                
                ### SAMPlE OR NOT SAMPLE ###
                if col!=0 and row!=0 and ( ((col%2)==1 and (row%2)==1) or ((col%2)==0 and (row%2)==0) ):
                    sample[:, :, row, col] = sample[:,:,[row,row-1,row-1],[col-1,col-1,col]].prod(dim=2)
                else:
                    x_hat = self.forward(sample)
                    sample[:, :, row, col] = self.sampler(x_hat[:, :, row, col])
            # periodic boundary conditions in real direction
            col=self.L2-1
            if row==0:
                x_hat = self.forward(sample)
                sample[:, :, row, col] = self.sampler(x_hat[:, :, row, col])
            elif (row%2)==0:
                sample[:, :, row, col] = sample[:,:,[row,row-1,row-1],[0,0,col]].prod(dim=2)
            elif (row%2)==1:
                sample[:, :, row, col] = sample[:,:,[row,row-1,row-1],[col-1,col-1,col]].prod(dim=2)
        # periodic boundary conditions in Trotter direction    
        row=self.L1-1
        # Sample the 0th spin in the last row
        x_hat = self.forward(sample)
        sample[:, :, row, 0] = self.sampler(x_hat[:, :, row, 0])
        # fill in all other spins in the last row
        for col in range(1,self.L2):
            if (col%2)==0:
                sample[:,:,-1,col] = sample[:,:,[0,0,-1],[col-1,col,col-1]].prod(dim=2)
            else:
                sample[:,:,-1,col] = sample[:,:,[-2,-2,-1],[col-1,col,col-1]].prod(dim=2)
        return sample

    def sample(self, batch_size):
        if self.spin_model=="qh_chain" and self.sample_method=="xyz":
            sample = self._xyz_sample(batch_size)
        elif self.spin_model=="qh_chain" and self.sample_method=="xxx":
            sample = self._xxx_sample(batch_size)
        else:
            sample = self._sample(batch_size)

        if self.sym_s and (self.spin_model=="ising" or self.spin_model=="qh_chain"):
            # Binary random int 0/1
            flip = torch.randint(
                2, [batch_size, 1, 1, 1],
                dtype=sample.dtype,
                device=sample.device) * 2 - 1
            sample *= flip
        return sample, None #x_hat

    def _log_prob(self, sample, x_hat):
        # scale x_hat to interval [epsilon, 1-epsilon]
        x_hat = (x_hat-0.5)*(1-2*self.epsilon) + 0.5

        if self.spin_model=="ising" or self.spin_model=="qh_chain":
            mask = (sample + 1) / 2
            log_prob = (torch.log(x_hat) * mask +
                        torch.log(1 - x_hat) * (1 - mask))
            if self.spin_model=="qh_chain":
                log_prob = log_prob[:,:,self.prob_mask]
            log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def log_prob(self, sample):
        x_hat = self.forward(sample)
        log_prob = self._log_prob(sample, x_hat)

        if self.sym_p and (self.spin_model=="ising" or self.spin_model=="qh_chain"):
            # Density estimation on inverted sample
            sample_inv = -sample
            x_hat_inv = self.forward(sample_inv)
            log_prob_inv = self._log_prob(sample_inv, x_hat_inv)
            log_prob = torch.logsumexp(
                torch.stack([log_prob, log_prob_inv]), dim=0)
            log_prob = log_prob - log(2)
        return log_prob

    def compute_energy(self, sample, ham="fm", lattice="s", boundary="periodic"):
        return utils.compute_energy(sample, ham, lattice, boundary, self.spin_model, 
                                    self.j, self.beta, penalty_78=self.penalty_78,horizontal_trotter=self.horizontal_trotter)

    def ignore_param(self, state):
        ignore_param_name_list = ['x_hat_mask', 'x_hat_bias']
        param_name_list = list(state.keys())
        for x in param_name_list:
            for y in ignore_param_name_list:
                if y in x:
                    state[x] = self.state_dict()[x]
                    break

    def load(self, path, map_location='cpu'):
        state = torch.load(path, map_location)
        self.ignore_param(state['net'])
        self.load_state_dict(state['net'])

    def __str__(self):
        return "PixelCNN - L ({},{}), net_depth {}, net_width {}, " \
               "half_kernel_size {}, bias {}, " \
               "sym_p {}, sym_s {}, res_block {}, x_hat_clip {}, " \
               "final_conv {}, device {}, epsilon {}".format(self.L1, self.L2, self.net_depth,
                                                 self.net_width, self.net_width,
                                                 self.half_kernel_size, self.bias,
                                                 self.sym_p, self.sym_s, self.res_block, self.x_hat_clip,
                                                 self.final_conv, self.device, self.epsilon)
