import logging
import torch
import math
import random
import itertools

import numpy as np

from numpy.random import rand
from tqdm import tqdm
from quantum_observable import q_ObservableMeter, q_Energy, q_Susceptibility, q_SpecificHeat, plaquette_dist 
from utils import _compute_energy_qh_chain, plaquette_cross_check, heisenberg_coupling


class MarkovChain:
    def __init__(self, lattice_size, beta, num_steps, equilibration_steps=0, save_interval=1,
                 verbose=False, 
                 spin_model="ising", q=None, j=None, horizontal_trotter=False, device=None,
                 observables=None):
        self.lattice_size = lattice_size
        self.spin_model = spin_model
        self.q = q
        self.num_steps = num_steps
        self.beta = beta
        self.equilibration_steps = equilibration_steps
        self.save_interval = save_interval
        self.verbose = verbose
        self.n_acc = 0
        self.observables = observables
        self.curr_step = 0
        if observables is not None:
             self.observables = observables
        if spin_model=="qh_chain":
            self.observables  = q_ObservableMeter({ "E": q_Energy(),
                                                    "C": q_SpecificHeat(self.beta),
                                                    "Chi": q_Susceptibility(self.beta),
                                                    "plaquette_dist": plaquette_dist()},
                                                    self.beta, j, self.lattice_size[0]/2 if not horizontal_trotter else self.lattice_size[1]/2, horizontal_trotter, device)


    def step(self, config):
        pass

    def run(self):
        # reset observables
        self.observables.reset()

        # initialize config
        config = self.initialize()

        # equilibration steps
        self.log("starting with equilibration steps")
        for _ in tqdm(range(self.equilibration_steps)):
            self.step(config)

        # main steps
        self.n_acc = 0
        self.curr_step = 0
        self.log("starting with main steps")
        for i in tqdm(range(self.num_steps)):
            config = self.step(config)
            self.curr_step += 1

            if i % self.save_interval == 0:
                self.observables.update(config)

    def initialize(self):
        self.observables.tag = str(self)
        if self.spin_model=="ising" or self.spin_model=="qh_chain":
            config = torch.from_numpy(2 * np.random.randint(2, size=(self.lattice_size[0], self.lattice_size[1])) - 1).float()
        return config
        

    def get_acc_perc(self):
        pass

    def log(self, message):
        if self.verbose:
            logging.info(message)



class LandauCullenMarkovChain(MarkovChain):
    def __init__(self, lattice_size, beta, j,
                 num_steps, equilibration_steps, save_interval, disturb, verbose):
        super().__init__(lattice_size, beta, num_steps, equilibration_steps, save_interval, verbose, j=j, spin_model="qh_chain")
        
        self.disturb = disturb

        self.trotter = lattice_size[0]
        m = self.trotter//2
        self.N = lattice_size[1]
        
        self.beta = beta
        self.k0,self.kp,self.km = heisenberg_coupling(j, beta, self.trotter/2)

        if j[0]==0:
            self.ground34 = True
        else:
            self.ground34 = False
        
        max_diagonal_steps = m-1  # in one direction
        
        # all possible combinations of left/up/right steps
        paths = np.array(list(itertools.product([-1, 0, 1], repeat=m*2)))
        
        # filter for allowed combinations of left/up/right steps
        mask = np.zeros(len(paths))==1
        for n_diagonal_steps in range(max_diagonal_steps+1):
            n_vertical_steps = 2*(m-n_diagonal_steps)

            mask = mask | (((paths==1).sum(axis=1)==n_diagonal_steps) \
                        & ((paths==-1).sum(axis=1)==n_diagonal_steps) \
                        & ((paths==0).sum(axis=1)==n_vertical_steps) )
        paths = paths[mask==1]
        
        # filter for paths with allowed left/right steps
        mask = np.zeros(len(paths))
        for i,path in enumerate(paths):
            mask[i] = plaquette_cross_check(path)
        paths = paths[mask==1]
        
        self.n_paths = paths.shape[0]
        print("number of paths", self.n_paths)
        
        # create spin flipping patterns
        even_flip_patterns = np.concatenate((np.zeros((paths.shape[0],1)),paths),axis=1).astype(int)[:,:-1]
        odd_flip_patterns = even_flip_patterns*-1
        self.patterns = [even_flip_patterns,odd_flip_patterns]

        
    def step(self, config, beta):
        for sweep in range(self.N):
            i = np.random.randint(0, self.N)
            pattern_choice = rand()
            if pattern_choice>0.5:
                new_config = self.local_square_flip(config, i)
            else:
                new_config = self.zig_zag_flip(config, i)
            
        
            new_E = _compute_energy_qh_chain(new_config, beta, self.k0, self.kp, self.km, 0, False)

            log_ratio = -beta*(new_E-self.current_E)
            
            r = math.log(rand())
            if log_ratio>=0 or r<log_ratio:
                self.n_acc += 1
                config = new_config
                self.current_E = new_E
                
        return config
                
    def initialize(self):
        config = torch.ones((1,1,self.trotter,self.N))
        if self.ground34:
            config[:,:,:,::2]=-1
            print("Starting with groundstate of only (3),(4) plaquettes")
        else:
            print("Starting with groundstate of only (1) plaquettes")
        self.current_E = _compute_energy_qh_chain(config, self.beta, self.k0, self.kp, self.km, 1, False)
        return config
        
    def local_square_flip(self,conf,i):
        #r = np.random.randint(low=0, high=self.trotter)
        if i%2==0:
            r = np.random.choice(np.arange(1,self.trotter,2))
        else:
            r = np.random.choice(np.arange(0,self.trotter,2))
        r_p_1 = (r+1)%self.trotter
        i_p_1 = (i+1)%self.N
        new_conf = conf.clone()
        new_conf[:,:,[r,r,r_p_1,r_p_1],[i,i_p_1,i_p_1,i]] *= -1
        return new_conf
            
    def zig_zag_flip(self,conf,i):
        p = np.random.randint(0,self.n_paths)
        pattern = self.patterns[i%2][p]
        new_conf = conf.clone()
        row_select = torch.arange(self.trotter-1,-1,-1, dtype=int)
        new_conf[:,:,row_select,(i+pattern.cumsum())%self.N] *= -1  
        return new_conf


    def run(self):
        # reset observables
        self.observables.reset()

        # initialize config
        config = self.initialize()

        # equilibration steps
        self.log("starting with equilibration steps")
        for _ in tqdm(range(self.equilibration_steps)):
            self.step(config, self.beta)

        # main steps
        self.n_acc = 0
        self.curr_step = 0
        self.log("starting with main steps")
        

        if self.disturb:
            for subset in range(20):
                for i in tqdm(range(self.num_steps//20)):
                    config = self.step(config, self.beta)
                    self.curr_step += 1

                    self.observables.update(config)
                # disturb system for 100 steps at a random temperature rather higher than current temperature
                beta_off = self.beta*np.random.weibull(1.5, 1)[0]
                for j in range(100):
                    self.step(config, beta_off)
        else:
            for i in tqdm(range(self.num_steps)):
                config = self.step(config, self.beta)
                self.curr_step += 1
                self.observables.update(config)
