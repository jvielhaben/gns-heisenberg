# Generative Neural Samplers for the Quantum Heisenberg Chain

This repository is accompanying our article [Generative Neural Samplers for the Quantum Heisenberg Chain](https://arxiv.org/abs/2012.10264), which shows how autoregressive models can sample configurations of a quantum Heisenberg chain via a classical approximation based on the Suzuki-Trotter transformation. In our paper, we present results for energy, specific heat and susceptibility for the isotropic XXX and the anisotropic XY chain that are in good agreement with Monte Carlo results.

The code builds on code by Nicoli et al. for their paper 

[Asymptotically unbiased estimation of physical observables with neural samplers](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.023304)

and on code by Wu et al. for their article

[Solving Statistical Mechanics Using Variational Autoregressive Networks](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.080602)

# Basic Usage
We provide some basic usage information on the example of a trotterized anisotropic XY chain on a 4x32 grid:
* train a PixelCNN model
```shell
python code/train.py --cuda --sampler pixelcnn --resnet --sample_method xyz --lattice_size 4 32 --spin_model qh_chain --j 0 1.000001 1 --beta 2.0 --batch_size 1024 --n_steps 10000
```
* run simple sampling for this model 
```shell
python code/run_simple_sampler.py --sample_beta 2.0 --cuda --num_configs 2024 --n_iter 2000  --model_dir runs/run_pixelcnn_4_32_20 --model_exist True
```
* run importance sampling for this model
```shell
python code/run_importance_sampler.py --sample_beta 2.0 --cuda --num_configs 2024 --n_iter 2000  --model_dir runs/run_pixelcnn_4_32_20 --model_exist True
```
* run a MCMC algorithm based on the article [Monte Carlo studies of one-dimensional quantum Heisenberg and XY models](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.27.297) by John J. Cullen and D. P. Landau
```shell
python code/run_mcmc.py --lattice_size 4 32 --j 0 1.000001 1 --sample_beta 2 --alg landau_cullen --num_steps 30000 --equi_steps 1000 --save_interval 1
```

# References

	@article{vielhaben2020generative,
	      title={Generative Neural Samplers for the Quantum Heisenberg Chain}, 
	      author={Johanna Vielhaben and Nils Strodthoff},
	      year={2020},
	      journal={arXiv preprint 2012.10264},
	      eprint={2012.10264},
	      archivePrefix={arXiv},
	      primaryClass={cond-mat.stat-mech}
	}

If you find this code useful also consider citing

	@article{PhysRevE.101.023304,
	  title = {Asymptotically unbiased estimation of physical observables with neural samplers},
	  author = {Nicoli, Kim A. and Nakajima, Shinichi and Strodthoff, Nils and Samek, Wojciech and M\"uller, Klaus-Robert and Kessel, Pan},
	  journal = {Phys. Rev. E},
	  volume = {101},
	  issue = {2},
	  pages = {023304},
	  numpages = {10},
	  year = {2020},
	  month = {Feb},
	  publisher = {American Physical Society},
	  doi = {10.1103/PhysRevE.101.023304},
	  url = {https://link.aps.org/doi/10.1103/PhysRevE.101.023304}
	}

	@article{PhysRevLett.122.080602,
	  title = {Solving Statistical Mechanics Using Variational Autoregressive Networks},
	  author = {Wu, Dian and Wang, Lei and Zhang, Pan},
	  journal = {Phys. Rev. Lett.},
	  volume = {122},
	  issue = {8},
	  pages = {080602},
	  numpages = {6},
	  year = {2019},
	  month = {Feb},
	  publisher = {American Physical Society},
	  doi = {10.1103/PhysRevLett.122.080602},
	  url = {https://link.aps.org/doi/10.1103/PhysRevLett.122.080602}
	}
