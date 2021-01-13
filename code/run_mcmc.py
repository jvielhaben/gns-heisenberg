import argparse
import logging
import torch
import os

from pixelcnn import PixelCNN
from markovchain import LandauCullenMarkovChain
from utils import show_setups_mcmc, plot_histograms
from json import load, dump

torch.set_printoptions(precision=6)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lattice_size', type=int, nargs='+', default=16, help='lattice size')
    argparser.add_argument('--lattice', type=str, choices=('s', 'tri'), default='s',
                           help='lattice type (either cubic (s) or triangular (tri)')
    argparser.add_argument('--j', type=float,  nargs='+', default=(1,1,1), help='coupling in x,y,z for Heisenberg chain')
    argparser.add_argument('--sample_beta', type=float, default=0.1, help='REAL beta value at which we want to estimate observables')
    argparser.add_argument('--model_beta', type=float, default=0.0, help='beta value at which the NN q is trained')
    argparser.add_argument('--eps', type=float, default=1e-07, help='epsilon regularization for q')
    argparser.add_argument('--alg', choices=['met', 'wolff', 'neural', 'landau_cullen'])
    argparser.add_argument('--disturb', action="store_true")
    argparser.add_argument('--num_steps', type=int, default=500, help='num steps')
    argparser.add_argument('--n_met_steps', type=int, default=0, help='num of metropolis steps')
    argparser.add_argument('--equi_steps', type=int, default=50000, help='equilibrating steps')
    argparser.add_argument('--num_configs', type=int, default=1000, help='bs for trove sampling in nip')
    argparser.add_argument('--save_interval', type=int, default=5, help='save every interval steps')
    argparser.add_argument('--verbose', type=bool, default=False, help='activate verbose mode')
    argparser.add_argument('--cuda', action="store_true", help='runs the code on GPU if available')
    argparser.add_argument('--resnet', action="store_true", help='To turn on if using a resnet with depth 6 as trained model')
    argparser.add_argument('--output', type=str, default='results_mcmc', help='path to store the results' )
    argparser.add_argument('--plot', action="store_true", help='Activate this flag to see the histogram dist of the variables')
    argparser.add_argument('--model_dir',help='model directory')
    argparser.add_argument('--model_exist', help='path to the model is given by model_dir, not outputdir/model_dir', default=False)
    argparser.add_argument('--outputdir',help='output directory for runs', default="runs")

    args, unknown = argparser.parse_known_args()
    print(vars(args))

    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if args.cuda else "cpu")
    show_setups_mcmc(args, logging)

    if args.alg == "landau_cullen":
        chain = LandauCullenMarkovChain(lattice_size=args.lattice_size, beta=args.sample_beta, j=args.j,
                                        num_steps=args.num_steps,equilibration_steps=args.equi_steps,save_interval=args.save_interval,
                                        disturb=args.disturb, verbose=args.verbose )
    chain.run()
    obs = chain.observables.aggregate()    

    if type(args.lattice_size) is list:
        file_name = '{}_mcmc_run_beta_{}_lattice_{}_{}'.format(args.alg, str(args.sample_beta).replace(".","_"), args.lattice_size[0], args.lattice_size[1])
    else:
        file_name = '{}_mcmc_run_beta_{}_lattice_{}'.format(args.alg, str(args.sample_beta).replace(".","_"), args.lattice_size)

    log_path = os.path.join(args.outputdir, args.output)

    chain.observables.save(os.path.join(log_path, file_name + '.pt'))

    if args.plot:
        img_path = os.path.join(log_path, file_name)
        plot_histograms(chain.observables, img_path)
        logging.info("plots have been saved!")

if __name__ == "__main__":
    main()
