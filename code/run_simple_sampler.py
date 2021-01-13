"""
PixelCNN model
based on code by Kim Nicoli et al.

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304
"""

import argparse
import torch
import logging
import os

from tqdm import tqdm

from pixelcnn import PixelCNN
from observable import ObservableMeter, Energy, Magnetization, Susceptibility, AbsMagnetization, \
    SpecificHeat, VariationalFreeEnergy, VariationalEntropy, get_iid_statistics
from quantum_observable import q_ObservableMeter, q_Energy, q_Susceptibility, q_SpecificHeat, plaquette_dist 
from utils import show_setups_q
from json import load

torch.set_printoptions(precision=6)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--lattice', type=str, choices=('s', 'tri'), default='s',
                           help='lattice type (either cubic (s) or triangular (tri)')
    argparser.add_argument('--ham', type=str, choices=('fm', 'afm'), default='fm',
                           help='ferromagnetic or antiferromagnetic')
    argparser.add_argument('--sample_beta', type=float, default=0.1, help='beta value at which to estimate variables')
    argparser.add_argument('--num_configs', type=int, default=100, help='num configs per each iteration')
    argparser.add_argument('--n_iter', type=int, default=5, help='num iterations')
    argparser.add_argument('--cuda', action="store_true", help='runs the code on GPU if available')
    argparser.add_argument('--verbose', action="store_true", help='activate verbose mode')
    argparser.add_argument('--output', type=str, default='results_sampling', help='path to store the results')
    argparser.add_argument('--plot', action="store_true", help='saves the histogram dist of the variables')
    argparser.add_argument('--model_dir',help='model directory')
    argparser.add_argument('--model_exist', help='path to the model is given by model_dir, not outputdir/model_dir', default=False)
    argparser.add_argument('--outputdir',help='output directory for runs', default="runs")

    args, unknown = argparser.parse_known_args()
    print(vars(args))
    
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if args.cuda else "cpu")

    
    if(args.model_exist):
        model_dir = args.model_dir
    else:
        model_dir = os.path.join(args.outputdir, args.model_dir)
        
    with open(os.path.join(model_dir,"commandline_args.txt"),"r") as f:
        model_param = load(f)

    if "spin_model" not in model_param.keys():
        model_param["spin_model"] = "ising"

    if "horizontal_trotter" not in model_param.keys():
        model_param["horizontal_trotter"] = False

    if "dropout_pp" not in model_param.keys():
        model_param["dropout_pp"] = 0.5

    if type(model_param["lattice_size"]) is int:
        model_param["lattice_size"] = (model_param["lattice_size"],model_param["lattice_size"])

    show_setups_q(args, model_param["lattice_size"], logging)

    if model_param["sampler"]=="pixelcnn":
        q = PixelCNN(L=model_param["lattice_size"], spin_model=model_param["spin_model"], 
            q=model_param["q"] if model_param["spin_model"]=="potts" else None,
            beta=args.sample_beta if model_param["spin_model"]=="qh_chain" else None, 
            j=model_param["j"] if model_param["spin_model"]=="qh_chain" else None, 
            sample_method=model_param["sample_method"] if model_param["spin_model"]=="qh_chain" else None,
            penalty_78=model_param["penalty_78"] if model_param["spin_model"]=="qh_chain" else None,
            horizontal_trotter=model_param["horizontal_trotter"] if model_param["spin_model"]=="qh_chain" else None,
            net_depth=model_param["convs"] if model_param["resnet"] else 3, net_width=model_param["nr_filter"],
            half_kernel_size=3 if model_param["resnet"] else 6, bias=True, sym_p=model_param["sym_p"], sym_s=model_param["sym_s"], res_block=model_param["resnet"],
            x_hat_clip=0, final_conv=False, epsilon=1e-07, device=device).to(device)

    q.load(os.path.join(model_dir,"best_model.pth"), map_location=device) #model_param.resnet,model_param.lattice_size, args.beta, lattice=args.lattice,

    if model_param["spin_model"]=="ising":
        meter = ObservableMeter(
            {
                "E": Energy(lattice=args.lattice, ham=args.ham, spin_model="ising"),
                "M": Magnetization(spin_model="ising"),
                "|M|": AbsMagnetization(),
                "Chi": Susceptibility(args.sample_beta),
                "Cv": SpecificHeat(args.sample_beta, model_param["lattice_size"]),

                "F": VariationalFreeEnergy(args.sample_beta, q, lattice=args.lattice, ham=args.ham),
                "S":  VariationalEntropy(q)
            },
        tag="DirectSampling {}".format(str(q)),
        stat_func=get_iid_statistics
        )
    elif model_param["spin_model"]=="qh_chain":
        meter = q_ObservableMeter(
            {
                "E": q_Energy(),
                "C": q_SpecificHeat(args.sample_beta),
                "Chi": q_Susceptibility(args.sample_beta),
                "plaquette_dist": plaquette_dist()
            },
            args.sample_beta, model_param["j"], model_param["lattice_size"][0]/2, q.horizontal_trotter, device
        )


    for _ in tqdm(range(args.n_iter)):
        with torch.no_grad():
            samples, _ = q.sample(args.num_configs)
            meter.update(samples)

    logging.info(meter.aggregate())

    if(args.model_exist):
        meter.save(os.path.join(args.outputdir, args.output, "sampling_beta{0}_lattice{1}.txt").format(str(args.sample_beta).replace(".",""), model_param["lattice_size"]),
        sample_spec=vars(args))
    else:
        meter.save(os.path.join(args.outputdir, args.model_dir, args.output, "sampling_beta{0}_lattice{1}.txt".format(str(args.sample_beta).replace(".",""), model_param["lattice_size"])),
        sample_spec=vars(args))

    if args.plot:
        if(args.model_exist):
            meter.save(os.path.join(args.outputdir, args.output, "sampling_beta{0}_lattice{1}.txt").format(str(args.sample_beta).replace(".",""), model_param["lattice_size"]),
            sample_spec=vars(args))
        else:
            img_path = os.path.join(args.outputdir, args.model_dir, args.output, "{0}_sampling_beta{1}_lattice{2}.".format(args.job_ID, str(args.sample_beta).replace(".",""), model_param["lattice_size"]))
        meter.plot_histograms(img_path)
        logging.info("plots have been saved!")


if __name__ == "__main__":
    main()
