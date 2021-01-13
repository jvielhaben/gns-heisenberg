"""
based on code by Kim Nicoli

Kim A. Nicoli, Shinichi Nakajima, Nils Strodthoff, Wojciech Samek, Klaus-Robert Müller, and Pan Kessel
Phys. Rev. E 101, 023304
"""

import argparse
import logging
import torch
import os
import json

from tqdm import tqdm
from tensorboardX import SummaryWriter

from pixelcnn import PixelCNN
from utils import args_to_str
from loss import KullbackLeiblerLoss
from observable import ObservableMeter, Energy, Magnetization, Susceptibility, AbsMagnetization, \
    SpecificHeat, VariationalFreeEnergy, VariationalEntropy, get_iid_statistics
from quantum_observable import q_ObservableMeter, q_Energy, q_Susceptibility, q_SpecificHeat
from utils import compute_q_qh_chain
import warnings
warnings.filterwarnings("ignore")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true", help='runs the code on GPU if available')
    parser.add_argument('--last_step', type=int, default=0, help='last training step. -1: continue with best model')
    parser.add_argument('omit_loading_cache', action="store_true", help="when resuming training of existing model, don't load cache of fastpixelcnnpp")
    parser.add_argument('--lattice_size', type=int, nargs='+', default=32, help='lattice size')
    parser.add_argument('--lattice', type=str, choices=('s', 'tri'), default='s',
                           help='lattice type (either cubic (s) or triangular (tri)')
    parser.add_argument('--spin_model', type=str, choices=('ising', 'potts', 'qh_chain'), default='ising')
    parser.add_argument('--sample_method', type=str, default="xyz", help='sampling method for Heisenberg chain')
    parser.add_argument('--horizontal_trotter', action="store_true", help='trotter direction on horizontal axis')
    parser.add_argument('--j', type=float,  nargs='+', default=(1,1,1), help='coupling in x,y,z for Heisenberg chain')
    parser.add_argument('--q', type=int, default=4, help='only relevant if spin_model==potts')
    parser.add_argument('--ham', type=str, choices=('fm', 'afm'), default='fm',
                           help='ferromagnetic or antiferromagnetic')
    parser.add_argument('--sym_s', type=bool, default=True, help='randomly switch to a symmetrical spin configuration in sampler')
    parser.add_argument('--sym_p', type=bool, default=True, help='add up all log probs of all symmetrical spin configurations')
    parser.add_argument('--save_interval', type=int, default=100000, help='save every interval steps')
    parser.add_argument('--beta', type=float, default=0.9, help='beta of Boltzmann distribution')
    parser.add_argument('--n_steps', help='number of iterations.', type=int, default=200000)
    parser.add_argument('--convs', help='number of conv layers.', type=int, default=6)
    parser.add_argument('--batch_size', help='number of samples', type=int, default=1000)
    parser.add_argument('--lr', help='initial learning rate.', type=float, default=1e-4)
    parser.add_argument('--lr_schedule', choices=['plateau'], help='learning rate scheduler')
    parser.add_argument('--n_warmup_steps', type=int, default=0, help="number of learning rate warm up steps")
    parser.add_argument('--weigh_loss', action='store_true', help='weigh loss according to number of (5,6) plaquettes for heisenberg chain')
    parser.add_argument('--stabilize_loss', action="store_false", help='variance reduction for loss')
    parser.add_argument('--penalty_78', type=float, default=1000.0, help='energy contribution for forbidden plaquettes in isotropic Heisenberg chain')
    parser.add_argument('--min_lr', help='minimum value for learning rate.', type=float, default=1e-6)
    parser.add_argument('--patience', help='set patience for lr scheduler.', type=int, default=100)
    parser.add_argument('--sampler', choices=['pixelcnn','pixelcnnpp','fastpixelcnnpp','igpt'], default="fastpixelcnnpp")
    parser.add_argument('--beta_anneal', type=float, default=0.0, help='rate of beta growth from 0 to final value, 0 for disabled')
    parser.add_argument('--eps', type=float, default=1e-07, help='epsilon regularization for sampler')
    parser.add_argument('--xhatclip', type=float, default=0.0, help='epsilon regularization for sampler')
    parser.add_argument('--resnet', action="store_true", help='turns on resnet sampler of depth 6')
    parser.add_argument('--ESS', action="store_true", help='Print Effective Sample Size during training')
    parser.add_argument('--best_model', action="store_true", help='Save the best model made on the loss value')
    parser.add_argument('--best_model_interval', type=int, default=1000, help='override the best model every n steps')
    parser.add_argument('--nr_filter',type=int,default=64, help='number of filter')

    
    # Directories
    parser.add_argument('--outputdir',help='output directory for runs', default="runs")
    parser.add_argument('--inputdir',help='input directory where pretrained model is located')

    args, unknown = parser.parse_known_args()
    print(vars(args))


    logging.basicConfig(level=logging.INFO)
    

    out_dir = "run_{0}_{1}_{2}_{3}".format(args.sampler, args.lattice_size[0], args.lattice_size[1], str(args.beta).replace(".",""))
    args_dir = os.path.join(args.outputdir, out_dir)

    os.makedirs(args_dir, exist_ok=True)



    with open(args_dir+'/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    writer = SummaryWriter(args_dir)
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.sampler=="pixelcnn":
        sampler = PixelCNN(L=args.lattice_size, spin_model=args.spin_model, q=args.q,
            beta=args.beta, j=args.j, sample_method=args.sample_method,
            penalty_78=args.penalty_78,horizontal_trotter=args.horizontal_trotter,
            net_depth=args.convs if args.resnet else 3, net_width=args.nr_filter,
            half_kernel_size=3 if args.resnet else 6, bias=True, sym_p=args.sym_p, sym_s=args.sym_s, res_block=args.resnet,
            x_hat_clip=args.xhatclip, final_conv=False, epsilon=args.eps, device=device
        ).to(device)

    if args.n_warmup_steps>0:
        lr_warmup_step = args.lr / args.n_warmup_steps

    optimizer = torch.optim.Adam(sampler.parameters(), lr=args.lr if args.n_warmup_steps==0 else 0)
    
    if args.ESS:
        ESS_filename = os.path.join(args_dir, "ESS.txt")

    if args.last_step!=0:
        if args.last_step > 0:
            previous_model_file = os.path.join(args.inputdir, "checkpoint_{}.pth".format(args.last_step))
            logging.info("training will be resumed from previous checkpoint {}".format(args.last_step))
        elif args.last_step == -1:
            previous_model_file = os.path.join(args.inputdir, "best_model.pth")
            logging.info("training will be resumed from previous best model")
            args.last_step = 0
        # load pretrained model dict
        state = torch.load(previous_model_file)
        pretrained_dict = state["net"]
        # 3. load the new state dict
        sampler.load_state_dict(pretrained_dict)
        optimizer.load_state_dict(state["optim"])

    kl_loss = KullbackLeiblerLoss(sampler, args.beta, args.lattice_size, args.batch_size,stabilize=args.stabilize_loss)
    if torch.cuda.device_count() > 1:
        print("Warning: deactivated multi-GPU training")
        #kl_loss.batch_size = args.batch_size // torch.cuda.device_count()
        #kl_loss = DataParallelCriterion(kl_loss)

    if args.lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.92, patience=args.patience, threshold=1e-4, min_lr=args.min_lr)

    # To track internal energy and magnetization during training
    if args.spin_model=="ising":
        meter = ObservableMeter(
            {
                "E": Energy(lattice=args.lattice, ham=args.ham, spin_model=args.spin_model),
                "|M|": AbsMagnetization()
            },
            tag="",
            stat_func=get_iid_statistics
        )
    elif args.spin_model=="qh_chain":
        meter = q_ObservableMeter(
            {
                "E": q_Energy(),
                "C": q_SpecificHeat(args.beta),
                "Chi": q_Susceptibility(args.beta)
            },
            args.beta, args.j, args.lattice_size[0]/2, sampler.horizontal_trotter, device
        )

    if args.spin_model!="qh_chain":
        args.weigh_loss = False

    with tqdm(range(args.n_steps)) as pbar:
        for step in pbar:
            actual_step = args.last_step + step
            optimizer.zero_grad()

            #kl_loss.beta = args.beta * (1 - args.beta_anneal ** actual_step)
            kl_loss.beta = args.beta * (1 - args.beta_anneal ** (actual_step+1))
            loss_reinforce, loss, actions, weight_hist, log_prob_hist, samples = kl_loss()
            

            if args.weigh_loss:
                with torch.no_grad():
                    plaquettes = compute_q_qh_chain(samples, ["PD"], args.beta, 1,1,1, horizontal_trotter=args.horizontal_trotter)[0]["PD"]
                    weights = plaquettes[:,[4,5]].sum(dim=1)
                    weights = weights / weights.sum()
                (loss_reinforce*weights).sum().backward()
            else:
                loss_reinforce.mean().backward()


            torch.nn.utils.clip_grad_norm_(sampler.parameters(), 1.0)

            if args.ESS:
                norm_weights_squared = (args.batch_size * weight_hist)**2
                ESS = 1/(norm_weights_squared.sum())
                if step % args.save_interval == 0:
                    print("\nEffective Sample Size: {}".format(ESS))
                with open(ESS_filename, 'a') as f:
                    f.write("{}\n".format(str(ESS.cpu().numpy())))
            
            # warm up learning rate
            if step<args.n_warmup_steps:
                for g in optimizer.param_groups:
                    g['lr'] += lr_warmup_step

            optimizer.step()

            if args.lr_schedule:
                scheduler.step(loss.var().sqrt())
            
            
            if step % 100 == 0:
                meter.update(samples)
                obs = meter.aggregate()
                pbar.set_description("step: {} loss: {:.2f}+-{:.2f} action: {:.2f}+-{:.2f} log_prob: {:.2f}+-{:.2f}".format(
                                    actual_step, loss.mean().item(), loss.var().sqrt().item(),
                                     torch.mean(actions).item(), actions.var().sqrt().item(),
                                     torch.mean(log_prob_hist).item(), log_prob_hist.var().sqrt().item())
                                     + str(meter))
                meter.reset()
            else:
                pbar.set_description(
                    "step: {} loss: {:.2f}+-{:.2f} action: {:.2f}+-{:.2f} log_prob: {:.2f}+-{:.2f}".format(
                        actual_step, loss.mean().item(), loss.var().sqrt().item(),
                        torch.mean(actions).item(), actions.var().sqrt().item(),
                        torch.mean(log_prob_hist).item(), log_prob_hist.var().sqrt().item()
                    )
                )
            
            # save best model at each checkpoint -> model with lowest loss variance
            if args.best_model:
                if step == 0:
                    best_model = {"net": sampler.state_dict(), "optim": optimizer.state_dict()}
                    model_loss = loss.mean().mean().item()

                if model_loss >= loss.mean().item():
                    best_model = {"net": sampler.state_dict(), "optim": optimizer.state_dict()}
                    model_loss = loss.mean().item()

            else:
                best_model = {"net": sampler.state_dict(), "optim": optimizer.state_dict()}
                model_loss = loss.mean().item()

            if step % args.best_model_interval == 0:
                best_model_path = os.path.join(args_dir, "best_model.pth".format(actual_step))
                path = os.path.dirname(best_model_path)
                os.makedirs(os.path.abspath(path), exist_ok=True)
                torch.save(best_model, best_model_path)

            writer.add_scalar('loss_reinf', loss_reinforce.mean().item(), actual_step)
            writer.add_scalar('loss_reinf_var', loss_reinforce.var().sqrt().item(), actual_step)
            writer.add_scalar('loss', loss.mean().item(), actual_step)
            writer.add_scalar('loss_var', loss.var().sqrt().item(), actual_step)
            writer.add_scalar('action', torch.mean(actions).item(), actual_step)
            writer.add_scalar('action_var', actions.var().sqrt().item(), actual_step)

            if step % args.save_interval == 0:
                print("loss, step: {}, {}\n".format(model_loss, actual_step))
                filename = os.path.join(args_dir, "checkpoint_{}.pth".format(actual_step))
                path = os.path.dirname(filename)
                os.makedirs(os.path.abspath(path), exist_ok=True)
                state = {"net": sampler.state_dict(), "optim": optimizer.state_dict()}
                torch.save(state, filename)
        if args.cuda:
            print("Memory allocated", torch.cuda.max_memory_allocated(device)/10**9)
if __name__ == "__main__":
    main()
