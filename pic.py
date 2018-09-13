import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--traj_prefix', type=str, help='Prefix for input trajectories. We assume the trajectories are named as follows: <traj_prefix>_[state index].dcd')
parser.add_argument('--top', type=str, help='Name of the topology file.')
parser.add_argument('--n_states', type=int, help='Number of MSM states.')
parser.add_argument('--out_file', type=str, help='Prefix of the output pickle file.')
args = parser.parse_args()

import numpy as np
import mdtraj as md
from pickle import dump
from pymbar.timeseries import subsampleCorrelatedData

t_ref = md.load(args.top)
t_ref_prot = t_ref.atom_slice(t_ref.topology.select('protein and not resname UNK'))
types = [str(i).split('-')[-1][0] for i in t_ref_prot.topology.atoms]

vdw_radii = {'C':0.170,'O':0.152,'N':0.155,'H':0.120,'S':0.180} # in nanometers
radii = np.array([vdw_radii[i] for i in types])

unk_pairs = t_ref.topology.select_pairs('resname UNK and name C','protein and not resname UNK').reshape([547,624,2])
wat_pairs = t_ref.topology.select_pairs('water and type O','protein and not resname UNK').reshape([3800,624,2])

def gamma_rt(cos,wat,r):
        """
        Calculate the preferential interaction coefficient (gamma) of a protein with water and a cosolvent.
        
        ***ALL DISTANCES ARE IN NANOMETERS***
        
        Input: cos, wat, r
          - cos : (T frames) X (N cosolvent molecules) array, the minimum distance of each cosolvent molecule to the protein Van der Waals surface for each frame.
          - wat : (T frames) X (M water molecules) array, the minimum distance of each water molecule to the protein Van der Waals surface for each frame. 
          - r : float, distance dividing the local and bulk domains of the solvent.
        
        Returns: gamma, sample
          - gamma : (T frames) array, gamma for the given r, for each inputted frame.
          - sample : list, the N_effective independent frames of gamma to be used for calculation of the time average of gamma. Obtained using the method of Chodera (2016).
          
        References:
          - BM Baynes and BL Trout. Proteins in mixed solvents: a molecular-level perspective. J. Phys. Chem. B. 107, 14058-14067 (2003).
          - D Shukla, C Shinde, and BL Trout. Molecular computations of preferential interaction coefficients of proteins. J. Phys. Chem. B. 113, 12546-12554 (2009).
          - JD Chodera. J. Chem. Theor. Comput. 12, 1799 (2016).
        """
        n_i_x = np.sum(cos > r,axis=1).astype(float)
        n_ii_x = np.sum(cos < r,axis=1).astype(float)
        n_i_w = np.sum(wat > r,axis=1).astype(float)
        n_ii_w = np.sum(wat < r,axis=1).astype(float)
        gamma = n_ii_x - n_ii_w * (n_i_x/n_i_w)
        sample = subsampleCorrelatedData(gamma)
        return gamma, sample

r_range = np.linspace(0,2.0,num=201) # Range of distances with which to calculate gamma
state_gamma_rt = {}
state_gamma_sample = {}
for state in range(args.n_states):
        t = md.load(args.traj_prefix + "_%i.dcd"%state,top=args.top)

        unk_dists = []
        for u in range(unk_pairs.shape[0]):
                ud = md.compute_distances(t,unk_pairs[u])
                min_ud = np.min(ud - radii,axis=1,keepdims=True)
                unk_dists.append(min_ud)
        unk_dists = np.hstack(unk_dists)

        wat_dists = []
        for w in range(wat_pairs.shape[0]):
                wd = md.compute_distances(t,wat_pairs[w])
                min_wd = np.min(wd - radii,axis=1,keepdims=True)
                wat_dists.append(min_wd)
        wat_dists = np.hstack(wat_dists)

        gamma = []
        sample = []
        for r in r_range:
                g, s = gamma_rt(unk_dists,wat_dists,r)
                gamma.append(g)
                sample.append(s)

        gamma = np.vstack(gamma)
        state_gamma_rt[state] = gamma
        state_gamma_sample[state] = sample

dump([r_range, state_gamma_rt, state_gamma_sample], open(args.out_file + ".pkl","wb"))
