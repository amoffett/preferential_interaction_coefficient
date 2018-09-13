import argparse
  
parser = argparse.ArgumentParser()
parser.add_argument('--pic_pickle', type=str, help='Name of the pickle file containing pic_protein.py output.')
parser.add_argument('--cosolvent', type=str, help='Cosolvent type. Currently, only 'urea' is accepted', default='urea')
parser.add_argument('--cosolvent_concentration'), type=float, help='Molal (moles cosolvent / kg water) concentration of cosolvent.')
parser.add_argument('--r_star', type=float, help='Dividing distance between local and bulk domains (nanometers).')
parser.add_argument('--temperature', type=float, help='Temperature (Kelvin)')
parser.add_argument('--out_file', type=str, help='Prefix of the output pickle file.')
args = parser.parse_args()

import numpy as np
import pickle as pkl
from scipy import constants

r_range, state_gamma_rt, state_gamma_sample = pkl.load(open(args.pic_pickle,"rb"))
R = (1/constants.calorie)/1000.0 * constants.R # gas constant in kcal/mol

if args.cosolvent == 'urea':
        gamma = .018
        alpha = .8309
        dlog_a_W_dm_W = lambda m: ((2*alpha*(gamma*m)/(1+gamma*m))-1/(1-((gamma*m)/(1+gamma*m))))*(gamma/(1+gamma*m)-(gamma**2*m)/(1+gamma*m)**2)
        m_W = 1000.0 / 18
else:
        raise ValueError('Only urea is supported as a cosolvent.')

r_index = np.argmin(np.abs(r_range-args.r_star))

state_mu = {}
for state in state_gamma_rt.keys():
        state_sample = state_gamma_sample[state]
        gamma = state_gamma_rt[state][r_index][state_sample].mean()
        mu = R * args.temperature * m_W * dlog_a_W_dm_W(args.cosolvent_concentration) * gamma
        state_mu[state] = mu

dump(state_mu, open(args.out_file + ".pkl","wb"))
