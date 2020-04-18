import pickle
import os


########################
# -- Set Parameters -- #
########################

# Load Parameters
# with open('../model_data/parameters.pickle', 'rb') as pickle_in:
#     parameters = pickle.load(pickle_in)

# Transmissibility (beta_jk - for each age group, for symptomatic/asymptomatic)
# beta = parameters['beta']

# Beta_home - home transmissibility:
with open('../Data/parameters/beta_home.pickle', 'rb') as pickle_in:
	beta_home = pickle.load(pickle_in)

# hospitalizations data for high risk with/without treatment and low risk:
with open('../Data/parameters/hospitalization.pickle', 'rb') as pickle_in:
	hospitalizations = pickle.load(pickle_in)


# Asymptomatic fraction
with open('../Data/parameters/f0_full.pickle', 'rb') as pickle_in:
	f0_full = pickle.load(pickle_in)

# Contact matrix dic
with open('../Data/parameters/C_calibration.pickle', 'rb') as pickle_in:
	C_calibration = pickle.load(pickle_in)

# Orthodox distribution
with open("Data/parameters/orthodox_dist.pickle", 'rb') as pickle_in:
	is_haredi = pickle.load(pickle_in)

# stay home index for behavior model
with open('Data/parameters/stay_home_idx.pickle', 'rb') as pickle_in:
	stay_home_idx = pickle.load(pickle_in)

# routine vector behavior model
with open('Data/parameters/routine_t.pickle', 'rb') as pickle_in:
    not_routine = pickle.load(pickle_in)

#  gama - transition rate between Is,Ia to R
gama = 1. / 7.

# delta - transition rate Ie to Is, Ia
delta = 1. / 2.

# sigma - transition rate E to Ie
sigma = 1. / 3.5


# Population size
with open('../Data/parameters/init_pop.pickle', 'rb') as pickle_in:
	population_size = pickle.load(pickle_in)

# fixing parameter for beta_home
psi = 1

# new - Transition rate out of H
nu = 1. / 9.

# Epsilon (small noise) - only for non-zero population groups
with open('../Data/parameters/eps_dict.pickle', 'rb') as pickle_in:
	eps = pickle.load(pickle_in)

# alpha - early infected infection factor
alpha = 1.0

# # Load immunity proportions
# with open('../model_data/immunity.pickle', 'rb') as pickle_in:
#     immunity = pickle.load(pickle_in)

# # Load log viral load
# with open('../model_data/viral_load.pickle', 'rb') as pickle_in:
#     VL = pickle.load(pickle_in)