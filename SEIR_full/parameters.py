import pickle


########################
# -- Set Parameters -- #
########################

# Load Parameters
# with open('../model_data/parameters.pickle', 'rb') as pickle_in:
#     parameters = pickle.load(pickle_in)


# hospitalizations probabilities for age-, risk-groups :
with open('../Data/parameters/hospitalization.pickle', 'rb') as pickle_in:
	hospitalizations = pickle.load(pickle_in)

# ventilation probabilities for age-, risk-groups :
with open('../Data/parameters/vents_proba.pickle', 'rb') as pickle_in:
	vents_proba = pickle.load(pickle_in)

# Asymptomatic fraction
with open('../Data/parameters/f0_full.pickle', 'rb') as pickle_in:
	f0_full = pickle.load(pickle_in)

# Contact matrix dic
with open('../Data/parameters/C_calibration.pickle', 'rb') as pickle_in:
	C_calibration = pickle.load(pickle_in)

# Orthodox distribution
with open("../Data/parameters/orthodox_dist.pickle", 'rb') as pickle_in:
	is_haredi = pickle.load(pickle_in)

# Arab distribution
with open("../Data/parameters/arab_dist.pickle", 'rb') as pickle_in:
	is_arab = pickle.load(pickle_in)

# stay home index for behavior model
with open('../Data/parameters/stay_home_idx.pickle', 'rb') as pickle_in:
	stay_home_idx = pickle.load(pickle_in)

# routine vector behavior model
with open('../Data/parameters/routine_t.pickle', 'rb') as pickle_in:
	not_routine = pickle.load(pickle_in)

# Population size
with open('../Data/parameters/init_pop.pickle', 'rb') as pickle_in:
	population_size = pickle.load(pickle_in)

# esp - model without sectors
with open('../Data/parameters/eps_by_region.pickle', 'rb') as pickle_in:
	eps_sector = pickle.load(pickle_in)


# Transmissibility (beta_jk - for each age group, for symptomatic/asymptomatic)
# beta = parameters['beta']

# Beta_home - home transmissibility:
# with open('../Data/parameters/beta_home.pickle', 'rb') as pickle_in:
# 	beta_home = pickle.load(pickle_in)
beta_home = 0.38/9

#  gama - transition rate between Is,Ia to R
gama = 1. / 7.

# delta - transition rate Ie to Is, Ia
delta = 1. / 2.

# sigma - transition rate E to Ie
sigma = 1. / 3.5

# fixing parameter for beta_home
psi = 1

# nu - Transition rate out of H
nu = 0.07247553272296633

# mu - Transition rate out of Vents
mu = 0.07142857142857142

# eta - Transition rate H_latent to H
eta = 0.8676140711537818

# xi - Transition rate Vents_latent to Vents
xi = 0.23638829809201067

# alpha - early infected infection factor
alpha = 1.0

pop_israel = 9136000

# # Load immunity proportions
# with open('../model_data/immunity.pickle', 'rb') as pickle_in:
#     immunity = pickle.load(pickle_in)

# # Load log viral load
# with open('../model_data/viral_load.pickle', 'rb') as pickle_in:
#     VL = pickle.load(pickle_in)