from .indices import *
from .helper_func import *
from .parameters import *
import pandas as pd
import numpy as np

import itertools
import pickle


#######################
# ---- Run Model ---- #
#######################

def run_model(
		beta_j,
		eps,
		f,
		days_in_season,
		alpha=alpha,
		beta_home = beta_home,
		sigma=sigma,
		delta=delta,
		gama=gama,
		population_size=population_size,
		C=C_calibration,
		psi=psi,
		rho=hospitalizations,
		nu=nu
	):
	"""
	Receives all model's data and parameters, runs it for a season and returns
	the results.
	:param beta_j:
	:param eps:
	:param f:
	:param days_in_season:
	:param alpha:
	:param beta_home:
	:param sigma:
	:param delta:
	:param gama:
	:param population_size:
	:param C:
	:param psi:
	:param rho:
	:param nu:
	:return:
	"""

	# Expand data - fitting beta_j size to (180X1)/
	beta_j = expand_partial_array(
		mapping_dic=age_ga_dict,
		array_to_expand=beta_j,
		size=len(GA)
	)
	# population size in each area
	init_region_pop = shrink_array_sum(
		mapping_dic=region_dict,
		array_to_shrink=population_size
	)

	# Initialize lists to save the states throughout the time steps
	S, E, Ie, Is, Ia, R, H = [], [], [], [], [], [], []

	# Initialize a list for the newly infected
	new_I, new_Is = [], []

	# Initialize a list for the lambdas
	L = []

	### -- Initialize the model states: -- ###
	# Initialize S_0 to population size of each age-group
	S.append(population_size.copy())

	# Initialize R - with only the naturally immune individuals
	R.append(np.zeros(len(N)))

	# subtract R0 from S0:
	S[-1] -= R[-1]

	# Initialize E  to 5*10**-4 ????
	E.append(np.zeros(len(N)))
	#             E[-1][:] = init_I

	# Initialize I (early) to 5*10**-4 ????
	Ie.append(np.zeros(len(N)))
	# Ie[-1][:] = eps[0]

	# Initialize I (asymptomatic) to 5*0.5*10**-4 ????
	Ia.append(np.zeros(len(N)))

	# Initialize I (symptomatic) to 5*0.5*10**-4 ????
	Is.append(np.zeros(len(N)))

	# Subtract I_0 and A_0 from S_0
	S[-1] -= (E[-1] + Ie[-1])

	# Zero newly infected on the first day of the season
	new_I.append(np.zeros(len(N)))
	new_Is.append(np.zeros(len(N)))

	# Initialize H, tracking compartment
	H.append(np.zeros(len(N)))

	### -- Run the model -- ###
	for t in range(days_in_season):
		# Calculate beta_home factor, current S_region/N_region and expand it
		# to mach (180X1).
		beta_home_factor = shrink_array_sum(
			mapping_dic=region_dict,
			array_to_shrink=S[-1]) \
			/ init_region_pop
		beta_home_factor = expand_partial_array(
			mapping_dic=region_ga_dict,
			array_to_expand=beta_home_factor,
			size=len(GA)
		)

		# Calculate lambda (High risk symptomatic + Low risk symptomatic +
		# Asymptomatic).
		contact_force = calculate_force_matriceis(
			t=t,
			C=C,
			Ie=Ie[-1],
			Ia=Ia[-1],
			Is=Is[-1],
			alpha=alpha
		)

		lambda_t = (beta_home * beta_home_factor * contact_force['home'] +
					beta_j * contact_force['out'])

		L.append(lambda_t)

		# fitting lambda_t size to (720X1)
		lambda_t = expand_partial_array(
			mapping_dic=region_age_dict,
			array_to_expand=lambda_t
		)

		# R(t)
		R.append(
			R[-1] + gama * (Is[-1] + Ia[-1])
		)

		# H(t)
		H.append(
			rho * Is[-1] - nu * H[-1]
		)

		# Is(t)
		# Save new_Is
		new_Is.append(
			(1 - f) * delta * Ie[-1]
		)
		# Calculate new i matrix for day t
		Is.append(
			Is[-1] + new_Is[-1] - gama * Is[-1]
		)

		# Ia(t)
		# Calculate new i matrix for day t
		Ia.append(
			Ia[-1] + f * delta * Ie[-1] - gama * Ia[-1]
		)

		# Ie(t)
		# Calculate new i matrix for day t
		Ie.append(Ie[-1] + sigma * E[-1] - delta * Ie[-1])

		# E(t)
		# Calculate new e matrix for day t
		E.append(eps[t] + E[-1] + lambda_t * S[-1] - sigma * E[-1])

		# S(t)
		# Save new_I
		new_I.append(lambda_t * S[-1])
		# Calculate current S
		S.append(S[-1] - lambda_t * S[-1])

	# Return the model results
	return {
		'S': np.array(S),
		'E': np.array(E),
		'Ie': np.array(Ie),
		'Ia': np.array(Ia),
		'Is': np.array(Is),
		'R': np.array(R),
		'new_I': np.array(new_I),
		'new_Is': np.array(new_Is),
		'L': np.array(L),
		'H': np.array(H),
	}


def run_sector_model(
		beta_j,
		eps,
		f,
		days_in_season,
		theta,
		is_haredi,
		alpha=alpha,
		beta_home = beta_home,
		sigma=sigma,
		delta=delta,
		gama=gama,
		population_size=population_size,
		C=C_calibration,
		psi=psi,
		rho=hospitalizations,
		nu=nu
	):
	"""
	Receives all model's data and parameters, runs it for a season and returns
	the results.
	:param beta_j:
	:param eps:
	:param f:
	:param days_in_season:
	:param theta:
	:param is_haredi:
	:param alpha:
	:param beta_home:
	:param sigma:
	:param delta:
	:param gama:
	:param population_size:
	:param C:
	:param psi:
	:param rho:
	:param nu:
	:return:
	"""

	# Expand data
	# fitting beta_j size to (180X1)
	beta_j = expand_partial_array(
		mapping_dic=age_ga_dict,
		array_to_expand=beta_j,
		size=len(GA)
	)
	# population size in each area
	init_region_pop = shrink_array_sum(mapping_dic=region_dict,array_to_shrink=population_size)
	# Initialize lists to save the states throughout the time steps
	S, E, Ie, Is, Ia, R, H = [], [], [], [], [], [], []

	# Initialize a list for the newly infected
	new_I, new_Is = [], []

	# Initialize a list fot the lambdas
	L = []

	# Run the model
	for t in range(days_in_season):
		# If first iteration - initialize all states
		if t % days_in_season == 0:
			# Initialize S_0 to population size of each age-group
			S.append(population_size.copy())
			# Initialize R - with only the naturally immune individuals
			R.append(np.zeros(len(N)))

			# subtract R0 from S0:
			S[-1] -= R[-1]

			# Initialize E  to 5*10**-4 ????
			E.append(np.zeros(len(N)))
			#             E[-1][:] = init_I

			# Initialize I (early) to 5*10**-4 ????
			Ie.append(np.zeros(len(N)))
			Ie[-1][:] = eps[t]

			# Initialize I (asymptomatic) to 5*0.5*10**-4 ????
			Ia.append(np.zeros(len(N)))

			# Initialize I (symptomatic) to 5*0.5*10**-4 ????
			Is.append(np.zeros(len(N)))

			# Subtract I_0 and A_0 from S_0
			S[-1] -= (E[-1] + Ie[-1])

			# Zero newly infected on the first day of the season
			new_I.append(np.zeros(len(N)))
			new_Is.append(np.zeros(len(N)))

			# Initialize H, tracking compartment
			H.append(np.zeros(len(N)))

		# Not a new season

		# Calculate beta_home factor, current S_region/N_region and expand it
		# to mach (180X1).
		beta_home_factor = shrink_array_sum(
			mapping_dic=region_dict,
			array_to_shrink=S[-1]) \
						   / init_region_pop
		beta_home_factor = expand_partial_array(
			mapping_dic=region_ga_dict,
			array_to_expand=beta_home_factor,
			size=len(GA)
		)

		# Calculate lambda (High risk symptomatic + Low risk symptomatic +
		# Asymptomatic)
		contact_force = calculate_force_matriceis(
			t=t,
			C=C,
			Ie=Ie[-1],
			Ia=Ia[-1],
			Is=Is[-1],
			alpha=alpha
		)

		lambda_t = (beta_home * beta_home_factor * contact_force['home'] +
					beta_j * theta * is_haredi * contact_force['out'])

		L.append(lambda_t)
		# fitting lambda_t size to (720X1)
		lambda_t = expand_partial_array(
			mapping_dic=region_age_dict,
			array_to_expand=lambda_t
		)

		# R(t)
		R.append(R[-1] + gama * (Is[-1] + Ia[-1]) - eps[t])

		# H(t)
		H.append(rho * Is[-1] - nu * H[-1])

		# Is(t)
		# Save new_Is
		new_Is.append((1 - f) * delta * Ie[-1])
		# Calculate new i matrix for day t
		Is.append(Is[-1] + new_Is[-1] - gama * Is[-1])

		# Ia(t)
		# Calculate new i matrix for day t
		Ia.append(Ia[-1] + f * delta * Ie[-1] - gama * Ia[-1])

		# Ie(t)
		# Calculate new i matrix for day t
		Ie.append(Ie[-1] + sigma * E[-1] - delta * Ie[-1])

		# E(t)
		# Calculate new e matrix for day t
		E.append(eps[t] + E[-1] + lambda_t * S[-1] - sigma * E[-1])

		# S(t)
		# Save new_I
		new_I.append(lambda_t * S[-1])
		# Calculate current S
		S.append(S[-1] - lambda_t * S[-1])

	# Return the model results
	return {
		'S': np.array(S),
		'E': np.array(E),
		'Ie': np.array(Ie),
		'Ia': np.array(Ia),
		'Is': np.array(Is),
		'R': np.array(R),
		'new_I': np.array(new_I),
		'new_Is': np.array(new_Is),
		'L': np.array(L),
		'H': np.array(H)
	}