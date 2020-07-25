import math
from typing import List

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
from scipy import optimize
import sys

from scipy.sparse import csr_matrix

sys.path.append('../SEIR_full/')
sys.path.append('..')
import SEIR_full as mdl
import SEIR_full.model_class as mdl
import datetime as dt
from scipy.stats import poisson
from scipy.stats import binom
import copy
import os
import time

#
# Functions and utilities to optimize policies.
#
#
# TODO: Add actual optimization
# TODO: Add tests cases for parameter variation at fitted model
# TODO: Add Monte-Carlo
# TODO: Add robustness check to Major Infection Events
# TODO: Add multi parameter optimization (criteria selection, periods, multiple criteria)
#
# author: Sergey Vichik


# Candidates for threshold: # [Is, new_Is, H, Vent]
# Therefore the weight matrix is nxm, where n is 4 (as above) and m is the number of ages

def run_global_policy(ind, model, policy_params, sim_time, is_pop, start=0):
	time_of_season = start
	pol_model = copy.deepcopy(model)
	with open('../Data/interventions/C_inter_' + policy_params['free_inter'] + '.pickle',
			  'rb') as pickle_in:
		C_free = pickle.load(pickle_in)

	with open(
			'../Data/interventions/stay_home_idx_inter_' + policy_params['free_inter'] + '.pickle',
			'rb') as pickle_in:
		stay_home_idx_free = pickle.load(pickle_in)

	with open(
			'../Data/interventions/routine_t_inter_' + policy_params['free_inter'] + '.pickle',
			'rb') as pickle_in:
		routine_t_free = pickle.load(pickle_in)

	with open(
			'../Data/interventions/transfer_pop_inter_' + policy_params['free_inter'] + '.pickle',
			'rb') as pickle_in:
		transfer_pop_free = pickle.load(pickle_in)

	with open('../Data/interventions/C_inter_' + policy_params[
		'stop_inter'] + '.pickle',
			  'rb') as pickle_in:
		C_stop = pickle.load(pickle_in)

	with open(
			'../Data/interventions/stay_home_idx_inter_' + policy_params[
				'stop_inter'] + '.pickle',
			'rb') as pickle_in:
		stay_home_idx_stop = pickle.load(pickle_in)

	res_mdl = pol_model.get_res()
	t_range = range(policy_params['policy_period'])
	applied_policies = []
	last_intervention_duration = dict(
		[(reg, 0) for reg in ind.region_dict.keys()])

	for period in range(int(float(sim_time)/policy_params['policy_period'])):
		if period != 0:
			transfer_pop_free = None
		regions_applied, policy = simple_policy_function(
			ind,
			policy_params,
			res_mdl,
			last_intervention_duration,
			is_pop,
		)
		for t in t_range:
			applied_policies.append(policy)
		C_apply, stay_home_idx_apply = apply_policy(
			ind,
			regions_applied,
			C_stop,
			stay_home_idx_stop,
			C_free,
			stay_home_idx_free,
			time_of_season + np.array(t_range)
		)
		res_mdl = pol_model.intervention(
			C=C_apply,
			days_in_season=len(t_range),
			stay_home_idx=stay_home_idx_apply,
			not_routine=routine_t_free,
			start=time_of_season,
			prop_dict=transfer_pop_free,
		)
		time_of_season += len(t_range)

	return res_mdl, applied_policies



def simple_policy_function(
		ind,
		policy_params,
		res_mdl,
		last_intervention_duration,
		is_pop,
	):
	intervention_policy = {}
	for i, region in enumerate(ind.region_dict.keys()):
		intervention_policy[region] = False
	regions_intervent = []

	# calculate global threshold:
	Vals = np.zeros([4, len(ind.age_dict.keys())])
	# [Is, new_Is, H, Vent]
	for j, age in enumerate(ind.age_dict.keys()):
		Vals[0][j] = (res_mdl['Is'][-1][ind.age_dict[age]]).sum()
		Vals[1][j] = (res_mdl['new_Is'][-1][ind.age_dict[age]]).sum()
		Vals[2][j] = (res_mdl['H'][-1][ind.age_dict[age]]).sum()
		Vals[3][j] = (res_mdl['Vents'][-1][ind.age_dict[age]]).sum()

	global_val_for_trhsh = 1000*(
		np.multiply(Vals, policy_params['weight_matrix']).sum())

	if ('global_thresh' in policy_params) and (
		policy_params['global_thresh'] == True
	):
		if ('global_lock' in policy_params) and (
			policy_params['global_lock'] == True

		):
			if (global_val_for_trhsh > policy_params['threshold']) and \
					(last_intervention_duration[list(ind.region_dict.keys())[0]] <
					 policy_params['max_duration']):
				for i, region in enumerate(ind.region_dict.keys()):
					intervention_policy[region] = True
					regions_intervent.append(region)
					last_intervention_duration[region] = \
					last_intervention_duration[region] + 1
			else:
				for i, region in enumerate(ind.region_dict.keys()):
					intervention_policy[region] = False
					last_intervention_duration[region] = 0
		else:
			if (global_val_for_trhsh > policy_params['threshold']):
				val_for_trhsh = {}
				for i, region in enumerate(ind.region_dict.keys()):
					Vals = np.zeros([4, len(ind.age_dict.keys())])
					for j, age in enumerate(ind.age_dict.keys()):
						# [Is, new_Is, H, Vent]
						Vals[0][j] = (
						res_mdl['Is'][-1][ind.region_age_dict[region, age]]).sum()
						Vals[1][j] = (
						res_mdl['new_Is'][-1][ind.region_age_dict[region, age]]).sum()
						Vals[2][j] = (res_mdl['H'][-1][
										 ind.region_age_dict[region, age]]).sum()
						Vals[3][j] = (res_mdl['Vents'][-1][
										 ind.region_age_dict[region, age]]).sum()

					val_for_trhsh[region] = (
						np.multiply(Vals, policy_params['weight_matrix'])).sum()
				sorted_cells = sorted(
					val_for_trhsh.keys(),
					key=val_for_trhsh.get,
					reverse=True,
				)
				count = 0
				for region in sorted_cells:
					if (last_intervention_duration[region] < policy_params['max_duration']):
						intervention_policy[region] = True
						regions_intervent.append(region)
						last_intervention_duration[region] += 1
						count += 1
						if count >= policy_params['num_of_regions']:
							break
					else:
						last_intervention_duration[region] = 0
	else:
		for i, region in enumerate(ind.region_dict.keys()):
			Vals = np.zeros([4, len(ind.age_dict.keys())])
			for j,age in enumerate(ind.age_dict.keys()):
				# [Is, new_Is, H, Vent]
				Vals[0][j] = (res_mdl['Is'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[1][j] = (res_mdl['new_Is'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[2][j] = (res_mdl['H'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[3][j] = (res_mdl['Vents'][-1][ind.region_age_dict[region,age]]).sum()

			val_for_trhsh = (np.multiply(Vals,policy_params['weight_matrix'])).sum()
			val_for_trhsh = (1000 * val_for_trhsh)/(mdl.population_size[ind.region_dict[region]].sum())
			if (val_for_trhsh > policy_params['threshold']) and (
					last_intervention_duration[region] < policy_params['max_duration']):
				intervention_policy[region] = True
				regions_intervent.append(region)
				last_intervention_duration[region] = last_intervention_duration[region] + 1
			else:
				intervention_policy[region] = False
				last_intervention_duration[region] = 0

	return regions_intervent, intervention_policy



def apply_policy(ind,regions_applied,C_inter,stay_home_idx_inter,C_no_inter,stay_home_idx_no_inter,t_range):
	"""" Assembles combined matrices according to regions under lockdown.
	Parameters
	----------
	t_range : times to build matrices for
	stay_home_idx_no_inter : stay_home_idx vector for no lockdown policy
	C_no_inter: Contact mixing patterns for no lockdown policy
	stay_home_idx_inter : stay_home_idx vector for lockdown policy
	C_inter : Contact mixing patterns for no lockdown policy
	regions_applied : list of regions in a lockdown
	 """
	C_apply = copy.deepcopy(C_no_inter)
	stay_home_idx_applied = copy.deepcopy(stay_home_idx_no_inter)

	# Assemble C matrix
	# Not sure if for loop is the best way to copy dictionaries.
	for key in C_inter.keys():
		C_mat = csr_matrix(mdl.isolate_areas(
			ind,
			(C_no_inter[key][0]).todense(),
			(C_inter[key][0]).todense(),
			regions_applied,
		))
		for t in t_range:
			C_apply[key][t] = C_mat

	# Assemble stay_home_idx vector
	for key in stay_home_idx_inter.keys():
		stay_home_idx = mdl.isolate_areas_vect(
			ind,
			stay_home_idx_no_inter[key][0],
			stay_home_idx_inter[key][0],
			regions_applied,
		)
		for t in t_range:
			stay_home_idx_applied[key][t] = stay_home_idx


	return C_apply,stay_home_idx_applied
