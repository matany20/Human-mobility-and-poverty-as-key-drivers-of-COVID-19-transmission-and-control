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

def evaluate_policy(ind,policy_params, horizon, return_data=False):
	""" Evaluates policy for operating in close loop locally for each region
	Parameters
	----------
	horizon: number of policy periods/cycles to evaluate
	policy_params: a dictionary with policy parameters.
			hosp_trhsh number of hospitalized per region to start local lockdown
	Returns
	---------
	number of beds for ventilators and economics indicator
	"""

	# basic models fit to examine:
	global num_applied_regions
	basic_parameters_list = [
		#     '70%',
		#     '75%',
		#     '80%',
		# 'ub',
			'base',
		#     'lb',
	]
	cal_parameters = pd.read_pickle('../Data/calibration/calibrattion_dict.pickle')
	cal_parameters = {key: cal_parameters[ind.cell_name][key] for key in basic_parameters_list}

	scen = 'Scenario2'
	start_inter = pd.Timestamp('2020-05-03')
	beginning = pd.Timestamp('2020-02-20')
	policy_period = policy_params['period']

	if ind.cell_name == '20':
		inter_name = '10_inter'
	else:
		inter_name = '250@10'

	with open('../Data/interventions/C_inter_' + inter_name + '.pickle', 'rb') as pickle_in:
		C_inter = pickle.load(pickle_in)

	with open('../Data/interventions/stay_home_idx_inter_' + inter_name + '.pickle', 'rb') as pickle_in:
		stay_home_idx_inter = pickle.load(pickle_in)

	with open('../Data/interventions/routine_t_inter_' + inter_name + '.pickle', 'rb') as pickle_in:
		routine_t_inter = pickle.load(pickle_in)

	with open('../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle', 'rb') as pickle_in:
		transfer_pop_inter = pickle.load(pickle_in)

	if ind.cell_name == '20':
		no_inter_name = '20@95_school_kid010'
	else:
		no_inter_name = '250@100_school_kid010' # '100_inter'
	with open('../Data/interventions/C_inter_' + no_inter_name + '.pickle', 'rb') as pickle_in:
		C_no_inter = pickle.load(pickle_in)

	with open('../Data/interventions/stay_home_idx_inter_' + no_inter_name + '.pickle', 'rb') as pickle_in:
		stay_home_idx_no_inter = pickle.load(pickle_in)

	with open('../Data/interventions/routine_t_inter_' + no_inter_name + '.pickle', 'rb') as pickle_in:
		routine_t_no_inter = pickle.load(pickle_in)

	with open('../Data/interventions/transfer_pop_inter_' + no_inter_name + '.pickle', 'rb') as pickle_in:
		transfer_pop_no_inter = pickle.load(pickle_in)

	if ind.cell_name == '20':
		current_inter_name = '20@75_no_school_no_kid010'
	else:
		current_inter_name = '250@75_no_risk60_no_school_kid010'
	with open('../Data/interventions/C_inter_' + current_inter_name + '.pickle', 'rb') as pickle_in:
		C_current_inter = pickle.load(pickle_in)

	with open('../Data/interventions/stay_home_idx_inter_' + current_inter_name + '.pickle', 'rb') as pickle_in:
		stay_home_idx_current_inter = pickle.load(pickle_in)

	with open('../Data/interventions/routine_t_inter_' + current_inter_name + '.pickle', 'rb') as pickle_in:
		routine_t_current_inter = pickle.load(pickle_in)

	with open('../Data/interventions/transfer_pop_inter_' + current_inter_name + '.pickle', 'rb') as pickle_in:
		transfer_pop_current_inter = pickle.load(pickle_in)


	time_steps_before_intervention = 0


	for key in cal_parameters.keys():
		time_in_season = 0
		def_model = mdl.Model_behave(
			ind=ind,
			beta_j=cal_parameters[key]['beta_j'],
			theta=cal_parameters[key]['theta'],
			beta_behave=cal_parameters[key]['beta_behave'],
			scen=scen,
		)
		res_mdl = def_model.predict(
			C=mdl.C_calibration,
			days_in_season=(start_inter - beginning).days,
			stay_home_idx=mdl.stay_home_idx,
			not_routine=mdl.not_routine,
			start=time_in_season,
		)
		time_in_season += (start_inter - beginning).days

		time_steps_before_intervention = (start_inter - beginning).days

		# change to 75 percent market for 1 week
		res_mdl = def_model.intervention(
			C=C_current_inter,
			days_in_season=7,
			stay_home_idx=stay_home_idx_current_inter,
			not_routine=routine_t_current_inter,
			prop_dict=transfer_pop_current_inter,
			start=time_in_season,
		)
		time_in_season += 7

		time_steps_before_intervention = time_steps_before_intervention + 7

		run_model = copy.deepcopy(def_model)
		num_applied_regions = 0
		t_range = range(policy_period)

		applied_policies = {}

		last_intervention_duration = dict([(reg,0) for reg in ind.region_dict.keys()])
		policy_params['applied_threshold'] = {}
		policy_params['vents_goal'] = 0

		for i in range(horizon):
			if 'threshold_table' in policy_params:
				policy_params['threshold'] = policy_params['threshold_table'][i]

			if 'start_KI' in policy_params:
				if i < policy_params['start_KI']: # do not close loop on threshold before needed
					integral_delta = policy_params['threshold']
					# store maximum number of vents in use
					if i > policy_params['start_KI']/2:
						policy_params['vents_goal'] = np.maximum(policy_params['vents_goal'],\
															 (((res_mdl['Vents'][-1]).sum()) * mdl.pop_israel))

				else:
					if i == policy_params['start_KI']: # Store the current vents use and use it as a goal
						policy_params['vents_goal'] = np.maximum(policy_params['vents_goal'],
																 (((res_mdl['Vents'][-1]).sum()) * mdl.pop_israel))
						print('Stored vents_goal ',policy_params['vents_goal'])
					else: # i is greater than start_KI, close the loop
						delta = ((((res_mdl['Vents'][-1]).sum()) * mdl.pop_israel) - policy_params['vents_goal'])
						integral_delta = np.minimum(integral_delta + np.maximum(0,policy_params['KIgain'] * delta),
													policy_params['limiter'])
						policy_params['threshold'] = np.maximum(0, policy_params['KPgain'] * delta) + integral_delta
						policy_params['threshold'] = np.maximum(np.minimum(policy_params['threshold'], policy_params['limiter']),0)

			policy_params['applied_threshold'].update({i:policy_params['threshold']})


			ratio_R = ((res_mdl['R'][-1]).sum() \
							 / ((res_mdl['S'][-1]).sum() + (res_mdl['R'][-1]).sum()))
			regions_applied, policy = policy_function(ind,policy_params, res_mdl,last_intervention_duration)
			num_applied_regions = num_applied_regions + len(regions_applied)

			applied_policies[i] = policy
			C_apply, stay_home_idx_apply  = apply_policy(ind,regions_applied,C_inter,stay_home_idx_inter,C_no_inter,stay_home_idx_no_inter,t_range)


			res_mdl = run_model.intervention(
				C=C_apply,
				days_in_season=policy_period,
				stay_home_idx=stay_home_idx_apply,
				not_routine=routine_t_no_inter,
				start=time_in_season,
			 #   prop_dict = transfer_pop_inter
			)
			time_in_season += policy_period

	product, product_1year, max_product, max_product_1year = EconomicCost(applied_policies,res_mdl,policy_period)

	max_beds = (((res_mdl['Vents'][(time_steps_before_intervention+round(policy_params['start_KI']/2)):]).sum(axis=1))*mdl.pop_israel).max()

	analysis = AnalyzeData(ind, policy_params, res_mdl, applied_policies,time_steps_before_intervention)
	analysis.update({'max_resp': max_beds, 'GDP': product, 'GDP_1year': product_1year, \
					 'max_GDP': max_product, 'max_GDP_1year': max_product_1year,\
					 ' ': ((res_mdl['Vents']).sum(axis=1))*mdl.pop_israel })

	if return_data: # parallel run cannot support returning a lot of data. so only return if asked for.
		analysis.update({'res_mdl':res_mdl, 'policies' : applied_policies})

	return analysis



def AnalyzeData(ind, policy_params, res_mdl, applied_policies,time_steps_before_intervention):

	# find herd immunization time - define as 60% recovered
	ratio_R =   ((res_mdl['R'][time_steps_before_intervention:]).sum(axis=1) \
	 	/ ((res_mdl['S'][time_steps_before_intervention:]).sum(axis=1) + (res_mdl['R'][time_steps_before_intervention:]).sum(axis=1)))

	analysis = {}
	if (np.argwhere(ratio_R > 0.3)).size > 0:
		analysis['days_immune'] = np.argwhere(ratio_R > 0.3 )[0]
	else:
		analysis['days_immune'] = np.nan

	# Lockdown statistics
	policy_image = np.zeros([len(applied_policies[0]), len(applied_policies) * policy_params['period']])
	for t in applied_policies:
		for i, region in enumerate(applied_policies[t]):
			policy_image[i][t * policy_params['period']:(t + 1) * policy_params['period']] \
				= np.ones([policy_params['period']]) * applied_policies[t][region]

	analysis['policy_image'] = policy_image
	analysis['applied_policy_ratio'] = policy_image.sum(axis=0 )/len(applied_policies[0])

	augmented_image = np.hstack((np.zeros([len(applied_policies[0]), 1]), policy_image, np.zeros([len(applied_policies[0]), 1])))
	max_lockdown_len = {}
	for i,region in enumerate(ind.region_dict):
		delta = np.argwhere(np.diff(augmented_image[i, :]) == -1) - np.argwhere(np.diff(augmented_image[i, :]) == 1)
		if delta.size > 0:
			max_lockdown_len[region] = np.amax(delta)
		else:
			max_lockdown_len[region] = 0

	analysis['max_lockdown_len'] = max_lockdown_len

	analysis['tot_lockdown_len'] = policy_image.sum(axis=1)


	if len(res_mdl['R'])> (time_steps_before_intervention+365):
		analysis['R_1year'] = (res_mdl['R'][time_steps_before_intervention+365]).sum()
	else:
		analysis['R_1year'] = (res_mdl['R'][-1]).sum()

	return analysis


def EconomicCost(applied_policies,res_mdl,policy_period):
	"""
	Computes an estimate of an economic cost for a given policy as applied.
	Parameters
	----------
	applied_policies : dictionary by time of dictionaries by regions
	res_mdl         : epidimiological data
	policy_period   : period in days of applied policies
	"""
	cell2salary = pd.read_csv('../Data/demograph/cell2salary.csv')
	cell2salary['tot_salary'] = (cell2salary['salary'] * cell2salary['employees'])
	salary = {}
	for i in range(len(cell2salary)):
		if type(cell2salary['cell_id'][i])==str:
			salary[cell2salary['cell_id'][i]] = cell2salary['tot_salary'][i] / 30  # convert per day
		else:
			salary[cell2salary['cell_id'][i].astype(str)] = cell2salary['tot_salary'][i]/30 # convert per day


	#average_salary_per_day = 10e3/30 # a place holder.
	#working_proportion = 0.4
	product = 0
	product_1year = 0
	max_product = 0
	max_product_1year = 0

	for t in applied_policies.keys():
		for region in applied_policies[t].keys():
			if applied_policies[t][region]==True:
				rate = 0.59 # from "משבר הקורונה –השלכות על שוק העבודה", slide 12
			else:
				rate = 1

			product = product+ salary[region]*rate*policy_period
			max_product = max_product+ salary[region]*1*policy_period
			if (t*policy_period > 365) and (product_1year == 0):
				# store 1 year data
				product_1year = product*365/(t*policy_period) # normalize period
				max_product_1year = max_product*365/(t*policy_period) # normalize period


	if product_1year==0:
		product_1year = product
		max_product_1year = max_product

	return product/1e9,product_1year/1e9,max_product,max_product_1year # translate to billions

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

def policy_function(ind,policy_params,res_mdl,last_intervention_duration):
	"""
	Returns a list of regions with policy applied and a dictionary of regions with policy decision for each
	Parameters
	----------
	res_mdl : known number of patients
	policy_params : dictionary with policy parameters
	"""
	intervention_policy = {}
	regions_intervent = []

	if ('global_decision' in policy_params) and (policy_params['global_decision']==True):
		Vals = np.zeros([4, len(ind.age_dict.keys())])
		# [Is, new_Is, H, Vent]
		for j,age in enumerate(ind.age_dict.keys()):
			Vals[0][j] = (res_mdl['Is'][-1][ind.age_dict[ age]]).sum()
			Vals[1][j] = (res_mdl['new_Is'][-1][ind.age_dict[ age]]).sum()
			Vals[2][j] = (res_mdl['Hosp_latent'][-1][ind.age_dict[ age]]).sum() + \
					 (res_mdl['H'][-1][ind.age_dict[ age]]).sum()
			Vals[3][j] = (res_mdl['Vents_latent'][-1][ind.age_dict[age]]).sum() + \
					 (res_mdl['Vents'][-1][ind.age_dict[age]]).sum()

		val_for_trhsh = (np.multiply(Vals, policy_params['weight_matrix'])).sum()
		if (val_for_trhsh > policy_params['threshold']) and \
			(last_intervention_duration[list(ind.region_dict.keys())[0]] < policy_params['max_duration']):
			for i, region in enumerate(ind.region_dict.keys()):
				intervention_policy[region] = True
				regions_intervent.append(region)
				last_intervention_duration[region] = last_intervention_duration[region] + 1
		else:
			for i, region in enumerate(ind.region_dict.keys()):
				intervention_policy[region] = False
				last_intervention_duration[region] = 0
	else:
		for i, region in enumerate(ind.region_dict.keys()):
			Vals = np.zeros([4, len(ind.age_dict.keys())])
			for j,age in enumerate(ind.age_dict.keys()):
				# [Is, new_Is, H, Vent]
				Vals[0][j] = (res_mdl['Is'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[1][j] = (res_mdl['new_Is'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[2][j] = (res_mdl['Hosp_latent'][-1][ind.region_age_dict[region,age]]).sum() +\
							(res_mdl['H'][-1][ind.region_age_dict[region,age]]).sum()
				Vals[3][j] = (res_mdl['Vents_latent'][-1][ind.region_age_dict[region,age]]).sum() +\
							(res_mdl['Vents'][-1][ind.region_age_dict[region,age]]).sum()

			val_for_trhsh = (np.multiply(Vals,policy_params['weight_matrix'])).sum()
			if (val_for_trhsh/mdl.population_size[ind.region_dict[region]].sum() \
					> policy_params['threshold']) and (last_intervention_duration[region] < policy_params['max_duration']):
				intervention_policy[region] = True
				regions_intervent.append(region)
				last_intervention_duration[region] = last_intervention_duration[region] + 1
			else:
				intervention_policy[region] = False
				last_intervention_duration[region] = 0


	return regions_intervent, intervention_policy

def SeasonalBetaFactor(date):
	day_in_year = date.timetuple().tm_yday

	# 20 percent decrease in beta in the middle of summer:
	beta_factor = 1+ min(0, math.cos(day_in_year/365*math.pi)*0.2)
	return  beta_factor


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