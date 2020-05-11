## Imports
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
import scipy
from scipy import optimize
from scipy.sparse import csr_matrix
import sys
import os
from SEIR_full.indices import *

#############################
# Generating interventions  #
#############################
## Must be run after cell parameters set to specific cell division

pct = [10]  #range(30, 105, 5)
no_risk = False
no_kid = False
kid_019 = False
kid_09 = False
kid_04 = True


""" Eras explanation:
-first days of routine from Feb 21st - March 13th
-first days of no school from March 14th - March 16th
-without school and work from March 17th - March 25th
-100 meters constrain from March 26th - April 2nd
-Bnei Brak quaranrine from April 3rd - April 6th
-
"""

### creating notations for intervention.
with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

### importing files for manipulation:
# full_mtx ordering
full_mtx_home = scipy.sparse.load_npz('../Data/base_contact_mtx/full_home.npz')

full_mtx_work = {
	'routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_routine.npz'),
	'no_school': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_school.npz'),
	'no_work': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_work.npz'),
	'no_100_meters': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_100_meters.npz'),
	'no_bb': scipy.sparse.load_npz('../Data/base_contact_mtx/full_work_no_bb.npz'),
}

full_mtx_leisure = {
	'routine': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_routine.npz'),
	'no_school': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_school.npz'),
	'no_work': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_work.npz'),
	'no_100_meters': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_100_meters.npz'),
	'no_bb': scipy.sparse.load_npz('../Data/base_contact_mtx/full_leisure_no_bb.npz'),
}
# stay_home ordering
sh_school = pd.read_csv('../Data/stay_home/no_school.csv', index_col=0)
sh_school.index = sh_school.index.astype(str)
sh_work = pd.read_csv('../Data/stay_home/no_work.csv', index_col=0)
sh_work.index = sh_work.index.astype(str)
sh_routine = pd.read_csv('../Data/stay_home/routine.csv', index_col=0)
sh_routine.index = sh_routine.index.astype(str)
sh_no_100_meters = pd.read_csv('../Data/stay_home/no_100_meters.csv', index_col=0)
sh_no_100_meters.index = sh_no_100_meters.index.astype(str)
sh_no_bb = pd.read_csv('../Data/stay_home/no_bb.csv', index_col=0)
sh_no_bb.index = sh_no_bb.index.astype(str)
# reordering and expanding vector for each period:
sh_school = sh_school['mean'].values
sh_school[1] = sh_school[0]

sh_work = sh_work['mean'].values
sh_work[1] = sh_work[0]

sh_no_100_meters = sh_no_100_meters['mean'].values
sh_no_100_meters[1] = sh_no_100_meters[0]

sh_no_bb = sh_no_bb['mean'].values
sh_no_bb[1] = sh_no_bb[0]

# expanding vectors:
sh_school = expand_partial_array(
	mapping_dic=ind.region_ga_dict,
	array_to_expand=sh_school,
	size=len(ind.GA),
)
sh_work = expand_partial_array(
	mapping_dic=ind.region_ga_dict,
	array_to_expand=sh_work,
	size=len(ind.GA),
)
sh_no_100_meters = expand_partial_array(
	mapping_dic=ind.region_ga_dict,
	array_to_expand=sh_no_100_meters,
	size=len(ind.GA),
)
sh_no_bb = expand_partial_array(
	mapping_dic=ind.region_ga_dict,
	array_to_expand=sh_no_bb,
	size=len(ind.GA),
)

for market_pct in pct:
	inter_name = inter2name(
		ind,
		market_pct,
		no_risk,
		no_kid,
		kid_019,
		kid_09,
		kid_04,
	)

	### setting C_inter:
	C_calibration = {}
	d_tot = 1200
	home_inter = []
	work_inter = []
	leis_inter = []
	home_no_inter = []
	work_no_inter = []
	leis_no_inter = []

	### setting stay_home:
	sh_t_work_inter = []
	sh_t_not_work_inter = []
	routine_vector_inter = []
	sh_t_non_inter = []
	routine_vector_non_inter = []

	# make baseline for intervention:
	if market_pct == 10:
		work = full_mtx_work['no_100_meters']
		leisure = full_mtx_leisure['no_100_meters']

		sh_work_inter_spec = sh_no_100_meters.copy()
		sh_not_work_inter_spec = sh_no_100_meters.copy()

	elif market_pct == 30:
		work = full_mtx_work['no_work']
		leisure = full_mtx_leisure['no_work']

		sh_work_inter_spec = sh_work.copy()
		sh_not_work_inter_spec = sh_work.copy()

	elif market_pct > 30:
		factor = (market_pct-30.0)/10.0
		work = full_mtx_work['no_work'] + \
			   (full_mtx_work['routine'] - full_mtx_work['no_work']) * factor/7.0
		leisure = full_mtx_leisure['no_work'] + \
				  (full_mtx_leisure['routine'] - full_mtx_leisure['no_work'])\
				  * factor/7.0

		sh_work_inter_spec = sh_work.copy() + \
							(np.ones_like(sh_work) - sh_work.copy()) \
							* factor / 7.0
		sh_not_work_inter_spec = sh_work.copy() + \
									(np.ones_like(sh_work) - sh_work.copy()) \
									* factor / 7.0

	else:
		print('market_pct value is not define!')
		sys.exit()

	# make inter for base
	if market_pct != 10:
		for group, idx in ind.age_ga_dict.items():
			if no_kid:
				if group in ['0-4', '5-9', '10-19']:
					work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
					work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

					sh_work_inter_spec[idx] = sh_no_100_meters[idx]
			elif kid_019:
				if group in ['0-4', '5-9', '10-19']:
					work[idx, :] = full_mtx_work['routine'][idx, :]
					work[:, idx] = full_mtx_work['routine'][:, idx]

					sh_work_inter_spec[idx] = 1
			elif kid_09:
				if group in ['0-4', '5-9']:
					work[idx, :] = full_mtx_work['routine'][idx, :]
					work[:, idx] = full_mtx_work['routine'][:, idx]

					sh_work_inter_spec[idx] = 1
				if group in ['10-19']:
					work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
					work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

					sh_work_inter_spec[idx] = sh_no_100_meters[idx]
			elif kid_04:
				if group in ['0-4']:
					work[idx, :] = full_mtx_work['routine'][idx, :]
					work[:, idx] = full_mtx_work['routine'][:, idx]

					sh_work_inter_spec[idx] = 1
				if group in ['5-9', '10-19']:
					work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
					work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

					sh_work_inter_spec[idx] = sh_no_100_meters[idx]
			if no_risk:
				if group in ['70+', '60-69']:
					work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
					work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

					leisure[idx, :] = full_mtx_leisure['no_100_meters'][idx, :]
					leisure[:, idx] = full_mtx_leisure['no_100_meters'][:, idx]

					sh_work_inter_spec[idx] = sh_no_100_meters[idx]
					sh_not_work_inter_spec[idx] = sh_no_100_meters[idx]

	for i in range(d_tot):
		home_inter.append(full_mtx_home)
		work_inter.append(work)
		leis_inter.append(leisure)

		sh_t_work_inter.append(sh_work_inter_spec)
		sh_t_not_work_inter.append(sh_not_work_inter_spec)
		routine_vector_inter.append(np.ones_like(sh_work))

	work = full_mtx_work['no_100_meters']
	leisure = full_mtx_leisure['no_100_meters']

	sh_t_non_inter_spec = sh_no_100_meters.copy()
	for i in range(d_tot):
		home_no_inter.append(full_mtx_home)
		work_no_inter.append(work)
		leis_no_inter.append(leisure)
		sh_t_non_inter.append(sh_no_100_meters)
		routine_vector_non_inter.append(np.ones_like(sh_work))

	C_calibration['home_inter'] = home_inter
	C_calibration['work_inter'] = work_inter
	C_calibration['leisure_inter'] = leis_inter
	C_calibration['home_non'] = home_no_inter
	C_calibration['work_non'] = work_no_inter
	C_calibration['leisure_non'] = leis_no_inter

	sh_calibration = {
		'non_inter':{
			'work': sh_t_non_inter,
			'not_work': sh_t_non_inter
		},
		'inter': {
			'work': sh_t_work_inter,
			'not_work': sh_t_not_work_inter,
		}
	}
	routine_vector_calibration = {
		'non_inter':{
			'work': [1]*d_tot,
			'not_work': [1]*d_tot,
		},
		'inter': {
			'work': [1]*d_tot,
			'not_work': [1]*d_tot,
		}
	}

	### make transfer to inter:
	transfer_pop = ind.region_risk_age_dict.copy()
	for region, risk, age in transfer_pop.keys():
		transfer_pop[(region, risk, age)] = 1
		if no_risk:
			if (risk == 'High') or (age in ['70+', '60-69']):
				transfer_pop[(region, risk, age)] = 0.0
	#     if (risk=='Low') and (age not in ['70+', '60-69']):
	#         transfer_pop[(region, risk, age)] = 1.0
	#     if (risk=='LOW') and (age in ['0-4']):
	#         transfer_pop[(region, risk, age)] = 1.0
	#     if (risk=='LOW') and (age in ['5-9']):
	#         transfer_pop[(region, risk, age)] = 2.0/5.0

	### Save

	with open('../Data/interventions/C_inter_' + inter_name + '.pickle', 'wb') as handle:
		pickle.dump(C_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('../Data/interventions/stay_home_idx_inter_' + inter_name + '.pickle', 'wb') as handle:
		pickle.dump(sh_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('../Data/interventions/routine_t_inter_' + inter_name + '.pickle', 'wb') as handle:
		pickle.dump(routine_vector_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle', 'wb') as handle:
		pickle.dump(transfer_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print('Done making ', inter_name)