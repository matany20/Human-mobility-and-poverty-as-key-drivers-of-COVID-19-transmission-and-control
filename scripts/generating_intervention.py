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

pct = [10, 100]# range(30, 105, 5)
no_risk = False
no_kid = True
kid_019 = True
kid_09 = False
kid_04 = False


""" Eras explanation:
-first days of routine from Feb 21st - March 13th
-first days of no school from March 14th - March 16th
-without school and work from March 17th - March 25th
-100 meters constrain from March 26th - April 2nd
-Bnei Brak quaranrine from April 3rd - April 6th
-full_lockdown from April 7th - April 16th
-release from April 17 - May 4th
-back2routine from May 5th - May 11th
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

sh_work = sh_work['mean'].values

sh_no_100_meters = sh_no_100_meters['mean'].values

sh_no_bb = sh_no_bb['mean'].values

# expanding vectors:
sh_school = expand_partial_array(
	mapping_dic=ind.region_gra_dict,
	array_to_expand=sh_school,
	size=len(ind.GRA),
)
sh_work = expand_partial_array(
	mapping_dic=ind.region_gra_dict,
	array_to_expand=sh_work,
	size=len(ind.GRA),
)
sh_no_100_meters = expand_partial_array(
	mapping_dic=ind.region_gra_dict,
	array_to_expand=sh_no_100_meters,
	size=len(ind.GRA),
)
sh_no_bb = expand_partial_array(
	mapping_dic=ind.region_gra_dict,
	array_to_expand=sh_no_bb,
	size=len(ind.GRA),
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
	sh_t_inter = []
	routine_vector_inter = []
	sh_t_non_inter = []
	routine_vector_non_inter = []

	# make baseline for intervention:
	if market_pct == 10:
		work = full_mtx_work['no_100_meters']
		leisure = full_mtx_leisure['no_100_meters']

		sh_inter_spec = sh_no_100_meters.copy()

	elif market_pct == 30:
		work = full_mtx_work['no_work']
		leisure = full_mtx_leisure['no_work']

		sh_inter_spec = sh_work.copy()

	elif market_pct > 30:
		factor = (market_pct-30.0)/10.0
		work = full_mtx_work['no_work'] + \
			   (full_mtx_work['routine'] - full_mtx_work['no_work']) * factor/7.0
		leisure = full_mtx_leisure['no_work'] + \
				  (full_mtx_leisure['routine'] - full_mtx_leisure['no_work'])\
				  * factor/7.0

		sh_inter_spec = sh_work.copy() + \
							(np.ones_like(sh_work) - sh_work.copy()) \
							* factor / 7.0

	else:
		print('market_pct value is not define!')
		sys.exit()

	# make inter for base
	if market_pct != 10:
		if no_kid:
			idx = list(ind.age_ga_dict['0-4']) + \
				list(ind.age_ga_dict['5-9']) + \
				list(ind.age_ga_dict['10-19'])
			work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
			work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

			idx = list(ind.age_gra_dict['0-4']) + \
				  list(ind.age_gra_dict['5-9']) + \
				  list(ind.age_gra_dict['10-19'])
			sh_inter_spec[idx] = sh_no_100_meters[idx]

		elif kid_019:
			idx = list(ind.age_ga_dict['0-4']) + \
				  list(ind.age_ga_dict['5-9']) + \
				  list(ind.age_ga_dict['10-19'])
			work[idx, :] = full_mtx_work['routine'][idx, :]
			work[:, idx] = full_mtx_work['routine'][:, idx]

			idx = list(ind.age_gra_dict['0-4']) + \
				  list(ind.age_gra_dict['5-9']) + \
				  list(ind.age_gra_dict['10-19'])
			sh_inter_spec[idx] = 1
		elif kid_09:
			idx09 = list(ind.age_ga_dict['0-4']) + \
				  list(ind.age_ga_dict['5-9'])
			work[idx09, :] = full_mtx_work['routine'][idx09, :]
			work[:, idx09] = full_mtx_work['routine'][:, idx09]

			idx09 = list(ind.age_gra_dict['0-4']) + \
					list(ind.age_gra_dict['5-9'])
			sh_inter_spec[idx09] = 1

			idx10_19 = list(ind.age_ga_dict['10-19'])
			work[idx10_19, :] = full_mtx_work['no_100_meters'][idx10_19, :]
			work[:, idx10_19] = full_mtx_work['no_100_meters'][:, idx10_19]

			idx10_19 = list(ind.age_gra_dict['10-19'])
			sh_inter_spec[idx10_19] = sh_no_100_meters[idx10_19]
		elif kid_04:
			idx04 = list(ind.age_ga_dict['0-4'])
			work[idx04, :] = full_mtx_work['routine'][idx04, :]
			work[:, idx04] = full_mtx_work['routine'][:, idx04]

			idx04 = list(ind.age_gra_dict['0-4'])
			sh_inter_spec[idx04] = 1

			idx5_19 = list(ind.age_ga_dict['5-9']) + \
				list(ind.age_ga_dict['10-19'])
			work[idx5_19, :] = full_mtx_work['no_100_meters'][idx5_19, :]
			work[:, idx5_19] = full_mtx_work['no_100_meters'][:, idx5_19]

			idx5_19 = list(ind.age_gra_dict['5-9']) + \
					  list(ind.age_gra_dict['10-19'])
			sh_inter_spec[idx5_19] = sh_no_100_meters[idx5_19]

	for i in range(d_tot):
		home_inter.append(full_mtx_home)
		work_inter.append(work)
		leis_inter.append(leisure)

		sh_t_inter.append(sh_inter_spec)
		routine_vector_inter.append(np.ones_like(sh_work))

	if no_risk:
		work = full_mtx_work['no_100_meters']
		leisure = full_mtx_leisure['no_100_meters']

		sh_t_non_inter_spec = sh_no_100_meters.copy()
	else:
		sh_t_non_inter_spec = sh_inter_spec

	for i in range(d_tot):
		home_no_inter.append(full_mtx_home)
		work_no_inter.append(work)
		leis_no_inter.append(leisure)
		sh_t_non_inter.append(sh_t_non_inter_spec)
		routine_vector_non_inter.append(np.ones_like(sh_work))

	C_calibration['home_inter'] = home_inter
	C_calibration['work_inter'] = work_inter
	C_calibration['leisure_inter'] = leis_inter
	C_calibration['home_non'] = home_no_inter
	C_calibration['work_non'] = work_no_inter
	C_calibration['leisure_non'] = leis_no_inter

	sh_calibration = {
		'Non-intervention': sh_t_non_inter,
		'Intervention': sh_t_inter,
	}
	routine_vector_calibration = {
		'Non-intervention':{
			'work': [1]*d_tot,
			'not_work': [1]*d_tot,
		},
		'Intervention': {
			'work': [1]*d_tot,
			'not_work': [1]*d_tot,
		}
	}

	### make transfer to inter:
	transfer_pop = ind.region_risk_age_dict.copy()
	for region, risk, age in transfer_pop.keys():
		transfer_pop[(region, risk, age)] = 1
		if (risk == 'High'):
			transfer_pop[(region, risk, age)] = 0.0
		if age in ['70+']:
			transfer_pop[(region, risk, age)] = 0.0
		if (risk == 'Low')and (age in ['60-69']):
			transfer_pop[(region, risk, age)] = 0.5
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