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

""" Eras explanation:
-first days of routine from Feb 21st - March 13th
-first days of no school from March 14th - March 16th
-without school and work from March 17th - March 25th
-100 meters constrain from March 26th - April 2nd
-Bnei Brak quaranrine from April 3rd
"""


### Setting parameters:
# market_pct of 10 means lockdown
market_pct = 10
no_risk = False
no_school = True
no_kid10 = False

### creating notations for intervention.
with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

inter_name = ind.cell_name + '@' + str(market_pct)
if  market_pct != 10:
	if no_risk:
		inter_name += '_no_risk60'
	if no_school:
		inter_name += '_no_school'
	else:
		inter_name += '_school'
	if not no_kid10:
		inter_name += '_kid010'
	else:
		inter_name += '_no_kid010'

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
stay_home_idx_school =  pd.read_csv('../Data/stay_home/no_school.csv',index_col=0)
stay_home_idx_school.index = stay_home_idx_school.index.astype(str)
stay_home_idx_work =  pd.read_csv('../Data/stay_home/no_work.csv',index_col=0)
stay_home_idx_work.index = stay_home_idx_work.index.astype(str)
stay_home_idx_routine =  pd.read_csv('../Data/stay_home/routine.csv',index_col=0)
stay_home_idx_routine.index = stay_home_idx_routine.index.astype(str)
stay_home_idx_no_100_meters =  pd.read_csv('../Data/stay_home/no_100_meters.csv',index_col=0)
stay_home_idx_no_100_meters.index = stay_home_idx_no_100_meters.index.astype(str)
stay_home_idx_no_bb =  pd.read_csv('../Data/stay_home/no_bb.csv',index_col=0)
stay_home_idx_no_bb.index = stay_home_idx_no_bb.index.astype(str)
# reordering and expanding vector for each period:
stay_home_idx_school = stay_home_idx_school['mean'].values
stay_home_idx_school[1] = stay_home_idx_school[0]

stay_home_idx_work = stay_home_idx_work['mean'].values
stay_home_idx_work[1] = stay_home_idx_work[0]

stay_home_idx_no_100_meters = stay_home_idx_no_100_meters['mean'].values
stay_home_idx_no_100_meters[1] = stay_home_idx_no_100_meters[0]

stay_home_idx_no_bb = stay_home_idx_no_bb['mean'].values
stay_home_idx_no_bb[1] = stay_home_idx_no_bb[0]
# expanding vectors:
stay_home_idx_school = expand_partial_array(mapping_dic=ind.region_ga_dict,array_to_expand=stay_home_idx_school,
                                              size=len(ind.GA))
stay_home_idx_work = expand_partial_array(mapping_dic=ind.region_ga_dict,array_to_expand=stay_home_idx_work,
                                              size=len(ind.GA))
stay_home_idx_no_100_meters = expand_partial_array(mapping_dic=ind.region_ga_dict,array_to_expand=stay_home_idx_no_100_meters,
                                              size=len(ind.GA))
stay_home_idx_no_bb = expand_partial_array(mapping_dic=ind.region_ga_dict,array_to_expand=stay_home_idx_no_bb,
                                              size=len(ind.GA))

### setting C_inter:
C_calibration = {}
d_tot = 500
home_inter = []
work_inter = []
leis_inter = []
home_no_inter = []
work_no_inter = []
leis_no_inter = []

### setting stay_home:
stay_idx_t_work_inter = []
stay_idx_t_not_work_inter = []
routine_vector_inter = []
stay_idx_t_non_inter = []
routine_vector_non_inter = []

# make baseline for intervention:
if market_pct == 10:
	work = full_mtx_work['no_100_meters']
	leisure = full_mtx_leisure['no_100_meters']

	stay_home_idx_work_inter_spec = stay_home_idx_no_100_meters.copy()
	stay_home_idx_not_work_inter_spec = \
		stay_home_idx_no_100_meters.copy()

elif market_pct == 30:
	work = full_mtx_work['no_work']
	leisure = full_mtx_leisure['no_work']

	stay_home_idx_work_inter_spec = stay_home_idx_work.copy()
	stay_home_idx_not_work_inter_spec = stay_home_idx_work.copy()

elif market_pct > 30:
	factor = (market_pct-30.0)/10.0
	work = full_mtx_work['no_work'] + \
		   (full_mtx_work['routine'] - full_mtx_work['no_work']) * factor/7.0
	leisure = full_mtx_leisure['no_work'] + \
			  (full_mtx_leisure['routine'] - full_mtx_leisure['no_work']) * factor/7.0

	stay_home_idx_work_inter_spec = stay_home_idx_work.copy() + \
									(np.ones_like(
										stay_home_idx_work) - stay_home_idx_work.copy()) \
									* factor / 7.0
	stay_home_idx_not_work_inter_spec = stay_home_idx_work.copy() + \
									(np.ones_like(
										stay_home_idx_work) - stay_home_idx_work.copy()) \
									* factor / 7.0

else:
	print('market_pct value is not define!')
	sys.exit()

# make inter for base
if market_pct != 10:
	for group, idx in ind.age_ga_dict.items():
		if group in ['0-4', '5-9']:
			if no_kid10:
				work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
				work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

				stay_home_idx_work_inter_spec[idx] = \
					stay_home_idx_no_100_meters[idx]
			else:
				work[idx, :] = full_mtx_work['routine'][idx, :]
				work[:, idx] = full_mtx_work['routine'][:, idx]

				stay_home_idx_work_inter_spec[idx] = 1
		if group in ['10-19']:
			if no_school:
				work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
				work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

				stay_home_idx_work_inter_spec[idx] = \
					stay_home_idx_no_100_meters[idx]
			else:
				work[idx, :] = full_mtx_work['routine'][idx, :]
				work[:, idx] = full_mtx_work['routine'][:, idx]

				stay_home_idx_work_inter_spec[idx] = 1

		if group in ['70+', '60-69']:
			if no_risk:
				work[idx, :] = full_mtx_work['no_100_meters'][idx, :]
				work[:, idx] = full_mtx_work['no_100_meters'][:, idx]

				leisure[idx, :] = full_mtx_leisure['no_100_meters'][idx, :]
				leisure[:, idx] = full_mtx_leisure['no_100_meters'][:, idx]

for i in range(d_tot):
	home_inter.append(full_mtx_home)
	work_inter.append(work)
	leis_inter.append(leisure)

work = full_mtx_work['no_100_meters']
leisure = full_mtx_leisure['no_100_meters']
for i in range(d_tot):
	home_no_inter.append(full_mtx_home)
	work_no_inter.append(work)
	leis_no_inter.append(leisure)

C_calibration['home_inter'] = home_inter
C_calibration['work_inter'] = work_inter
C_calibration['leisure_inter'] = leis_inter
C_calibration['home_non'] = home_no_inter
C_calibration['work_non'] = work_no_inter
C_calibration['leisure_non'] = leis_no_inter

for i in range(d_tot):
	stay_idx_t_work_inter.append(stay_home_idx_work_inter_spec)
	stay_idx_t_not_work_inter.append(stay_home_idx_not_work_inter_spec)
	routine_vector_inter.append(np.ones_like(stay_home_idx_work))

for i in range(d_tot):
	stay_idx_t_non_inter.append(stay_home_idx_no_100_meters)
	routine_vector_non_inter.append(np.ones_like(stay_home_idx_work))

stay_idx_calibration = {
	'non_inter':{
		'work': stay_idx_t_non_inter,
		'not_work': stay_idx_t_non_inter
	},
	'inter': {
		'work': stay_idx_t_work_inter,
		'not_work': stay_idx_t_not_work_inter,
	}
}
routine_vector_calibration = {
	'non_inter':{
		'work': [1]*500,
		'not_work': [1]*500,
	},
	'inter': {
		'work': [1]*500,
		'not_work': [1]*500,
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
	pickle.dump(stay_idx_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Data/interventions/routine_t_inter_' + inter_name + '.pickle', 'wb') as handle:
	pickle.dump(routine_vector_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle', 'wb') as handle:
	pickle.dump(transfer_pop, handle, protocol=pickle.HIGHEST_PROTOCOL)