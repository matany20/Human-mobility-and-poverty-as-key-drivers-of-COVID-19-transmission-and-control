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
from SEIR_full import model_class as mdl

####################################
# Generating model Single Results  #
####################################
## Must be run after cell parameters set to specific cell division.
## Must be run after interventions generator.

today_time_smp = '28_4_20_20_36'

# load indices
with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

# interventions to examine:
inter_list = [
	'20@50_no_school_kid010',
	'20@75_no_school_kid010',
	'20@100_no_school_kid010',
	'20@100',
]

# parameters to examine:
parameters_list = [
	'ub',
	'base',
	'lb',
]
# intresting dates:
dates = [
	pd.Timestamp('2020-04-23'),
	pd.Timestamp('2020-05-01'),
	pd.Timestamp('2020-05-09'),
	pd.Timestamp('2020-05-17'),
	pd.Timestamp('2020-05-25'),
	pd.Timestamp('2020-06-02'),
	pd.Timestamp('2020-06-10'),
	pd.Timestamp('2020-06-18'),
	pd.Timestamp('2020-06-26'),
	pd.Timestamp('2020-07-15'),
	pd.Timestamp('2020-08-01'),
]

### make results:
results = {}
scen = 'Scenario2'
start_inter = pd.Timestamp('2020-05-03')
beginning = pd.Timestamp('2020-02-20')

cal_parameters = pd.read_pickle('../Data/calibration/calibrattion_dict.pickle')
cal_parameters = {key: cal_parameters[key] for key in parameters_list}

for key in cal_parameters.keys():
	time_in_season = 0
	model = mdl.Model_behave(
		ind=ind,
		beta_j=cal_parameters[key]['beta_j'],
		theta=cal_parameters[key]['theta'],
		beta_behave=cal_parameters[key]['beta_behave'],
		eps=mdl.eps_sector[scen],
		f=mdl.f0_full[scen],
	)

	res_mdl = model.predict(
		C=mdl.C_calibration,
		days_in_season=(start_inter - beginning).days,
		stay_home_idx=mdl.stay_home_idx,
		not_routine=mdl.not_routine,
		start=time_in_season,
	)
	time_in_season += (start_inter - beginning).days

	for inter_name in inter_list:
		# First intervention
		with open('../Data/interventions/C_inter_' + inter_name + '.pickle',
				  'rb') as pickle_in:
			C_inter = pickle.load(pickle_in)

		with open(
				'../Data/interventions/stay_home_idx_inter_' + inter_name + '.pickle',
				'rb') as pickle_in:
			stay_home_idx_inter = pickle.load(pickle_in)

		with open(
				'../Data/interventions/routine_t_inter_' + inter_name + '.pickle',
				'rb') as pickle_in:
			routine_t_inter = pickle.load(pickle_in)

		with open(
				'../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle',
				'rb') as pickle_in:
			transfer_pop_inter = pickle.load(pickle_in)

		model_inter = copy.deepcopy(model)
		res_mdl = model_inter.intervention(
			C=C_inter,
			days_in_season=300,
			#             days_in_season=(dates[-1]-start_inter).days,
			#             days_in_season=inter2_timing - (start_inter-beginning).days,
			stay_home_idx=stay_home_idx_inter,
			not_routine=routine_t_inter,
			prop_dict=transfer_pop_inter,
			start=time_in_season,
		)
		time_in_season += 300
		for i, vent in enumerate(res_mdl['Vents']):
			res_mdl['Vents'][i] = vent + (
						(60.0 / mdl.pop_israel) * vent) / vent.sum()

		print(key, ' parameters, intervention: ', inter_name, ' we got:')
		for date in dates:
			print(date, 'Respiratoy cases: ', (((res_mdl['Vents'])[(
						date - beginning).days].sum()) * mdl.pop_israel))

		results[(inter_name, key)] = res_mdl

try:
    os.mkdir('../Data/results')
except:
    pass
with open('../Data/results/' + today_time_smp + '.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

