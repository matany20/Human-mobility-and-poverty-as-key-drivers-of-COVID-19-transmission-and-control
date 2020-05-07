import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
from scipy import optimize
import sys
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

with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

##################
# prepare preset##
##################
pct_ramp = [30,100]#range(30, 105, 5)
single_pct = 75 # the single to compare ramp to.

deg_param = {
	'inter_max':mdl.inter2name(ind, 100, no_risk=False),
	'deg_rate': 0.02,
	'max_deg_rate': 50,
}
# deg_param = None

# interventions to examine:
inter_list = [mdl.inter2name(ind, x, no_risk=True) for x in pct_ramp]
inter_list[-1] = mdl.inter2name(ind, pct_ramp[-1], no_risk=True)

time2last_inter = 90 # dayes

# parameters to examine:
parameters_list = [
#     '70%',
#     '75%',
#     '80%',
#     'ub',
	'base',
#     'lb',
]

results = {}

scen = 'Scenario2'
start_inter = pd.Timestamp('2020-05-03')
beginning = pd.Timestamp('2020-02-20')

cal_parameters = pd.read_pickle('../Data/calibration/calibrattion_dict.pickle')
cal_parameters = {key: cal_parameters[key] for key in parameters_list}

for key in cal_parameters.keys():
	model = mdl.Model_behave(
		ind=ind,
		beta_j=cal_parameters[key]['beta_j'],
		theta=cal_parameters[key]['theta'],
		beta_behave=cal_parameters[key]['beta_behave'],
		eps=mdl.eps_sector[scen],
		f=mdl.f0_full[scen],
	)

	model.predict(
		C=mdl.C_calibration,
		days_in_season=(start_inter - beginning).days,
		stay_home_idx=mdl.stay_home_idx,
		not_routine=mdl.not_routine,
	)
	delta_t = int(time2last_inter / (float(len(inter_list) - 1)))
	inter_times = [delta_t] * (len(inter_list) - 1)
	res_rmp = mdl.multi_inter_by_name(
		ind,
		model,
		mdl.pop_israel,
		inter_list,
		inter_times,
		sim_length=500,
		fix_vents=False,
		deg_param=deg_param,
	)

	inter_list = [
		mdl.inter2name(ind, single_pct, no_risk=True),
		inter_list[-1],
	]
	res_single = mdl.multi_inter_by_name(
		ind,
		model,
		mdl.pop_israel,
		inter_list,
		[time2last_inter],
		sim_length=500,
		fix_vents=False,
		deg_param=deg_param,
	)
	#     # fix 60 offset
	#     for i, vent in enumerate(res_mdl['Vents']):
	#         res_mdl['Vents'][i] = vent + ((60.0/mdl.pop_israel)*vent)/vent.sum()

	print(key, ' parameters, we got:')

