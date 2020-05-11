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

last = 100
final = mdl.inter2name(ind, last, no_risk=False)

pct_ramp = range(30, last+5, 5)
single_pct_list = [100] # the single to compare ramp to.
kid_comb = [
#     [
#         {
#             'no_kid': False,
#             'kid_019': True,
#         },
#     ],
    [
        {
            'no_kid': False,
            'kid_019': False,
            'kid_09': False,
            'kid_04': True,
        },
        {
            'no_kid': False,
            'kid_019': False,
            'kid_09': True,
        },
        {
            'no_kid': False,
            'kid_019': True,
        },
    ],
]

time2last_inter = 150 # dayes

deg_param = {
    'inter_max':mdl.inter2name(ind, last, no_risk=False),
#     'deg_rate': last/float(time2last_inter),
    'deg_rate': 20,
    'max_deg_rate': last,
}
# deg_param = None

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

inter_lists = []
for pct, kid_list in itertools.product(single_pct_list, kid_comb):
	inter_lists.append([
						   mdl.inter2name(ind, pct, no_risk=True, **kid_dict)
						   for kid_dict in kid_list
					   ] + [final])

# for i in range(len(inter_lists)):
#     inter_lists[i][-1] = final

inter_times_lists = []
for pct, kid_list in itertools.product(single_pct_list, kid_comb):
	base_time = int(time2last_inter / (float(len(kid_list))))
	time_list = []
	for i, kid_dict in enumerate(kid_list):
		if i == len(kid_list) - 1:
			cur_time = time2last_inter - base_time * (len(kid_list) - 1)
		else:
			cur_time = base_time
		time_list.append(cur_time)
	inter_times_lists.append(time_list)

scen = 'Scenario2'
start_inter = pd.Timestamp('2020-05-03')
beginning = pd.Timestamp('2020-02-20')

cal_parameters = pd.read_pickle('../Data/calibration/calibrattion_dict.pickle')
cal_parameters = {key: cal_parameters[ind.cell_name][key] for key in
				  parameters_list}

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

	_, model = mdl.multi_inter_by_name(
		ind,
		model,
		mdl.pop_israel,
		[mdl.inter2name(ind, 30, no_risk=False, no_kid=False, )],
		sim_length=7,
		fix_vents=False,
	)

	for i in range(len(inter_times_lists)):
		res_mdl, _ = mdl.multi_inter_by_name(
			ind,
			model,
			mdl.pop_israel,
			inter_lists[i],
			inter_times_lists[i],
			sim_length=1200,
			fix_vents=False,
			deg_param=deg_param,
			no_pop=True,
		)
		results[(key, i)] = res_mdl
	#     # fix 60 offset
	#     for i, vent in enumerate(res_mdl['Vents']):
	#         res_mdl['Vents'][i] = vent + ((60.0/mdl.pop_israel)*vent)/vent.sum()

	print(key, ' parameters, we got:')