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
from PolicyOptimization import EvaluatePolicy as pol

isr_pop = 9136000

with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

parameters_list = [
#     '70%',
#     '75%',
#     '80%',
#     'ub',
#     'base',
#     'lb',
	(1,'-'),
#     (1,29),
]
policy_params_list = []
# policy_params = {
#     'policy_period': 7,
#     'stop_inter': mdl.inter2name(ind, 10),
#     'free_inter': mdl.inter2name(ind, 100, no_risk=False),
#     'deg_param': None,
#     'global_thresh': False,
#     'max_duration': 2,
#     'threshold': 1000,
# }
policy_params = {
	'policy_period': 7,
	'stop_inter': mdl.inter2name(ind, 10),
	'free_inter': mdl.inter2name(ind, 100, no_risk=False),
	'deg_param': None,
	'global_decision': True,
	'global_thresh': True,
	'max_duration': 2,
	'threshold': 5e2,
}
# [Is, new_Is, H, Vent]
policy_params['weight_matrix'] = np.zeros([4,9])
policy_params['weight_matrix'][1][5:] = 0.25



policy_params_list.append(policy_params)

start_inter = pd.Timestamp('2020-05-08')
beginning = pd.Timestamp('2020-02-20')

cal_parameters = pd.read_pickle('../Data/calibration/calibrattion_dict.pickle')
cal_parameters = {key : cal_parameters[ind.cell_name][key] for key in parameters_list}
res_mdl_glob = []
pol_states = []
for scen_idx, phase in cal_parameters.keys():
	if phase == '-':
		seasonality = False
		phi=0
	else:
		seasonality = True
		phi=phase
	model = mdl.Model_behave(
		ind=ind,
		beta_j=cal_parameters[(scen_idx, phase)]['beta_j'],
		theta=cal_parameters[(scen_idx, phase)]['theta'],
		beta_behave=cal_parameters[(scen_idx, phase)]['beta_behave'],
		mu=cal_parameters[(scen_idx, phase)]['mu'],
		nu=cal_parameters[(scen_idx, phase)]['nu'],
		eta=cal_parameters[(scen_idx, phase)]['eta'],
		xi=cal_parameters[(scen_idx, phase)]['xi'],
		scen=mdl.num2scen(scen_idx),
		seasonality=seasonality,
		phi=phi,
	)

	res = model.predict(
		C=mdl.C_calibration,
		days_in_season=(start_inter-beginning).days,
		stay_home_idx=mdl.stay_home_idx,
		not_routine=mdl.not_routine,
	)
	for policy_params in policy_params_list:
		res_mdl_glob_i, pol_states_i = pol.run_global_policy(
			ind,
			model,
			policy_params,
			200 - (start_inter-beginning).days,
			mdl.pop_israel,
			start=(start_inter-beginning).days,
		)
		res_mdl_glob.append(res_mdl_glob_i)
		pol_states.append(pol_states_i)

