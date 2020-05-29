import numpy as np
import pandas as pd
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
from PolicyOptimization import EvaluatePolicy as pol
import datetime as dt
from scipy.stats import poisson
from scipy.stats import binom
import copy
import os
import time
from operator import add

#################################
# Generating Lockdowns Results  #
#################################
## Must be run after cell parameters set to specific cell division.
## Must be run after interventions generator.

# load indices
with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

parameters_list = [
#     '70%',
#     '75%',
#     '80%',
#     'ub',
#     'base',
#     'lb',
#     (3,'-'),
# 	(1,'-'),
# 	(1,29),
# 	(2,'-'),
# 	(2,29),
# 	(3,60),
# 	(3,90),
# 	(1, 60),
# 	(2, 60),
# 	(3, 60),
# 	(1, -1),
# 	(2, -1),
# 	(3, -1),
	(4, 60),
]

stops_type = [
	(True, 'all'),
	# (True, 'risk'),
	# (True, 'kid'),
	(False, 'all'),
	(False, 'risk'),
    (False, 'kid'),
	# (False, 'kid_risk'),
	# (True, 'kid_risk'),
]

threshs = [
	0.1,
	# 0.175,
	# 0.25,
	0.5,
	# 1,
	# 2,
	# 3,
	# 4,
	# 5,
	1000,
]

start_inter = pd.Timestamp('2020-05-08')
beginning = pd.Timestamp('2020-02-20')

policy_params_general = {
	'policy_period': 7,
	'stop_inter': mdl.inter2name(ind, 10),
	'free_inter': mdl.inter2name(ind, 100, no_risk=False),
	'deg_param': None,
	'global_thresh': False,
	'max_duration': 100,
	'threshold': 0.5,
}
policy_params_general['weight_matrix'] = np.zeros([4,9])
policy_params_general['weight_matrix'][1][:] = 1

start_inter = pd.Timestamp('2020-05-08')
beginning = pd.Timestamp('2020-02-20')

cal_parameters = pd.read_pickle('../Data/calibration/calibration_dict.pickle')
cal_parameters = {key: cal_parameters[ind.cell_name][key] for key in
				  parameters_list}
res_mdl = {}
pol_states = {}
for scen_idx, phase in cal_parameters.keys():
	if phase == '-':
		seasonality = False
		phi = 0
	else:
		seasonality = True
		phi = phase
	model = mdl.Model_behave(
		ind=ind,
		beta_j=cal_parameters[(scen_idx, phase)]['beta_j'],
		theta=cal_parameters[(scen_idx, phase)]['theta'],
		beta_behave=cal_parameters[(scen_idx, phase)]['beta_behave'],
		# mu=cal_parameters[(scen_idx, phase)]['mu'],
		# nu=cal_parameters[(scen_idx, phase)]['nu'],
		# eta=cal_parameters[(scen_idx, phase)]['eta'],
		# xi=cal_parameters[(scen_idx, phase)]['xi'],
		scen=mdl.num2scen(scen_idx),
		seasonality=seasonality,
		phi=phi,
	)

	res = model.predict(
		C=mdl.C_calibration,
		days_in_season=(start_inter - beginning).days,
		stay_home_idx=mdl.stay_home_idx,
		not_routine=mdl.not_routine,
	)
	for glob, stop in stops_type:
		res_mdl[(scen_idx, phase, stop)] = []
		pol_states[(scen_idx, phase, stop)] = []
		for thresh in threshs:
			policy_params = copy.deepcopy(policy_params_general)
			policy_params['threshold'] = thresh
			policy_params['global_thresh'] = glob
			policy_params['global_lock'] = glob
			if stop == 'all':
				policy_params['stop_inter'] = mdl.inter2name(ind, 10)
			elif stop == 'kid':
				policy_params['stop_inter'] = mdl.inter2name(ind, 100,
															 no_risk=False,
															 no_kid=True)
			elif stop == 'risk':
				policy_params['stop_inter'] = mdl.inter2name(ind, 100,
															 no_risk=True)
				policy_params['free_inter'] = mdl.inter2name(ind, 100,
															 no_risk=False)
			elif stop == 'nothing':
				policy_params['stop_inter'] = mdl.inter2name(ind, 100,
															 no_risk=False)
				policy_params['free_inter'] = mdl.inter2name(ind, 100,
															 no_risk=False)

			print('Doing: ', ' '.join(
				[str(x) for x in [scen_idx, phase, stop, glob, thresh]]))
			res_mdl_glob_i, pol_states_i = pol.run_global_policy(
				ind,
				model,
				policy_params,
				3 * 365 + 1,
				mdl.pop_israel,
				start=(start_inter - beginning).days,
			)
			with open('../Data/results/quarantine_mod_res' +
					  '_'.join([str(x) for x in
								[scen_idx, phase, stop, glob, thresh]]) +
					  '.pickle', 'wb') as handle:
				pickle.dump(res_mdl_glob_i, handle,
							protocol=pickle.HIGHEST_PROTOCOL)
			with open('../Data/results/quarantine_regions' +
					  '_'.join([str(x) for x in
								[scen_idx, phase, stop, glob, thresh]]) +
					  '.pickle', 'wb') as handle:
				pickle.dump(pol_states_i, handle,
							protocol=pickle.HIGHEST_PROTOCOL)
#             res_mdl[(scen_idx, phase, stop, glob, thresh)] = res_mdl_glob_i.copy()
#             pol_states[(scen_idx, phase, stop, glob, thresh)] = pol_states_i.copy()