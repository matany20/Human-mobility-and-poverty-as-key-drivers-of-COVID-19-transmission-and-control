from .indices import *
from .model import *
from .utils import *
import pandas as pd
import numpy as np
import itertools
import pickle


def errorfunc(tpl, data, f, season_length, eps, alpha, ind):
	"""

	:param tpl:
	:param data:
	:param f:
	:param season_length:
	:param eps:
	:param alpha:
	:return:
	"""

	# setting parameters
	# beta_home = tpl[5]
	beta = np.array([
		tpl[0],
		tpl[0],
		tpl[1],
		tpl[2],
		tpl[2],
		tpl[3],
		tpl[3],
		tpl[4],
		tpl[4]
	])

	# Run model with given parameters
	model_result = run_model(
		beta_home=0.38/9.,
		beta_j=beta,
		eps=eps,
		f=f,
		alpha=alpha,
		days_in_season=season_length
	)

	new_cases_model = model_result['new_Is']
	model_results_cal = np.zeros((season_length + 1, len(ind.A)))
	# Calculated total symptomatic (high+low) per age group (adding as columns)
	for i, age_group in enumerate(ind.age_dict.keys()):
		model_results_cal[:, i] = new_cases_model[
								  	:,
								  ind.age_dict[age_group]
								  ].sum(axis=1)

	return np.log(MSE(data, model_results_cal))


def print_stat_fit(fit_results_object):
	"""
	The function gets optimization results object and print additional info
	about the optimization.
	:param fit_results_object:
	:return:
	"""

	print('minimized value:', fit_results_object.fun)
	print('fitted parameters: Beta={}'.format(fit_results_object.x[:]))
	print('num of sampling the target function:', fit_results_object.nfev)

	return 0
