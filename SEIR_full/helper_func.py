import numpy as np
from .indices import *

def MSE(
		data,
		model_data_processed
	):
	"""

	:param data:
	:param model_data_processed:
	:return:
	"""
	mse = (((data - model_data_processed[:data.shape[0],:].copy()) ** 2).mean(axis=0)).sum()
	return mse

def shrink_array_sum(
		mapping_dic,
		array_to_shrink
	):
	"""
    The function gets mapping_dic - indices to assign in the shrunk array
	(with key granularity based on array_to_expand), and array_to_shrink -
	the shrunk array's values will be based on this array. Returns and shrunk
	array shape (len(mapping_dic),1) based on the array_to_shrink.
	:param mapping_dic:
	:param array_to_shrink:
	:return:
	"""

	# Assigning values to the shrinked array
	shrunk_array = np.zeros(len(mapping_dic))
	for i, k in enumerate(mapping_dic.keys()):
		shrunk_array[i] = array_to_shrink[mapping_dic[k]].sum()

	return shrunk_array


def calculate_force_matriceis(
		C,
		Ie,
		Is,
		Ia,
		t,
		alpha,
		beta_behave=None,
		stay_home_idx=None,
		not_routine=None

	):
	"""
	Calculates the by-product of all C-correlation tenzors with the sick
	population for each time - home, work, leisure, fitting each j (age group)
	and k (area) indexes.
	:param C: C-correlation tenzors
	:param Ie:
	:param Is:
	:param Ia:
	:param t:
	:param alpha:
	:param beta_behave:
	:param stay_home_idx:
	:param not_routine:
	:return:
	"""
	# Calculating beta_behave components:
	if not((beta_behave is None) and (stay_home_idx is None) and (not_routine is None)):
		behave_componnet_inter_no_work = (beta_behave * stay_home_idx['inter']['not_work'][t]) ** \
										 not_routine['inter']['not_work'][t]

		behave_componnet_non_no_work = (beta_behave * stay_home_idx['non_inter']['not_work'][t]) ** \
										 not_routine['non_inter']['not_work'][t]

		behave_componnet_inter_work = (beta_behave * stay_home_idx['inter']['work'][t]) ** \
									   not_routine['inter']['work'][t]

		behave_componnet_non_work = (beta_behave * stay_home_idx['non_inter']['work'][t]) ** \
									   not_routine['non_inter']['work'][t]

	force_home = (behave_componnet_inter_no_work *\
				  C['home_inter'][t].T.dot((Ie[inter_risk_dict['Intervention', 'Low']] + Ie[inter_risk_dict['Intervention', 'High']]) * alpha +
											Is[inter_risk_dict['Intervention', 'Low']] + Is[inter_risk_dict['Intervention', 'High']] +
											Ia[inter_risk_dict['Intervention', 'Low']] + Ia[inter_risk_dict['Intervention', 'High']]) +

				   behave_componnet_non_no_work *\
				  C['home_non'][t].T.dot((Ie[inter_risk_dict['Non-intervention', 'Low']] + Ie[inter_risk_dict['Non-intervention', 'High']]) * alpha +
										 Is[inter_risk_dict['Non-intervention', 'Low']] +Is[inter_risk_dict[ 'Non-intervention', 'High']] +
										 Ia[inter_risk_dict['Non-intervention', 'Low']] + Ia[inter_risk_dict['Non-intervention', 'High']]))

	force_out = (behave_componnet_inter_work * \
				  C['work_inter'][t].T.dot((Ie[inter_risk_dict['Intervention', 'Low']] + Ie[inter_risk_dict['Intervention', 'High']]) * alpha +
										   Is[inter_risk_dict['Intervention', 'Low']] + Is[inter_risk_dict['Intervention', 'High']] +
										   Ia[inter_risk_dict['Intervention', 'Low']] + Ia[inter_risk_dict['Intervention', 'High']]) +

				  behave_componnet_non_work *\
				  C['work_non'][t].T.dot((Ie[inter_risk_dict['Non-intervention', 'Low']] + Ie[inter_risk_dict['Non-intervention', 'High']]) * alpha +
										 Is[inter_risk_dict['Non-intervention', 'Low']] + Is[inter_risk_dict['Non-intervention', 'High']] +
										 Ia[inter_risk_dict['Non-intervention', 'Low']] + Ia[inter_risk_dict['Non-intervention', 'High']]) +

				 behave_componnet_inter_no_work *\
				 C['leisure_inter'][t].T.dot((Ie[inter_risk_dict['Intervention', 'Low']] + Ie[inter_risk_dict['Intervention', 'High']]) * alpha +
											  Is[inter_risk_dict['Intervention', 'Low']] + Is[inter_risk_dict['Intervention', 'High']] +
											  Ia[inter_risk_dict['Intervention', 'Low']] + Ia[inter_risk_dict['Intervention', 'High']]) +

				 behave_componnet_non_no_work*\
				 C['leisure_non'][t].T.dot((Ie[inter_risk_dict['Non-intervention', 'Low']] + Ie[inter_risk_dict['Non-intervention', 'High']]) * alpha +
											Is[inter_risk_dict['Non-intervention', 'Low']] + Is[inter_risk_dict['Non-intervention', 'High']] +
											Ia[inter_risk_dict['Non-intervention', 'Low']] + Ia[inter_risk_dict['Non-intervention', 'High']]))
	return {
		'out': force_out,
		'home': force_home
	}


def print_stat_fit_behave(fit_results_object):

	"""The function gets sci optimization results object and print additional info about the optimization
	:param fit_results_object:
	:return:
	"""
	print('minimized value:', fit_results_object.fun)
	print('Fitted parameters:\n Beta={0}\n Theta={1},\n Beta_behave={2}'.format(fit_results_object.x[:4],
																				  fit_results_object.x[4],
																				  fit_results_object.x[5]))
	print('num of sampling the target function:', fit_results_object.nfev)
