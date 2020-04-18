import numpy as np
from scipy.sparse import csr_matrix
from .indices import *
from scipy.stats import poisson
from scipy.stats import binom


def ML_bin(
		data,
		model_pred,
		threshold = 5,
		approx=True,
	):
	"""
	This function calculates the log-likelihood of the Bin approximation of the
	measurment of illnes in israel.
	It assumes the number of tests is n_{j,k,t}, the probability for getting a
	result is p_{j,k,t} - the model prediction, and the data point is q_{j,k,t}.
	in total the likelihood P(X=q)~Bin(n,p) per data point.
	For cells (specific t,j,k triplet) of not sufficient number of tests:
	with n_{j,k,t} < threshold the likelihood will be ignored.
	:param data: np.array of 4 dimensions :
					axis 0: n, q - representing different values:
							starting from  total tests, and
							than positives rate.
					axis 1: t - time of sample staring from the first day in
							quastion calibrated to the model.
					axis 2: k - area index
					axis 3: j - age index
			data - should be smoothed.
			(filled with zeros where no test accured)
	:param model_pred: np.ndarray of 3 dimensions representing the probability:
						axis 1: t - time of sample staring from the first day
						 		in quastion calibrated to the model.
						axis 2: k - area index
						axis 3: j - age index
	:return: the -log-likelihood of the data given the prediction of the model.
	"""
	n = data[0, :, :, :]
	q = data[1, :, :, :]
	p = model_pred
	if approx:
		##				###
		# poison approx. ##
		##				###
		ll = -poisson.logpmf(
			k=n * q,
			mu=n * p,
		)
	else:
		##				###
		# Binomial dist. ##
		##				###
		ll = -binom.logpmf(
			k=n * q,
			n=n,
			p=p,
		)

	# cut below threshold values
	ll = ll * (n > threshold)

	return ll.sum()


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


def create_C_mtx_leisure_work(
		od_mat,
		base_mat,
		age_dist_area,
		eye_mat=None,
		stay_home_idx=None
	):
	"""
	This function creates Contact matrix for a specific scenario with or w/o
	stay_home_idx
	:param od_mat:
	:param base_mat:
	:param age_dist_area:
	:param eye_mat:
	:param stay_home_idx:
	:return:
	"""
	full_C = pd.DataFrame(
		index=pd.MultiIndex.from_tuples(list(MI.values()),
										names=['age', 'area', 'age']),
		columns=od_mat.index
	)
	# fill the matrix:
	if stay_home_idx is None:
		factor = 1
	for index in list(full_C.index):
		if factor != 1:
			try:
				factor = stay_home_idx.loc[index[1]]['mean']
			except:
				factor = 1
		tmp1 = age_dist_area[index[2]]
		tmp1.index = list(G.values())
		if (index[0] in ['0-4', '5-9']) and (eye_mat is not None):
			tmp2 = eye_mat.loc[index[1]] * \
				   base_mat.loc[index[0]][index[2]] * factor
		else:
			tmp2 = od_mat.loc[index[1]] * \
				   base_mat.loc[index[0]][index[2]] * factor
		full_C.loc[index] = tmp2.divide(tmp1)

	return csr_matrix(
		full_C.unstack().reorder_levels(['area', 'age']).sort_index().values.astype(float)
	)


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
