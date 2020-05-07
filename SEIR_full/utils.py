import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import poisson
from scipy.stats import binom
import pandas as pd
import pickle
import copy


def get_opposite_dict(dic, keys):
	"""The function gets a dict and new keys list and returns a dictionary in which keys as keys,
	and values are the keys from dic """
	res_dict = {}

	for new_key in keys:
		new_val_lst = []
		for old_k, old_v in dic.items():
			if all(i in old_v for i in new_key):
				new_val_lst.append(old_k)
		if len(new_key) == 1:
			res_dict[new_key[0]] = new_val_lst
		else:
			res_dict[new_key] = new_val_lst
	return res_dict


def expand_partial_array(mapping_dic, array_to_expand, size):
	"""The function gets mapping_dic - indeces to assign in the expanded array (with key granularity based on
	array_to_expand), and array_to_expand - the expanded array's values will be based on this array. Returns
	and expanded array shape (len(N),1) based on the array_to_expand"""

	# Creating dictionary that maps the indices in the array to expand
	#     small_mapping_dic = {k:v[0] for k,v in mapping_dic.items()}

	# Assigning values to the full array
	full_array = np.zeros(size)
	for i, k in enumerate(mapping_dic.keys()):
		full_array[mapping_dic[k]] = array_to_expand[i]

	return full_array


def divide_population(
		ind,
		prop_dict,
		vector_to_switch,
	):
	"""
	The function move population from non-Intervention group to Intervention group
	:param prop_dict: key = (region,risk,age), value=prop to shift
	:param vector_to_switch: the current population distribution
	:param low_only: shift low-risk groups only (boolean)
	:return: vector with new population distribution.
	"""
	new_distribution = vector_to_switch.copy()
	for region, risk, age in prop_dict.keys():
		# indices of intervention group and non-intervention group
		inter_idx = ind.inter_region_risk_age_dict['Intervention',region,risk,age]
		non_inter_idx = ind.inter_region_risk_age_dict['Non-intervention', region, risk, age]

		new_distribution[inter_idx] = vector_to_switch[non_inter_idx] * prop_dict[region,risk,age]
		new_distribution[non_inter_idx] = vector_to_switch[non_inter_idx] * (1 - prop_dict[region, risk, age])


	return new_distribution


def ML_Bin(
		data,
		model_pred,
		threshold = 5,
		approx=True,
		factor=1,
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
	n = data[0, :, :]
	q = factor*data[1, :, :]
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
	ll = np.nan_to_num(ll,
				  nan=0,
				  posinf=0,
				  neginf=0)
	ll = ll * (n > threshold)
	return ll.sum()


def ML_Pois_Naiv(
		data,
		model_pred,
	):
	"""
	This function calculates the log-likelihood of the POISSON of the
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
	factor = 1
	q = data
	p = factor*model_pred

	##				###
	# poison approx. ##
	##				###
	ll = -poisson.logpmf(
		k=q,
		mu=p,
	)

	# cut below threshold values
	ll = np.nan_to_num(ll,
				  nan=0,
				  posinf=0,
				  neginf=0)
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
	mse = (((data - model_data_processed) ** 2).mean(axis=0)).sum()
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
		ind,
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
		index=pd.MultiIndex.from_tuples(list(ind.MI.values()),
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
		tmp1.loc[tmp1 == 0] = 1
		tmp1.index = list(ind.G.values())
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


def Beta2beta_j(Beta):
	return np.array([Beta[0], Beta[0], Beta[0],
					Beta[1], Beta[1],
					Beta[2], Beta[2],
					Beta[3], Beta[3]])


def calculate_force_matriceis(
		ind,
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
				  C['home_inter'][t].T.dot((Ie[ind.inter_risk_dict['Intervention', 'Low']] + Ie[ind.inter_risk_dict['Intervention', 'High']]) * alpha +
											Is[ind.inter_risk_dict['Intervention', 'Low']] + Is[ind.inter_risk_dict['Intervention', 'High']] +
											Ia[ind.inter_risk_dict['Intervention', 'Low']] + Ia[ind.inter_risk_dict['Intervention', 'High']]) +

				   behave_componnet_non_no_work *\
				  C['home_non'][t].T.dot((Ie[ind.inter_risk_dict['Non-intervention', 'Low']] + Ie[ind.inter_risk_dict['Non-intervention', 'High']]) * alpha +
										 Is[ind.inter_risk_dict['Non-intervention', 'Low']] +Is[ind.inter_risk_dict[ 'Non-intervention', 'High']] +
										 Ia[ind.inter_risk_dict['Non-intervention', 'Low']] + Ia[ind.inter_risk_dict['Non-intervention', 'High']]))

	force_out = (behave_componnet_inter_work * \
				  C['work_inter'][t].T.dot((Ie[ind.inter_risk_dict['Intervention', 'Low']] + Ie[ind.inter_risk_dict['Intervention', 'High']]) * alpha +
										   Is[ind.inter_risk_dict['Intervention', 'Low']] + Is[ind.inter_risk_dict['Intervention', 'High']] +
										   Ia[ind.inter_risk_dict['Intervention', 'Low']] + Ia[ind.inter_risk_dict['Intervention', 'High']]) +

				  behave_componnet_non_work *\
				  C['work_non'][t].T.dot((Ie[ind.inter_risk_dict['Non-intervention', 'Low']] + Ie[ind.inter_risk_dict['Non-intervention', 'High']]) * alpha +
										 Is[ind.inter_risk_dict['Non-intervention', 'Low']] + Is[ind.inter_risk_dict['Non-intervention', 'High']] +
										 Ia[ind.inter_risk_dict['Non-intervention', 'Low']] + Ia[ind.inter_risk_dict['Non-intervention', 'High']]) +

				 behave_componnet_inter_no_work *\
				 C['leisure_inter'][t].T.dot((Ie[ind.inter_risk_dict['Intervention', 'Low']] + Ie[ind.inter_risk_dict['Intervention', 'High']]) * alpha +
											  Is[ind.inter_risk_dict['Intervention', 'Low']] + Is[ind.inter_risk_dict['Intervention', 'High']] +
											  Ia[ind.inter_risk_dict['Intervention', 'Low']] + Ia[ind.inter_risk_dict['Intervention', 'High']]) +

				 behave_componnet_non_no_work*\
				 C['leisure_non'][t].T.dot((Ie[ind.inter_risk_dict['Non-intervention', 'Low']] + Ie[ind.inter_risk_dict['Non-intervention', 'High']]) * alpha +
											Is[ind.inter_risk_dict['Non-intervention', 'Low']] + Is[ind.inter_risk_dict['Non-intervention', 'High']] +
											Ia[ind.inter_risk_dict['Non-intervention', 'Low']] + Ia[ind.inter_risk_dict['Non-intervention', 'High']]))
	return {
		'out': force_out,
		'home': force_home
	}


def isolate_areas(
		ind,
		base_mtx,
		isolation_mtx,
		area_lst,
	):
	"""
	The function gets base_mtx and switches its rows and columns representing the areas in area_lst with thw
	matching rows and columns from isolation_mtx.
	:param base_mtx: numpy array matrix
	:param isolation_matx: numpy array matrix with the same shape as base_mtx
	:param area_lst: areas to put in/out quarantine
	:return: base_mtx with the matching rows and columns from isolation_mtx
	"""
	base_mtx = base_mtx.copy()

	for area in area_lst:
		# switch rows:
		base_mtx[ind.region_ga_dict[area], :] = isolation_mtx[ind.region_ga_dict[area], :]
		# switch columns:
		base_mtx[:, ind.region_ga_dict[area]] = isolation_mtx[:, ind.region_ga_dict[area]]
	return base_mtx

def multi_inter_by_name(
		ind,
		model,
		is_pop,
		inter_names,
		inter_times = None,
		sim_length=300,
		fix_vents=True,
		deg_param=None,
		no_pop = False,
	):
	if len(inter_names) == 1:
		inter_times = []
		inter_times.append(sim_length)
		sim_left = sim_length
	else:
		inter_times.append(sim_length - np.sum(inter_times))

	model_inter = copy.deepcopy(model)
	if deg_param is not None:
		with open('../Data/interventions/C_inter_' + deg_param['inter_max'] + '.pickle',
				  'rb') as pickle_in:
			C_max = pickle.load(pickle_in)

	for i, inter_name in enumerate(inter_names):
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

		if i != 0 or no_pop:
			transfer_pop_inter = None
		else:
			with open(
					'../Data/interventions/transfer_pop_inter_' + inter_name + '.pickle',
					'rb') as pickle_in:
				transfer_pop_inter = pickle.load(pickle_in)

		sim_left = inter_times[i]
		if deg_param is not None:
			if i==0:
				curr_time = 0
			else:
				curr_time = np.sum(inter_times[:i])
			C_inter, stay_home_idx_inter = policy_degredation(
				ind,
				C_inter,
				C_max,
				stay_home_idx_inter,
				deg_param['deg_rate'],
				range(curr_time, curr_time + sim_left, 1),
				deg_param['max_deg_rate'],
				True,
			)
		res_mdl = model_inter.intervention(
			C=C_inter,
			days_in_season=sim_left,
			stay_home_idx=stay_home_idx_inter,
			not_routine=routine_t_inter,
			prop_dict=transfer_pop_inter,
		)

	if fix_vents:
		for i, vent in enumerate(res_mdl['Vents']):
			res_mdl['Vents'][i] = vent + (
					(60.0 / is_pop) * vent) / vent.sum()

	return (res_mdl, model_inter)


def automatic_global_stop_inter(model, is_pop, inter_name, closed_inter, thresh=4960, thresh_var='Vents', sim_length=300):
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

	#Seconed intervention
	with open('../Data/interventions/C_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
		C_close = pickle.load(pickle_in)

	with open('../Data/interventions/stay_home_idx_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
		stay_home_idx_close = pickle.load(pickle_in)

	with open('../Data/interventions/routine_t_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
		routine_t_close = pickle.load(pickle_in)

	with open('../Data/interventions/transfer_pop_inter_' + closed_inter + '.pickle', 'rb') as pickle_in:
		transfer_pop_close = pickle.load(pickle_in)

	# run normal intervention of letting everybody out
	model_inter = copy.deepcopy(model)
	res_mdl = model_inter.intervention(
		C=C_inter,
		days_in_season=sim_length,
		#             days_in_season=(dates[-1]-start_inter).days,
		#             days_in_season=inter2_timing - (start_inter-beginning).days,
		stay_home_idx=stay_home_idx_inter,
		not_routine=routine_t_inter,
		prop_dict=transfer_pop_inter,
	)

	days_to_closing = (res_mdl[thresh_var].sum(
		axis=1) * is_pop < thresh).sum() - 14
	days_to_closing -= len(model.S)

	# run normal intervention of letting everybody out until sec inter.
	model_inter = copy.deepcopy(model)
	model_inter.intervention(
		C=C_inter,
		days_in_season=days_to_closing,
		#             days_in_season=(dates[-1]-start_inter).days,
		#             days_in_season=inter2_timing - (start_inter-beginning).days,
		stay_home_idx=stay_home_idx_inter,
		not_routine=routine_t_inter,
		prop_dict=transfer_pop_inter,
	)
	res_mdl_close = model_inter.intervention(
		C=C_close,
		days_in_season=sim_length-days_to_closing,
		stay_home_idx=stay_home_idx_close,
		not_routine=routine_t_close,
		prop_dict=None,
	)

	for i, vent in enumerate(res_mdl['Vents']):
		res_mdl['Vents'][i] = vent + (
					(60.0 / is_pop) * vent) / vent.sum()
	for i, vent in enumerate(res_mdl_close['Vents']):
		res_mdl_close['Vents'][i] = vent + (
					(60.0 / is_pop) * vent) / vent.sum()

	return (res_mdl, res_mdl_close, days_to_closing)


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


def print_stat_fit_hosp(fit_results_object, tracking='hosp'):

	"""The function gets sci optimization results object and print additional info about the optimization
	:param fit_results_object:
	:return:
	"""
	print('minimized value:', fit_results_object.fun)
	if tracking == 'hosp':
		print('Fitted parameters:\n Eta={0}\n Nu={1},\n '.format(fit_results_object.x[0],
																				  fit_results_object.x[1]))

	elif tracking == 'vents':
		print('Fitted parameters:\n Xi={0}\n Mu={1},\n '.format(fit_results_object.x[0],
																				  fit_results_object.x[1]))
	print('num of sampling the target function:', fit_results_object.nfev)


def policy_degredation(
	ind,
	C,
	C_max,
	sh_idx,
	deg_rate,
	t,
	max_deg_rate,
	risk_deg,
	):

	new_c = {}
	new_sh = {}
	if risk_deg:
		for key in C.keys():
			new_c[key] = []
			C_mat_step = (C_max[key][0] - C[key][0])
			for curr_t in t:
				C_mat = C[key][0].copy()
				factor = np.minimum(
					curr_t * deg_rate,
					max_deg_rate) / 100.0
				if key in ['home_non', 'work_non', 'leisure_non']:
					C_mat += C_mat_step * factor
				else:
					for group, idx in ind.age_ga_dict.items():
						if group in ['70+', '60-69']:
							C_mat[idx, :] += C_mat_step[idx, :] * factor
							C_mat[:, idx] += C_mat_step[:, idx] * factor
				new_c[key].append(C_mat)
		for key in sh_idx.keys():
			new_sh[key] = {}
			for key2 in sh_idx[key].keys():
				new_sh[key][key2] = []
				sh_idx_step = (np.ones_like(sh_idx[key][key2][0]) -
							   sh_idx[key][key2][0])
				for curr_t in t:
					sh_idx_vec = sh_idx[key][key2][0].copy()
					factor = np.minimum(
						curr_t * deg_rate,
						max_deg_rate) / 100.0
					if key == 'non_inter':
						sh_idx_vec += sh_idx_step * factor
					else:
						for group, idx in ind.age_ga_dict.items():
							if group in ['70+', '60-69']:
								sh_idx_vec[idx] += sh_idx_step[idx] * factor
					new_sh[key][key2].append(sh_idx_vec)
	return (new_c, new_sh)


def inter2name(
		ind,
		pct,
		slipped_pct=None,
		no_risk=False,
		no_school=False,
		no_kid010=False,
		risk_slipping=False,
		world_slipping=False
	):
	inter_name = ind.cell_name + '@' + str(pct)
	if pct != 10:
		if no_risk:
			inter_name += '_no_risk60'
		if no_school:
			inter_name += '_no_school'
		else:
			inter_name += '_school'
		if not no_kid010:
			inter_name += '_kid010'
		else:
			inter_name += '_no_kid010'
		if risk_slipping:
			inter_name += '_slipped_risk('+str(slipped_pct)+')'
		if world_slipping:
			inter_name += '_slipped_world('+str(slipped_pct)+')'
	return inter_name

