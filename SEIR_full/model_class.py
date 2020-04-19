from .indices import *
from .helper_func import *
from .parameters import *
import pandas as pd
import numpy as np
from scipy import optimize
import itertools
import pickle


#######################
# ---- Model Class---- #
#######################

class Model_behave:

	def __init__(
			self,
			beta_j=np.array(
				[0.02927922, 0.02927922, 0.02927922, 0.04655266, 0.04655266,
				0.05775265, 0.05775265, 0.18444245, 0.18444245]),
			theta=2.826729434860104,
			beta_behave=0.5552998605894367,
			is_haredi=is_haredi,
			alpha=alpha,
			beta_home=beta_home,
			sigma=sigma,
			delta=delta,
			gama=gama,
			population_size=population_size,
			rho=hospitalizations,
			nu=nu,
			eps=eps,
			f=f0_full,
	):
		"""
		Receives all model's hyper-parameters and creates model object
		the results.
		:param self:
		:param is_haredi:
		:param alpha:
		:param beta_home:
		:param sigma:
		:param delta:
		:param gama:
		:param population_size:
		:param rho:
		:param nu:
		:param eps:
		:param f:
		"""
		# defining parameters:
		self.init_region_pop = shrink_array_sum(
			mapping_dic=region_dict,
			array_to_shrink=population_size
		)

		self.beta_j = beta_j
		self.theta = theta
		self.beta_behave = beta_behave
		self.is_haredi = is_haredi
		self.alpha = alpha
		self.beta_home = beta_home
		self.delta = delta
		self.gama = gama
		self.rho = rho
		self.nu = nu
		self.sigma = sigma
		self.eps = eps
		self.f = f
		self.population_size = population_size.copy()

		# defining model compartments:
		self.reset(self.population_size, self.eps)

		# counter for training
		self.fit_iter_count = 0

	def reset(
			self,
			population_size=None,
			eps=None
		):
		"""
		Reset object's SEIR compartments.
		:param population_size:
		:param eps:
		"""
		if population_size is None:
			population_size = self.population_size
		if eps is None:
			eps = self.eps

		# defining model compartments:
		self.S, self.E, self.Ie, self.Is, self.Ia, self.R, \
		self.H, self.new_Is, self.L = [], [], [], [], [],[], [], [], []

		# Initializing compartments:
		# Initialize S_0 to population size of each age-group
		self.S.append(population_size.copy())

		# Initialize R - with only the naturally immune individuals
		self.R.append(np.zeros(len(N)))

		# subtract R0 from S0:
		self.S[-1] -= self.R[-1]

		# Initialize E
		self.E.append(np.zeros(len(N)))

		# Initialize I (early)
		self.Ie.append(np.zeros(len(N)))

		# Initialize I (asymptomatic)
		self.Ia.append(np.zeros(len(N)))

		# Initialize I (symptomatic)
		self.Is.append(np.zeros(len(N)))

		# Subtract E(0) and Ie(0) from S(0)
		self.S[-1] -= (self.E[-1] + self.Ie[-1])

		# Zero newly infected on the first day of the season
		self.new_Is.append(np.zeros(len(N)))

		# Initialize H, tracking compartment
		self.H.append(np.zeros(len(N)))


	def fit(
			self,
			p0,
			bnds,
			data,
			C=C_calibration,
			days_in_season=70,
			stay_home_idx=stay_home_idx,
			not_routine=not_routine,
			method='TNC',
			maxiter=200,
			date_lst=pd.date_range('2020-02-20', periods=100, freq='d'),
			start='2020-03-20',
			end='2020-04-13',
			loss_func='MSE',
			factor=1,
		):

		self.reset()
		self.fit_iter_count = 0

		res_fit = optimize.minimize(
			self.calc_loss,
			p0,
			bounds=bnds,
			method=method,
			args=(
				data,
				C,
				days_in_season,
				stay_home_idx,
				not_routine,
				date_lst,
				start,
				end,
				loss_func,
				factor,
			),
			options={'maxiter': maxiter},
		)
		fitted_params = res_fit.x

		# run the fitted model:
		fitted_beta = np.array(
			[fitted_params[0], fitted_params[0], fitted_params[0],
			 fitted_params[1], fitted_params[1], \
			 fitted_params[2], fitted_params[2], fitted_params[3],
			 fitted_params[3]])
		theta = fitted_params[4]
		beta_behave = fitted_params[5]

		self.update({
			'beta_j': fitted_beta,
			'theta': theta,
			'beta_behave': beta_behave,
		})
		return res_fit


	def calc_loss(
			self,
			tpl,
			data,
			C,
			days_in_season,
			stay_home_idx,
			not_routine,
			date_lst=pd.date_range('2020-02-20',periods=70, freq='d'),
			start='2020-03-20',
			end='2020-04-13',
			loss_func='MSE',
			factor=1,
		):
		"""
		Calibrates the model to data
		:param self:
		:param tpl:
		:param C:
		:param days_in_season:
		:param stay_home_idx:
		:param not_routine:
		:param date_lst:
		:param start:
		:param end:
		:param loss_func: loss function to minimize 'MSE' or 'BIN' or 'POIS'
		:returns: loss functions' value
		"""
		# update counter of runs since last fit action
		self.fit_iter_count = self.fit_iter_count + 1

		# setting parameters
		beta_j = np.array([tpl[0], tpl[0], tpl[0], tpl[1], tpl[1], tpl[2], tpl[2], tpl[3], tpl[3]])

		# Run model with given parameters
		model_res = self.predict(
			C=C,
			days_in_season=days_in_season,
			stay_home_idx=stay_home_idx,
			not_routine=not_routine,
			beta_j=beta_j,
			beta_behave=tpl[5],
			theta=tpl[4]
		)

		new_cases_model = model_res['new_Is']
		# if np.isnan(new_cases_model).any():
		# 	print(np.isnan(new_cases_model).sum())
		# 	print('beta_j: ', beta_j)
		# 	print('beta_behave: ', tpl[5])
		# 	print('theta: ', tpl[4])

		model_results_cal = np.zeros((days_in_season + 1, len(region_age_dict)))

		# Calculated total symptomatic (high+low) per age group (adding as columns)
		for i, key in enumerate(region_age_dict.keys()):
			model_results_cal[:, i] = new_cases_model[:, region_age_dict[key]].sum(axis=1)

		# Taking relevant time frame from model
		start_idx = int(np.where(date_lst == start)[0])
		end_idx = int(np.where(date_lst == end)[0])
		model_for_loss = model_results_cal[start_idx:end_idx + 1, :].copy()

		if loss_func == "MSE":
			# fixing data to be proportion of israel citizens
			data_specific = data[1]/9136000
			loss = np.log(MSE(data_specific, model_for_loss))

		elif loss_func == "BIN":
			# fixing data to be proportion of tests
			data_specific = data.copy()
			data_specific[1] = (data_specific[1] / data_specific[0])
			data_specific = np.nan_to_num(data_specific,
										  nan=0,
										  posinf=0,
										  neginf=0)
			# fixing model_out to be proportion of cell j,k
			pop_jk = shrink_array_sum(region_age_dict, self.population_size)
			model_for_loss_specific = model_for_loss.copy()
			model_for_loss_specific = model_for_loss_specific / pop_jk
			loss = ML_Bin(data_specific, model_for_loss_specific, approx=False,
						  factor=factor)

		elif loss_func == "POIS":
			# fixing data to be proportion of tests
			data_specific = data.copy()
			data_specific[1] = (data_specific[1] / data_specific[0])
			data_specific = np.nan_to_num(data_specific,
										  nan=0,
										  posinf=0,
										  neginf=0)
			# fixing model_out to be proportion of cell j,k
			pop_jk = shrink_array_sum(region_age_dict, self.population_size)
			model_for_loss_specific = model_for_loss.copy()
			model_for_loss_specific = model_for_loss_specific / pop_jk
			loss = ML_Bin(data_specific, model_for_loss_specific, approx=True)

		elif loss_func == "POIS_NAIV":
			# fixing data to be proportion of israel citizens
			data_specific = data[1]
			# fixing model output to be new sick people
			model_for_loss_specific = model_for_loss* 9136000
			loss = ML_Pois_Naiv(data_specific, model_for_loss_specific)

		self.reset()
		if self.fit_iter_count % 50 == 0:
			print('iter: ', self.fit_iter_count,' loss: ', loss)
		return loss


	def update(self, new_param):
		"""
		update object's attributes.
		:param self:
		:param new_param: dictionary object, key=attribute name, value= attribute value to assign
		"""
		for key, value in new_param.items():
			setattr(self, key, value)

	def intervention(
			self,
			C,
			beta_j,
			beta_behave,
			theta,
			days_in_season,
			stay_home_idx,
			not_routine,
			prop_dict,
			disable_beta_behave=np.zeros(len(GA))
		):
		"""
		Receives  model's parameters and run intervention for days_in_season days
		the results.
		:param self:
		:param beta_j:
		:param beta_behave:
		:param days_in_season:
		:param stay_home_idx:
		:param not_routine:
		:param disable_beta_behave:
		"""
		# Shifting population in compartments:
		self.S.append(divide_population(prop_dict, self.S[-1]))
		self.E.append(divide_population(prop_dict, self.E[-1]))
		self.Ie.append(divide_population(prop_dict, self.Ie[-1]))
		self.Ia.append(divide_population(prop_dict, self.Ia[-1]))
		self.Is.append(divide_population(prop_dict, self.Is[-1]))
		self.R.append(divide_population(prop_dict, self.R[-1]))

		self.predict(
			C=C,
			beta_j=beta_j,
			beta_behave=beta_behave,
			theta=theta,
			days_in_season=days_in_season,
			stay_home_idx=stay_home_idx,
			not_routine=not_routine,
			disable_beta_behave = disable_beta_behave,
		)


	def predict(
			self,
			C,
			days_in_season,
			stay_home_idx,
			not_routine,
			beta_j=None,
			beta_behave=None,
			theta=None,
			disable_beta_behave=np.zeros(len(GA)),
		):
		"""
		Receives  model's parameters and run model for days_in_season days
		the results.
		:param C:
		:param days_in_season:
		:param stay_home_idx:
		:param not_routine:
		:param beta_j:
		:param beta_behave:
		:param theta:
		:return:
		"""

		if beta_j is None:
			beta_j = self.beta_j
		if beta_behave is None:
			beta_behave = self.beta_behave
		if theta is None:
			theta = self.theta

		beta_j = expand_partial_array(
			mapping_dic=age_ga_dict,
			array_to_expand=beta_j,
			size=len(GA)
		)

		for t in range(days_in_season):

			# Calculate beta_home factor, current S_region/N_region and expand it
			# to mach (180X1).
			beta_home_factor = shrink_array_sum(
				mapping_dic=region_dict,
				array_to_shrink=self.S[-1]
			)
			beta_home_factor = beta_home_factor / self.init_region_pop
			beta_home_factor = expand_partial_array(
				mapping_dic=region_ga_dict,
				array_to_expand=beta_home_factor,
				size=len(GA)
			)

			# Calculate lambda (High risk symptomatic + Low risk symptomatic +
			# Asymptomatic)
			contact_force = self.calculate_force_matriceis(
				t=t,
				C=C,
				beta_behave=beta_behave,
				stay_home_idx=stay_home_idx,
				not_routine=not_routine,
				disable_beta_behave=disable_beta_behave,
			)

			lambda_t = (self.beta_home * beta_home_factor * contact_force['home'] +
					   beta_j * (theta * self.is_haredi + 1 - self.is_haredi) * contact_force['out'])

			# preventing from lambda to become nan where there is no population.
			#lambda_t[np.isnan(lambda_t)] = 0

			self.L.append(lambda_t)

			# fitting lambda_t size to (720X1)
			lambda_t = expand_partial_array(
				mapping_dic=region_age_dict,
				array_to_expand=lambda_t
			)

			# R(t)
			self.R.append(self.R[-1] + self.gama * (self.Is[-1] + self.Ia[-1]))

			# H(t)
			self.H.append(self.H[-1] + (self.rho * self.gama) * self.Is[-1] - self.nu * self.H[-1])

			# Is(t)
			# Save new_Is
			self.new_Is.append((1 - self.f) * self.delta * self.Ie[-1])

			# Calculate new i matrix for day t
			self.Is.append(self.Is[-1] + self.new_Is[-1] - self.gama * self.Is[-1])

			# Ia(t)
			# Calculate new i matrix for day t
			self.Ia.append(self.Ia[-1] + self.f * self.delta * self.Ie[-1] - self.gama * self.Ia[-1])

			# Ie(t)
			# Calculate new i matrix for day t
			self.Ie.append(self.Ie[-1] + self.sigma * self.E[-1] - self.delta * self.Ie[-1])

			# E(t)
			# Calculate new e matrix for day t
			self.E.append(self.eps[t] + self.E[-1] + lambda_t * self.S[-1] - self.sigma * self.E[-1])

			# S(t)
			# Calculate current S
			self.S.append(self.S[-1] - lambda_t * self.S[-1])

			# Return the model results
		return {
			'S': np.array(self.S),
			'E': np.array(self.E),
			'Ie': np.array(self.Ie),
			'Ia': np.array(self.Ia),
			'Is': np.array(self.Is),
			'R': np.array(self.R),
			'new_Is': np.array(self.new_Is),
			'L': np.array(self.L),
			'H': np.array(self.H),
		}


	def calculate_force_matriceis(
			self,
			t,
			C,
			stay_home_idx,
			not_routine,
			beta_behave,
			disable_beta_behave=np.zeros(len(GA)),
		):
		# Calculating beta_behave components:
		behave_componnet_factor = (beta_behave * stay_home_idx['inter']['not_work'][t])
		if (type(behave_componnet_factor)==float) or \
				(type(behave_componnet_factor) == np.float64):
			if behave_componnet_factor==0:
				behave_componnet_factor = 1
		else:
			behave_componnet_factor[behave_componnet_factor==0] = 1
		behave_componnet_inter_no_work = np.power(
			(1 / behave_componnet_factor),
			 disable_beta_behave) * \
			(beta_behave *
			stay_home_idx['inter']['not_work'][t]) ** \
			not_routine['inter']['not_work'][t]

		behave_componnet_non_no_work = (beta_behave * stay_home_idx['non_inter']['not_work'][t]) ** \
										not_routine['non_inter']['not_work'][t]

		behave_componnet_factor = (beta_behave * stay_home_idx['inter']['work'][t])
		if (type(behave_componnet_factor) == float) or \
				(type(behave_componnet_factor) == np.float64):
			if behave_componnet_factor == 0:
				behave_componnet_factor = 1
		else:
			behave_componnet_factor[behave_componnet_factor == 0] = 1

		behave_componnet_inter_work = np.power((1. / behave_componnet_factor),
													disable_beta_behave) *\
										(beta_behave * stay_home_idx['inter']['work'][t]) ** \
										not_routine['inter']['work'][t]

		behave_componnet_non_work = (beta_behave * stay_home_idx['non_inter']['work'][t]) ** \
									not_routine['non_inter']['work'][t]

		force_home = ((behave_componnet_inter_no_work * \
						C['home_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
												self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Is[-1][inter_risk_dict['Intervention', 'High']] +
												self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Ia[-1][inter_risk_dict['Intervention', 'High']])) +

					behave_componnet_non_no_work * \
					(C['home_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
											self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
											self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Ia[-1][inter_risk_dict['Non-intervention', 'High']])))

		force_out = ((behave_componnet_inter_work * \
					C['work_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
											self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
											self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
											self.Is[-1][inter_risk_dict['Intervention', 'High']] +
											self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
											self.Ia[-1][inter_risk_dict['Intervention', 'High']]) +

					(behave_componnet_non_work * \
					C['work_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
											self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
											self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
											self.Ia[-1][inter_risk_dict['Non-intervention', 'High']])) +

					(behave_componnet_inter_no_work * \
					C['leisure_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
												self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Is[-1][inter_risk_dict['Intervention', 'High']] +
												self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
												self.Ia[-1][inter_risk_dict['Intervention', 'High']])) +

					(behave_componnet_non_no_work * \
					C['leisure_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
												self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
												self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
												self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
												self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
												self.Ia[-1][inter_risk_dict['Non-intervention', 'High']]))))
		return {
			'out': force_out,
			'home': force_home
		}
