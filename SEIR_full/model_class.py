from .utils import *
from .parameters import *
from .plot_utils import *
import pandas as pd
import numpy as np
from scipy import optimize

#######################
# ---- Model Class---- #
#######################


class Model_behave:

	def __init__(
			self,
			ind,
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
			chi=vents_proba,
			nu=nu,
			mu=mu,
			eta=eta,
			xi=xi,
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
		:param rho: probability of getting hospitalized
		:param nu: hospitalization recovery rate
		:param chi: probability of getting ventilator supports
		:param mu: ventilator recovery rate
		:param eta: rate of transition between hosp_latent to H
		:param xi: rate of transition between vents_latent to ventilated
		:param eps:
		:param f:
		"""
		# define indices
		self.ind = ind

		# defining parameters:
		self.init_region_pop = shrink_array_sum(
			mapping_dic=ind.region_dict,
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
		self.chi = chi
		self.mu = mu
		self.eta = eta
		self.xi = xi
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
		self.S, self.E, self.Ie, self.Is, self.Ia, self.R, self.Vents_latent, \
		self.H, self.new_Is, self.L, self.Vents, self.Hosp_latent = [], [], [], [], [],[], [], [], [], [], \
																	[], []

		# Initializing compartments:
		# Initialize S_0 to population size of each age-group
		self.S.append(population_size.copy())

		# Initialize R - with only the naturally immune individuals
		self.R.append(np.zeros(len(self.ind.N)))

		# subtract R0 from S0:
		self.S[-1] -= self.R[-1]

		# Initialize E
		self.E.append(np.zeros(len(self.ind.N)))

		# Initialize I (early)
		self.Ie.append(np.zeros(len(self.ind.N)))

		# Initialize I (asymptomatic)
		self.Ia.append(np.zeros(len(self.ind.N)))

		# Initialize I (symptomatic)
		self.Is.append(np.zeros(len(self.ind.N)))

		# Subtract E(0) and Ie(0) from S(0)
		self.S[-1] -= (self.E[-1] + self.Ie[-1])

		# Zero newly infected on the first day of the season
		self.new_Is.append(np.zeros(len(self.ind.N)))

		# Initialize H(0), Vents(0), H_latent(0), Vents_latent(0) tracking compartment
		self.H.append(np.zeros(len(self.ind.N)))
		self.Vents.append(np.zeros(len(self.ind.N)))
		self.Hosp_latent.append(np.zeros(len(self.ind.N)))
		self.Vents_latent.append(np.zeros(len(self.ind.N)))




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
			mapper=None,
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
				mapper,
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

	def fit_tracking(
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
			tracking='hosp'
		):

		self.reset()
		self.fit_iter_count = 0

		res_fit = optimize.minimize(
			self.calc_loss_tracking,
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
				tracking
			),
			options={'maxiter': maxiter},
		)
		fitted_params = res_fit.x

		# run the fitted model:
		if tracking == 'hosp':
			eta = fitted_params[0]
			nu = fitted_params[1]

			self.update({
				'nu': nu,
				'eta': eta
			})

		elif tracking == 'vents':
			xi = fitted_params[0]
			mu = fitted_params[1]

			self.update({
				'mu': mu,
				'xi': xi
			})

		return res_fit


	def calc_loss_tracking(
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
			tracking='hosp'
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
		:param tracking: tracking compartment to calibrate 'hosp' or 'vents'
		:returns: loss functions' value
		"""
		# update counter of runs since last fit action
		self.fit_iter_count = self.fit_iter_count + 1

		if tracking == 'hosp':
			# Run model with given parameters
			model_res = self.predict(
				C=C,
				days_in_season=days_in_season,
				stay_home_idx=stay_home_idx,
				not_routine=not_routine,
				eta=tpl[0],
				nu=tpl[1]
			)
			new_cases_model = model_res['H'].sum(axis=1)

		elif tracking == 'vents':
			# Run model with given parameters
			model_res = self.predict(
				C=C,
				days_in_season=days_in_season,
				stay_home_idx=stay_home_idx,
				not_routine=not_routine,
				xi=tpl[0],
				mu=tpl[1]
			)
			new_cases_model = model_res['Vents'].sum(axis=1)

		# if np.isnan(new_cases_model).any():
		# 	print(np.isnan(new_cases_model).sum())
		# 	print('beta_j: ', beta_j)
		# 	print('beta_behave: ', tpl[5])
		# 	print('theta: ', tpl[4])


		# Taking relevant time frame from model
		start_idx = int(np.where(date_lst == start)[0])
		end_idx = int(np.where(date_lst == end)[0])
		model_for_loss = new_cases_model[start_idx:end_idx + 1].copy()

		if loss_func == "MSE":
			# fixing data to be proportion of israel citizens
			data_specific = data/pop_israel
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
			pop_jk = shrink_array_sum(self.ind.region_age_dict, self.population_size)
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
			pop_jk = shrink_array_sum(self.ind.region_age_dict, self.population_size)
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
			mapper=None
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
		:param mapper: different dict to match model's results
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
		if mapper == None:
			model_results_cal = np.zeros((days_in_season + 1, len(self.ind.region_age_dict)))

			# Calculated total symptomatic (high+low) per age group (adding as columns)
			for i, key in enumerate(self.ind.region_age_dict.keys()):
				model_results_cal[:, i] = new_cases_model[:, self.ind.region_age_dict[key]].sum(axis=1)

		else:
			model_results_cal = np.zeros((days_in_season + 1, len(mapper)))

			# Calculated total symptomatic (high+low) per age group (adding as columns)
			for i, key in enumerate(mapper.keys()):
				model_results_cal[:, i] = new_cases_model[:, mapper[key]].sum(axis=1)

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
			pop_jk = shrink_array_sum(self.ind.region_age_dict, self.population_size)
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
			pop_jk = shrink_array_sum(self.ind.region_age_dict, self.population_size)
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
			days_in_season,
			stay_home_idx,
			not_routine,
			prop_dict=None,
		):
		"""
		Receives  model's parameters and run intervention for days_in_season days
		the results.
		:param self:
		:param days_in_season:
		:param stay_home_idx:
		:param not_routine:
		:param disable_beta_behave:
		"""
		# Shifting population in compartments:
		if prop_dict is not None:
			self.S.append(divide_population(self.ind, prop_dict, self.S.pop()))
			self.E.append(divide_population(self.ind, prop_dict, self.E.pop()))
			self.Ie.append(divide_population(self.ind, prop_dict, self.Ie.pop()))
			self.Ia.append(divide_population(self.ind, prop_dict, self.Ia.pop()))
			self.Is.append(divide_population(self.ind, prop_dict, self.Is.pop()))
			self.R.append(divide_population(self.ind, prop_dict, self.R.pop()))
		# make sure no new sick will pop up during intervention
		self.eps = np.zeros((days_in_season, len(self.ind.N)))

		return self.predict(
			C=C,
			days_in_season=days_in_season,
			stay_home_idx=stay_home_idx,
			not_routine=not_routine,
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
			xi=None,
			mu=None,
			eta=None,
			nu=None
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
		:param xi:
		:param mu:
		:param eta:
		:param nu
		:return:
		"""

		if beta_j is None:
			beta_j = self.beta_j
		if beta_behave is None:
			beta_behave = self.beta_behave
		if theta is None:
			theta = self.theta
		if xi is None:
			xi = self.xi
		if mu is None:
			mu = self.mu
		if eta is None:
			eta = self.eta
		if nu is None:
			nu = self.nu

		beta_j = expand_partial_array(
			mapping_dic=self.ind.age_ga_dict,
			array_to_expand=beta_j,
			size=len(self.ind.GA)
		)

		for t in range(days_in_season):

			# Calculate beta_home factor, current S_region/N_region and expand it
			# to mach (180X1).
			beta_home_factor = shrink_array_sum(
				mapping_dic=self.ind.region_dict,
				array_to_shrink=self.S[-1]
			)
			beta_home_factor = beta_home_factor / self.init_region_pop
			beta_home_factor = expand_partial_array(
				mapping_dic=self.ind.region_ga_dict,
				array_to_expand=beta_home_factor,
				size=len(self.ind.GA)
			)

			# Calculate lambda (High risk symptomatic + Low risk symptomatic +
			# Asymptomatic)
			contact_force = self.calculate_force_matriceis(
				t=t,
				C=C,
				beta_behave=beta_behave,
				stay_home_idx=stay_home_idx,
				not_routine=not_routine,
			)
			lambda_t = (self.beta_home * beta_home_factor * contact_force['home'] +
					   beta_j * (theta * self.is_haredi + 1 - self.is_haredi) * contact_force['out'])

			# preventing from lambda to become nan where there is no population.
			#lambda_t[np.isnan(lambda_t)] = 0

			self.L.append(lambda_t)

			# fitting lambda_t size to (720X1)
			lambda_t = expand_partial_array(
				mapping_dic=self.ind.region_age_dict,
				array_to_expand=lambda_t,
				size=len(self.ind.N),
			)

			# R(t)
			self.R.append(self.R[-1] + self.gama * (self.Is[-1] + self.Ia[-1]))

			# H(t)
			self.H.append(self.H[-1] + eta * self.Hosp_latent[-1] - nu * self.H[-1])

			# Vents(t)
			self.Vents.append(self.Vents[-1] + xi * self.Vents_latent[-1] - mu * self.Vents[-1])

			# H_latent(t)
			self.Hosp_latent.append(self.Hosp_latent[-1] + (self.rho * self.gama) * self.Is[-1] -
									eta * self.Hosp_latent[-1])

			# Vents_latent(t)
			self.Vents_latent.append(self.Vents_latent[-1] + (self.chi * self.gama) * self.Is[-1] -
									 xi * self.Vents_latent[-1])

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
			'Hosp_latent': np.array(self.Hosp_latent),
			'Vents': np.array(self.Vents),
			'Vents_latent': np.array(self.Vents_latent),
		}

	def get_res(self):
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
			'Hosp_latent': np.array(self.Hosp_latent),
			'Vents': np.array(self.Vents),
			'Vents_latent': np.array(self.Vents_latent),
		}

	def calculate_force_matriceis(
			self,
			t,
			C,
			stay_home_idx,
			not_routine,
			beta_behave,
		):
		# Calculating beta_behave components:
		behave_componnet_inter_no_work = \
			np.power(beta_behave * stay_home_idx['inter']['not_work'][t],
					 not_routine['inter']['not_work'][t])

		behave_componnet_non_no_work = \
			np.power(beta_behave * stay_home_idx['non_inter']['not_work'][t],
					 not_routine['non_inter']['not_work'][t])

		behave_componnet_inter_work = \
			np.power(beta_behave * stay_home_idx['inter']['work'][t],
					 not_routine['inter']['work'][t])

		behave_componnet_non_work = \
			np.power(beta_behave * stay_home_idx['non_inter']['work'][t],
					 not_routine['non_inter']['work'][t])
		force_home = ((behave_componnet_inter_no_work * \
					   (C['home_inter'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'High']]) * self.alpha +
												self.Is[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Is[-1][self.ind.inter_risk_dict['Intervention', 'High']] +
												self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'High']])).reshape(-1)) +

					(behave_componnet_non_no_work * \
					(C['home_non'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
											self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'High']] +
											self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'High']])).reshape(-1)))

		force_out = ((behave_componnet_inter_work * \
					C['work_inter'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
											self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'High']]) * self.alpha +
											self.Is[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
											self.Is[-1][self.ind.inter_risk_dict['Intervention', 'High']] +
											self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
											self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'High']]) +

					(behave_componnet_non_work * \
					C['work_non'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
											self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'High']] +
											self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
											self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'High']])) +

					(behave_componnet_inter_no_work * \
					C['leisure_inter'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Ie[-1][self.ind.inter_risk_dict['Intervention', 'High']]) * self.alpha +
												self.Is[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Is[-1][self.ind.inter_risk_dict['Intervention', 'High']] +
												self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'Low']] +
												self.Ia[-1][self.ind.inter_risk_dict['Intervention', 'High']])) +

					(behave_componnet_non_no_work * \
					C['leisure_non'][t].T.dot((self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
												self.Ie[-1][self.ind.inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
												self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
												self.Is[-1][self.ind.inter_risk_dict['Non-intervention', 'High']] +
												self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'Low']] +
												self.Ia[-1][self.ind.inter_risk_dict['Non-intervention', 'High']]))))
		return {
			'out': force_out,
			'home': force_home
		}
