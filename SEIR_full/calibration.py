from .indices import *
from SEIR_full.model_class import *
from .utils import *
import pandas as pd
import numpy as np
import itertools
import pickle


def make_calibration(
		scen,
		phase,
		ind,
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):
	# load mapper:
	# loads mapping 250 to 20:
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_mapper.pickle',
			'rb') as pickle_in:
		region_250_to_30 = pickle.load(pickle_in)
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_age_mapper.pickle',
			'rb') as pickle_in:
		region_age_250_to_30 = pickle.load(pickle_in)
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_ga_mapper.pickle',
			'rb') as pickle_in:
		region_ga_250_to_30 = pickle.load(pickle_in)

	# get the sick data:
	data, pos = load_data(ind, region_250_to_30, region_age_250_to_30)

	start_data = pos.index[0]
	end_data = pos.index[-1]
	# fit params
	date_lst = pd.date_range(start_sim, end_data)
	if (phase is None) or (phase == '-'):
		seasonality = False
		phi = 0
	else:
		seasonality = True
		phi = phase

	#make default model
	model_ml = Model_behave(
		ind=ind,
		beta_j=np.array(
			[0.10167117, 0.10167117, 0.10167117, 0.05606748, 0.05606748,
			 0.04709613, 0.04709613, 0.25676354, 0.25676354]) * 0.4,
		theta=2.0684200685446243,
		beta_behave=0.38,
		scen=scen,
		seasonality=seasonality,
		phi=phi,
	)

	# Model Fitting
	if no_haredim:
		p0 = (
		0.04066847, 0.02242699, 0.01883845, 0.10270542, 1,
		0.38)  # initial guess
		bnds = ((0, 0.3), (0, 0.2), (0, 0.2), (0, 0.6), (1, 1),
				(0, 1))  # boundries for variables
	elif no_mobility:
		p0 = (
		0.04066847, 0.02242699, 0.01883845, 0.10270542, 2.0684200685446243,
		0.38)  # initial guess
		bnds = ((0, 1000), (0, 1000), (0, 1000), (0, 1000), (1.5, 3),
				(0, 1))  # boundries for variables
	else:
		p0 = (0.04066847, 0.02242699, 0.01883845, 0.10270542, 2.0684200685446243,
			  0.38)  # initial guess
		bnds = ((0, 0.3), (0, 0.2), (0, 0.2), (0, 0.6), (1.5, 3),
				(0, 1))  # boundries for variables
	if no_mobility:
		C = {}
		sh = {}
		for key in C_calibration.keys():
			C[key] = [np.ones_like(C_calibration[key][0])]*len(C_calibration[key])
		for key in stay_home_idx.keys():
			sh[key] = [np.ones_like(stay_home_idx[key][0])] * len(
				stay_home_idx[key])
	else:
		C = C_calibration
		sh = stay_home_idx
	res_fit_ml = model_ml.fit(
		C=C,
		stay_home_idx=sh,
		mapper=region_age_250_to_30,
		p0=p0,
		bnds=bnds,
		data=data,
		date_lst=date_lst,
		start=start_data,
		end=end_data,
		loss_func='MSE',
		maxiter=85,
		factor=1,
	)
	# print_stat_fit_behave(res_fit_ml)

	save_cal(res_fit_ml, ind, scen, phase, no_mobility, no_haredim)

	show_calibration_param(
		ind,
		scen_idx=int(scen[-1]),
		phase=phase,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)


def make_track_calibration(
		scen,
		phase,
		ind,
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):

	df = pd.read_csv('../Data/sick/daily_hospital_resp.csv')
	df.index = pd.to_datetime(df['Date'], dayfirst=True)
	resp = df['resp'].dropna()
	hosp = df['hospitalized'].dropna()

	make_spec_track(
		state='H',
		data=hosp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)

	show_calibration_track_spec(
		state='H',
		data=hosp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)

	make_spec_track(
		state='Vents',
		data=resp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)

	show_calibration_track_spec(
		state='Vents',
		data=resp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)


def make_spec_track(
		state,
		data,
		scen,
		phase,
		ind,
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):

	# fit params
	if (phase is None) or (phase == '-'):
		seasonality = False
		phi = 0
	else:
		seasonality = True
		phi = phase

	scen_idx = int(scen[-1])
	start_data = data.index[0]
	end_data = data.index[-1]
	date_lst = pd.date_range(start_sim, end_data)

	if state == 'H':
		# Model Fitting
		p0 = (1 / 7., 0.1)  # initial guess
		# bnds = ((0.05, 1.0), (0.05, 0.25))  # boundries for variables
		bnds = ((0.0, 1.0), (0.0, 1.0))  # boundries for variables
	elif state == 'Vents':
		# Model Fitting
		p0 = (0.05, 1 / 9.9)  # initial guess
		# bnds = ((0.00, 1), (1 / 21., 1 / 10.))  # boundries for variables
		bnds = ((0.00, 1.0), (0.0, 1.0))  # boundries for variables

	cal_parameters = pd.read_pickle(
		'../Data/calibration/calibration_dict.pickle')

	model = Model_behave(
		ind=ind,
		beta_j=cal_parameters[ind.cell_name][(scen_idx, phase)]['beta_j'],
		theta=cal_parameters[ind.cell_name][(scen_idx, phase)]['theta'],
		beta_behave=cal_parameters[ind.cell_name][(scen_idx, phase)][
			'beta_behave'],
		scen=scen,
		seasonality=seasonality,
		phi=phi,
	)

	model.reset()
	if state == 'H':
		state_str = 'hosp'
	elif state == 'Vents':
		state_str = 'vents'
	res_fit_track = model.fit_tracking(
		p0=p0,
		bnds=bnds,
		data=data.values,
		date_lst=date_lst,
		days_in_season=len(date_lst),
		start=start_data,
		end=end_data,
		loss_func='MSE',
		maxiter=85,
		factor=1,
		tracking=state_str
	)

	save_cal_track(res_fit_track, ind, scen, phase, state, no_mobility, no_haredim)
	print_stat_fit_hosp(res_fit_track, tracking=state_str)


def load_data(ind, region_250_to_30, region_age_250_to_30):
	first = pd.Timestamp('2020-03-03')
	last = pd.Timestamp('2020-05-01')
	tot = pd.read_csv('../Data/sick/all_tests_by_age_county.csv')
	pos = pd.read_csv('../Data/sick/smooth_sick_by_age_calibration_county.csv')
	pos.set_index('Unnamed: 1', inplace=True)
	pos.columns = ['county'] + list(pos.columns)[1:]
	pos.sort_values(by='county', inplace=True)
	pos = pos.pivot(columns='county')
	pos.columns = pos.columns.swaplevel(0, 1)
	pos = pos.reindex(
		pd.MultiIndex.from_product([region_250_to_30.keys(), ind.A.values()]),
		axis=1)
	pos.index = pd.to_datetime(pos.index)
	pos = pos.reindex(
		pd.date_range(pos.index.values.min(), pos.index.values.max(),
					  freq='d'), fill_value=0)
	pos.fillna(0, inplace=True)

	begining = pos.index.values.min()
	ending = pos.index.values.max()
	pos = pos.reindex(pd.date_range(begining, ending, freq='d'), fill_value=0)

	# Shifting data by 5 days for fit
	pos_0303 = pos.loc[
			   '2020-03-03':].copy()  # starting the fit from March 3rd 2020

	shift_date_list = pd.date_range('2020-02-27', freq='d',
									periods=pos_0303.shape[0])

	pos_0303['shift_date'] = shift_date_list

	pos_0303.set_index('shift_date', inplace=True)
	data = np.ndarray((2, len(pos_0303), len(region_age_250_to_30.keys())))
	data[0] = np.zeros(
		pos_0303.shape)  # insert the tested positive specimens only for relevant calibration period
	data[1] = pos_0303

	return (data, pos_0303)


def show_calibration(
		ind,
		scen_idx=1,
		phase='-',
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):

	show_calibration_param(
		ind,
		scen_idx=scen_idx,
		phase=phase,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)

	show_calibration_track(
		scen=num2scen(scen_idx),
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)


def show_calibration_param(
		ind,
		scen_idx=1,
		phase='-',
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):
	print('Results for ', scen_idx, ' and for seasonality ', phase, ':')
	# parameters:
	# load mapper:
	# loads mapping 250 to 20:
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_mapper.pickle',
			'rb') as pickle_in:
		region_250_to_30 = pickle.load(pickle_in)
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_age_mapper.pickle',
			'rb') as pickle_in:
		region_age_250_to_30 = pickle.load(pickle_in)
	with open(
			'../Data/division_choice/250_to_30/250_to_30_region_ga_mapper.pickle',
			'rb') as pickle_in:
		region_ga_250_to_30 = pickle.load(pickle_in)

	# get the sick data:
	data, pos = load_data(ind, region_250_to_30, region_age_250_to_30)

	start_data = pos.index[0]
	end_data = pos.index[-1]
	# fit params
	date_lst = pd.date_range(start_sim, end_data)
	if (phase is None) or (phase == '-'):
		seasonality = False
		phi = 0
	else:
		seasonality = True
		phi = phase

	cal_parameters = pd.read_pickle(
		'../Data/calibration/calibration_dict.pickle')
	if no_haredim:
		cal_tpl = (scen_idx, phase, 'no_haredim')
	elif no_mobility:
		cal_tpl = (scen_idx, phase, 'no_mobility')
	else:
		cal_tpl = (scen_idx, phase)
	model = Model_behave(
		ind=ind,
		beta_j=cal_parameters[ind.cell_name][cal_tpl]['beta_j'],
		theta=cal_parameters[ind.cell_name][cal_tpl]['theta'],
		beta_behave=cal_parameters[ind.cell_name][cal_tpl][
			'beta_behave'],
		scen=num2scen(scen_idx),
		seasonality=seasonality,
		phi=phi,
	)
	if no_mobility:
		C = {}
		sh = {}
		for key in C_calibration.keys():
			C[key] = [np.ones_like(C_calibration[key][0])] * len(
				C_calibration[key])
		for key in stay_home_idx.keys():
			sh[key] = [np.ones_like(stay_home_idx[key][0])] * len(
				stay_home_idx[key])
	else:
		C = C_calibration
		sh = stay_home_idx
	loss = model.calc_loss(
		(model.beta_j[0],
		 model.beta_j[3],
		 model.beta_j[5],
		 model.beta_j[7],
		 model.theta,
		 model.beta_behave,),
		data,
		C,
		sh,
		not_routine,
		date_lst,
		start_data,
		end_data,
		'MSE',
		factor=1,
		mapper=region_age_250_to_30,
	)

	print(
		'Fit Loss: {3}\nFitted parameters:\n Beta={0}\n Theta={1},\n,Beta_behave={2}, '.format(
			cal_parameters[ind.cell_name][cal_tpl]['beta_j'][[0, 3, 5, 7]],
			cal_parameters[ind.cell_name][cal_tpl]['theta'],
			cal_parameters[ind.cell_name][cal_tpl][
				'beta_behave'],
			loss,)
	)

	# generate fitting graph:
	res_mdl_ml = model.predict(
		C=C,
		days_in_season=len(date_lst),
		stay_home_idx=sh,
		not_routine=not_routine,
	)

	plot_calibrated_total_model(
		pd.DataFrame(data[1]) / pop_israel,
		res_mdl_ml, date_lst,
		start=start_data,
		end=end_data
	)
	plt.vlines(x=14, ymin=0, ymax=0.0001)
	plt.vlines(x=23 - 5, ymin=0, ymax=0.0001, color='r')

	# plotting the data and model
	fig, ax = plot_calibrated_model_region(
		mdl_mapper=region_250_to_30,
		data=data,
		mdl_data=res_mdl_ml['new_Is'],
		date_list=date_lst,
		start=start_data,
		end=end_data,
		region_name={k: k for k in
					 region_250_to_30.keys()},
		loss_func='MSE',
		ind=ind,
		data_mapper=region_ga_250_to_30
	)
	fig.set_size_inches((30, 30))
	plt.tight_layout()
	plt.show()

	# plot age fit:
	age_cal = [
		['0-4', '5-9', '10-19'],
		['20-29', '30-39'],
		['40-49', '50-59'],
		['60-69', '70+'],
	]
	age_name = [
		'0-19',
		'20-39',
		'40-59',
		'60+',
	]
	age_pct = []
	age_pct_data = []
	age_model = []
	data_age = (pos.sum(axis=0).unstack(level=0).sum(axis=1)) / (
		pos.sum(axis=0).unstack(level=0).sum(axis=1).sum())
	for age_list in age_cal:
		idx = []
		for age in age_list:
			idx += list(ind.age_dict[age])
		age_pct.append(res_mdl_ml['new_Is'][:, idx].sum() / res_mdl_ml[
			'new_Is'].sum())
		age_pct_data.append(data_age.loc[age_list].sum())

	# set width of bar
	barWidth = 0.25

	# set height of bar

	# Set position of bar on X axis
	r1 = np.arange(len(age_pct))
	r2 = [x - barWidth for x in r1]
	r3 = [x + barWidth for x in r2]

	# Make the plot
	plt.bar(r2, age_pct, color='#557f2d', width=barWidth, edgecolor='white',
			label='model')
	plt.bar(r3, age_pct_data, color='#2d7f5e', width=barWidth,
			edgecolor='white', label='data')

	# Add xticks on the middle of the group bars
	plt.xlabel('group', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(age_name))], age_name)

	# Create legend & Show graphic
	plt.legend()
	plt.show()


def show_calibration_track(
		scen,
		phase,
		ind,
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):

	df = pd.read_csv('../Data/sick/daily_hospital_resp.csv')
	df.index = pd.to_datetime(df['Date'], dayfirst=True)
	resp = df['resp'].dropna()
	hosp = df['hospitalized'].dropna()

	show_calibration_track_spec(
		state='H',
		data=hosp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)

	show_calibration_track_spec(
		state='Vents',
		data=resp,
		scen=scen,
		phase=phase,
		ind=ind,
		start_sim=start_sim,
		no_mobility=no_mobility,
		no_haredim=no_haredim,
	)



def show_calibration_track_spec(
		state,
		data,
		scen,
		phase,
		ind,
		start_sim='2020-02-20',
		no_mobility=False,
		no_haredim=False,
	):

	# fit params
	if (phase is None) or (phase == '-'):
		seasonality = False
		phi = 0
	else:
		seasonality = True
		phi = phase

	scen_idx = int(scen[-1])
	start_data = data.index[0]
	end_data = data.index[-1]
	date_lst = pd.date_range(start_sim, end_data)

	print('Results for ', scen_idx, ' and for seasonality ', phase, 'for tracking ', state, ':')

	cal_parameters = pd.read_pickle(
		'../Data/calibration/calibration_dict.pickle')

	if no_haredim:
		cal_tpl = (scen_idx, phase, 'no_haredim')
	elif no_mobility:
		cal_tpl = (scen_idx, phase, 'no_mobility')
	else:
		cal_tpl = (scen_idx, phase)

	print(state)
	if state == 'H':
		state_str = 'hosp'
		eta = cal_parameters[ind.cell_name][cal_tpl]['eta']
		nu = cal_parameters[ind.cell_name][cal_tpl]['nu']
		xi=9
		mu=0
		param_tpl = (eta, nu)

	elif state == 'Vents':
		print(cal_parameters[ind.cell_name][cal_tpl]['xi'])
		state_str = 'vents'
		xi = cal_parameters[ind.cell_name][cal_tpl]['xi']
		mu = cal_parameters[ind.cell_name][cal_tpl]['mu']
		eta=0
		nu=0
		param_tpl = (xi, mu)

	model = Model_behave(
		ind=ind,
		beta_j=cal_parameters[ind.cell_name][cal_tpl]['beta_j'],
		theta=cal_parameters[ind.cell_name][cal_tpl]['theta'],
		beta_behave=cal_parameters[ind.cell_name][cal_tpl][
			'beta_behave'],
		eta=eta,
		nu=nu,
		xi=xi,
		mu=mu,
		scen=scen,
		seasonality=seasonality,
		phi=phi,
	)

	model.reset()

	if no_mobility:
		C = {}
		sh = {}
		for key in C_calibration.keys():
			C[key] = [np.ones_like(C_calibration[key][0])] * len(
				C_calibration[key])
		for key in stay_home_idx.keys():
			sh[key] = [np.ones_like(stay_home_idx[key][0])] * len(
				stay_home_idx[key])
	else:
		C = C_calibration
		sh = stay_home_idx

	loss = model.calc_loss_tracking(
		param_tpl,
		data.values,
		C,
		len(date_lst),
		sh,
		not_routine,
		date_lst,
		start_data,
		end_data,
		'MSE',
		1,
		state_str,
	)

	if state_str == 'hosp':
		print(
			'Fit Loss: {2}\nFitted parameters:\n Eta={0}\n Nu={1},\n '.format(
				cal_parameters[ind.cell_name][cal_tpl]['eta'],
				cal_parameters[ind.cell_name][cal_tpl]['nu'],
				loss
			)
		)
	elif state_str == 'vents':
		print('Fit Loss: {2}\nFitted parameters:\n Xi={0}\n Mu={1},\n '.format(
			cal_parameters[ind.cell_name][cal_tpl]['xi'],
			cal_parameters[ind.cell_name][cal_tpl]['mu'],
			loss,
		))

	res_mdl = model.predict(
		C=C,
		days_in_season=200,
		stay_home_idx=sh,
		not_routine=not_routine,
	)
	plot_hospitalizations_calibration(
		res_mdl, data, date_lst,
		start_date=start_data,
		end_date=end_data,
		tracking=state_str,
	)
