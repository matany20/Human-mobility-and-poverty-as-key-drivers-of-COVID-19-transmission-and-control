from .indices import *
from .parameters import *
from matplotlib import pyplot as plt
import datetime
import numpy as np
import pandas as pd
from .helper_func import *
from matplotlib.patches import Patch


########################
# -- Plot functions -- #
########################

def plot_I_by_age(
		mdl_res,
		with_asym=False,
		sym_only=False,
	):
	"""

	:param mdl_res:
	:param with_asym:
	:param sym_only:
	:return:
	"""

	Is = mdl_res['Is']
	Ia = mdl_res['Ia']

	# dictionary of arrays to plot
	plot_dict ={}
	if with_asym:
		for age in A.values():
			ylabel = 'sym/asym'
			plot_dict[age + ' sym'] = Is[:, age_dict[age]].sum(axis=1)*pop_israel
			plot_dict[age + ' asym'] = Ia[:, age_dict[age]].sum(axis=1)*pop_israel

	elif sym_only:
		ylabel = 'sym only'
		for age in A.values():
			plot_dict[age + ' sym'] = Is[:, age_dict[age]].sum(axis=1)*pop_israel

	else:
		ylabel = 'all'
		for age in A.values():
			plot_dict[age] = (Is[:, age_dict[age]].sum(axis=1) + \
							 Ia[:, age_dict[age]].sum(axis=1))*pop_israel

	fig = plt.figure(figsize=(15,10))
	ax = plt.subplot()
	ax.set_ylabel('new Spreaders cases (' + ylabel + ') [#]', fontsize=35)
	ax.set_title('Spreaders Cases Global by Age', fontsize=50)
	ax.set_xlabel('Time [d]', fontsize=35)
	plot_df = pd.DataFrame.from_dict(plot_dict)

	# plot
	plot_df.plot(ax=ax)
	plt.show()
	plt.close()
	return fig, ax


def plot_R_by_age(
		mdl_res,
		with_asym=False,
		sym_only=False,
	):
	"""

	:param mdl_res:
	:param with_asym:
	:param sym_only:
	:return:
	"""

	R = mdl_res['R']

	# dictionary of arrays to plot
	plot_dict ={}
	for age in A.values():
		plot_dict[age] = R[:, age_dict[age]].sum(axis=1)*pop_israel

	fig = plt.figure(figsize=(15,10))
	ax = plt.subplot()
	ax.set_ylabel('Total Recovered cases [#]', fontsize=35)
	ax.set_title('Recovered Cases Global by Age', fontsize=50)
	ax.set_xlabel('Time [d]', fontsize=35)
	plot_df = pd.DataFrame.from_dict(plot_dict)

	# plot
	plot_df.plot(ax=ax)
	plt.show()
	plt.close()
	return fig, ax


def plot_I_by_age_region(
		mdl_res,
		with_asym=False,
		sym_only=False,
	):
	"""

	:param mdl_res:
	:param with_asym:
	:param sym_only:
	:return:
	"""

	Is = mdl_res['Is']
	Ia = mdl_res['Is']

	plot_dict ={} # dictionary of arrays to plot

	if with_asym:
		fig, axes = plt.subplots(3, 3)
		for ax, groups in zip(axes.flat, range(9)):
			plot_dict = {}
			for age in A.values():
				for s in G.values():
					plot_dict[s + ' sym'] = Is[
												:,
												region_age_dict[s ,age],
											].sum(axis=1)
					plot_dict[s + ' asym'] = Ia[
											 	:,
											 	region_age_dict[s ,age],
											 ].sum(axis=1)

			plot_df = pd.DataFrame.from_dict(plot_dict)
			plot_df.plot(ax=ax, title=age)

	elif sym_only:
		fig, axes = plt.subplots(3, 3)
		for ax, groups in zip(axes.flat, range(9)):
			plot_dict = {}
			for age in A.values():
				for s in G.values():
					plot_dict[s + ' sym'] = Is[
												:,
												region_age_dict[s ,age],
											].sum(axis=1)

			plot_df = pd.DataFrame.from_dict(plot_dict)
			plot_df.plot(ax=ax, title=age)

	else:
		fig, axes = plt.subplots(3, 3)
		for ax, groups in zip(axes.flat, range(9)):
			plot_dict = {}
			for age in A.values():
				for s in G.values():
					plot_dict[s] = Is[:, age_dict[age]].sum(axis=1) + \
								   Ia[:, age_dict[age]].sum(axis=1)

			plot_df = pd.DataFrame.from_dict(plot_dict)
			plot_df.plot(ax=ax)
			ax.get_legend().remove()

	plt.tight_layout()
	plt.show()
	plt.close()

	return fig, axes


def plot_calibrated_model(
		data,
		mdl_data,
		date_list,
		season_length
	):
	"""
	The function gets the results of the model and plot for each age group the
	model results and the data.
	:param data:
	:param mdl_data:
	:param date_list:
	:param season_length:
	:return:
	"""

	model_tot_dt = np.zeros((season_length + 1, len(A)))
	# Calculated total symptomatic (high+low) per age group (adding as columns)
	plot_dict = {}
	plot_dict['dates'] = date_list
	for i, age_group in enumerate(age_dict.keys()):
		model_tot_dt[:, i] = mdl_data[:, age_dict[age_group]].sum(axis=1)
		plot_dict[A[i] + '_mdl'] = mdl_data[
									:len(date_list),
									age_dict[age_group],
								   ].sum(axis=1)
		plot_dict[A[i] + '_dt'] = data[:, i]

	plot_df = pd.DataFrame.from_dict(plot_dict)

	fig, axes = plt.subplots(3, 3, figsize=(16, 10))

	for ax, groups in zip(axes.flat, range(9)):
		plot_df.plot(
			x='dates',
			y=[A[groups] + '_mdl', A[groups] + '_dt'],
			style=['-', '.'],
			ax=ax
		)
		ax.set_xticklabels(
			labels=plot_df.dates.values[::5],
            rotation=70,
			rotation_mode="anchor",
			ha="right"
		)
	plt.ticklabel_format(axis='y', style='sci', useMathText=True)
	plt.tight_layout()
	plt.show()
	plt.close()

def plot_calibrated_model_region(
		data,
		mdl_data,
		date_list,
		region_name,
		start='2020-03-20',
		end='2020-04-13',
		loss_func='MSE'
		):
	""" The function gets the results of the model and plot for each region
     the model results and the data normalized by region population. Data format is of region-age
	:param data:
	:param mdl_data:
	:param date_list:
	:param start:
	:param end:
	:param region_name:
	:param loss_func:
	:return:
	"""
	if loss_func == "MSE":
		# fixing data to be proportion of israel citizens
		data_specific = data[1] / 9136000
	elif loss_func == "BIN" or loss_func == "POIS":
		# fixing data to be proportion of tests
		data_specific = data.copy()
		data_specific[1] = (data[1] / data[0]).fillna(0).replace(
			[np.inf, -np.inf], 0)
		# fixing model_out to be proportion of cell j,k
		pop_jk = shrink_array_sum(region_age_dict, population_size)
		mdl_data_specific = mdl_data.copy()
		mdl_data_specific = mdl_data_specific / pop_jk
	elif loss_func == 'POIS_NAIV':
		# fixing data to be proportion of israel citizens
		data_specific = data[1]
		# fixing model output to be new sick people
		mdl_data_specific = mdl_data * 9136000

	#index to cut model's data
	start_idx = int(np.where(date_list == start)[0])
	end_idx = int(np.where(date_list == end)[0])
	plot_dict = {}
	for key in region_dict.keys():
		plot_dict[key + '_mdl'] = mdl_data[start_idx:end_idx+1, region_dict[key]].sum(axis=1) / \
								  population_size[region_dict[key]].sum()
		plot_dict[key + '_dt'] = data_specific[:, region_ga_dict[key]].sum(axis=1) / \
								 population_size[region_dict[key]].sum()

	plot_df = pd.DataFrame.from_dict(plot_dict)
	plot_df.set_index(date_list[start_idx:end_idx+1],inplace=True)

	fig, axes = plt.subplots(int(np.ceil(len(region_dict)/3)), 3, figsize=(15,15))

	for ax, key in zip(axes.flat, region_dict.keys()):

		plot_df.plot(y=[key + '_mdl', key + '_dt'],
					 style=['-', '.'],
					 # c=['b', 'r'],
					 linewidth=3,
					 markersize=12,
					 ax=ax,
					 label=['Model', 'Data'])
		ax.set_title('Region {}'.format(region_name[key]))
		ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=20, rotation_mode="anchor", ha="right")
		ax.legend(frameon=False)
		# ax.set_ylim(0,0.001)
		ax.ticklabel_format(axis='y', style='sci', useMathText=True, scilimits=(0,0))

	return fig, axes

def plot_calibrated_total_model(data,
								mdl_data,
								date_list,
								start='2020-03-20',
								end='2020-04-13'
								):

	""" The function gets the results of the model and plot the model results and the data,
	on country level.
	Data format is of region-age
	:param data:
	:param mdl_data:
	:param date_list:
	:param start:
	:param end:
	:return:
	"""

	#index to cut model's data
	start_idx = int(np.where(date_list == start)[0])
	end_idx = int(np.where(date_list == end)[0])

	# creating DF
	model_data_cal = mdl_data['new_Is'][start_idx:end_idx + 1,:].sum(axis=1)
	data_tot = data.sum(axis=1).to_frame()
	data_tot.columns = ['Data']
	data_tot['Model_tot'] = model_data_cal

	#Plot
	fig = plt.figure()
	ax = plt.subplot()
	data_tot.plot(style=['.', '-'], ax=ax)
	ax.set_title('Country level calibration plot')

	return fig, ax


def make_respiratory_warning(
		res_mdl,
		days_of_inter,
		time=14,
		thresh=700
):
	ubRespiration = res_mdl['H'].sum(axis=1)
	ubRespiration = ubRespiration[days_of_inter:] * pop_israel
	x = ubRespiration[time:]
	y = ubRespiration[:-time]
	max_idx = x.argmax()
	x = x[:max_idx]
	y = y[:max_idx]

	def find_y(x, y, value):
		before = x[x < value].max()
		after = x[x > value].min()
		idx = x[x < value].argmax()
		x_ratio = (value - before) / (after - before)
		return (y[idx + 1] - y[idx]) * x_ratio + y[idx]

	def plt_malben(ax, x, y, x_list, zorder):
		for thresh in x_list:
			if thresh < x.max():
				y_thresh = find_y(x, y, thresh)
				ax.axvline(x=thresh, ymin=0,
						   ymax=(y_thresh - y.min()) / (y.max() - x.min()), c='k',
						   linewidth=2, linestyle='--', zorder=zorder)
				ax.axhline(y=y_thresh, xmin=0,
						   xmax=(thresh - x.min()) / (x.max() - x.min()), c='k',
						   linewidth=2, linestyle='--', zorder=zorder)

	fig, ax = plt.subplots(figsize=(15, 10))

	plt.rcParams.update({'font.size': 40})
	ax.set_xlim([x.min(), x.max()])
	ax.set_ylim([y.min(), y.max()])
	ax.set_xlabel('רתומ םימשנומ ףס', fontsize=35)
	str1 = '\n' + ' הארתהל םימשנומ רפסמ'
	str2 = 'שארמ םימי' + str(time)
	ax.set_ylabel(str2 + str1, fontsize=35)
	ax.set_title('ןוחטב ףס', fontsize=50)
	xticks = list(range(thresh, int(x.min()), -200)) + list(
		range(thresh, int(x.max()), 200))
	ax.set_xticks(xticks)
	if thresh < x.max():
		ax.scatter(x=[thresh], y=[find_y(x, y, thresh)], c='r', s=400, zorder=4)

	ax.plot(x, y, c='k', linewidth=10, zorder=3)
	plt_malben(ax, x, y, range(thresh, thresh + 1200, 400), 2)
	plt.show()
	plt.rcParams.update({'font.size': 20})


def plot_respiration_cases(res_mdl):
	fig, ax = plt.subplots(figsize=(15, 10))
	ax.plot(((res_mdl['Vents']).sum(axis=1))*pop_israel)
	ax.set_ylabel('Resipratory cases [#]', fontsize=35)
	ax.set_title('Respiratory Cases Global', fontsize=50)
	ax.set_xlabel('Time [d]', fontsize=35)
	plt.show()


def plot_hospitalization_cases(res_mdl):
	fig, ax = plt.subplots(figsize=(15, 10))
	ax.plot(((res_mdl['H']).sum(axis=1))*pop_israel)
	ax.set_ylabel('Hospitalization cases [#]', fontsize=35)
	ax.set_title('Hospitalization Cases Global', fontsize=50)
	ax.set_xlabel('Time [d]', fontsize=35)
	plt.show()


def plot_hospitalizations_calibration(res_mdl,data,date_lst, start_date, end_date, tracking='hosp'):

	# index to cut model's data
	start_idx = int(np.where(date_lst == start_date)[0])
	end_idx = int(np.where(date_lst == end_date)[0])
	# print(start_idx)
	# print(end_idx)
	# creating DF
	if tracking == 'hosp':
		model_data_cal = res_mdl['H'][start_idx:end_idx + 1].sum(axis=1) * pop_israel
		y_label = 'Number of hospitalizations'
	elif tracking == 'vents':
		model_data_cal = res_mdl['Vents'][start_idx:end_idx + 1].sum(axis=1) * pop_israel
		y_label = 'Number of Ventilators in use'

	plot_dict={}
	plot_dict['Model'] = model_data_cal
	plot_dict['date'] = date_lst[start_idx:end_idx + 1]
	plot_dict['Data'] = data

	# print('len model:',len(model_data_cal))
	# print('len date:', len(date_lst[start_idx:end_idx+1]))
	# print('len data:', len(data))

	plot_df = pd.DataFrame.from_dict(plot_dict)
	plot_df.set_index('date',inplace=True)

	# Plot
	fig = plt.figure()
	ax = plt.subplot()
	plot_df.plot(style=['-', '.'], ax=ax)
	ax.set_title('Country level calibration plot')
	ax.set_ylabel(y_label)

	return fig, ax