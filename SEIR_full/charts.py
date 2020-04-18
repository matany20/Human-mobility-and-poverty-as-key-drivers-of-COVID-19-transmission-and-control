from .indices import *
from .parameters import *
from matplotlib import pyplot as plt
import datetime
import numpy as np
import pandas as pd
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
	Ia = mdl_res['Is']

	# dictionary of arrays to plot
	plot_dict ={}

	if with_asym:
		for age in A.values():
			plot_dict[age + ' sym'] = Is[:, age_dict[age]].sum(axis=1)
			plot_dict[age + ' asym'] = Ia[:, age_dict[age]].sum(axis=1)

	elif sym_only:
		for age in A.values():
			plot_dict[age + ' sym'] = Is[:, age_dict[age]].sum(axis=1)

	else:
		for age in A.values():
			plot_dict[age] = Is[:, age_dict[age]].sum(axis=1) + \
							 Ia[:, age_dict[age]].sum(axis=1)

	fig = plt.figure(figsize=(10,10))
	ax = plt.subplot()
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
		):
	""" The function gets the results of the model and plot for each region
     the model results and the data normalized by region population. Data format is of region-age
	:param data:
	:param mdl_data:
	:param date_list:
	:param start:
	:param end:
	:param region_name:
	:return:
	"""

	#index to cut model's data
	start_idx = int(np.where(date_list == start)[0])
	end_idx = int(np.where(date_list == end)[0])
	plot_dict = {}
	for key in region_dict.keys():
		plot_dict[key + '_mdl'] = mdl_data[start_idx:end_idx+1, region_dict[key]].sum(axis=1) / \
								  population_size[region_dict[key]].sum()
		plot_dict[key + '_dt'] = data[:, region_ga_dict[key]].sum(axis=1) / \
								 population_size[region_dict[key]].sum()

	plot_df = pd.DataFrame.from_dict(plot_dict)
	plot_df.set_index(date_list[start_idx:end_idx+1],inplace=True)

	fig, axes = plt.subplots(np.ceil(len(region_dict))/4, 4)

	for ax, key in zip(axes.flat, region_dict.keys()):

		plot_df.plot(y=[key + '_mdl', key + '_dt'], style=['-', '.'],
					 ax=ax,label=['Model', 'Data'])
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
