from .indices import *
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