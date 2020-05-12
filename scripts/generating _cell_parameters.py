## Imports
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
import scipy
from scipy import optimize
from scipy.sparse import csr_matrix
import sys
import os
from SEIR_full.indices import *

#############################################
# Generating parameters files based on tazs #
#############################################

cell_name = '250'
isr_pop = 9136000

# add functions
def make_pop(df):
	df = df.iloc[:, 0:-2]
	return df.sum(axis=0)


def make_pop_religion(df):
	df = df.iloc[:, 1:8].multiply(df['tot_pop'], axis='index')
	return df.sum(axis=0)


def robust_max(srs, n=3):
	sort = sorted(srs)
	return np.mean(sort[-n:])


def robust_min(srs, n=3):
	sort = sorted(srs)
	return np.mean(sort[:n])


def weighted_std(values, weights):
	average = np.average(values, weights=weights)
	variance = np.average((values - average) ** 2, weights=weights)
	return np.sqrt(variance)


def avg_by_dates(df, from_date, to_date, weights=None):
	filtered = df[(df.index >= from_date) & (df.index <= to_date)]
	if weights is None:
		return filtered.describe().T[['mean', 'std', 'min', 'max']]

	weights = pd.Series(weights)
	stats = filtered.describe().T[['min', 'max']]
	stats['mean'] = filtered.apply(
		lambda col: np.average(col, weights=weights))
	stats['std'] = filtered.apply(
		lambda col: weighted_std(col, weights=weights))
	return stats


def wheighted_average(df):
	tot = df['tot_pop'].sum()
	return (df['cases_prop'].sum() / tot)


def create_demograph_age_dist_empty_cells(ind):
	### Creating demograph/age_dist
	pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2)
	ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]
	pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
	pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
	pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
	pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
	pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
		pop_dist['id']).join(pop_dist['tot_pop'])
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
	pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
	pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'],
													  axis=0)

	taz2cell = pd.read_excel(
		'../Data/division_choice/' + ind.cell_name + '/taz2cell.xlsx')
	taz2cell = taz2cell[['taz_id', 'cell_id']]
	taz2cell.columns = ['id', 'new_id']

	pop_cell = pop_dist.merge(taz2cell, left_on='id', right_on='id')
	pop_cell['new_id'] = pop_cell['new_id'].astype(str)
	pop_cell.sort_values(by='new_id')

	pop_cell = pop_cell.groupby(by='new_id').apply(lambda df: make_pop(df))
	pop_cell['10-19'] = pop_cell['10-14'] + pop_cell['15-19']
	pop_cell['20-29'] = pop_cell['20-24'] + pop_cell['25-29']
	pop_cell['30-39'] = pop_cell['30-34'] + pop_cell['35-39']
	pop_cell['40-49'] = pop_cell['40-44'] + pop_cell['45-49']
	pop_cell['50-59'] = pop_cell['50-54'] + pop_cell['55-59']
	pop_cell['60-69'] = pop_cell['60-64'] + pop_cell['65-69']
	pop_cell['70+'] = pop_cell['70-74'] + pop_cell['75+']
	pop_cell = pop_cell[list(ind.A.values())]
	pop_cell = pop_cell / pop_cell.sum().sum()
	pop_cell.reset_index(inplace=True)
	pop_cell.columns = ['cell_id'] + list(ind.A.values())

	## empty cells file to save
	try:
		os.mkdir('../Data/demograph')
	except:
		pass
	empty_cells = pop_cell[pop_cell.sum(axis=1) == 0]['cell_id']
	empty_cells.to_csv('../Data/demograph/empty_cells.csv')

	empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
		'cell_id'].astype(str)
	pop_cell = pop_cell[
		pop_cell['cell_id'].apply(lambda x: x not in empty_cells.values)]
	pop_cell.to_csv('../Data/demograph/age_dist_area.csv')


def create_paramaters_ind(ind):
	ind.update_empty()
	## empty cells file to save
	try:
		os.mkdir('../Data/parameters')
	except:
		pass
	with open('../Data/parameters/indices.pickle', 'wb') as handle:
		pickle.dump(ind, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return ind


def create_demograph_religion(ind):
	### creating demograph/religion
	religion2taz = pd.read_csv('../Data/raw/religion2taz.csv')
	religion2taz.sort_values(by='taz_id', inplace=True)
	religion2taz.columns = ['id', 'Orthodox', 'Druze', 'Other', 'Sacular',
							'Muslim', 'Christian']
	religion2taz['Jewish'] = religion2taz['Orthodox'] + religion2taz['Sacular']
	taz2cell = pd.read_excel(
		'../Data/division_choice/' + ind.cell_name + '/taz2cell.xlsx')
	taz2cell = taz2cell[['taz_id', 'cell_id']]
	taz2cell.columns = ['id', 'new_id']
	religion2taz = religion2taz.merge(taz2cell, on='id')
	religion2taz['new_id'] = religion2taz['new_id'].astype(str)
	religion2taz.sort_values(by='new_id', inplace=True)
	pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2)
	ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]

	pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
	pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
	pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
	pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
	pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
		pop_dist['id']).join(pop_dist['tot_pop'])
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
	pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
	pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'],
													  axis=0)
	pop_dist = pop_dist[['id', 'tot_pop']]

	religion2taz = religion2taz.merge(pop_dist, on='id')
	religion2taz.sort_values(by='id', inplace=True)

	# fixing religion city factor
	if ind.cell_name == '20':
		cell_num = len(list(set(religion2taz['new_id'])))
		factor = pd.DataFrame({'new_id': list(set(religion2taz['new_id'])),
							   'orth_factor': [1] * cell_num,
							   'arab_factor': [1] * cell_num, }).sort_values(
			by='new_id')
		factor = factor.reset_index().drop(['index'], axis=1)
		factor.iloc[:, 1] = pd.Series(
			[1,
			 0.48 / 0.41,
			 0.13 / 0.04,
			 0.05 / 0.02,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1,
			 0.1 / 0.05,
			 1,
			 1,
			 1,
			 0.82 / 0.6,
			 1,
			 1,
			 1,
			 1,
			 0.24 / 0.36])
		factor.iloc[:, 2] = pd.Series(
			[0.38 / 0.01,
			 1,
			 0.106 / 0.01,
			 0.36 / 0.14,
			 0.65 / 0.4,
			 1.1 / 0.38,
			 1.1 / 0.1,
			 0.15 / 0.07,
			 0.6 / 0.3,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1,
			 1.3 / 0.8,
			 1])

		religion2taz = religion2taz.merge(factor, on='new_id')
		religion2taz['Orthodox'] = religion2taz['Orthodox'] * religion2taz[
			'orth_factor']
		religion2taz['Sacular'] = religion2taz['Sacular'] - religion2taz[
			'Orthodox'] * (religion2taz['orth_factor'] - 1)
		religion2taz['Muslim'] = religion2taz['Muslim'] * religion2taz[
			'arab_factor']

	religion2taz = religion2taz.groupby(by='new_id').apply(make_pop_religion)
	tmp = religion2taz[
		['Druze', 'Other', 'Muslim', 'Christian', 'Jewish']].sum(axis=1)
	tmp.loc[tmp == 0] = 1
	religion2taz = religion2taz.divide(tmp, axis=0)
	religion2taz.reset_index(inplace=True)
	religion2taz.columns = ['cell_id', 'Orthodox', 'Druze', 'Other', 'Sacular',
							'Muslim', 'Christian', 'Jewish']
	religion2taz['cell_id'] = religion2taz['cell_id'].astype(str)
	empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
		'cell_id'].astype(str)
	religion2taz = religion2taz[
		religion2taz['cell_id'].apply(lambda x: x not in empty_cells.values)]
	religion2taz.to_csv('../Data/demograph/religion_dis.csv')


def create_stay_home(ind):
	## Creating stay_home/ALL
	home = pd.read_csv('../Data/raw/Summary_Home_0_TAZ.txt', delimiter='\t',
					   encoding='utf-16')
	home.columns = ['date', 'taz_id', 'stay', 'out']
	home['date'] = pd.to_datetime(home['date'], dayfirst=True)
	home['stay'] = home['stay'].apply(lambda x: x.replace(',', '')).astype(int)
	home['out'] = home['out'].apply(lambda x: x.replace(',', '')).astype(int)
	home['total'] = home['stay'] + home['out']
	home['out_pct'] = home['out'] / home['total']

	taz2cell = pd.read_excel(
		'../Data/division_choice/' + ind.cell_name + '/taz2cell.xlsx')
	home = home.merge(taz2cell, on='taz_id')
	home['cell_id'] = home['cell_id'].astype(str)
	empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
		'cell_id'].astype(str)
	home = home[
		home['cell_id'].apply(lambda x: x not in empty_cells.values)]

	home_cell = home.groupby(['date', 'cell_id'])[
		['stay', 'out', 'total']].sum().reset_index()
	home_cell['out_pct'] = home_cell['out'] / home_cell['total']
	pivoted = pd.pivot_table(home_cell, index='date', columns='cell_id',
							 values='out_pct')
	pivoted[pivoted.index >= '2020-03-07'][
		np.random.choice(pivoted.columns, 5)].plot()
	global_max = pivoted.apply(robust_max)
	global_min = pivoted.apply(robust_min)
	span = global_max - global_min
	relative_rate = pivoted.apply(lambda row: (row - global_min) / span,
								  axis=1)

	result = dict()
	result['routine'] = avg_by_dates(relative_rate, '2020-02-02', '2020-02-29')
	result['no_school'] = avg_by_dates(relative_rate, '2020-03-14',
									   '2020-03-16',
									   weights={'2020-03-14': 2 / 7,
												'2020-03-15': 2.5 / 7,
												'2020-03-16': 2.5 / 7})
	result['no_work'] = avg_by_dates(relative_rate, '2020-03-17', '2020-03-25',
									 weights={
									 i: 1 / 14 if i.day in [17, 18, 24,
															25] else 1 / 7
									 for i in pd.date_range('2020-03-17',
															'2020-03-25')})
	result['no_100_meters'] = avg_by_dates(relative_rate, '2020-03-26',
										   '2020-04-02',
										   weights={i: 1 / 14 if i.day in [26,
																		   2] else 1 / 7
													for i in
													pd.date_range('2020-03-26',
																  '2020-04-02')})
	result['no_bb'] = avg_by_dates(relative_rate, '2020-04-03', '2020-04-06',
								   weights={
								   i: 5 / 14 if i.day in [5, 6] else 1 / 7
								   for i in
								   pd.date_range('2020-04-03', '2020-04-06')})
	# save
	try:
		os.mkdir('../Data/stay_home')
	except:
		pass
	result['routine'].to_csv('../Data/stay_home/routine.csv')
	result['no_school'].to_csv('../Data/stay_home/no_school.csv')
	result['no_work'].to_csv('../Data/stay_home/no_work.csv')
	result['no_100_meters'].to_csv('../Data/stay_home/no_100_meters.csv')
	result['no_bb'].to_csv('../Data/stay_home/no_bb.csv')


def create_demograph_sick_pop(ind):
	### Creating demograph/sick_pop.csv
	taz2sick = pd.read_csv('../Data/sick/taz2sick.csv')

	taz2cell = pd.read_excel(
		'../Data/division_choice/' + ind.cell_name + '/taz2cell.xlsx')
	pop_dist = pd.read_excel('../Data/raw/pop2taz.xlsx', header=2)
	ages_list = ['Unnamed: ' + str(i) for i in range(17, 32)]
	pop_dist = pop_dist[['אזור 2630', 'גילאים'] + ages_list]
	pop_dist.columns = ['id'] + list(pop_dist.iloc[0, 1:])
	pop_dist = pop_dist.drop([0, 2631, 2632, 2633])
	pop_dist['tot_pop'] = pop_dist.iloc[:, 1:].sum(axis=1)
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 0] = 1
	pop_dist = pop_dist.iloc[:, 1:-1].div(pop_dist['tot_pop'], axis=0).join(
		pop_dist['id']).join(pop_dist['tot_pop'])
	pop_dist['tot_pop'].loc[pop_dist['tot_pop'] == 1] = 0
	pop_dist['tot_pop'] = pop_dist['tot_pop'] / pop_dist['tot_pop'].sum()
	pop_dist.iloc[:, :-2] = pop_dist.iloc[:, :-2].mul(pop_dist['tot_pop'],
													  axis=0)
	pop_dist = pop_dist[['id', 'tot_pop']]

	taz2sick = taz2sick.merge(taz2cell, on='taz_id')
	taz2sick = taz2sick.merge(pop_dist, left_on='taz_id', right_on='id')
	taz2sick['cell_id'] = taz2sick['cell_id'].astype(str)
	empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
		'cell_id'].astype(str)
	taz2sick = taz2sick[
		taz2sick['cell_id'].apply(lambda x: x not in empty_cells.values)]
	# taz2sick['cases_prop'] = taz2sick['cases_prop'] * taz2sick['tot_pop']

	taz2sick = taz2sick.groupby(by='cell_id')[['cases_prop']].apply(
		sum)
	taz2sick.name = 'cases_prop'
	taz2sick.to_csv('../Data/demograph/sick_prop.csv')


def create_stay_idx_routine(ind):
	### Data loading
	# import data:
	stay_home_idx_school = pd.read_csv('../Data/stay_home/no_school.csv',
									   index_col=0)
	stay_home_idx_school.index = stay_home_idx_school.index.astype(str)
	stay_home_idx_work = pd.read_csv('../Data/stay_home/no_work.csv',
									 index_col=0)
	stay_home_idx_work.index = stay_home_idx_work.index.astype(str)
	stay_home_idx_routine = pd.read_csv('../Data/stay_home/routine.csv',
										index_col=0)
	stay_home_idx_routine.index = stay_home_idx_routine.index.astype(str)
	stay_home_idx_no_100_meters = pd.read_csv(
		'../Data/stay_home/no_100_meters.csv', index_col=0)
	stay_home_idx_no_100_meters.index = stay_home_idx_no_100_meters.index.astype(
		str)
	stay_home_idx_no_bb = pd.read_csv('../Data/stay_home/no_bb.csv',
									  index_col=0)
	stay_home_idx_no_bb.index = stay_home_idx_no_bb.index.astype(str)

	# reordering and expanding vector for each period:
	stay_home_idx_school = stay_home_idx_school['mean'].values
	stay_home_idx_school[1] = stay_home_idx_school[0]

	stay_home_idx_work = stay_home_idx_work['mean'].values
	stay_home_idx_work[1] = stay_home_idx_work[0]

	stay_home_idx_no_100_meters = stay_home_idx_no_100_meters['mean'].values
	stay_home_idx_no_100_meters[1] = stay_home_idx_no_100_meters[0]

	stay_home_idx_no_bb = stay_home_idx_no_bb['mean'].values
	stay_home_idx_no_bb[1] = stay_home_idx_no_bb[0]

	# expanding vectors:
	stay_home_idx_school = expand_partial_array(mapping_dic=ind.region_ga_dict,
												array_to_expand=stay_home_idx_school,
												size=len(ind.GA))
	stay_home_idx_work = expand_partial_array(mapping_dic=ind.region_ga_dict,
											  array_to_expand=stay_home_idx_work,
											  size=len(ind.GA))
	stay_home_idx_no_100_meters = expand_partial_array(
		mapping_dic=ind.region_ga_dict,
		array_to_expand=stay_home_idx_no_100_meters,
		size=len(ind.GA))
	stay_home_idx_no_bb = expand_partial_array(mapping_dic=ind.region_ga_dict,
											   array_to_expand=stay_home_idx_no_bb,
											   size=len(ind.GA))
	# preparing model objects:
	stay_idx_t = []
	routine_vector = []
	d_tot = 500

	# first days of routine from Feb 21st - March 13th
	d_routin = 9 + 13
	for i in range(d_routin):
		stay_idx_t.append(1.0)
		routine_vector.append(0)

	# first days of no school from March 14th - March 16th
	d_school = 3
	for i in range(d_school):
		stay_idx_t.append(stay_home_idx_school)
		routine_vector.append(1)

	# without school and work from March 17th - March 25th
	d_work = 9
	for i in range(d_work):
		stay_idx_t.append(stay_home_idx_work)
		routine_vector.append(1)

	# 100 meters constrain from March 26th - April 2nd
	d_100 = 8
	for i in range(d_100):
		stay_idx_t.append(stay_home_idx_no_100_meters)
		routine_vector.append(1)

	# Bnei Brak quaranrine from April 3rd - April 18th
	d_bb = 16
	for i in range(d_bb):
		stay_idx_t.append(stay_home_idx_no_bb)
		routine_vector.append(1)

	# Back to 30% market like no school and no work - April 19th forward
	for i in range(d_tot - (d_routin + d_school + d_work + d_100 + d_bb)):
		stay_idx_t.append(stay_home_idx_work)
		routine_vector.append(1)

	stay_idx_calibration = {
		'non_inter': {
			'work': stay_idx_t,
			'not_work': stay_idx_t
		},
		'inter': {
			'work': [0] * 500,
			'not_work': [0] * 500,
		}
	}

	routine_vector_calibration = {
		'non_inter': {
			'work': routine_vector,
			'not_work': routine_vector
		},
		'inter': {
			'work': [1] * 500,
			'not_work': [1] * 500,
		}
	}

	# save objects
	with open('../Data/parameters/stay_home_idx.pickle', 'wb') as handle:
		pickle.dump(stay_idx_calibration, handle,
					protocol=pickle.HIGHEST_PROTOCOL)

	with open('../Data/parameters/routine_t.pickle', 'wb') as handle:
		pickle.dump(routine_vector_calibration, handle,
					protocol=pickle.HIGHEST_PROTOCOL)


def create_full_matices(ind):
	### Full Matrixes
	with (
	open('../Data/division_choice/' + ind.cell_name + '/mat_macro_model_df.pickle',
		 'rb')) as openfile:
		OD_dict = pickle.load(openfile)

	base_leisure = pd.read_csv('../Data/raw/leisure_mtx.csv', index_col=0)
	base_work = pd.read_csv('../Data/raw/work_mtx.csv', index_col=0)
	base_school = pd.read_csv('../Data/raw/school_mtx.csv', index_col=0)

	religion_dist = pd.read_csv('../Data/demograph/religion_dis.csv',
								index_col=0)
	age_dist_area = pd.read_csv('../Data/demograph/age_dist_area.csv',
								index_col=0)
	home_secularism = pd.read_excel('../Data/raw/secularism_base_home.xlsx',
									index_col=0)
	home_haredi = pd.read_excel('../Data/raw/haredi_base_home.xlsx',
								index_col=0)
	home_arabs = pd.read_excel('../Data/raw/arabs_base_home.xlsx', index_col=0)

	# fix_shahaf_bug
	if ind.cell_name == '250':
		if len(str(OD_dict[list(OD_dict.keys())[0]].columns[0])) == 6:
			print('shahaf bug returned!!!!')
			for k in OD_dict.keys():
				OD_dict[k].columns = pd.Index(ind.G.values())
		if len(str(OD_dict[list(OD_dict.keys())[0]].index[0])) == 6:
			for k in OD_dict.keys():
				OD_dict[k].index = pd.Index(ind.G.values())

	# make sure index of area is string
	for k in OD_dict.keys():
		OD_dict[k].columns = OD_dict[k].columns.astype(str)
		OD_dict[k].index = OD_dict[k].index.astype(str)
		OD_dict[k] = OD_dict[k].filter(list(ind.G.values()), axis=1)
		OD_dict[k] = OD_dict[k].filter(list(ind.G.values()), axis=0)

	############ 21.2-14.3 #############
	full_leisure_routine = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['routine', 2],
		base_mat=base_leisure,
		age_dist_area=age_dist_area
	)

	############ 14.3-16.3 #############
	full_leisure_no_school = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_school', 2],
		base_mat=base_leisure,
		age_dist_area=age_dist_area
	)

	############ 17.3-25.3 #############
	full_leisure_no_work = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_work', 2],
		base_mat=base_leisure,
		age_dist_area=age_dist_area
	)

	############ 26.3-2.4 #############
	full_leisure_no_100_meters = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_100_meters', 2],
		base_mat=base_leisure,
		age_dist_area=age_dist_area
	)

	############ 3.4-6.4 #############
	full_leisure_no_bb = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_bb', 2],
		base_mat=base_leisure,
		age_dist_area=age_dist_area
	)

	# save matrix
	try:
		os.mkdir('../Data/base_contact_mtx')
	except:
		pass
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_routine.npz',
						  full_leisure_routine)
	scipy.sparse.save_npz(
		'../Data/base_contact_mtx/full_leisure_no_school.npz',
		full_leisure_no_school)
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_work.npz',
						  full_leisure_no_work)
	scipy.sparse.save_npz(
		'../Data/base_contact_mtx/full_leisure_no_100_meters.npz',
		full_leisure_no_100_meters)
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_leisure_no_bb.npz',
						  full_leisure_no_bb)

	# creating school- work matrix;
	base_work_school = base_work.copy()
	base_work_school.loc['0-4'] = base_school.loc['0-4']
	base_work_school.loc['5-9'] = base_school.loc['5-9']
	base_work_school['0-4'] = base_school['0-4']
	base_work_school['5-9'] = base_school['5-9']
	# creating eye matrix
	eye_OD = OD_dict['routine', 1].copy()

	for col in eye_OD.columns:
		eye_OD[col].values[:] = 0
	eye_OD.values[tuple([np.arange(eye_OD.shape[0])] * 2)] = 1

	############ 21.2-14.3 #############
	full_work_routine = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['routine', 1],
		base_mat=base_work_school,
		age_dist_area=age_dist_area,
		eye_mat=eye_OD
	)

	############ 14.3-16.3 #############
	full_work_no_school = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_school', 1],
		base_mat=base_work_school,
		age_dist_area=age_dist_area,
		eye_mat=eye_OD
	)

	############ 17.3-25.3 #############
	full_work_no_work = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_work', 1],
		base_mat=base_work_school,
		age_dist_area=age_dist_area,
		eye_mat=eye_OD
	)

	############ 26.3-2.4 #############
	full_work_no_100_meters = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_100_meters', 1],
		base_mat=base_work_school,
		age_dist_area=age_dist_area,
		eye_mat=eye_OD
	)

	############ 3.4-6.4 #############
	full_work_no_bb = create_C_mtx_leisure_work(
		ind=ind,
		od_mat=OD_dict['no_bb', 1],
		base_mat=base_work_school,
		age_dist_area=age_dist_area,
		eye_mat=eye_OD
	)

	# save matrix
	try:
		os.mkdir('../Data/base_contact_mtx')
	except:
		pass
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_routine.npz',
						  full_work_routine)
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_school.npz',
						  full_work_no_school)
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_work.npz',
						  full_work_no_work)
	scipy.sparse.save_npz(
		'../Data/base_contact_mtx/full_work_no_100_meters.npz',
		full_work_no_100_meters)
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_work_no_bb.npz',
						  full_work_no_bb)

	## Home Matices
	full_home = pd.DataFrame(
		index=pd.MultiIndex.from_tuples(list(ind.MI.values()),
										names=['age', 'area', 'age']),
		columns=OD_dict['routine', 0].index)

	religion_dist.set_index('cell_id', inplace=True)
	religion_dist.index = religion_dist.index.astype(str)

	# fill the matrix:
	for index in list(full_home.index):
		religion_area = religion_dist.loc[index[1]].copy()
		cell_val = religion_area['Orthodox'] * home_haredi.loc[index[0]][
			index[2]] + \
				   religion_area['Sacular'] * home_secularism.loc[index[0]][
					   index[2]] + \
				   religion_area['Christian'] * home_arabs.loc[index[0]][
					   index[2]] + \
				   religion_area['Other'] * home_secularism.loc[index[0]][
					   index[2]] + \
				   religion_area['Druze'] * home_arabs.loc[index[0]][
					   index[2]] + \
				   religion_area['Muslim'] * home_arabs.loc[index[0]][index[2]]
		full_home.loc[index] = (eye_OD.loc[index[1]] * cell_val) / \
							   age_dist_area[index[2]]

	full_home = csr_matrix(full_home.unstack().reorder_levels(
		['area', 'age']).sort_index().values.astype(float))
	# save matrix
	try:
		os.mkdir('../Data/base_contact_mtx')
	except:
		pass
	scipy.sparse.save_npz('../Data/base_contact_mtx/full_home.npz', full_home)


def create_parameters_indices(ind):
	with open('../Data/parameters/indices.pickle', 'wb') as handle:
		pickle.dump(ind, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_f0(ind):
	### Asymptomatic
	asymp = pd.read_csv('../Data/raw/asymptomatic_proportions.csv',
						index_col=0)
	f0_full = {}  # dict that contains the possible scenarios

	# asymptomatic with risk group, high risk with 0
	f_init = np.zeros(len(list(itertools.product(ind.R.values(), ind.A.values()))))
	for i in [1, 2, 3]:
		f_tmp = f_init.copy()
		f_tmp[9:] = asymp['Scenario ' + str(i)].values[:-1]
		f0_full['Scenario' + str(i)] = expand_partial_array(ind.risk_age_dict,
															f_tmp, len(ind.N))
	# Save
	try:
		os.mkdir('../Data/parameters')
	except:
		pass
	with open('../Data/parameters/f0_full.pickle', 'wb') as handle:
		pickle.dump(f0_full, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_eps_dict(ind, age_dist):
	### Initial illness
	# Age dist. positive specimens
	age_dist_area = pd.read_csv('../Data/demograph/age_dist_area.csv')
	age_dist_area.drop(['Unnamed: 0'], axis=1, inplace=True)
	age_dist_area.set_index('cell_id', inplace=True)
	age_dist_area = age_dist_area.stack()

	init_pop = expand_partial_array(ind.region_age_dict, age_dist_area.values,
									len(ind.N))
	init_pop[ind.inter_dict['Intervention']] = 0

	risk_pop = pd.read_csv('../Data/raw/risk_dist.csv')
	risk_pop.set_index('Age', inplace=True)
	risk_pop['High'] = risk_pop['risk']
	risk_pop['Low'] = 1 - risk_pop['risk']
	risk_pop.drop(['risk'], axis=1, inplace=True)
	risk_pop = risk_pop.stack()
	risk_pop.index = risk_pop.index.swaplevel(0, 1)
	risk_pop = risk_pop.unstack().stack()
	for (r, a), g_idx in zip(ind.risk_age_dict.keys(),
							 ind.risk_age_dict.values()):
		init_pop[g_idx] = init_pop[g_idx] * risk_pop[r, a]

	# Age distribution:
	pop_dist = init_pop
	# Save
	with open('../Data/parameters/init_pop.pickle', 'wb') as handle:
		pickle.dump(pop_dist, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# risk distribution by age:
	risk_dist = pd.read_csv('../Data/raw/population_size.csv')
	init_I_dis_italy = pd.read_csv('../Data/raw/init_i_italy.csv')[
						   'proportion'].values[:-1]
	f_init = pd.read_pickle('../Data/parameters/f0_full.pickle')
	eps_t = {}
	init_I_IL = {}
	init_I_dis = {}
	for i in [1, 2, 3]:
		scen = 'Scenario' + str(i)
		f_init_i = f_init[scen][:(len(ind.R) * len(ind.A))]
		init_I_IL[scen] = (491. / (1 - (
					f_init_i * risk_dist['pop size'].values).sum())) / 9136000.
		init_I_dis[scen] = init_I_dis_italy * init_I_IL[scen]
	for i in [1, 2, 3]:
		scen = 'Scenario' + str(i)
		eps_t[scen] = []
		for i in range(1000):
			eps_t[scen].append(init_I_dis[scen][i] * pop_dist if i < len(
				init_I_dis[scen]) else 0)

	# Save
	with open('../Data/parameters/eps_dict.pickle', 'wb') as handle:
		pickle.dump(eps_t, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# # Save
	# with open('../Data/parameters/init_I_IL.pickle', 'wb') as handle:
	# 	pickle.dump(init_I_IL, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_eps_by_region_prop(ind, age_dist):
	### eps by region proportion
	risk_dist = pd.read_csv('../Data/raw/population_size.csv')
	init_I_dis_italy = pd.read_csv('../Data/raw/init_i_italy.csv')[
						   'proportion'].values[:-1]
	f_init = pd.read_pickle('../Data/parameters/f0_full.pickle')
	eps_t = {}
	init_I_IL = {}
	init_I_dis = {}
	for i in [1, 2, 3]:
		scen = 'Scenario' + str(i)
		f_init_i = f_init[scen][:(len(ind.R) * len(ind.A))]
		init_I_IL[scen] = (491. / (1 - (
				f_init_i * risk_dist['pop size'].values).sum())) / isr_pop
		init_I_dis[scen] = init_I_dis_italy * init_I_IL[scen]


	# Loading data
	region_prop = pd.read_csv('../Data/demograph/sick_prop.csv', index_col=0)[
		'cases_prop'].copy()
	region_prop.index = region_prop.index.astype(str)
	risk_prop = pd.read_csv('../Data/raw/risk_dist.csv', index_col=0)[
		'risk'].copy()
	eps_t_region = {}
	for sc, init_I in zip(init_I_dis.keys(), init_I_dis.values()):
		eps_temp = []
		for t in range(1000):
			if t < len(init_I):
				# empty array for day t
				day_vec = np.zeros(len(ind.N))
				# fill in the array, zero for intervention groups
				for key in ind.N.keys():
					if ind.N[key][0] == 'Intervention':
						day_vec[key] = 0
					else:
						day_vec[key] = init_I[t] * region_prop[ind.N[key][1]] * \
									   age_dist[ind.N[key][3]] * \
									   (risk_prop[ind.N[key][3]] ** (
											   1 - (ind.N[key][2] == 'Low'))) * \
									   ((1 - risk_prop[ind.N[key][3]]) ** (
											   ind.N[key][2] == 'Low'))
				eps_temp.append(day_vec)
			else:
				eps_temp.append(0.0)

			eps_t_region[sc] = eps_temp
	# save eps:
	with open('../Data/parameters/eps_by_region.pickle', 'wb') as handle:
		pickle.dump(eps_t_region, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_hosptialization(ind):
	### hospitalization
	hosp_init = pd.read_csv('../Data/raw/hospitalizations.csv')
	hosp = expand_partial_array(ind.risk_age_dict, hosp_init['pr_hosp'].values,
								len(ind.N))
	# Save
	with open('../Data/parameters/hospitalization.pickle', 'wb') as handle:
		pickle.dump(hosp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_vents_proba(ind):
	### Ventilation
	vents_init = pd.read_csv('../Data/raw/vent_proba.csv')
	vent = expand_partial_array(ind.risk_age_dict, vents_init['pr_vents'].values,
								len(ind.N))
	# Save
	with open('../Data/parameters/vents_proba.pickle', 'wb') as handle:
		pickle.dump(vent, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_C_calibration(ind):
	### Calibration contact matrix
	full_mtx_home = scipy.sparse.load_npz(
		'../Data/base_contact_mtx/full_home.npz')

	full_mtx_work = {
		'routine': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_work_routine.npz'),
		'no_school': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_work_no_school.npz'),
		'no_work': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_work_no_work.npz'),
		'no_100_meters': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_work_no_100_meters.npz'),
		'no_bb': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_work_no_bb.npz'),
	}

	full_mtx_leisure = {
		'routine': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_leisure_routine.npz'),
		'no_school': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_leisure_no_school.npz'),
		'no_work': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_leisure_no_work.npz'),
		'no_100_meters': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_leisure_no_100_meters.npz'),
		'no_bb': scipy.sparse.load_npz(
			'../Data/base_contact_mtx/full_leisure_no_bb.npz'),
	}
	C_calibration = {}
	d_tot = 500
	# no intervation are null groups
	home_no_inter = []
	work_no_inter = []
	leis_no_inter = []

	for i in range(d_tot):
		home_no_inter.append(
			csr_matrix((full_mtx_home.shape[0], full_mtx_home.shape[1])))
		work_no_inter.append(csr_matrix((full_mtx_work['routine'].shape[0],
										 full_mtx_work['routine'].shape[1])))
		leis_no_inter.append(csr_matrix((full_mtx_leisure['routine'].shape[0],
										 full_mtx_leisure['routine'].shape[
											 1])))

	# Intervantion
	home_inter = []
	work_inter = []
	leis_inter = []

	# first days of routine from Feb 21st - March 13th
	d_rout = 9 + 13
	for i in range(d_rout):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['routine'])
		leis_inter.append(full_mtx_leisure['routine'])

	# first days of no school from March 14th - March 16th
	d_no_school = 3
	for i in range(d_no_school):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['no_school'])
		leis_inter.append(full_mtx_leisure['no_school'])

	# without school and work from March 17th - March 25th
	d_no_work = 9
	for i in range(d_no_work):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['no_work'])
		leis_inter.append(full_mtx_leisure['no_work'])

	# 100 meters constrain from March 26th - April 2nd
	d_no_100_meters = 8
	for i in range(d_no_100_meters):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['no_100_meters'])
		leis_inter.append(full_mtx_leisure['no_100_meters'])

	# Bnei Brak quaranrine from April 3rd - April 18th
	d_bb = 16
	for i in range(d_bb):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['no_bb'])
		leis_inter.append(full_mtx_leisure['no_bb'])

	# Back to 30% market like no school and no work - April 19th forward
	for i in range(
			d_tot - d_no_school - d_rout - d_no_work - d_no_100_meters - d_bb):
		home_inter.append(full_mtx_home)
		work_inter.append(full_mtx_work['no_work'])
		leis_inter.append(full_mtx_leisure['no_work'])

	C_calibration['home_inter'] = home_no_inter
	C_calibration['work_inter'] = work_no_inter
	C_calibration['leisure_inter'] = leis_no_inter
	C_calibration['home_non'] = home_inter
	C_calibration['work_non'] = work_inter
	C_calibration['leisure_non'] = leis_inter

	# Save
	with open('../Data/parameters/C_calibration.pickle', 'wb') as handle:
		pickle.dump(C_calibration, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_parameters_is_haredim(ind):
	### Haredim vector
	hared_dis = pd.read_csv('../Data/demograph/religion_dis.csv', index_col=0)[
		['cell_id', 'Orthodox']].copy()
	hared_dis.set_index('cell_id', inplace=True)
	hared_dis.index = hared_dis.index.astype(str)
	# Creating model orthodox dist. and save it as pickle
	model_orthodox_dis = np.zeros(len(ind.GA))
	for i in ind.GA.keys():
		model_orthodox_dis[i] = hared_dis.loc[str(ind.GA[i][0])]
	with open('../Data/parameters/orthodox_dist.pickle', 'wb') as handle:
		pickle.dump(model_orthodox_dis, handle,
					protocol=pickle.HIGHEST_PROTOCOL)

def create_parameters_is_arab(ind):
	### Arabs vector
	arab_dis = pd.read_csv('../Data/demograph/religion_dis.csv', index_col=0)[
		['cell_id', 'Druze', 'Muslim', 'Christian']].copy()
	arab_dis.set_index('cell_id', inplace=True)
	arab_dis.index = arab_dis.index.astype(str)
	arab_dis = arab_dis.sum(axis=1)
	# Creating model arab dist. and save it as pickle
	model_arab_dis = np.zeros(len(ind.GA))
	for i in ind.GA.keys():
		model_arab_dis[i] = arab_dis.loc[str(ind.GA[i][0])]
	with open('../Data/parameters/arab_dist.pickle', 'wb') as handle:
		pickle.dump(model_arab_dis, handle,
					protocol=pickle.HIGHEST_PROTOCOL)


### define indices
ind = Indices(cell_name)

age_dist = {'0-4': 0.02, '5-9': 0.02, '10-19': 0.11, '20-29': 0.23,
				'30-39': 0.15, '40-49': 0.14, '50-59': 0.14, '60-69': 0.11,
				'70+': 0.08}

create_demograph_age_dist_empty_cells(ind)

ind = create_paramaters_ind(ind)

create_demograph_religion(ind)

create_stay_home(ind)

create_demograph_sick_pop(ind)

create_stay_idx_routine(ind)

create_full_matices(ind)

create_parameters_indices(ind)

create_parameters_f0(ind)

create_parameters_eps_dict(ind, age_dist)

create_parameters_eps_by_region_prop(ind, age_dist)

create_parameters_hosptialization(ind)

create_parameters_vents_proba(ind)

create_parameters_C_calibration(ind)

create_parameters_is_haredim(ind)

create_parameters_is_arab(ind)


