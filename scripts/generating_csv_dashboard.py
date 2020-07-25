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
import SEIR_full as mdl
import SEIR_full.model_class as mdl
import datetime as dt

###############################
# Generating dashboard files  #
###############################
## Must be run after model results.

result_stmp = '28_4_20_20_36'
scens = [
    '20@50_no_school_kid010',
    '20@75_no_school_kid010',
    '20@100_no_school_kid010',
    '20@100',
]

with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

with open('../Data/results/' + result_stmp + '.pickle', 'rb') as pickle_file:
	inter_data = pickle.load(pickle_file)

region_age_ga_dict = mdl.get_opposite_dict(ind.GA,list(itertools.product(ind.G.values(),ind.A.values())))
region_index = {v:k for k,v in ind.G.items()}
init_region_pop = mdl.shrink_array_sum(mapping_dic=ind.region_dict,array_to_shrink=mdl.population_size)
init_region_age_pop = mdl.shrink_array_sum(mapping_dic=ind.region_age_dict,array_to_shrink=mdl.population_size)

### generate map file
def map_table(res_mdl, inter_code, col_name='mean', start_day='2020-02-20'):
	tot_I = res_mdl['Is'] + res_mdl['Ia'] + res_mdl['Ie']
	date_range = pd.date_range(start_day, periods=tot_I.shape[0], freq='d')

	# Creating table for map, by region:
	table = pd.DataFrame(index=pd.MultiIndex.from_tuples(
		itertools.product(date_range, ind.G.values(),
						  list(ind.A.values()) + ['total']),
		names=['date', 'region', 'age']),
						 columns=[col_name])
	# filling df, cases per 1K
	for index in list(table.index):
		#         print(index)
		#         print(type(tot_I[int(np.where(date_range == index[0])[0]),mdl.region_dict[index[1]]].sum()))
		#         print([int(np.where(date_range == index[0])[0]),mdl.region_dict[index[1]]])

		# if total area value needed
		if index[2] == 'total':
			table.loc[index] = (tot_I[int(np.where(date_range == index[0])[0]),
									  ind.region_dict[index[1]]].sum()
								/ init_region_pop[
									region_index[index[1]]]) * 1000
		else:
			table.loc[index] = (tot_I[int(np.where(date_range == index[0])[0]),
									  ind.region_age_dict[
										  index[1], index[2]]].sum()
								/ init_region_age_pop[region_age_ga_dict[
						index[1], index[2]]]) * 1000

	# calculate weekly cases
	# 'daysoffset' will container the weekday, as integers
	table['daysoffset'] = table.apply(lambda x: x.name[0].weekday(), axis=1)
	# We apply, row by row (axis=1) a timedelta operation
	table['week_start'] = table.apply(
		lambda x: x.name[0] - dt.timedelta(days=x['daysoffset']), axis=1)

	table.reset_index(inplace=True)
	table_groupped = table.groupby(['week_start', 'region', 'age'])[col_name]
	table_groupped = table_groupped.apply(lambda x: (x.max() + x.min()) / 2)
	table_groupped = table_groupped.to_frame()
	table_groupped['scenario_idx'] = inter_code

	return table_groupped[pd.to_datetime('today'):]


model_sets = ['base']
inter_dict_map = {}
for scen in scens:
	for p in model_sets:
		inter_dict_map[scen] = map_table(
			inter_data[scen, p],
			scen,
			col_name=p,
			start_day='2020-02-20',
		).reset_index()
inter_df_list = [v for v in inter_dict_map.values()]
inter_df_concat = pd.concat(inter_df_list)

# Save
inter_df_concat.to_csv('../Data/results/map_data_upload_280420.csv')


### Generate age distribution:
def age_distribution_table(
		res_mdl,
		inter_code,
		col_name='mean',
		start_day='2020-02-20'
	):
	tot_I = res_mdl['Is'] + res_mdl['Ia'] + res_mdl['Ie']
	date_range = pd.date_range(start_day, periods=tot_I.shape[0], freq='d')

	# Creating table for map, by region:
	table = pd.DataFrame(index=pd.MultiIndex.from_tuples(
		itertools.product(date_range, list(ind.A.values())),
		names=['date', 'age']),
						 columns=[col_name])

	for index in list(table.index):
		table.loc[index] = (tot_I[int(np.where(date_range == index[0])[0]),
								  ind.age_dict[index[1]]].sum()) * mdl.pop_israel

	# calculate weekly cases
	# 'daysoffset' will container the weekday, as integers
	table['daysoffset'] = table.apply(lambda x: x.name[0].weekday(), axis=1)
	# We apply, row by row (axis=1) a timedelta operation
	table['week_start'] = table.apply(
		lambda x: x.name[0] - dt.timedelta(days=x['daysoffset']), axis=1)

	table.reset_index(inplace=True)
	table_groupped = table.groupby(['week_start', 'age'])[col_name]
	table_groupped = table_groupped.apply(
		lambda x: np.ceil((x.max() + x.min()) / 2))
	table_groupped = table_groupped.to_frame()
	table_groupped['scenario_idx'] = inter_code

	return table_groupped[pd.to_datetime('today'):]


def combine_ub_lb_ml(ub_df, lb_df, ml_df, col_list=['ub', 'lb', 'ml']):
	combined_df = ml_df.copy()
	combined_df = combined_df.merge(ub_df, left_index=True, right_index=True)
	combined_df = combined_df.merge(lb_df, left_index=True, right_index=True)
	#     print(combined_df)
	# fixing LB/Mean/UB
	combined_df['new_lb'] = combined_df[col_list].apply(lambda x: x.min(),
														axis=1)
	combined_df['new_ub'] = combined_df[col_list].apply(lambda x: x.max(),
														axis=1)
	combined_df['new_ml'] = combined_df[col_list].apply(
		lambda x: sorted([x[0], x[1], x[2]])[1], axis=1)

	return combined_df[['new_lb', 'new_ub', 'new_ml', 'scenario_idx']]


model_sets = [
	'base',
	'ub',
	'lb'
]
inter_dict_age_dis = {}
for scen in scens:
	curr_inter_dict = {}
	for p in model_sets:
		curr_inter_dict[p] = age_distribution_table(
			inter_data[scen,p],
			scen,
			col_name=p,
			start_day='2020-02-20',
		)

	# combining ub,lb,ml to one DF:
	inter_dict_age_dis[scen] = combine_ub_lb_ml(
		curr_inter_dict['ub'],
		curr_inter_dict['lb'],
		curr_inter_dict['base'],
		col_list=model_sets,
	).reset_index()
inter_df_list_age_dis = [v for v in inter_dict_age_dis.values()]
inter_df_age_dis = pd.concat(inter_df_list_age_dis)
# Save
inter_df_age_dis.to_csv('../Data/results/age_data_upload_280420.csv')


### Ventilators file:
def vents_graph(res_mdl, inter_code, tracking='Vents', col_name='mean',
				curr_vents=None, start_day='2020-02-20',
				inter_start_date='2020-04-20', fix_vents=False):
	tot_vents = np.ceil(res_mdl[tracking].sum(axis=1) * mdl.pop_israel)
	date_range = pd.date_range(start_day, periods=tot_vents.shape[0], freq='d')
	# fixing patients no.
	if fix_vents:
		model_vents_proj = tot_vents[
			int(np.where(date_range == inter_start_date)[0])]
		tot_vents += max(curr_vents - model_vents_proj, 0)

	# creates DF with results
	vents_dict = {}
	vents_dict['date'] = date_range
	vents_dict[col_name] = tot_vents
	vents_df = pd.DataFrame.from_dict(vents_dict)

	# 'daysoffset' will container the weekday, as integers
	vents_df['daysoffset'] = vents_df['date'].apply(lambda x: x.weekday())
	# We apply, row by row (axis=1) a timedelta operation
	vents_df['week_start'] = vents_df.apply(
		lambda x: x['date'] - dt.timedelta(days=x['daysoffset']), axis=1)

	vents_df_grop = vents_df.groupby('week_start')[col_name]
	vents_df_grop = vents_df_grop.max().to_frame()
	vents_df_grop['scenario_idx'] = inter_code
	return vents_df_grop[pd.to_datetime('today'):]


inter_dict_vents_dict = {}
for scen in scens:
	curr_inter_dict = {}
	for p in model_sets:
		curr_inter_dict[p] = vents_graph(
			inter_data[scen,p],
			scen,
			col_name=p,
			curr_vents=113,
			start_day='2020-02-20',
			fix_vents=True,
		)


	# combining ub,lb,ml to one DF:
	inter_dict_vents_dict[scen] = combine_ub_lb_ml(
		curr_inter_dict['ub'],
		curr_inter_dict['lb'],
		curr_inter_dict['base'],
		col_list=model_sets
	).reset_index()
inter_df_list_vents_dis = [v for v in inter_dict_vents_dict.values()]
inter_df_vents_dis = pd.concat(inter_df_list_vents_dis)
# Save
inter_df_vents_dis.to_csv('../Data/results/vents_data_upload_280420.csv')


### Hospitalization file:
inter_dict_vents_dict = {}
for scen in scens:
	curr_inter_dict = {}
	for p in model_sets:
		curr_inter_dict[p] = vents_graph(
			inter_data[scen,p],
			scen,
			tracking='H',
			col_name=p,
			curr_vents=113,
			start_day='2020-02-20',
			fix_vents=True,
		)


	# combining ub,lb,ml to one DF:

	inter_dict_vents_dict[scen] = combine_ub_lb_ml(
		curr_inter_dict['ub'],
		curr_inter_dict['lb'],
		curr_inter_dict['base'],
		col_list=model_sets,
	).reset_index()
inter_df_list_vents_dis = [v for v in inter_dict_vents_dict.values()]
inter_df_vents_dis = pd.concat(inter_df_list_vents_dis)
# Save
inter_df_vents_dis.to_csv('../Data/results/hospitalization_data_upload_280420.csv')