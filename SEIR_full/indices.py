import itertools
import numpy as np
import pandas as pd

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


def make_new_indices(empty_list=None):
	#define cells
	global cell
	cell = pd.read_excel(
		'../Data/division_choice/' + cell_name + '/cell2name.xlsx')

	# remove empty cells from indices
	if empty_list!=None:
		cell = cell[cell['cell_id'] not in empty_list]
	# set area indices
	global G
	G = {i: str(k) for i, k in enumerate(list(cell['cell_id'].values))}

	# set cell names dict
	cell.set_index('cell_id', inplace=True)
	cell.index = cell.index.astype(str)
	cell = cell.to_dict()['cell_name']

	# All combination:
	global N
	N = {
		i: group for
		i, group in
		enumerate(itertools.product(
			M.values(),
			G.values(),
			R.values(),
			A.values(),
		))
	}

	# Region and age combination - for beta_j
	global GA
	GA = {
		i: group for
		i, group in
		enumerate(itertools.product(
			G.values(),
			A.values(),
		))
	}

	global MI
	MI = {
		i: group for
		i, group in
		enumerate(itertools.product(
			A.values(),
			G.values(),
			A.values(),
		))
	}


	# Opposite indices dictionaries:
	global inter_dict
	inter_dict = get_opposite_dict(
		N,
		[[x] for x in list(M.values())],
	)

	global risk_dict
	risk_dict = get_opposite_dict(
		N,
		[[x] for x in list(R.values())],
	)

	global region_age_dict
	region_age_dict = get_opposite_dict(
		N,
		list(itertools.product(
			G.values(),
			A.values(),
		)),
	)

	global inter_region_risk_age_dict
	inter_region_risk_age_dict = get_opposite_dict(
		N,
		list(itertools.product(
			M.values(),
			G.values(),
			R.values(),
			A.values(),
		))
	)

	global region_risk_age_dict
	region_risk_age_dict = get_opposite_dict(
		N,
		list(itertools.product(
			G.values(),
			R.values(),
			A.values(),
		))
	)

	global inter_risk_dict
	inter_risk_dict = get_opposite_dict(
		N,
		list(itertools.product(
			M.values(),
			R.values(),
		)),
	)

	global age_dict
	age_dict = get_opposite_dict(
		N,
		[[x] for x in list(A.values())],
	)

	global risk_age_dict
	risk_age_dict = get_opposite_dict(
		N,
		list(itertools.product(
			R.values(),
			A.values(),
		)),
	)

	global age_ga_dict
	age_ga_dict = get_opposite_dict(
		GA,
		[[x] for x in list(A.values())],
	)

	global region_dict
	region_dict = get_opposite_dict(
		N,
		[[x] for x in list(G.values())],
	)

	global region_ga_dict
	region_ga_dict = get_opposite_dict(
		GA,
		[[x] for x in list(G.values())],
	)



#######################
# --- Set indices --- #
#######################

cell_name = '20'

# Age groups
A = {
	0: '0-4',
	1: '5-9',
	2: '10-19',
	3: '20-29',
	4: '30-39',
	5: '40-49',
	6: '50-59',
	7: '60-69',
	8: '70+',
}

# Risk groups
R = {
	0: 'High',
	1: 'Low',
}

# Intervention groups
M = {
	0: 'Intervention',
	1: 'Non-intervention',
}

## changbale indices due to empty cells
cell = {}
G = {}
N = {}
GA = {}
MI = {}
inter_dict = {}
risk_dict = {}
region_age_dict = {}
inter_region_risk_age_dict = {}
region_risk_age_dict = {}
inter_risk_dict = {}
age_dict = {}
risk_age_dict = {}
age_ga_dict = {}
region_dict = {}
region_ga_dict = {}
make_new_indices()