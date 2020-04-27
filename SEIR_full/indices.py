import itertools
import numpy as np
import pandas as pd
from .utils import *


class Indices:

	def __init__(self, cell_name='20'):

		self.cell_name = cell_name
		# Age groups
		self.A = {
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
		self.R = {
			0: 'High',
			1: 'Low',
		}

		# Intervention groups
		self.M = {
			0: 'Intervention',
			1: 'Non-intervention',
		}
		self.cell = {}
		self.G = {}
		self.N = {}
		self.GA = {}
		self.MI = {}
		self.inter_dict = {}
		self.risk_dict = {}
		self.region_age_dict = {}
		self.inter_region_risk_age_dict = {}
		self.region_risk_age_dict = {}
		self.inter_risk_dict = {}
		self.age_dict = {}
		self.risk_age_dict = {}
		self.age_ga_dict = {}
		self.region_dict = {}
		self.region_ga_dict = {}
		self.make_new_indices()


	def make_new_indices(self, empty_list=None):
		#define cells
		self.cell = pd.read_excel(
			'../Data/division_choice/' + self.cell_name + '/cell2name.xlsx')
		self.cell['cell_id'] = self.cell['cell_id'].astype(str)

		# remove empty cells from indices
		if not(empty_list is None):
			self.cell = self.cell[
				self.cell['cell_id'].apply(lambda x: x not in empty_list.values)]
		# set area indices
		self.G = {i: str(k) for i, k in enumerate(list(self.cell['cell_id'].values))}
		# set cell names dict
		self.cell.set_index('cell_id', inplace=True)
		self.cell.index = self.cell.index.astype(str)
		self.cell = self.cell.to_dict()['cell_name']

		# All combination:
		self.N = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.M.values(),
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		}

		# Region and age combination - for beta_j
		self.GA = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.G.values(),
				self.A.values(),
			))
		}

		self.MI = {
			i: group for
			i, group in
			enumerate(itertools.product(
				self.A.values(),
				self.G.values(),
				self.A.values(),
			))
		}


		# Opposite indices dictionaries:
		self.inter_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.M.values())],
		)

		self.risk_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.R.values())],
		)

		self.region_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.G.values(),
				self.A.values(),
			)),
		)

		self.inter_region_risk_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.M.values(),
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		)

		self.region_risk_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.G.values(),
				self.R.values(),
				self.A.values(),
			))
		)

		self.inter_risk_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.M.values(),
				self.R.values(),
			)),
		)

		self.age_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.A.values())],
		)

		self.risk_age_dict = get_opposite_dict(
			self.N,
			list(itertools.product(
				self.R.values(),
				self.A.values(),
			)),
		)

		self.age_ga_dict = get_opposite_dict(
			self.GA,
			[[x] for x in list(self.A.values())],
		)

		self.region_dict = get_opposite_dict(
			self.N,
			[[x] for x in list(self.G.values())],
		)

		self.region_ga_dict = get_opposite_dict(
			self.GA,
			[[x] for x in list(self.G.values())],
		)

	def update_empty(self):
		empty_cells = pd.read_csv('../Data/demograph/empty_cells.csv')[
			'cell_id'].astype(str)
		self.make_new_indices(empty_cells)
