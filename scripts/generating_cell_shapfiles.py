## Imports
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib import pyplot as plt
import datetime
from scipy import optimize
import os
import geopandas as gpd
from geopandas import GeoSeries

############################################
# Generating Cell shap files based on tazs #
############################################

# This file generates the geography of cells based on tazs geography.

## Consts
cell_name = '20'

## actual script:
TAZs = gpd.read_file('../Data/GIS/2630_TAZ_POLY_only/2630_TAZ_POLY_only.shp')
TAZs.sort_values(by='ID', inplace=True)

TAZ2cell = pd.read_excel('../Data/division_choice/' + cell_name +'/taz2cell.xlsx')

TAZ2cell_geo = TAZs.merge(TAZ2cell, right_on='taz_id', left_on='ID')[['taz_id', 'geometry', 'cell_id']]

cell_geo = TAZ2cell_geo.groupby(by='cell_id').apply(lambda f: GeoSeries(f['geometry']).unary_union)
cell_geo = cell_geo.reset_index()
cell_geo.columns = ['cell_id', 'geometry']
cell_geo = gpd.GeoDataFrame(cell_geo)
cell_names = pd.read_excel('../Data/division_choice/' + cell_name +'/cell2name.xlsx')
cell_names[['cell_id', 'cell_name']]
cell_geo = cell_geo.merge(cell_names, on='cell_id')[['cell_id', 'cell_name', 'geometry']]

import os
os.mkdir('../Data/GIS/'+ cell_name + '_poly')
cell_geo.to_file('../Data/GIS/'+ cell_name + '_poly/' +cell_name + '_poly.shp')
