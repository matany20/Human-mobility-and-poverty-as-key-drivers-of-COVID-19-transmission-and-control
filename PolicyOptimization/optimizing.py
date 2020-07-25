import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import itertools
import pickle
from matplotlib import pyplot as plt
import datetime
from scipy import optimize
import sys
sys.path.append('../SEIR_full/')
sys.path.append('..')
import SEIR_full as mdl
import SEIR_full.model_class as mdl
import datetime as dt
from scipy.stats import poisson
from scipy.stats import binom
import scipy
import copy
import os
import time
from EvaluatePolicy import evaluate_policy

policy_params = {}
policy_params['threshold']=2e-4
policy_params['max_duration']=2
policy_params['period']=5

# [Is, new_Is, H, Vent]
policy_params['weight_matrix'] = np.zeros([4,9])
policy_params['weight_matrix'][0][5:] = 0.25*10
with (open('../Data/parameters/indices.pickle', 'rb')) as openfile:
	ind = pickle.load(openfile)

policy_params['global_decision'] = True
policy_params['start_KI'] = 60/policy_params['period']
policy_params['KIgain'] = -0.003 / 2000 * 0.03
policy_params['KPgain'] = -0.003 / 2000 * 0.1
policy_params['limiter'] = 0.001


res = evaluate_policy(ind, policy_params, 30)

print(res['max_resp'],res['GDP'])