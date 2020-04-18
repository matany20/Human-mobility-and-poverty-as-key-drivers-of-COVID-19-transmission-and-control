from .indices import *
from .helper_func import *
from .parameters import *
import pandas as pd
import numpy as np

import itertools
import pickle


#######################
# ---- Model Class---- #
#######################

class Model_behave:

    def __init__(
            self,
            stay_home_idx=stay_home_idx,
            is_haredi=is_haredi,
            not_routine=not_routine,
            alpha=alpha,
            beta_home=beta_home,
            sigma=sigma,
            delta=delta,
            gama=gama,
            population_size=population_size,
            C=C_calibration,
            psi=psi,
            rho=hospitalizations,
            nu=nu
    ):
        """
        Receives all model's data and parameters, runs it for a season and returns
        the results.
        :param self:
        :param stay_home_idx:
        :param is_haredi:
        :param not_routine:
        :param alpha:
        :param beta_home:
        :param sigma:
        :param delta:
        :param gama:
        :param population_size:
        :param C:
        :param psi:
        :param rho:
        :param nu:
        :return:
        """