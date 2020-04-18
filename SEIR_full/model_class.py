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
            is_haredi=is_haredi,
            alpha=alpha,
            beta_home=beta_home,
            sigma=sigma,
            delta=delta,
            gama=gama,
            population_size=population_size,
            rho=hospitalizations,
            nu=nu,
            eps=eps,
            f=f0_full,
    ):
        """
        Receives all model's hyper-parameters and creates model object
        the results.
        :param self:
        :param is_haredi:
        :param alpha:
        :param beta_home:
        :param sigma:
        :param delta:
        :param gama:
        :param population_size:
        :param psi:
        :param rho:
        :param nu:
        :param eps:
        :param f:
        """
        # defining parameters:
        self.init_region_pop = shrink_array_sum(
                                                mapping_dic=region_dict,
                                                array_to_shrink=population_size
        )

        self.is_haredi = is_haredi
        self.alpha = alpha
        self.beta_home = beta_home
        self.delta = delta
        self.gama = gama
        self.rho = rho
        self.nu = nu
        self.sigma = sigma
        self.eps = eps
        self.f = f
        self.population_size = population_size.copy()

        # defining model compartments:
        self.reset(self.population_size, self.eps)


    def reset(self, population_size, eps):
        """
                Reset object's SEIR compartments.
                :param population_size:
                :param eps:
                """

        # defining model compartments:
        self.S, self.E, self.Ie, self.Is, self.Ia, self.R, self.H, self.new_Is, self.L = [], [], [], [], [],\
                                                                                         [], [], [], []

        # Initializing compartments:
        # Initialize S_0 to population size of each age-group
        self.S.append(population_size.copy())

        # Initialize R - with only the naturally immune individuals
        self.R.append(np.zeros(len(N)))

        # subtract R0 from S0:
        self.S[-1] -= self.R[-1]

        # Initialize E
        self.E.append(np.zeros(len(N)))

        # Initialize I (early)
        self.Ie.append(np.zeros(len(N)))

        # Initialize I (asymptomatic)
        self.Ia.append(np.zeros(len(N)))

        # Initialize I (symptomatic)
        self.Is.append(np.zeros(len(N)))

        # Subtract E(0) and Ie(0) from S(0)
        self.S[-1] -= (self.E[-1] + self.Ie[-1])

        # Zero newly infected on the first day of the season
        self.new_Is.append(np.zeros(len(N)))

        # Initialize H, tracking compartment
        self.H.append(np.zeros(len(N)))

    def fit(self,
            tpl,
            data,
             C,
            days_in_season,
            stay_home_idx,
            not_routine,
            date_lst,
            start='2020-03-20',
            end='2020-04-13',
            type='MSE'
            ):
        """
                Calibrates the model to data
                :param self:
                :param tpl:
                :param C:
                :param days_in_season:
                :param stay_home_idx:
                :param not_routine:
                :param date_lst:
                :param start:
                :param end:
                :param type: loss function to minimize 'MSE' or 'loss'
                :returns: loss functions' value
                        """

        # setting parameters
        beta_j = np.array([tpl[0], tpl[0], tpl[0], tpl[1], tpl[1], tpl[2], tpl[2], tpl[3], tpl[3]])

        # Run model with given parameters
        model_res = self.predict(C=C, beta_j=beta_j, beta_behave=tpl[5], theta=tpl[4],
                                 days_in_season=days_in_season, stay_home_idx=stay_home_idx,
                                 not_routine=not_routine)

        new_cases_model = model_res['new_Is']
        model_results_cal = np.zeros((days_in_season + 1, len(region_age_dict)))

        # Calculated total symptomatic (high+low) per age group (adding as columns)

        for i, key in enumerate(region_age_dict.keys()):
            model_results_cal[:, i] = new_cases_model[:, region_age_dict[key]].sum(axis=1)

        # Taking relevant time frame from model
        start_idx = int(np.where(date_lst == start)[0])
        end_idx = int(np.where(date_lst == end)[0])
        model_for_loss = model_results_cal[start_idx:end_idx + 1, :].copy()

        if type == "MSE":
            return np.log(MSE(data, model_for_loss))

        elif type == "loss":
            pass # need to complete

        else:
            return


    def update(self, new_data):
        """
        update object's attributes.
        :param self:
        :param new_data: dictionary object, key=attribute name, value= attribute value to assign
                """
        for key, value in new_data.items():
            setattr(self, key, value)

    def intervention(self,
                     C,
                     beta_j,
                     beta_behave,
                     theta,
                     days_in_season,
                     stay_home_idx,
                     not_routine
                     ):
        """
                Receives  model's parameters and run intervention for days_in_season days
                the results.
                :param self:
                :param beta_j:
                :param beta_behave:
                :param days_in_season:
                :param stay_home_idx:
                :param not_routine:
                """

        self.predict(C=C, beta_j=beta_j, beta_behave=beta_behave, theta=theta, days_in_season=days_in_season,
                     stay_home_idx=stay_home_idx,not_routine=not_routine)

    def predict(self,
                C,
                beta_j,
                beta_behave,
                theta,
                days_in_season,
                stay_home_idx,
                not_routine
                ):
        """
        Receives  model's parameters and run model for days_in_season days
        the results.
        :param self:
        :param beta_j:
        :param beta_behave:
        :param days_in_season:
        :param stay_home_idx:
        :param not_routine:
                """
        beta_j = expand_partial_array(
            mapping_dic=age_ga_dict,
            array_to_expand=beta_j,
            size=len(GA)
        )

        for t in range(days_in_season):

            # Calculate beta_home factor, current S_region/N_region and expand it
            # to mach (180X1).
            beta_home_factor = shrink_array_sum(
                mapping_dic=region_dict,
                array_to_shrink=self.S[-1]
            )
            beta_home_factor = beta_home_factor / self.init_region_pop
            beta_home_factor = expand_partial_array(
                mapping_dic=region_ga_dict,
                array_to_expand=beta_home_factor,
                size=len(GA)
            )

            # Calculate lambda (High risk symptomatic + Low risk symptomatic +
            # Asymptomatic)
            contact_force = self.calculate_force_matriceis(
                t=t,
                C=C,
                alpha=alpha,
                beta_behave=beta_behave,
                stay_home_idx=stay_home_idx,
                not_routine=not_routine
            )

            lambda_t = (self.beta_home * beta_home_factor * contact_force['home'] +
                        beta_j * (theta * self.is_haredi + 1 - self.is_haredi) * contact_force['out'])

            # preventing from lambda to become nan where there is no population.
            #lambda_t[np.isnan(lambda_t)] = 0

            self.L.append(lambda_t)

            # fitting lambda_t size to (720X1)
            lambda_t = expand_partial_array(
                mapping_dic=region_age_dict,
                array_to_expand=lambda_t
            )

            # R(t)
            self.R.append(self.R[-1] + self.gama * (self.Is[-1] + self.Ia[-1]) - self.eps[t])

            # H(t)
            self.H.append(self.H[-1] + (self.rho * self.gama) * self.Is[-1] - self.nu * self.H[-1])

            # Is(t)
            # Save new_Is
            self.new_Is.append((1 - self.f) * self.delta * self.Ie[-1])

            # Calculate new i matrix for day t
            self.Is.append(self.Is[-1] + self.new_Is[-1] - self.gama * self.Is[-1])

            # Ia(t)
            # Calculate new i matrix for day t
            self.Ia.append(self.Ia[-1] + self.f * self.delta * self.Ie[-1] - self.gama * self.Ia[-1])

            # Ie(t)
            # Calculate new i matrix for day t
            self.Ie.append(self.Ie[-1] + self.sigma * self.E[-1] - self.delta * self.Ie[-1])

            # E(t)
            # Calculate new e matrix for day t
            self.E.append(self.eps[t] + self.E[-1] + lambda_t * self.S[-1] - self.sigma * self.E[-1])

            # S(t)
            # Calculate current S
            self.S.append(self.S[-1] - lambda_t * self.S[-1])

            # Return the model results
        return {
            'S': np.array(self.S),
            'E': np.array(self.E),
            'Ie': np.array(self.Ie),
            'Ia': np.array(self.Ia),
            'Is': np.array(self.Is),
            'R': np.array(self.R),
            'new_Is': np.array(self.new_Is),
            'L': np.array(self.L),
            'H': np.array(self.H)
        }

    def calculate_force_matriceis(
                self,
                t,
                C,
                alpha,
                beta_behave,
                stay_home_idx,
                not_routine
            ):
        # Calculating beta_behave components:
        behave_componnet_inter_no_work = (beta_behave * stay_home_idx['inter']['not_work'][t]) ** \
                                         not_routine['inter']['not_work'][t]

        behave_componnet_non_no_work = (beta_behave * stay_home_idx['non_inter']['not_work'][t]) ** \
                                       not_routine['non_inter']['not_work'][t]

        behave_componnet_inter_work = (beta_behave * stay_home_idx['inter']['work'][t]) ** \
                                      not_routine['inter']['work'][t]

        behave_componnet_non_work = (beta_behave * stay_home_idx['non_inter']['work'][t]) ** \
                                    not_routine['non_inter']['work'][t]

        force_home = ((behave_componnet_inter_no_work * \
                       C['home_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
                                                 self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
                                                self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
                                                self.Is[-1][inter_risk_dict['Intervention', 'High']] +
                                                self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
                                                self.Ia[-1][inter_risk_dict['Intervention', 'High']])) +

                      behave_componnet_non_no_work * \
                      (C['home_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                               self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
                                              self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                              self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
                                              self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                              self.Ia[-1][inter_risk_dict['Non-intervention', 'High']])))

        force_out = ((behave_componnet_inter_work * \
                      C['work_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
                                                self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
                                               self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
                                               self.Is[-1][inter_risk_dict['Intervention', 'High']] +
                                               self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
                                               self.Ia[-1][inter_risk_dict['Intervention', 'High']]) +

                      (behave_componnet_non_work * \
                       C['work_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                               self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
                                              self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                              self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
                                              self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                              self.Ia[-1][inter_risk_dict['Non-intervention', 'High']])) +

                      (behave_componnet_inter_no_work * \
                       C['leisure_inter'][t].T.dot((self.Ie[-1][inter_risk_dict['Intervention', 'Low']] +
                                                    self.Ie[-1][inter_risk_dict['Intervention', 'High']]) * self.alpha +
                                                   self.Is[-1][inter_risk_dict['Intervention', 'Low']] +
                                                   self.Is[-1][inter_risk_dict['Intervention', 'High']] +
                                                   self.Ia[-1][inter_risk_dict['Intervention', 'Low']] +
                                                   self.Ia[-1][inter_risk_dict['Intervention', 'High']])) +

                      (behave_componnet_non_no_work * \
                       C['leisure_non'][t].T.dot((self.Ie[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                                  self.Ie[-1][inter_risk_dict['Non-intervention', 'High']]) * self.alpha +
                                                 self.Is[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                                 self.Is[-1][inter_risk_dict['Non-intervention', 'High']] +
                                                 self.Ia[-1][inter_risk_dict['Non-intervention', 'Low']] +
                                                 self.Ia[-1][inter_risk_dict['Non-intervention', 'High']]))))
        return {
            'out': force_out,
            'home': force_home
        }

    






