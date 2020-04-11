import itertools
import numpy as np

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


def expand_partial_array(mapping_dic, array_to_expand, size=720):
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


#######################
# --- Set indices --- #
#######################

# Age groups
A = {0: '0-4', 1: '5-9', 2: '10-19', 3: '20-29', 4: '30-39', 5: '40-49', 6: '50-59', 7: '60-69', 8: '70+'}

# Risk groups
R = {0: 'High', 1: 'Low'}

# Intervention groups
M = {0: 'Intervention', 1: 'Non-intervention'}

# region groups
G = {0: '11', 1: '11_betshemesh', 2: '21', 3: '22', 4: '23', 5: '24', 6: '29', 7: '31', 8: '32', 9: '41', 10: '42', 11: '43', 12: '44', 13: '51',
     14: '51_tlv', 15: '51_bb', 16: '61', 17: '62', 18: '62_arab', 19: '71'}

# All combination:
N = {i: group for i, group in enumerate(itertools.product(M.values(), G.values(), R.values(), A.values()))}

# Region and age combination - for bea_j
GA = {i: group for i, group in enumerate(itertools.product(G.values(),A.values()))}

# Opposite indices dictionaries:
inter_dict = get_opposite_dict(N, [['Intervention'], ['Non-intervention']])
risk_dict = get_opposite_dict(N, [['High'], ['Low']])
region_age_dict = get_opposite_dict(N,list(itertools.product(G.values(), A.values())))
inter_risk_dict = get_opposite_dict(N,list(itertools.product(M.values(), R.values())))
age_dict = get_opposite_dict(N, [['0-4'], ['5-9'], ['10-19'], ['20-29'], ['30-39'], ['40-49'], ['50-59'], ['60-69'], ['70+']])
risk_age_dict = get_opposite_dict(N,list(itertools.product(R.values(), A.values())))
age_ga_dict = get_opposite_dict(GA, [['0-4'], ['5-9'], ['10-19'], ['20-29'], ['30-39'], ['40-49'], ['50-59'], ['60-69'], ['70+']])
region_dict = get_opposite_dict(N, [['11'], ['11_betshemesh'], ['21'], ['22'], ['23'], ['24'],
                                               ['29'], ['31'], ['32'], ['41'], ['42'], ['43'], ['44'], ['51'],
                                               ['51_tlv'], ['51_bb'], ['61'], ['62'], ['62_arab'], ['71']])

region_ga_dict = get_opposite_dict(GA, [['11'], ['11_betshemesh'], ['21'], ['22'], ['23'], ['24'],
                                               ['29'], ['31'], ['32'], ['41'], ['42'], ['43'], ['44'], ['51'],
                                               ['51_tlv'], ['51_bb'], ['61'], ['62'], ['62_arab'], ['71']])
