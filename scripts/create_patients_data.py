from datetime import *
import numpy as np
from scripts.county_utils import *

days_shift = 6
jer_orth_prob = 0.75
smooth_window = 7
smooth_center = True


data = pd.read_csv('../Data/sick/patients_to_20_4.csv')
for col in data.columns:
    if col.endswith('date'):
        data[col] = pd.to_datetime(data[col], dayfirst=True).dt.date

data['start_date'] = data['result_date'] - timedelta(days=days_shift)

# map city name to county_sector
data['new_cityname'] = data['new_cityname'].apply(
    lambda x: 'ירושלים-חרדים' if (x == 'ירושלים') and (np.random.rand() <= jer_orth_prob) else x)
data['county'] = data.new_cityname.apply(get_county).astype(str)

# age groups
def get_age_group(age):
    if pd.isnull(age):
        return 'NA'
    dec = int(age / 10)
    if dec >= 7:
        return '70+'
    if dec == 0:
        return '0-4' if age < 5 else '5-9'
    return f'{dec}0-{dec}9'

data['age_group'] = data['AGE'].apply(get_age_group)

def get_dist(col, value2drop='NA'):
    dist = pd.get_dummies(col).sum().drop(value2drop)
    dist /= dist.sum()
    return dist

def replace_na(x, dist, value2replace='NA'):
    if x == value2replace:
        return np.random.choice(dist.index, p=dist)
    else:
        return x

county_dist = get_dist(data.county)
data.county = data.county.apply(lambda x: replace_na(x, county_dist))

ages_dist = get_dist(data.age_group)
data.age_group = data.age_group.apply(lambda x: replace_na(x, ages_dist))

ages = pd.get_dummies(data['age_group'])
joint = data.join(ages)

# add new dates
new_dates = pd.read_csv('../Data/sick/patients_26_4_to_15_5.csv').set_index('city')
new_dates.columns = pd.to_datetime(new_dates.columns)
new_dates = new_dates.T.dropna(axis=1)

new_dates['ירושלים-חרדים'] = new_dates['ירושלים'] * jer_orth_prob
new_dates['ירושלים'] = new_dates['ירושלים'] * (1 - jer_orth_prob)
new_dates = new_dates.drop_duplicates()
date_range = pd.date_range(new_dates.index.min(), new_dates.index.max())
new_dates = new_dates.reindex(date_range).interpolate()
concat = new_dates.diff()
concat.index = pd.to_datetime(concat.index).date - timedelta(days=days_shift)
concat = concat.T
concat['county'] = [get_county(i) for i in concat.index]

# pivot
age_dist_by_county = joint.groupby('county')[list(ages.columns)].sum().dropna()
age_dist_by_county = age_dist_by_county.div(age_dist_by_county.sum(axis=1), axis=0)
new_dates_piv = concat.groupby('county').sum()


results = list()
for col in new_dates_piv.columns:
    df = age_dist_by_county.apply(lambda x: new_dates_piv[col] * x, axis=0)
    df = df.reset_index()
    df['start_date'] = col
    results.append(df)

append = pd.concat(results)

piv = joint.groupby(['start_date', 'county'])[list(ages.columns)].sum().reset_index().append(
    append.reset_index()).groupby(['start_date', 'county']).sum().reset_index().drop('index', axis=1)

national = pd.read_csv('../Data/sick/national_daily_cases.csv', index_col='Date', squeeze=True)
national.index = pd.to_datetime(national.index, dayfirst=True).date

coef = national.shift(-days_shift) / piv.groupby('start_date').sum().sum(axis=1)
coef = coef.fillna(1)

piv_fixed = piv.set_index('start_date').groupby('county').apply(lambda df: df.groupby(level='start_date').apply(lambda x: coef[x.index[0]] * x))

# smoothing
def smooth(county_df, window=smooth_window, center=smooth_center):
    grouped = county_df.groupby('start_date')[list(ages.columns)].sum()
    date_range = pd.date_range(grouped.index.min(), grouped.index.max())
    full_df = pd.DataFrame(0, index=date_range, columns=grouped.columns)
    full_df = (full_df + grouped).fillna(0)
    cumsum = full_df.cumsum()
    rolling = cumsum.rolling(window, center=center).mean().dropna()
    return rolling.diff()

smoothen = piv_fixed.groupby('county').apply(smooth).dropna().clip(lower=0)
smoothen.to_csv('../Data/sick/smooth_sick_by_age_calibration_county.csv')
