import pandas as pd

taz_details = pd.read_csv('../Data/sick/city2county_sector.csv')
taz_details['city'] = taz_details['city'].astype(str)
taz_details['city'] = taz_details['city'].apply(lambda x: x.replace('קריית', 'קרית'))
taz_details['city'] = taz_details['city'].apply(lambda x: x.replace('מעיין', 'מעין'))
taz_details['city'] = taz_details['city'].apply(lambda x: x.replace('ייה', 'יה'))
taz_details.loc[taz_details.city == 'תל אביב -יפו', 'city'] = 'תל אביב - יפו'
taz_details.loc[taz_details.city == 'נצרת עילית', 'city'] = 'נוף הגליל'
taz_details.loc[taz_details.city == 'אבו גוש ידידה קרית יערים (מוסד)', 'city'] = 'אבו גוש ידידה קרית יערים(מוסד)'

city2county = taz_details.set_index('city')['calibration_county'].to_dict()
city2county['גבעת אבני'] = city2county['לביא']
city2county['מחנה יפה'] = city2county['באר שבע']


def get_county(cityname):
    if pd.isnull(cityname):
        return 'NA'
    if ('(שבט)' in cityname) or (cityname == 'מסעודין אל-עזאזמה'):
        return '62_arab'
    if cityname in city2county:
        return city2county[cityname]
    cityname2 = cityname.replace('"', '').replace("'", '').replace('  ', ' ')
    if cityname2 in city2county:
        return city2county[cityname2]
    if cityname2.replace('-', ' ') in city2county:
        return city2county[cityname2.replace('-', ' ')]
    for k in city2county.keys():
        if (cityname in k) or (cityname2 in k):
            return city2county[k]
    return 'NA'

