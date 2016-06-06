# -*- coding: utf-8 -*-
"""Transform the Interpolated Indicator data set to DDF format
Link: http://esa.un.org/unpd/wpp/Download/Standard/Interpolated/
"""

import pandas as pd
import re
from index import create_index_file

# configuration of file path
source = '../source/WPP2015_INT_F01_ANNUAL_DEMOGRAPHIC_INDICATORS.XLS'
out_dir = '../../'  # path for outputs


# functions for building ddf files
def to_concept_id(s):
    '''convert a string to lowercase alphanumeric + underscore id for concepts'''
    s1 = re.sub(r'[/ -\.\*";]+', '_', s.strip())
    s1 = re.sub(r'\[.*\]', '', s1)
    s1 = s1.replace('\n', '')

    if s1[-1] == '_':
        s1 = s1[:-1]

    return s1.lower()


def extract_concept_discrete(data):
    """extract discrete concepts from source data.
    data can be estimated or medium variant data, because the columns
    of both sheet are same.
    """

    # headers for dataframe and csv output
    headers = ['concept', 'name', 'concept_type']

    discrete = data.columns[:5]
    discrete_id = list(map(to_concept_id, discrete))

    # build data frame
    dis_df = pd.DataFrame([], columns=headers)
    dis_df['concept'] = discrete_id
    dis_df['name'] = discrete
    dis_df['concept_type'] = ['string', 'string', 'string', 'entity domain', 'time']

    return dis_df


def extract_concept_continuous(data):
    """extract continuous concepts from source data.
    data can be estimated or medium variant data, because the columns
    of both sheet are same.
    """

    # headers
    headers = ['concept', 'name', 'concept_type', 'unit']

    cont = data.columns[5:]

    # because the name and unit of the concept are both in the column name
    # we will seperate them.
    res = {}

    for c in cont:
        name, unit = re.match(r"(.*) *\((.*)\)", c).groups()
        res[name.strip()] = unit

    # now build the data frame
    cont_df = pd.DataFrame([], columns=headers)
    cont_df['name'] = res.keys()
    cont_df['unit'] = res.values()
    cont_df['concept'] = cont_df['name'].apply(to_concept_id)
    cont_df['concept_type'] = 'measure'

    return cont_df


def extract_entities_country(data):
    """extract the country entities from source data"""

    # headers for dataframe and csv output
    headers = ['country_code', 'major_area_region_country_or_area']

    # build dataframe
    ent = data[['Country code', 'Major area, region, country or area *']]
    ent = ent.drop_duplicates()
    ent.columns = headers

    return ent


def extract_datapoints_country_year(data):
    """extract data points by country and year"""

    # rename the columns to concept_id
    cols = data.columns[5:]
    rename = {}

    for c in cols:
        name, unit = re.match(r"(.*) *\((.*)\)", c).groups()
        rename[c] = name

    data = data.rename(columns=rename)
    new_cols = list(map(to_concept_id, data.columns))
    data.columns = new_cols

    # extract data for each concept
    res_dp = {}

    for c in new_cols[5:]:
        df = data[['country_code', 'reference_date_1_january_31_december', c, 'variant']]
        res_dp[c] = df

    return res_dp


def extract_notes(data, notes):
    """extract the notes from source data"""

    headers = ['country_code', 'variant', 'notes']

    # first get the notes id and contents
    res_notes = {}

    for n in notes['Notes']:
        g1, g2 = re.match(r'\((.{1,2})\) * (.*)', n).groups()
        res_notes[g1] = g2

    df = data.set_index('Country code')
    df = df[['Variant', 'Notes']].dropna().drop_duplicates()

    df['Notes'] = df['Notes'].apply(lambda x: res_notes[str(x)])
    df = df.reset_index()
    df.columns = headers

    return df


if __name__ == '__main__':
    import os

    print('reading source files...')
    est = pd.read_excel(source, sheetname='ESTIMATES', skiprows=16, index_col=0)
    mva = pd.read_excel(source, sheetname='MEDIUM VARIANT', skiprows=16, index_col=0)
    notes = pd.read_excel(source, sheetname='NOTES')

    print('creating concepts ddf files...')
    discrete = extract_concept_discrete(est)
    discrete.to_csv(os.path.join(out_dir, 'ddf--concepts--discrete.csv'),
                    index=False)
    continuous = extract_concept_continuous(est)
    continuous.to_csv(os.path.join(out_dir, 'ddf--concepts--continuous.csv'),
                      index=False)

    print('creating entities ddf files...')
    entities = extract_entities_country(mva)
    entities.to_csv(os.path.join(out_dir, 'ddf--entities--country_code.csv'),
                    index=False, encoding='utf8')

    print('creating data point ddf files...')
    data = pd.concat([est, mva])
    res = extract_datapoints_country_year(data)
    for c, df in res.items():
        path = os.path.join(out_dir, 'ddf--datapoints--'+c+'--by--country_code--year.csv')
        df.to_csv(path, index=False)

    print('creating notes files...')
    notes_df = extract_notes(data, notes)
    notes_df.to_csv(os.path.join(out_dir, 'ddf--notes.csv'), index=False, encoding='utf8')

    print('generating index file...')
    create_index_file(out_dir, os.path.join(out_dir, 'ddf--index.csv'))

