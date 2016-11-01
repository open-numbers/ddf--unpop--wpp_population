# -*- coding: utf-8 -*-
"""Transform the Interpolated Indicator data set to DDF format
Link: http://esa.un.org/unpd/wpp/Download/Standard/Interpolated/
"""

import pandas as pd
import numpy as np
import os
from ddf_utils.str import to_concept_id, format_float_digits
from ddf_utils.index import create_index_file


# configuration of file path
source_t = '../source/WPP2015_INT_F03_1_POPULATION_BY_AGE_ANNUAL_BOTH_SEXES.XLS'
source_m = '../source/WPP2015_INT_F03_2_POPULATION_BY_AGE_ANNUAL_MALE.XLS'
source_f = '../source/WPP2015_INT_F03_3_POPULATION_BY_AGE_ANNUAL_FEMALE.XLS'
out_dir = '../../'  # path for outputs


def read_cleanup(source, gender):
    data_est = pd.read_excel(source, sheetname='ESTIMATES', skiprows=16, na_values='…')
    data_var = pd.read_excel(source, sheetname='MEDIUM VARIANT', skiprows=16, na_values='…')

    # rename/drop some columns.
    # for 80+ and 100+ groups, rename to 80plus and 100plus
    data_est = data_est.drop(['Index', 'Notes'], axis=1)
    data_var = data_var.drop(['Index', 'Notes'], axis=1)

    data_est = data_est.rename(columns={'80+': '80plus',
                                        '100+': '100plus'})
    data_var = data_var.rename(columns={'100+': '100plus'})  # todo: no use to rename for now.

    # insert Gender column and rearrange the order
    col_est_1 = data_est.columns[:4]
    col_est_2 = data_est.columns[4:]

    col_var_1 = data_var.columns[:4]
    col_var_2 = data_var.columns[4:]

    cols_est = [*col_est_1, 'Gender', *col_est_2]
    cols_var = [*col_var_1, 'Gender', *col_var_2]

    data_est['Gender'] = gender
    data_var['Gender'] = gender

    return (data_est[cols_est], data_var[cols_var])


def extract_concepts(data):
    """extract concept from one of the dataframes."""
    data_ = data.rename(columns={
        'Major area, region, country or area *': 'Name',
        'Reference date (as of 1 July)': 'Year'
    })

    concept_name = list(data_.columns[:5])
    concept_name.append('Population')
    concept_name.append('Age')
    concepts = list(map(to_concept_id, concept_name))

    # now construct the dataframe
    cdf = pd.DataFrame([], columns=['concept', 'concept_type', 'name'])
    cdf['concept'] = concepts
    cdf['name'] = concept_name

    cdf['concept_type'] = 'string'

    # population
    cdf['concept_type'].iloc[5] = 'measure'

    # entity domains
    cdf['concept_type'].iloc[[2, 4, 6]] = 'entity_domain'

    # year
    cdf['concept_type'].iloc[3] = 'time'
    cdf['name'].iloc[3] = 'Reference date (as of 1 July)'

    return cdf


def extract_entities_country(data_est, data_var):
    """extract country entities from source.

    data_est is data from estimates tab.
    data_var is from medium variant tab.

    we assume that both tab should have same entities.
    """
    data_est.columns = list(map(to_concept_id, data_est.columns))
    data_var.columns = list(map(to_concept_id, data_var.columns))

    entity = data_est[['major_area_region_country_or_area', 'country_code']].copy()
    entity = entity.rename(columns={'major_area_region_country_or_area': 'name'})
    entity = entity.drop_duplicates()

    entity_2 = data_var[['major_area_region_country_or_area', 'country_code']].copy()
    entity_2 = entity_2.rename(columns={'major_area_region_country_or_area': 'name'})
    entity_2 = entity_2.drop_duplicates()

    if len(entity) != len(entity_2):
        print('Warning: entities not same in the excel tabs.')

        ent = pd.concat([entity, entity_2])
        return ent.drop_duplicates()

    return entity


def extract_entities_gender():
    """no more information about gender in source, just create that"""
    df = pd.DataFrame([], columns=['gender', 'name'])
    df['gender'] = ['male', 'female']
    df['name'] = ['Male', 'Female']

    return df


def extract_entities_age(data_est):
    """extract ages from estimates tab of source data."""

    df = pd.DataFrame([], columns=['age', 'name'])
    df['age'] = data_est.columns[5:]

    df['name'] = 'Age ' + df['age']
    return df


def extract_datapoints(dflist):
    """make datapoint file with all dataframe in dflist."""

    to_concat = []

    for df in dflist:
        e = df.drop(['Variant', 'Major area, region, country or area *'], axis=1)
        e = e.set_index([
            'Country code', 'Reference date (as of 1 July)', 'Gender'])
        e.columns.name = 'Age'
        df_new = e.stack().reset_index().rename(columns={0: 'Population'})
        to_concat.append(df_new)

    df_all = pd.concat(to_concat, ignore_index=True)
    df_all = df_all.rename(columns={'Reference date (as of 1 July)': 'Year'})
    df_all.columns = list(map(to_concept_id, df_all.columns))

    # make age column sort correctly by changing to categorial dtype.
    df_all['age'] = df_all['age'].astype('category', categories=list(df_all['age'].unique()), ordered=True)

    df_all = df_all.sort_values(by=['country_code', 'year', 'age', 'gender'])

    # the only duplicates are in year 2015. There are both esitmated and observed data.
    # But both are same so we can drop them.
    df_all = df_all.drop_duplicates()
    # assert not np.any(df_all.duplicated(['country_code', 'year', 'age', 'gender']))

    return df_all


if __name__ == '__main__':

    print('reading source data...')
    print('\tboth sexes...')
    est_t, var_t = read_cleanup(source_t, 'both_sexes')
    print('\tmale...')
    est_m, var_m = read_cleanup(source_m, 'male')
    print('\tfemale...')
    est_f, var_f = read_cleanup(source_f, 'female')

    print('creating datapoint file...')
    dflist = [est_m, var_m, est_f, var_f]
    df_mf = extract_datapoints(dflist)
    for geo, idxs in df_mf.groupby(by='country_code').groups.items():
        path = os.path.join(out_dir, 
                            'ddf--datapoints--population--by--country_code-{}--year--gender--age.csv'.format(geo))
        to_save = df_mf.ix[idxs]
        to_save = to_save.sort_values(by=['country_code', 'year'])
        to_save.ix[idxs].to_csv(path, index=False, float_format='%.15g')

    df_t = extract_datapoints([est_t, var_t])
    df_t = df_t.drop('gender', axis=1)  # we don't need gender = both sexes in datapoint
    for geo, idxs in df_t.groupby(by='country_code').groups.items():
        path = os.path.join(out_dir, 
                            'ddf--datapoints--population--by--country_code-{}--year--age.csv'.format(geo))
        to_save = df_t.ix[idxs]
        to_save = to_save.sort_values(by=['country_code', 'year'])
        to_save.ix[idxs].to_csv(path, index=False, float_format='%.15g')

    print('creating concepts files...')
    concepts = extract_concepts(est_t)
    path = os.path.join(out_dir, 'ddf--concepts.csv')
    concepts.to_csv(path, index=False)

    print('creating entities files...')
    country = extract_entities_country(est_t, var_t)
    path = os.path.join(out_dir, 'ddf--entities--country_code.csv')
    country.to_csv(path, index=False)

    gender = extract_entities_gender()
    path = os.path.join(out_dir, 'ddf--entities--gender.csv')
    gender.to_csv(path, index=False)

    age = extract_entities_age(est_t)
    path = os.path.join(out_dir, 'ddf--entities--age.csv')
    age.to_csv(path, index=False)

    print('creating index files...')
    create_index_file(out_dir)
