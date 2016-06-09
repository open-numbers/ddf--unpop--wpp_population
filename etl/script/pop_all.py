# -*- coding: utf-8 -*-
"""Transform the Interpolated Indicator data set to DDF format
Link: http://esa.un.org/unpd/wpp/Download/Standard/Interpolated/
"""

import pandas as pd
import numpy as np
import os
from ddf_utils.str import to_concept_id
from ddf_utils.index import create_index_file


# configuration of file path
source_t = '../source/WPP2015_INT_F03_1_POPULATION_BY_AGE_ANNUAL_BOTH_SEXES.XLS'
source_m = '../source/WPP2015_INT_F03_2_POPULATION_BY_AGE_ANNUAL_MALE.XLS'
source_f = '../source/WPP2015_INT_F03_3_POPULATION_BY_AGE_ANNUAL_FEMALE.XLS'
out_dir = '../../'  # path for outputs


def read_cleanup(source, gender):
    data_est = pd.read_excel(source, sheetname='ESTIMATES', skiprows=16, na_values='…')
    data_var = pd.read_excel(source, sheetname='MEDIUM VARIANT', skiprows=16, na_values='…')

    data_est['Gender'] = gender
    data_var['Gender'] = gender

    # rename/drop some columns.
    # for 80+ and 100+ groups, rename to 80plus and 100plus
    data_est = data_est.drop(['Index', 'Notes'], axis=1)
    data_var = data_var.drop(['Index', 'Notes'], axis=1)

    data_est = data_est.rename(columns={'80+': '80plus', '100+': '100plus'})
    data_var = data_var.rename(columns={'100+': '100plus'})

    return (data_est, data_var)


def extract_concepts(data):

    data_ = data.rename(columns={
        'Major area, region, country or area *': 'Name',
        'Reference date (as of 1 July)': 'year'
    })
    concept_name = data_.columns[:4]
    concept_name.append('Unit')
    concept_name.append('Population')
    concept_name.append('Gender')
    concept_name.append('Age')
    concepts = map(to_concept_id, concept_name)

    # now construct the dataframe
    cdf = pd.DataFrame([], columns=['concept', 'concept_type', 'name'])
    cdf['concept'] = concepts
    cdf['name'] = concept_name

    cdf['concept_type'] = 'string'
    cdf['concept_type'].iloc[6] = 'measure'

    cdf['concept_type'].iloc[7:] = 'entity_domain'

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
    df['gender'] = ['both_sexes', 'male', 'female']
    df['name'] = ['Both sexes', 'Male', 'Female']

    return df


def extract_entities_age(data_est):
    """extract ages from estimates tab of source data."""

    df = pd.DataFrame([], columns=['age', 'name'])
    df['age'] = data_est.columns[4:]

    df['name'] = 'Age ' + df['age']
    return df


if __name__ == '__main__':
    data_

