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
source = '../source/WPP2015_INT_F03_1_POPULATION_BY_AGE_ANNUAL_BOTH_SEXES.XLS'
out_dir = '../../'  # path for outputs


def extract_entities_country(data_est, data_var):
    """extract country entities from source.

    data_est is data from estimates tab.
    data_var is from medium variant tab.

    we assume that both tab should have same entities.
    """
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


def extract_concepts(data_est, data_var):
    """extract concepts from source.

    data_est is data from estimates tab.
    data_var is from medium variant tab.

    this function will change the input dataframes in place.
    """

    # rename the columns. Because in the source file the column names for data
    # are only the age number, so we need to add more info to them.
    cols_1 = data_est.columns
    cols_new_1 = [*cols_1[:4], *(map(lambda x: 'Total Population aged ' + x + ' (Number)', cols_1[4:]))]

    cols_2 = data_var.columns
    cols_new_2 = [*cols_2[:4], *(map(lambda x: 'Total Population aged ' + x + ' (Number)', cols_2[4:]))]

    # the only difference should be the 80plus group
    # which is in Estimates but not in medium variant.
    assert len(cols_new_1) == len(cols_new_2) + 1

    data_est.columns = cols_new_1
    data_var.columns = cols_new_2

    data_est.columns = data_est.columns.map(to_concept_id)
    data_var.columns = data_var.columns.map(to_concept_id)

    data_est = data_est.rename(columns={'reference_date_as_of_1_july': 'year'})
    data_var = data_var.rename(columns={'reference_date_as_of_1_july': 'year'})

    # now construct the concepts dataframe.
    concs = pd.DataFrame([], columns=['name', 'concept'])

    concs['name'] = cols_new_1
    concs['concept'] = data_est.columns
    concs['concept_type'] = 'string'

    concs['concept_type'].iloc[2] = 'entity_domain'
    concs['concept_type'].iloc[3] = 'time'
    concs['concept_type'].iloc[4:] = 'measure'

    concs['unit'] = 'thousands'
    concs['unit'].iloc[:4] = np.nan

    concs = concs.append(pd.DataFrame([
        ['unit', 'Unit', 'string', np.nan],
        ['name', 'Name', 'string', np.nan]], columns=concs.columns))

    return concs.sort_index()


def extract_datapoints(data_est, data_var):
    """extract datapoints

    data_est is data from estimates tab.
    data_var is from medium variant tab.
    """
    dp_1 = data_est.drop('major_area_region_country_or_area', axis=1)
    dp_2 = data_var.drop('major_area_region_country_or_area', axis=1)

    dp_1 = dp_1.rename(columns={'reference_date_as_of_1_july': 'year'})
    dp_2 = dp_2.rename(columns={'reference_date_as_of_1_july': 'year'})

    dp = pd.concat([dp_1, dp_2])

    dp = dp.set_index(['country_code', 'year', 'variant'])

    for name, df in dp.iteritems():
        yield (name, df.reset_index().sort_values(by=['country_code', 'year']).dropna())


if __name__ == '__main__':

    print('reading source files...')
    data_est = pd.read_excel(source, sheetname='ESTIMATES', skiprows=16, na_values='…')
    data_var = pd.read_excel(source, sheetname='MEDIUM VARIANT', skiprows=16, na_values='…')

    # rename/drop some columns.
    # for 80+ and 100+ groups, rename to 80plus and 100plus
    data_est = data_est.drop(['Index', 'Notes'], axis=1)
    data_var = data_var.drop(['Index', 'Notes'], axis=1)

    data_est = data_est.rename(columns={'80+': '80plus', '100+': '100plus'})
    data_var = data_var.rename(columns={'100+': '100plus'})

    print('creating concept files...')
    path = os.path.join(out_dir, 'ddf--concepts.csv')
    concepts = extract_concepts(data_est, data_var)
    concepts.to_csv(path, index=False)

    print('creating entities files...')
    path = os.path.join(out_dir, 'ddf--entities--country_code.csv')
    ent = extract_entities_country(data_est, data_var)
    ent.to_csv(path, index=False)

    print('creating datapoint files...')
    for name, df in extract_datapoints(data_est, data_var):
        path = os.path.join(out_dir, 'ddf--datapoints--{}--by--country_code--year.csv'.format(name))
        df.to_csv(path, index=False)

    print('creating index file...')
    create_index_file(out_dir)

    print('Done.')
