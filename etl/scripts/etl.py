# -*- coding: utf-8 -*-

"""
A script to parse UN WPP data files from https://population.un.org/wpp2019/Download/Standard/Population/
"""

import os
import os.path as osp
import logging
import numpy as np
import pandas as pd

from functools import partial

from ddf_utils.model.ddf import Entity, EntityDomain, Concept
from ddf_utils.str import format_float_digits, to_concept_id


logger = logging.getLogger("etl")

location_source = '../source/WPP2019_F01_LOCATIONS.XLSX'
metadata_file = '../source/metadata.xlsx'

output_dir = '../../'

CONCEPTS = dict()
ENTITYDOMAINS = dict()
AGE_GROUP_MAPPING = dict()


def remove_separators(df):
    """removing 'Label/Separator' from df"""
    if "Type" in df.columns:
        return df[df['Type'] != 'Label/Separator']
    return df


def read_un_xls(fp, **kwargs):
    df = pd.read_excel(fp,  **kwargs)
    return remove_separators(df)


def merge_xls_variants(fp, v1, v2, dims, axis=0):
    df1 = read_un_xls(fp, skiprows=16, sheet_name=v1).set_index(dims)
    df2 = read_un_xls(fp, skiprows=16, sheet_name=v2).set_index(dims)

    if axis == 0:
        df = pd.concat([df1, df2], axis=axis, sort=False)
        return df.reset_index().drop_duplicates(subset=dims)
    else:
        df = pd.merge(df1, df2, how='outer', left_index=True, right_index=True)
        return df.reset_index()


def create_output_dir(d, exist_ok=True):
    p = os.path.join(output_dir, d)
    os.makedirs(p, exist_ok=exist_ok)
    return p


def location_metadata() -> pd.DataFrame :
    loc = read_un_xls(location_source, sheet_name='DB', dtype=str)
    # print(loc.columns)
    res = loc.loc[:, :"WB_NoIncomeGroup"].copy()
    # cleanup spaces in strings
    for c in res.columns:
        res[c] = res[c].str.strip()
    return res


def create_geo_domain(loc_: pd.DataFrame) -> EntityDomain:
    """create the geo domain for location list"""
    domain = EntityDomain(id='geo', entities=[], props={'name': 'Geographic Locations'})
    loc = loc_.copy()

    loc = loc.set_index('LocID')

    # speical mapping for un developing groups
    special_un_group = {
        '1636': 'un_landlocked_group',
        '1637': 'un_landlocked_group'
    }

    # mapping for parent id and its entity set
    parentid_eset_mapping = {
        '1803': 'un_development_group',
        '902': 'un_less_developed_region',
        '1802': 'wb_income_group',
        '1517': 'wb_middle_income_country',
        '1840': 'geographic_region'
    }

    # mapping for location type and its entity set
    loctype_eset_mapping = {
        '12': 'sdg_region',
        '3': 'geographic_subregion',
        '4': 'country',
        '24': 'sdg_subregion'
    }

    # property columns and entity set mappings
    colname_entityset_mapping = {
        'ISO3_Code': 'iso3',
        'SubRegID': 'geographic_subregion',
        'SDGRegID': 'sdg_region',
        'GeoRegID': 'geographic_region',
        'MoreDev': 'un_development_group',
        'LessDev': 'un_development_group',
        'LeastDev': 'un_less_developed_region',
        'oLessDev': 'un_less_developed_region',
        # 'LessDev_ExcludingChina': 'un_development_group',  # NOTE: this overlaps with LessDev so we don't keep it.
        'LLDC': 'un_landlocked_group',
        'SIDS': 'un_landlocked_group',
        'WB_HIC': 'wb_income_group',
        'WB_MIC': 'wb_income_group',
        'WB_LIC': 'wb_income_group',
        'WB_NoIncomeGroup': 'wb_income_group',
        'WB_UMIC': 'wb_middle_income_country',
        'WB_LMIC': 'wb_middle_income_country'
    }

    for l, row in loc.iterrows():
        if l == '900':  # world
            ent = Entity(id=l, sets=['global'], props={'name': 'WORLD'}, domain='geo')
        else:
            loctype = row['LocType']
            if loctype == '25':  # Label/Separator
                continue

            if l in special_un_group:
                sets = [special_un_group[l]]
            elif loctype in loctype_eset_mapping:
                sets = [loctype_eset_mapping[loctype]]
            else:
                parentid = row['ParentID']
                sets = [parentid_eset_mapping[parentid]]

            props = dict()
            props['name'] = row['Location']
            for k, v in colname_entityset_mapping.items():
                if not pd.isnull(row[k]):
                    assert v not in props.keys(), f"duplicated key {v} for {l}: {row[k]}, {props[v]}"
                    props[v] = row[k]
            ent = Entity(id=l, sets=sets, props=props, domain='geo')

        domain.add_entity(ent)


    return domain


def assert_path_not_exist(p):
    assert not os.path.exists(p), f"{p} already exists!"


def serve_dp(df, concept, by, path, split_domain_set=None):
    df[concept] = df[concept].map(format_float_digits)
    cols = [*by, concept]

    if split_domain_set:
        d, s = split_domain_set
        domain = ENTITYDOMAINS[d]
        for eset in domain.entity_sets:
            if eset == s:
                ents = [e.id for e in domain[eset]]
                for g in df[df[d].isin(ents)][d].unique():
                    by_new = by.copy()
                    by_new[by.index(d)] = f'{s}-{g}'
                    cols_new = cols.copy()
                    cols_new[by.index(d)] = eset
                    by_str = '--'.join(by_new)
                    fp = os.path.join(path, f'ddf--datapoints--{concept}--by--{by_str}.csv')
                    assert_path_not_exist(fp)
                    df_ = df[df[d] == g].copy()
                    df_ = df_.rename(columns={d: eset})
                    df_[cols_new].to_csv(fp, index=False)
            else:
                ents = [e.id for e in domain[eset]]
                by_new = by.copy()
                by_new[by.index(d)] = eset
                cols_new = cols.copy()
                cols_new[by.index(d)] = eset
                by_str = '--'.join(by_new)
                fp = os.path.join(path, f'ddf--datapoints--{concept}--by--{by_str}.csv')
                assert_path_not_exist(fp)
                df_ = df[df[d].isin(ents)].copy()
                df_ = df_.rename(columns={d: eset})
                df_[cols_new].to_csv(fp, index=False)
    else:
        by_str = '--'.join(by)
        fp = os.path.join(path, f'ddf--datapoints--{concept}--by--{by_str}.csv')
        assert_path_not_exist(fp)
        df[cols].to_csv(fp, index=False)


def combine_male_female(df_male, df_female):
    df_male = append_col(df_male, gender=1)
    df_female = append_col(df_female, gender=2)
    return pd.concat([df_male, df_female], sort=False, ignore_index=True)


def start_year(yearspan):
    if isinstance(yearspan, str):
        return yearspan.split("-")[0]
    return yearspan


def standardise_yearincolumn(data, dims, concept, fiveyr=False, drop_columns=None, rename=None):
    """pre-processing for xls with year in column"""
    def _rem(s):
        if s.endswith('_x') or s.endswith('_y'):
            return s[:-2]
        else:
            return s

    df = data.copy()
    if drop_columns:
        # also drops pd.merge result, with _x and _y
        dc = list()
        for d in drop_columns:
            if d in df.columns:
                dc.append(d)
            if d + '_x' in df.columns:
                dc.append(d + '_x')
            if d + '_y' in df.columns:
                dc.append(d + '_y')
        if len(dc) > 0:
            df = df.drop(dc, axis=1)
    if rename:
        df = df.rename(columns=rename)

    df = df.set_index(dims).stack()
    df = df.reset_index()
    cols = [*dims, 'time', concept]
    df.columns = cols
    # handle duplicated columns in time
    df['time'] = df['time'].map(_rem)
    if fiveyr:
        df['time'] = df['time'].map(start_year).map(int)
    else:
        df['time'] = df['time'].map(int)
    df = df.drop_duplicates()
    return df


def append_col(df_, **kwargs):
    df = df_.copy()
    for k, v in kwargs.items():
        df[k] = v
    return df


def age_group_to_entity_id(s):
    if 'age' in s:
        return AGE_GROUP_MAPPING[s]
    else:
        return s.replace('-', '_').replace('+', 'plus')


def standardise_ageincolumn(data, dims, concept, age='age1yearinterval',
                            fiveyr=False, drop_columns=None, rename=None):
    global ENTITYDOMAINS
    if age not in ENTITYDOMAINS:
        ENTITYDOMAINS[age] = EntityDomain(id=age, entities=[])

    all_ages = []

    def age_column_to_entity(s):
        sid = age_group_to_entity_id(s)
        all_ages.append((sid, s))
        return sid

    df = data.copy()
    if drop_columns:
        dc = list()
        for d in drop_columns:
            if d in df.columns:
                dc.append(d)
        if len(dc) > 0:
            df = df.drop(dc, axis=1)
    if rename:
        df = df.rename(columns=rename)

    df = df.set_index(dims).stack()
    df = df.reset_index()
    cols = [*dims, age, concept]
    df.columns = cols
    if fiveyr:
        df['time'] = df['time'].map(start_year).map(int)
    else:
        df['time'] = df['time'].map(int)

    df[age] = df[age].map(age_column_to_entity)

    # update age domain
    age_df = pd.DataFrame(all_ages, columns=[age, 'name']).drop_duplicates()
    for _, row in age_df.iterrows():
        ent = Entity(id=row[age], domain=age, sets=[], props={'name': row['name']})
        ENTITYDOMAINS[age].add_entity(ent)

    return df


def standardise_multiindicator(data, dims, rename, drop_columns=None):
    df = data.copy()
    if drop_columns:
        dc = list()
        for d in drop_columns:
            if d in df.columns:
                dc.append(d)
        if len(dc) > 0:
            df = df.drop(dc, axis=1)
    df = df.rename(columns=rename)
    df = df.set_index(dims)

    res = dict()
    for c in df:
        res[c] = df[[c]].reset_index()
    return res


# TODO: move this to ddf_utils?
def serve_entity_domain(domain, out_dir, split_sets=False):
    name = domain.id
    if split_sets:
        for s in domain.entity_sets:
            ent = domain.get_entity_set(s)
            ent_df = pd.DataFrame.from_records([e.to_dict() for e in ent])
            ent_df.to_csv(f'../../ddf--entities--{name}--{s}.csv', index=False)
    else:
        ent = domain.entities
        ent_df = pd.DataFrame.from_records([e.to_dict() for e in ent])
        ent_df.to_csv(f'../../ddf--entities--{name}.csv', index=False)


def select_func(filetype, freq):
    age = None
    dims = ['geo', 'time']
    fiveyr = False

    if filetype == 'age1incolumn':
        f = standardise_ageincolumn
        age = 'age1yearinterval'
    elif filetype == 'age5incolumn':
        f = standardise_ageincolumn
        age = 'age5yearinterval'
    elif filetype == 'agebroadincolumn':
        f = standardise_ageincolumn
        age = 'agebroad'
    elif filetype == 'year1incolumn':
        f = standardise_yearincolumn
        dims = ['geo']
    elif filetype == 'year5incolumn':
        f = standardise_yearincolumn
        dims = ['geo']
    else:
        raise ValueError(f'{filetype} not recongized')

    if freq == '5yr':
        fiveyr = True

    if age:
        return partial(f, age=age, dims=dims, fiveyr=fiveyr)
    else:
        return partial(f, dims=dims, fiveyr=fiveyr)


def get_by(filetype, freq, gender=False):
    if 'year' in filetype:
        by = ['geo', 'time']
    elif 'age1' in filetype:
        by = ['geo', 'time', 'age1yearinterval']
    elif 'age5' in filetype:
        by = ['geo', 'time', 'age5yearinterval']
    elif 'agebroad' in filetype:
        by = ['geo', 'time', 'agebroad']
    else:
        raise ValueError(f'{filetype} not recongized')
    if freq == '5yr':
        by.append('freq')
    if gender:
        by.append('gender')

    return by


def process_file(filename, indicator, filetype, freq, rename, drop_cols):
    func = select_func(filetype, freq)
    if 'year' in filetype:
        data = merge_xls_variants(filename, 'ESTIMATES', 'MEDIUM VARIANT',
                                  dims=['Country code'], axis=1)
        df = func(data=data, rename=rename, drop_columns=drop_cols, concept=indicator)
    elif 'age' in filetype:
        if freq == '1yr':
            data = merge_xls_variants(filename, 'ESTIMATES', 'MEDIUM VARIANT',
                                      dims=['Country code', 'Reference date (as of 1 July)'])
        else:
            try:
                data = merge_xls_variants(filename, 'ESTIMATES', 'MEDIUM VARIANT',
                                          dims=['Country code', 'Period'])
            except KeyError:
                data = merge_xls_variants(filename, 'ESTIMATES', 'MEDIUM VARIANT',
                                          dims=['Country code', 'Reference date (as of 1 July)'])
        df = func(data=data, rename=rename, drop_columns=drop_cols, concept=indicator)
    if freq == '5yr':
        df = append_col(df, freq='5yr')

    return df


def process_one_line_metadata(md, indicator, filetype, freq, rename, drop_cols, serve=True, split_domain_set=('geo', None)):
    filename = os.path.join('../source', md.iloc[0]['file'])
    print(filename)
    df = process_file(filename, indicator, filetype, freq, rename, drop_cols)
    if serve:
        serve_dp_2(df, indicator, filetype, freq, split_domain_set=split_domain_set)
    return df


def serve_dp_2(df, indicator, filetype, freq, gender=False, split_domain_set=('geo', None)):
    by = get_by(filetype, freq, gender=gender)
    by_str = '--'.join(by)
    path = create_output_dir(f'{indicator}--by--{by_str}')
    cols = [*by, indicator]
    df = df[cols]
    # double checking
    if freq == '5yr':
        assert 1951 not in df['time'].values
    else:
        assert 1951 in df['time']
    serve_dp(df, indicator, by, path, split_domain_set=split_domain_set)


def process_file_demograph():
    print('running on demography indicators...')
    demograph_mappings = pd.read_excel('../source/metadata.xlsx', sheet_name='DemographyFormat')

    # update concepts
    global CONCEPTS
    for _, row in demograph_mappings.iterrows():
        indicator = row['indicator']
        if indicator not in CONCEPTS:
            CONCEPTS[indicator] = Concept(id=indicator, concept_type='measure', props={'name': row['name']})

    drop_cols = ['Index', 'Variant', 'Region, subregion, country or area *',
                 'Notes', 'Type', 'Parent code', 'Total']
    rename = {'Country code': 'geo',
              'Period': 'time',
              'Reference date (as of 1 July)': 'time',
              'Reference date (1 January - 31 December)': 'time'}

    # demography indicators
    demograph_data = merge_xls_variants('../source/WPP2019_INT_F01_ANNUAL_DEMOGRAPHIC_INDICATORS.xlsx',
                                        'ESTIMATES', 'MEDIUM VARIANT',
                                        dims=['Country code', 'Reference date (1 January - 31 December)'])
    df_dict = standardise_multiindicator(demograph_data, ['geo', 'time'], rename, drop_cols)
    gs = demograph_mappings.groupby(['indicator'])
    for g, md in gs:
        print(g)
        nogender = md[pd.isnull(md['gender'])]
        assert len(nogender) == 1
        name = md.iloc[0]['name']
        by = ['geo', 'time']
        by_str = '--'.join(by)
        path = create_output_dir(f'{g}--by--{by_str}')
        cols = [*by, g]
        df = df_dict[name].rename(columns={name: g})[cols]
        df[g] = df[g].map(format_float_digits)
        serve_dp(df, g, by, path, split_domain_set=('geo', None))

        hasgender = md[~pd.isnull(md['gender'])]
        if hasgender.empty:
            continue
        by = ['geo', 'time', 'gender']
        by_str = '--'.join(by)
        path = create_output_dir(f'{g}--by--{by_str}')
        cols = [*by, g]
        male_name = md[md['gender'] == 'male'].iloc[0]['name']
        female_name = md[md['gender'] == 'female'].iloc[0]['name']
        df_male = df_dict[male_name].rename(columns={male_name: g}).assign(gender=1)[cols]
        df_female = df_dict[female_name].rename(columns={female_name: g}).assign(gender=2)[cols]
        df = pd.concat([df_male, df_female], sort=False, ignore_index=True)
        serve_dp(df, g, by, path, split_domain_set=('geo', None))


def process_file_dep_ratio():
    print("running on dependency ratio indicators..")
    drop_cols = ['Index', 'Variant', 'Region, subregion, country or area *',
                 'Notes', 'Type', 'Parent code', 'Total']
    rename = {'Country code': 'geo',
              'Period': 'time',
              'Reference date (as of 1 July)': 'time',
              'Reference date (1 January - 31 December)': 'time'}
    # dep ratios
    dependency_meta = pd.read_excel('../source/metadata.xlsx', sheet_name='DependencyFormat')
    dependency_mappings = dependency_meta.set_index('name')['indicator'].to_dict()
    # update concepts
    global CONCEPTS
    for _, row in dependency_meta.iterrows():
        indicator = row['indicator']
        if indicator not in CONCEPTS:
            CONCEPTS[indicator] = Concept(id=indicator, concept_type='measure', props={'name': row['name']})

    dependency_total_data = merge_xls_variants(
        '../source/WPP2019_INT_F02C_1_ANNUAL_POPULATION_INDICATORS_DEPENDENCY_RATIOS_BOTH_SEXES.xlsx',
        'ESTIMATES', 'MEDIUM VARIANT',
        dims=['Country code', 'Reference date (as of 1 July)'])
    dependency_male_data = merge_xls_variants(
        '../source/WPP2019_INT_F02C_2_ANNUAL_POPULATION_INDICATORS_DEPENDENCY_RATIOS_MALE.xlsx',
        'ESTIMATES', 'MEDIUM VARIANT',
        dims=['Country code', 'Reference date (as of 1 July)'])
    dependency_female_data = merge_xls_variants(
        '../source/WPP2019_INT_F02C_3_ANNUAL_POPULATION_INDICATORS_DEPENDENCY_RATIOS_FEMALE.xlsx',
        'ESTIMATES', 'MEDIUM VARIANT',
        dims=['Country code', 'Reference date (as of 1 July)'])

    by = ['geo', 'time']
    by_str = '--'.join(by)
    total_dict = standardise_multiindicator(dependency_total_data, by, rename, drop_cols)
    for k, df in total_dict.items():
        indicator = dependency_mappings[k]
        path = create_output_dir(f'{indicator}--by--{by_str}')
        cols = [*by, indicator]
        df_ = df.rename(columns={k: indicator})
        serve_dp(df_, indicator, by, path, split_domain_set=('geo', None))

    male_dict = standardise_multiindicator(dependency_male_data, by, rename, drop_cols)
    female_dict = standardise_multiindicator(dependency_female_data, by, rename, drop_cols)

    assert len(male_dict.keys()) == len(female_dict.keys())

    for k, male_df in male_dict.items():
        indicator = dependency_mappings[k]
        female_df = female_dict[k]

        by_gender = ['geo', 'time', 'gender']
        by_str = '--'.join(by_gender)
        path = create_output_dir(f'{indicator}--by--{by_str}')
        cols = [*by_gender, indicator]

        mdf = male_df.rename(columns={k: indicator}).assign(gender=1)[cols]
        fdf = female_df.rename(columns={k: indicator}).assign(gender=2)[cols]

        df = pd.concat([mdf, fdf], sort=False, ignore_index=True)

        serve_dp(df, indicator, by_gender, path, split_domain_set=('geo', None))


def create_agebroad_domain(m: pd.DataFrame):
    ents = []
    for _, row in m.iterrows():
        ents.append(Entity(id=row['agebroad'], props={'name': row['name']}, domain='agebroad', sets=[]))

    return EntityDomain(id='agebroad', entities=ents)


def process_file_with_one_indicator():
    print("running on files with one indicator...")
    meta = pd.read_excel('../source/metadata.xlsx', sheet_name='New')
    global AGE_GROUP_MAPPING
    global ENTITYDOMAINS
    global CONCEPTS
    age_group_md = pd.read_excel('../source/metadata.xlsx', sheet_name='BroadAgeMap')
    AGE_GROUP_MAPPING = age_group_md.set_index('name')['agebroad'].to_dict()
    # generate ageboard entities
    ENTITYDOMAINS['agebroad'] = create_agebroad_domain(age_group_md)

    # processing files with only one indicator
    gs = meta.groupby(['indicator', 'type', 'freq'])

    drop_cols = ['Index', 'Variant', 'Region, subregion, country or area *',
                 'Notes', 'Type', 'Parent code', 'Total']
    rename = {'Country code': 'geo',
              'Period': 'time',
              'Reference date (as of 1 July)': 'time',
              'Reference date (1 January - 31 December)': 'time'}
    for g, md in gs:
        indicator, filetype, freq = g
        print(g)
        if filetype == 'multipleindicator':  # will process these later
            print("skipped for later")
            continue
        if indicator not in CONCEPTS:
            CONCEPTS[indicator] = Concept(id=indicator, concept_type='measure', props={'name': md['table_name'].unique()[0]})
        # handel file without gender dimension
        nogender = md[pd.isnull(md['gender'])]
        assert len(nogender) == 1
        if indicator == 'population' and freq == '1yr':
            process_one_line_metadata(nogender, indicator, filetype, freq, rename, drop_cols, split_domain_set=('geo', 'country'))
        else:
            process_one_line_metadata(nogender, indicator, filetype, freq, rename, drop_cols)

        # then, files with gender dimension
        hasgender = md[~pd.isnull(md['gender'])]
        if hasgender.empty:
            continue
        male_md = hasgender[hasgender['gender'] == 'male']
        male_df = process_one_line_metadata(male_md, indicator, filetype, freq, rename,
                                            drop_cols, serve=False)

        female_md = hasgender[hasgender['gender'] == 'female']
        female_df = process_one_line_metadata(female_md, indicator, filetype, freq,
                                              rename, drop_cols, serve=False)
        combine_df = combine_male_female(male_df, female_df)
        if indicator == 'population' and freq == '1yr':
            serve_dp_2(combine_df, indicator, filetype, freq, gender=True, split_domain_set=('geo', 'country'))
        else:
            serve_dp_2(combine_df, indicator, filetype, freq, gender=True)


def main():
    global ENTITYDOMAINS
    global CONCEPTS
    global AGE_GROUP_MAPPING

    loc = location_metadata()
    domain = create_geo_domain(loc)
    ENTITYDOMAINS['geo'] = domain

    # add freq/gender domain
    ENTITYDOMAINS['freq'] = EntityDomain(id='freq',
                                         entities=[Entity(id='5yr', props={'name': '5yearly'},
                                                          domain='freq', sets=[])],
                                         props={'name': 'Freq'})
    ENTITYDOMAINS['gender'] = EntityDomain(id='gender',
                                           entities=[Entity(id='1', props={'name': 'male'},
                                                            domain='gender', sets=[]),
                                                     Entity(id='2', props={'name': 'female'},
                                                            domain='gender', sets=[])],
                                           props={'name': 'Gender'})

    # process datapoints
    process_file_demograph()
    process_file_dep_ratio()
    process_file_with_one_indicator()

    # serving entities
    for k, domain in ENTITYDOMAINS.items():
        CONCEPTS[k] = Concept(id=k, concept_type='entity_domain', props=domain.props)
        if len(domain.entity_sets) > 0:
            for s in domain.entity_sets:
                concept_name = s.replace('_', ' ').title().replace('Un', 'UN').replace('Wb', 'WB').replace('Sdg', 'SDG')
                print(concept_name)
                CONCEPTS[s] = Concept(id=s, concept_type='entity_set', props={'name': concept_name, 'domain': k})
                df = pd.DataFrame.from_dict(domain.to_dict(eset=s))
                df.to_csv(osp.join(output_dir, f'ddf--entities--{k}--{s}.csv'), index=False)
        else:
            df = pd.DataFrame.from_dict(domain.to_dict())
            df.to_csv(osp.join(output_dir, f'ddf--entities--{k}.csv'), index=False)

    # serving concepts
    missing_concepts = {'time': ['Time', 'time'],
                        'name': ['Name', 'string'],
                        'domain': ['Domain', 'string'],
                        'iso3': ['ISO3', 'string']}
    for k, v in missing_concepts.items():
        CONCEPTS[k] = Concept(id=k, concept_type=v[1], props={'name': v[0]})

    cdf = pd.DataFrame.from_records([c.to_dict() for c in CONCEPTS.values()])
    cdf.sort_values(by='concept').to_csv(osp.join(output_dir, 'ddf--concepts.csv'), index=False)


if __name__ == '__main__':
    main()
