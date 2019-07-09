#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'etl/notebooks'))
	print(os.getcwd())
except:
	pass

#%%
import pandas as pd
from ddf_utils.str import to_concept_id

#%%
import os

def create_dir(c, by):
    p = '../../{}--by--{}'.format(c, '--'.join(by))
    os.makedirs(p, exist_ok=True)
    return p

#%%
get_ipython().system('ls ../source/csv')


#%%
popbyagesex = pd.read_csv('../source/csv/WPP2019_PopulationBySingleAgeSex.csv', encoding='latin1')


#%%
popbyagesex.head()


#%%
from ddf_utils.str import to_concept_id


#%%
popbyagesex['AgeGrpSpan'].unique()


#%%
popbyagesex[popbyagesex['AgeGrpSpan'] == -1]['AgeGrp'].unique()


#%%
agegroup = popbyagesex[['AgeGrp']].drop_duplicates()
agegroup['age1yearinterval'] = agegroup['AgeGrp']
agegroup.columns = ['name', 'age1yearinterval']


#%%
agegroup.head()


#%%
agegroup[['age1yearinterval', 'name']].to_csv('../../ddf--entities--age1yearinterval.csv', index=False)


#%%
variants = popbyagesex[['VarID', 'Variant']].drop_duplicates()


#%%
variants


#%%
popbyagesex.columns


#%%
dp = popbyagesex[['LocID', 'Time', 'AgeGrp', 'PopMale', 'PopFemale', 'PopTotal']].copy()
# dp['AgeGrp'] = dp['AgeGrp'].map(to_concept_id)
dp.columns = ['location', 'time', 'age1yearinterval', 'pop_male', 'pop_female', 'pop_total']


#%%
dp = dp.set_index(['location', 'time', 'age1yearinterval'])


#%%
dfs = []
for c in dp:
    sex = c.split('_')[-1]
    df = dp[[c]].copy()
    df.columns = ['population']
    df['sex'] = sex
    df = df.set_index('sex', append=True)
    dfs.append(df)


#%%
dp_merged = pd.concat(dfs, sort=False)


#%%
dp_merged.head()


#%%
dp_merged.query('location == 900').head()


#%%
location_metadata = pd.read_excel('../source/WPP2019_F01_LOCATIONS.XLSX', skiprows=16)


#%%
location_metadata.columns


#%%
loc = location_metadata[['Region, subregion, country or area*',
                         'Location code',
                         'ISO3 Alpha-code',
                         'Parent Location code',
                         'Code',
                         'Name']].copy()


#%%
loc.head()


#%%
loc[loc['Name'] == 'Country/Area'].head()


#%%
countries = loc[loc['Name'] == 'Country/Area'].copy()
countries.columns = ['name', 'country', 'iso3', 'parent', 'code', 'type']
countries = countries[['country', 'name', 'iso3']]


#%%
countries.head()


#%%
countries['is--country'] = 'TRUE'


#%%
# countries.to_csv('../../ddf--entities--geo--country.csv', index=False)
countries[['country', 'name', 'iso3']].to_csv('../../ddf--entities--country.csv', index=False)


#%%
dp_country = dp_merged.reset_index()


#%%
dp_country = dp_country.rename(columns={'location': 'country', 'sex': 'gender'})


#%%
dp_country.head()


#%%
dp_total = dp_country[dp_country.gender == 'total']
dp_total = dp_total[['country', 'time', 'age1yearinterval', 'population']]

for c in countries['country'].unique():
    df = dp_total[dp_total['country'] == c]
    if not df.empty:
        df.to_csv(f'../../populations--by--country--time--age1yearinterval/ddf--datapoints--population--by--country-{c}--time--age1yearinterval.csv'
                  , index=False)


#%%
dp_genders = dp_country[dp_country.gender.isin(['male', 'female'])]

for c in countries['country'].unique():
    df = dp_genders[dp_genders['country'] == c]
    if not df.empty:
        df.to_csv(
            f'../../populations--by--country--time--age1yearinterval--gender/ddf--datapoints--population--by--country-{c}--time--age1yearinterval--gender.csv'
            , index=False)


#%%
countries.columns


#%%
peroidInd = pd.read_csv('../source/csv/WPP2019_Period_Indicators_Medium.csv', encoding='latin1')


#%%
peroidInd.head()


#%%
pi_concepts = {
    "TFR          ": "Total fertility (live births per woman)",
    "NRR          ": "Net reproduction rate (surviving daughters per woman)",
    "CBR          ": "Crude birth rate (births per 1,000 population)",
    "Births       ": "Number of births, both sexes combined (thousands)",
    "LEx          ": "Life expectancy at birth for both sexes combined (years)",
    "IMR          ": "Infant mortality rate, q(1), for both sexes combined (infant deaths per 1,000 live births)",
    "Q5           ": "Under-five mortality, 5q0, for both sexes combined (deaths under age five per 1,000 live births)",
    "CDR          ": "Crude death rate (deaths per 1,000 population)",
    "Deaths       ": "Number of deaths, both sexes combined (thousands)",
    "CNMR         ": "Net migration rate (per 1,000 population)",
    "NetMigrations": "Net number of migrants, both sexes combined (thousands)",
    "GrowthRate   ": "Average annual rate of population change (percentage)",
    "NatIncr      ": "Rate of natural increase (per 1,000 population)",
    "SRB          ": "Sex ratio at birth (male births per female births)",
    "MAC          ": "Female mean age of childbearing (years)"
}


#%%
pi_concepts = dict([(k.strip(), v) for k, v in pi_concepts.items()])


#%%
pi_concepts


#%%
c = pd.DataFrame.from_dict(pi_concepts, orient='index')
c.index = c.index.map(to_concept_id)
c['concept_type'] = 'measure'
c['domain'] = ''

print(c.to_csv())


#%%
peroidInd['VarID'].unique()


#%%
peroidInd['MidPeriod'].unique()


#%%
peroidInd_cty = peroidInd[peroidInd['LocID'].isin(countries.country)].drop(['Location', 'VarID', 'Variant', 'MidPeriod'], axis=1).copy()


#%%
peroidInd_cty['time'] = peroidInd_cty['Time'].map(lambda x: x.split('-')[0])


#%%
peroidInd_cty = peroidInd_cty.drop('Time', axis=1).set_index(['LocID', 'time'])


#%%
peroidInd_cty.head()


#%%
peroidInd_cty.index.names = ['country', 'time']


#%%
deaths_sex = peroidInd_cty[['DeathsMale', 'DeathsFemale']].copy()


#%%
deaths_sex_dp = []

m = deaths_sex[['DeathsMale']].copy()
m.columns = ['deaths']
m['gender'] = 'male'
m['freq'] = '5yearly'
m = m.set_index(['gender', 'freq'], append=True)

f = deaths_sex[['DeathsFemale']].copy()
f.columns = ['deaths']
f['gender'] = 'female'
f['freq'] = '5yearly'
f = f.set_index(['gender', 'freq'], append=True)

deaths_sex_dp.append(m)
deaths_sex_dp.append(f)


#%%
deaths_sex_dp = pd.concat(deaths_sex_dp, sort=False).dropna()


#%%
deaths_sex_dp.to_csv('../../deaths--by--country--time--gender--freq/ddf--datapoints--deaths--by--country--time--gender--freq.csv')


#%%
deaths_sex = peroidInd_cty[['LExMale', 'LExFemale']].copy()


#%%
deaths_sex_dp = []

m = deaths_sex[['LExMale']].copy()
m.columns = ['lex']
m['gender'] = 'male'
m['freq'] = '5yearly'
m = m.set_index(['gender', 'freq'], append=True)

f = deaths_sex[['LExFemale']].copy()
f.columns = ['lex']
f['gender'] = 'female'
f['freq'] = '5yearly'
f = f.set_index(['gender', 'freq'], append=True)

deaths_sex_dp.append(m)
deaths_sex_dp.append(f)


#%%
deaths_sex_dp = pd.concat(deaths_sex_dp, sort=False).dropna()


#%%
deaths_sex_dp.shape


#%%
get_ipython().system('mkdir ../../lex--by--country--time--gender--freq')


#%%
deaths_sex_dp.to_csv('../../lex--by--country--time--gender--freq/ddf--datapoints--lex--by--country--time--gender--freq.csv')


#%%
# other period indicators
peroidInd_cty_2 = peroidInd_cty.drop(['DeathsMale', 'DeathsFemale', 'LExMale', 'LExFemale'], axis=1)


#%%
for c in peroidInd_cty_2:
    c_ = to_concept_id(c)
    p = create_dir(c_, ['country', 'time', 'freq'])
    df = peroidInd_cty_2[[c]]
    df.columns = [c_]
    df = df.dropna()
    df['freq'] = '5yearly'
    df = df.set_index('freq', append=True)
    df.to_csv(os.path.join(p, 'ddf--datapoints--{}--by--{}.csv'.format(c_, '--'.join(['country', 'time', 'freq']))))


#%%
mort = pd.read_csv('../source/csv/WPP2019_LifeTable.csv', encoding='latin1')


#%%
mort.head()


#%%
mort.columns


#%%
mort['AgeGrp'].unique()


#%%
age5 = mort[['AgeGrp']].drop_duplicates()


#%%
age5.columns = ['name']
age5['age5yearinterval'] = age5['name'].map(to_concept_id)


#%%
age5[['age5yearinterval', 'name']].to_csv('../../ddf--entities--age5yearinterval.csv', index=False)


#%%
mort_concepts = {
    "mx": "Central death rate, nmx, for the age interval (x, x+n)",
    "qx": "Probability of dying (nqx), for an individual between age x and x+n",
    "px": "Probability of surviving, (npx), for an individual of age x to age x+n",
    "lx": "Number of survivors, (lx), at age (x) for 100000 births",
    "dx": "Number of deaths, (ndx), between ages x and x+n",
    "Lx": "Number of person-years lived, (nLx), between ages x and x+n",
    "Sx": "Survival ratio (nSx) corresponding to proportion of the life table population in age group (x, x+n) who are alive n year later",
    "Tx": "Person-years lived, (Tx), above age x",
    "ex": "Expectation of life (ex) at age x, i.e., average number of years lived subsequent to age x by those reaching age x",
    "ax": "Average number of years lived (nax) between ages x and x+n by those dying in the interval"
}


#%%
mort_concepts_rename = {
    "mx": "Central death rate",
    "qx": "Probability of dying",
    "px": "Probability of surviving",
    "lx": "Number of survivors",
    "dx": "Number of deaths",
    "Lx": "Number of person-years lived",
    "Sx": "Survival ratio",
    "Tx": "Person-years lived",
    "ex": "Expectation of life",
    "ax": "Average number of years lived"
}

mort_concepts_rename = dict([(k, to_concept_id(v)) for k, v in mort_concepts_rename.items()])


#%%
mort_concepts_rename


#%%
mort_concepts = dict([(mort_concepts_rename[k], v) for k, v in mort_concepts.items()])


#%%
mort_concepts


#%%
mort_concepts_df = pd.DataFrame.from_dict(mort_concepts, orient='index')
mort_concepts_df['concept_type'] = 'measure'
mort_concepts_df['domain'] = ''

print(mort_concepts_df.to_csv())


#%%
mort_df = mort[['LocID', 'Time', 'Sex']]


#%%
mort['VarID'].unique()


#%%
mort.columns


#%%
mort['Sex'].unique()


#%%
mort['SexID'].unique()


#%%
mort_total = mort[mort['Sex'] == 'Total'].copy()


#%%
mort_total = mort_total[['LocID', 'Time', 'AgeGrp', 'mx', 'qx', 'px', 'lx',
                         'dx', 'Lx', 'Sx', 'Tx', 'ex', 'ax']]


#%%
mort_total = mort_total.rename(columns={'LocID': 'country', 'Time': 'time', 'AgeGrp': 'age5yearinterval'}).rename(columns=mort_concepts_rename)


#%%
mort_total = mort_total[mort_total['country'].isin(countries.country)]
mort_total['age5yearinterval'] = mort_total['age5yearinterval'].map(to_concept_id)
mort_total['time'] = mort_total['time'].map(lambda x: x.split('-')[0])


#%%
mort_total = mort_total.set_index(['country', 'time', 'age5yearinterval'])

#%%
mort_total['freq'] = '5yearly'
mort_total = mort_total.set_index('freq', append=True)

#%%
for c in mort_total:
    by = mort_total.index.names
    p = create_dir(c, by=by)
    df = mort_total[[c]].dropna()
    df.to_csv(os.path.join(p, 'ddf--datapoints--{}--by--{}.csv'.format(c, '--'.join(by))))


#%%
mort_sex = mort[mort['Sex'] != 'Total'].copy()
mort_sex = mort_sex[['LocID', 'Time', 'AgeGrp', 'Sex', 'mx', 
                     'qx', 'px', 'lx', 'dx', 'Lx', 'Sx', 'Tx', 'ex', 'ax']]
mort_sex = mort_sex.rename(
    columns={'LocID': 'country', 
            'Time': 'time', 
            'Sex': 'gender', 
            'AgeGrp': 'age5yearinterval'}).rename(columns=mort_concepts_rename)

#%%
mort_sex = mort_sex[mort_sex['country'].isin(countries.country)]
mort_sex['age5yearinterval'] = mort_sex['age5yearinterval'].map(to_concept_id)
mort_sex['time'] = mort_sex['time'].map(lambda x: x.split('-')[0])
mort_sex['gender'] = mort_sex['gender'].str.lower()
mort_sex['freq'] = '5yearly'


#%%
mort_sex = mort_sex.set_index(['country', 'time', 'gender', 'age5yearinterval', 'freq'])

#%%
for c in mort_sex:
    by = mort_sex.index.names
    p = create_dir(c, by=by)
    df = mort_sex[[c]].dropna()
    df.to_csv(os.path.join(p, 'ddf--datapoints--{}--by--{}.csv'.format(c, '--'.join(by))))

#%%
