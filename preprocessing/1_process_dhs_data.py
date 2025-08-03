import pandas as pd
import configparser
import os
from tqdm import tqdm

# Transform the DHS data exported by the R script into a simplified version
# The simplified version is grouped at the cluster level and contains only the relevant information
# that is, cluster ID, coordinates, rural/urban status, region ID, country, survey name, month and year of the survey,
# and the IWI (International Wealth Index) of the cluster.

def get_survey_name(source):
    dhs_num, survey_name, dhs_type, _ = source.split('/')[-4:]
    return survey_name + ' ' + dhs_type

def fix_group(g):

    # Get the common cluster information
    first = g.iloc[0]
    cluster = first[['ClusterID', 'lon', 'lat', 'rural', 'RegionID', 'country']]

    # Get a interpretable survey name
    cluster['survey'] = get_survey_name(first['source'])

    # Get the mean year and month of the survey
    cluster['month'] = int(g['month.of.interview'].mean())
    cluster['year'] = int(g['year.of.interview'].mean())
    cluster['iwi'] = g['iwi'].mean()

    return cluster

if __name__ == '__main__':

    print('Reading DHS data and transforming it to a simplified version...')

    # Set up the tqdm progress bar
    tqdm.pandas()

    # Read config file
    config = configparser.ConfigParser()
    config.read('../config.ini')
    DATA_DIR = config['PATHS']['DATA_DIR']

    # Read the exported DHS data
    df = pd.read_csv(os.path.join(DATA_DIR, 'raw_household_dhs_data.csv'))

    # Convert the 'rural' column to boolean
    df['rural'] = df['rural'] = df['rural'].apply(lambda x: x == 'Rural')

    # A few samples are missing GPS coordinates, i.e., their 'lat' and 'lon' values are zero. We drop these
    df = df[~((abs(df['lat']) < 0.01) & (abs(df['lon']) < 0.01))]
    df = df.reset_index(drop=True)

    # Group data at the cluster level and retain only the relevant information
    new_df = df.groupby(['ClusterID', 'source']).progress_apply(fix_group).reset_index(drop=True)
    new_df.columns = ['cluster_id', 'lon', 'lat', 'rural', 'region_id', 'country', 'survey', 'month', 'year', 'iwi']
    
    # Remove slashes from cluster ID's for easier processing
    new_df['cluster_id'] = new_df['cluster_id'].apply(lambda x: str(x).replace('/', '.'))

    new_df.to_csv(os.path.join(DATA_DIR, 'dhs_data.csv'), index=False)