import pandas as pd

# Internal file for how we transform "the" lab DHS csv file to the simplified version used in the project
# This is a one-time operation, so it is not necessary to run this file again
# The file is kept for documentation purposes

if __name__ == '__main__':
    df = pd.read_csv('/mimer/NOBACKUP/groups/globalpoverty1/markus/temporal-vit/dhs_data.csv')

    def get_survey_name(source):
        dhs_num, survey_name, dhs_type, _ = source[96:].split('/')
        return survey_name + ' ' + dhs_type

    def fix_group(g):

        # Get the common cluster information
        first = g.iloc[0]
        cluster = first[['ClusterID', 'lon', 'lat', 'rural', 'RegionID', 'country']]
        cluster.columns = ['cluster_id', 'lon', 'lat', 'rural', 'region_id', 'country']

        # Get a interpretable survey name
        cluster['survey'] = get_survey_name(first['source'])

        # Get the mean year and month of the survey
        cluster['month'] = int(g['month.of.interview'].mean())
        cluster['year'] = int(g['year.of.interview'].mean())
        cluster['iwi'] = g['iwi'].mean()

        return cluster


    new_df = df.groupby(['ClusterID', 'source']).apply(fix_group).reset_index(drop=True)
    new_df.to_csv('/mimer/NOBACKUP/groups/globalpoverty1/markus/impute_aware_ate/dhs_data.csv', index=False)