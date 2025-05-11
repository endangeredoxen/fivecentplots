import json

# v0.5
with open('/Users/steve/Desktop/0001_v0.5.json', 'r') as file:
    data_v0_5 = json.load(file)
df_0p5 = pd.DataFrame(data_v0_5['benchmarks'])
df_0p5['rounds'] = df_0p5['stats'].apply(lambda x: x['rounds'])
df_0p5 = df_0p5.loc[df_0p5.rounds < 500]
df_0p5['data'] = df_0p5['stats'].apply(lambda x: x['data'])
df_0p5['plot_type'] = df_0p5['fullname'].apply(lambda x: x.split('/')[1].split('::')[0].split('test_')[1].split('.py')[0])
df_0p5 = df_0p5[['plot_type', 'name', 'data']]
df_0p5 = df_0p5.explode('data')
df_0p5['version'] = '0.5'
# df_0p5.set_index(['plot_type', 'name'])

# v0.6
with open('/Users/steve/Desktop/0001_v0.6.json', 'r') as file:
    data_v0_6 = json.load(file)
df_0p6 = pd.DataFrame(data_v0_6['benchmarks'])
df_0p6['rounds'] = df_0p6['stats'].apply(lambda x: x['rounds'])
df_0p6 = df_0p6.loc[df_0p6.rounds < 500]
df_0p6['data'] = df_0p6['stats'].apply(lambda x: x['data'])
df_0p6['plot_type'] = df_0p6['fullname'].apply(lambda x: x.split('/')[1].split('::')[0].split('test_')[1].split('.py')[0])
df_0p6 = df_0p6[['plot_type', 'name', 'data']]
df_0p6 = df_0p6.explode('data')
df_0p6['version'] = '0.6'
# df_0p6.set_index(['plot_type', 'name'])

# First, concatenate the dataframes vertically
combined_df = pd.concat([df_0p5, df_0p6], ignore_index=True)

# Create a key from plot_type and name
combined_df['key'] = combined_df['plot_type'] + '_' + combined_df['name']

# Count how many versions each key appears in
key_counts = combined_df.groupby('key')['version'].nunique()

# Filter to keep only keys that appear in both versions (count = 2)
keys_in_both = key_counts[key_counts == 2].index.tolist()

# Filter the combined dataframe to keep only rows with those keys
result_df = combined_df[combined_df['key'].isin(keys_in_both)]

# Drop the temporary key column
result_df = result_df.drop('key', axis=1)

# Optional: Sort by plot_type, name, and version for better readability
df = result_df.sort_values(['plot_type', 'name', 'version'])


fcp.set_theme('gray')
fcp.KWARGS['engine'] = 'mpl'

# fcp.boxplot(df, y='data', groups=['plot_type', 'name', 'version'], filter='plot_type=="plot"', ax_size='auto')
fcp.boxplot(df, y='data', groups=['plot_type', 'name', 'version'], filter='plot_type=="plot"', ax_size='auto',
            ymax='q0.99', ymin=0, legend='version')

df.loc[:, ['plot_type', 'version', 'data']].groupby(['plot_type', 'version']).mean()

fcp.boxplot(df, y='data', groups=['plot_type', 'version'], ymax='q0.99', ymin=0, legend='version')

# drop plots that are not in both versions
# stats

## bugs
# ymin not being applied
# auto didn't work
# plotly takes forever

compare = pd.pivot_table(df.groupby(['plot_type', 'name', 'version']).mean().reset_index(), values='data',
                         index=['plot_type', 'name'], columns='version')