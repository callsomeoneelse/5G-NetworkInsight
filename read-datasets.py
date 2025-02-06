import os
import pandas as pd

# Read the data
files = [f for f in os.listdir('data') if f.endswith('.csv') and f.endswith('kml.csv')]

datasets = []
for f in files:
    print(f'Processing {f}')
    dataset = pd.read_csv(os.path.join('data', f), low_memory=False)
    datasets.append(dataset)

# Merge the dataframes
df = pd.concat(datasets)
print(df.head())

# Save the dataframe
df.to_csv('data/all_data.csv', index=False)