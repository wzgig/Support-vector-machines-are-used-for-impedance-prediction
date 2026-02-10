import pandas as pd
import numpy as np

INPUT_CSV = 'equivalent_circuit_parameters_optimized_accurate_Y11.csv'
TARGET_BRANCH_TYPE = 'RL_Series'
TARGET_BRANCH_ID = 'e'

df = pd.read_csv(INPUT_CSV)
mask = (df['Branch_Type'] == TARGET_BRANCH_TYPE) & (df['Branch_ID'] == TARGET_BRANCH_ID)
df = df[mask].copy()

threshold = 10000
df['is_extreme'] = df['R'].abs() > threshold

print('Extreme count:', df['is_extreme'].sum())
print('\nExtreme Sample:')
print(df[df['is_extreme']][['P', 'Q', 'V', 'xi', 'R']].head())

print('\nFeature Stats (Extreme):')
print(df[df['is_extreme']][['P', 'Q', 'V', 'xi']].describe())

print('\nFeature Stats (Normal):')
print(df[~df['is_extreme']][['P', 'Q', 'V', 'xi']].describe())
