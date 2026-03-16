import pandas as pd
from pathlib import Path
path = Path('data/processed/pillar_scores_v3_2_bin.csv')
df = pd.read_csv(path)
print(df['year'].min(), df['year'].max())
summary = df.groupby('year')['grade'].agg(['count','min','median','max'])
print(summary.head(10))
print(summary.tail(10))
print('\nBefore 2014:')
print(summary.loc[summary.index < 2014])
