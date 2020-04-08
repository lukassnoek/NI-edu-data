import os.path as op
import pandas as pd
from glob import glob


rating_files = sorted(glob(op.join('exp/logs/sub*_task-rating*.tsv')))
ratings = dict()
for f in rating_files[:2]:
    sub = op.basename(f).split('_')[0]
    df = pd.read_csv(f, sep='\t')
    df = df.query("event_type == 'fix'").loc[:, ['face_id', 'rating_type', 'rating']].dropna(how='any', axis=0)
    df = df.groupby(['face_id', 'rating_type']).mean()
    df = df.reset_index().pivot(index='face_id', columns='rating_type', values='rating')
    df.index = df.index.astype(int)
    if sub not in ratings.keys():
        ratings[sub] = [df]
    else:
        ratings[sub].append(df)


for sub, dfs in ratings.items():
    assert(all(dfs[0].index.equals(df.index) for df in dfs))
    df_concat = pd.concat(dfs)
    df_av = df_concat.groupby(df_concat.index).mean()


