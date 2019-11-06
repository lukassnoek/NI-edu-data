import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

N_face = 40

rootdir = op.dirname(op.dirname(__file__))
ratings = pd.read_csv(op.join(rootdir, 'data', 'london_face_dataset', 'london_faces_ratings.csv'))
ratings = ratings.loc[:, [col for col in ratings.columns if col[0] == 'X']]

info = pd.read_csv(op.join(rootdir, 'data', 'london_face_dataset', 'london_faces_info.csv')).set_index('face_id')
mean_rat = ratings.mean(axis=0)
info.loc[mean_rat.index, 'attractiveness'] = mean_rat.values
info.loc['X036', 'face_eth'] = 'east_asian/white'
info = info.dropna(how='any', axis=0)

N_per_group = N_face // 4
best_var = 0
for i in range(1000):
    white_m = info.query("face_eth == 'white' & face_sex == 'male'").sample(N_per_group, replace=False)
    white_f = info.query("face_eth == 'white' & face_sex == 'female'").sample(N_per_group, replace=False)
    nwhite_m = info.query("face_eth != 'white' & face_sex == 'male'").sample(N_per_group, replace=False)
    nwhite_f = info.query("face_eth != 'white' & face_sex == 'female'").sample(N_per_group, replace=False)
    df = pd.concat((white_m, white_f, nwhite_m, nwhite_f), axis=0)
    var = np.var(df.attractiveness)
    if var > best_var:
        best_var = var
        best_df = df

dfs = []
for exp in ['neutral', 'smiling']:
    df = best_df.copy()
    df['expression'] = exp
    dfs.append(df)

df = pd.concat(dfs, axis=0)
df['face_id'] = [str(s[1:]) for s in df.index]
df.index = df.index + '_' + df.expression

n_runs = 6
n_per_run = df.shape[0] // n_runs
all_df = []
for sub in ['sub-' + str(i).zfill(2) for i in range(1, 21)]:
    subjects = []
    for ses in [1, 2]:
        sessions = []
        run = 1
        for part in [0, 1, 2]:
            this_df = df.copy()
            this_df['sub_id'] = sub
            for xrun in range(1, int(n_runs / 3)+1):
                runs = []
                for exp in ['neutral', 'smiling']:
                    for sex in ['male', 'female']:
                        
                        tmp = this_df.query('expression == @exp & face_sex == @sex').sample(10, replace=False)
                        tmp['ses'] = ses
                        tmp['run'] = run
                        tmp['trial_type'] = tmp.face_id + '_' + exp
                        tmp = tmp.sample(frac=1, replace=False)
                        runs.append(tmp)
                        this_df = this_df.drop(tmp.index)

                this_run = pd.concat(runs, axis=0).sample(frac=1, replace=False)
                catch_trial = np.random.choice(this_run.index, size=5, replace=False)
                this_run.loc[catch_trial, 'catch'] = 1
                sessions.append(this_run)
                run += 1
        subjects.append(pd.concat(sessions))
    sub_df = pd.concat(subjects, axis=0)
    sub_df.loc[sub_df.catch.isna(), 'catch'] = 0
    all_df.append(sub_df)

df = pd.concat(all_df, axis=0)
df.catch = df.catch.astype(int)
df.to_csv(op.join(op.dirname(__file__), 'faces_SINGLETRIAL.tsv'), index=None, sep='\t')