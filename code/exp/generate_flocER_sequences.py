from optimize_design import optimize_design
import os
import pandas as pd
import numpy as np
import os.path as op
from joblib import Parallel, delayed
from glob import glob

SUBS = 20
RUNS = 4
N = 65
rf_args = dict(basis_set='fourier', interval=[0, 20], n_regressors=9)
sub_names = ['sub-' + str(sub).zfill(2) for sub in range(1, SUBS+1)]
stim_files_dir = op.join(op.dirname(__file__), 'stim_selection')

all_dfs = []
for sub in sub_names:

    out = Parallel(n_jobs=4)(delayed(optimize_design)(   
        N=N,
        P=5,
        cond_names=['child', 'corridor', 'word', 'body', 'instrument'] if run % 2 == 0
                    else ['adult', 'house', 'number', 'limb', 'car'],
        use_mseq=False,
        isi_type='pseudo_exponential',
        min_isi=2,
        stim_dur=0.4,
        wait2start=6,
        TR=2,
        search_attempts=10000,
        noise_level=1,
        true_betas=np.array([1, 1, 1, 1, 1]),
        output_dir=None,
        output_str=None,
        **rf_args
        ) for run in range(1, RUNS+1))

    dfs = [this_run[0] for this_run in out]
    for i, df in enumerate(dfs):
        df['run'] = i+1

    new_dfs = []
    for run, df in enumerate(dfs):
        baseline = pd.DataFrame(dict(trial_type='baseline', duration=5.9, onset=0, isi=0.1, run=run+1, stim_name='baseline'), index=[0])
        df = pd.concat([baseline, df, baseline], axis=0, sort=True)
        df.index = np.arange(0, df.shape[0])

        df.loc[:, 'same'] = [False] + [df.loc[i, 'trial_type'] == df.loc[i-1, 'trial_type'] for i in range(1, df.shape[0])]
        n_probe = int(N * 0.05)
        probe_trials = np.random.choice(df.query('same').index.tolist(), size=n_probe, replace=False)
        df.loc[probe_trials, 'task_probe'] = 1
        new_dfs.append(df)

    df = pd.concat(new_dfs).drop('same', axis=1)
    df.loc[df.task_probe.isna(), 'task_probe'] = 0
    df.loc[:, 'sub_id'] = sub
    df.index = np.arange(0, df.shape[0])
    for cond in [con for con in df.trial_type.unique() if con != 'baseline']:
        n_stim = df.loc[df.trial_type == cond].shape[0]
        floc_dir = op.join(op.dirname(op.dirname(__file__)), 'data', 'fLoc', 'stimuli')
        imgs = np.random.choice(glob(op.join(floc_dir, cond, '*.jpg')), size=n_stim, replace=False)
        df.loc[df.trial_type == cond, 'stim_name'] = [op.basename(img) for img in imgs]
    
    df.loc[:, 'stim_name'] = [df.loc[0, 'stim_name']] + [df.loc[i, 'stim_name'] if df.loc[i, 'task_probe'] == 0
                              else df.loc[i-1, 'stim_name'] for i in range(1, df.shape[0])]

    all_dfs.append(df)

df = pd.concat(all_dfs, axis=0)
df.to_csv(op.join(op.dirname(__file__), 'fLoc_ER.tsv'), sep='\t', index=False)