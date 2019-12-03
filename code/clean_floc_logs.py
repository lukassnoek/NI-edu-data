import pandas as pd
import os.path as op
import numpy as np
from glob import glob


logs = sorted(glob('exp/logs/*floc*.tsv'))
for log in logs:
    task = 'flocBLOCKED' if 'flocBLOCKED' in log else 'flocER'
    print(f'Processing {op.basename(log)} ...')
    df = pd.read_csv(log, sep='\t')
    idx = ~df['event_type'].isin(['ISI', 'baseline', 'pulse'])
    df = df.loc[idx, :]

    df.loc[df['event_type'] == 'response', 'trial_type'] = 'response'
    df.loc[df['trial_type'] == 'response', 'response_hand'] = ['left' if r in ['b', 'y', 'g', 'r'] else 'right' for r in df.loc[df['trial_type'] == 'response', 'response']]

    df = df.loc[:, ['onset', 'duration', 'trial_type', 'stim_name', 'task_probe']]
    df.loc[df['trial_type'] == 'response', 'duration'] = 0.1  # model as impulse
    df.loc[df['trial_type'] == 'response', ['stim_name', 'task_probe']] = 'n/a'
    df.index = np.arange(df.shape[0])


    for idx, row in df.loc[df['task_probe'] == 1, :].iterrows():
        
        onset_probe = row['onset']
        next_row = df.iloc[idx+1, :]
        i, rtype, rtime = 1, 'miss', 'n/a'
        
        def get_thresh(task, row, next_row):
            if task == 'flocBLOCKED':
                return next_row['onset'] - row['onset'] < 5
            else:
                return i < 2

        while get_thresh(task, row, next_row) and df.shape[0] > (idx+i):
            nrow = df.iloc[idx+i, :]
            if nrow['trial_type'] == 'response':
                rtime = nrow['onset'] - row['onset']
                rtype = 'hit'
                break
            i += 1

        df.loc[idx, 'response_time'] = rtime
        df.loc[idx, 'response_accuracy'] = rtype

    df = df.fillna('n/a')
    prop_correct = np.mean(df.loc[df['task_probe'] == 1, 'response_accuracy'] == 'hit')
    print(f"Proportion correct (hit): {prop_correct:.3f}")

    baselog = op.basename(log)
    sub, ses = baselog.split('_')[:2]
    save_dir = f'../{sub}/{ses}/func/{baselog}'
    df.to_csv(save_dir, sep='\t')

    conds = np.unique(df['trial_type'])
    for con in conds:
        tmp = df.query("trial_type == @con")
        tmp['weight'] = 1
        tmp.loc[:, 'duration'] = tmp.loc[:, 'duration'].round(1)
        tmp = tmp.loc[:, ['onset', 'duration', 'weight']]
        tmp.to_csv(
            save_dir.replace('_events.tsv', f'_condition-{con}_events.txt'), 
            header=False, index=False, sep='\t'
        )

