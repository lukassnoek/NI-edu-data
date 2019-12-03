import pandas as pd
import os.path as op
import numpy as np
from glob import glob

button2score = dict(d=-4, n=-3, w=-2, e=-1, b=1, y=2, g=3, r=4)
logs = sorted(glob('exp/logs/*face*.tsv'))
for log in logs:
    baselog = op.basename(log)
    sub, ses = baselog.split("_")[:2]
    if sub == 'sub-01':
        continue

    print(f'Processing {baselog} ...')
    df = pd.read_csv(log, sep='\t')
    idx = ~df['event_type'].isin(['fix', 'baseline', 'pulse'])
    df = df.loc[idx, :]
    df.loc[df['event_type'] == 'response', 'trial_type'] = 'response'
    df.loc[df['trial_type'] == 'response', 'response_hand'] = ['left' if r in ['b', 'y', 'g', 'r'] else 'right' for r in df.loc[df['trial_type'] == 'response', 'response']]
    df.loc[df['trial_type'] == 'response', 'rating_score'] = [button2score[r] for r in df.loc[df['trial_type'] == 'response', 'response']]
    df.loc[df['event_type'] == 'rating', 'trial_type'] = 'rating'
    df = df.loc[:, ['onset', 'duration', 'trial_type', 'expression', 'face_id', 'face_age', 'face_sex', 'face_eth', 'attractiveness',  'catch', 'rating_score', 'response_hand']]
 
    df.loc[df['trial_type'] == 'response', 'duration'] = 0.1  # model as impulse
    df.loc[df['trial_type'] == 'response', ['catch', 'expression', 'face_id', 'face_age', 'face_sex', 'face_eth', 'attractiveness']] = 'n/a' 
    df.loc[df['trial_type'] == 'rating', ['catch', 'expression', 'face_id', 'face_age', 'face_sex', 'face_eth', 'attractiveness']] = 'n/a'

    df = df.rename(columns={'attractiveness': 'average_attractiveness'})
    stim_idx = ~df.loc[:, 'trial_type'].isin(['rating', 'response'])
    tmp = df.loc[stim_idx, 'average_attractiveness']
    df.loc[stim_idx, 'average_attractiveness'] = (tmp - tmp.mean()) / tmp.std()
    df.index = np.arange(df.shape[0])
    df = df.fillna('n/a')
    save_dir = f'../{sub}/{ses}/func/{baselog}'
    df.to_csv(save_dir, sep='\t')

   
    for exp1 in ['neutral', 'smiling']:
        for exp2 in ['male', 'female']:
            tmp = df.query("expression == @exp1 & face_sex == @exp2").loc[:, ['onset', 'duration']]
            tmp['weight'] = 1
            tmp.loc[:, 'duration'] = tmp.loc[:, 'duration'].round(2)
            tmp.to_csv(
                save_dir.replace('_events.tsv', f'_condition-{exp1}{exp2}_events.txt'),
                header=False, index=False, sep='\t'
            )

    #for hand in ['left', 'right']:
    #    tmp = df.query("response_hand == @hand").loc[:, ['onset', 'duration']]
    #    tmp['weight'] = 1
    #    tmp.to_csv(
    #        save_dir.replace('_events.tsv', f'_condition-response{hand}_events.txt'),
    #        header=False, index=False, sep='\t'
    #    )

    tmp = df.query("face_sex == 'male' or face_sex == 'female'").loc[:, ['onset', 'duration', 'average_attractiveness']]
    tmp['weight'] = tmp['average_attractiveness']
    tmp = tmp.drop('average_attractiveness', axis=1)
    tmp.to_csv(
        save_dir.replace('_events.tsv', f'_condition-attractivenessmodulated_events.txt'),
        header=False, index=False, sep='\t'
    )
    
    #tmp['weight'] = 1
    #tmp.to_csv(
    #    save_dir.replace('_events.tsv', f'_condition-attractivenessunmodulated_events.txt'),
    #    header=False, index=False, sep='\t'
    #)
