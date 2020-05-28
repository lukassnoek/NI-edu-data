import pandas as pd
import os.path as op
import numpy as np
from glob import glob

button2score = dict(d=-4, n=-3, w=-2, e=-1, b=1, y=2, g=3, r=4)
logs = sorted(glob('exp/logs/*face*.tsv'))
for log in logs:
    baselog = op.basename(log)
    sub, ses = baselog.split("_")[:2]
    #if sub == 'sub-01':
    #    continue

    rating_files = sorted(glob(op.join(f'exp/logs/{sub}*_task-rating*.tsv')))
    ratings = []
    for f in rating_files:
        df = pd.read_csv(f, sep='\t')
        df = df.query("event_type == 'fix'").loc[:, ['face_id', 'rating_type', 'rating']].dropna(how='any', axis=0)
        df = df.groupby(['face_id', 'rating_type']).mean()
        df = df.reset_index().pivot(index='face_id', columns='rating_type', values='rating')
        df.index = df.index.astype(int)
        ratings.append(df)

    if ratings:
        assert(all(ratings[0].index.equals(df.index) for df in ratings))
        ratings_df = pd.concat(ratings)
        ratings_df = ratings_df.groupby(ratings_df.index).mean()

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

    idx = np.array([True if '_' in s else False for s in df['trial_type']])
    df['trial_type'] = [s.replace('_', '') for s in df['trial_type']]
    df.loc[idx, 'trial_type'] = [f'{str(i).zfill(2)}STIM{n}' for i, n in enumerate(df.loc[idx, 'trial_type'])]

    df = df.rename(columns={'attractiveness': 'average_attractiveness'})
    stim_idx = ~df.loc[:, 'trial_type'].isin(['rating', 'response'])
    tmp = df.loc[stim_idx, 'average_attractiveness']
    df.loc[stim_idx, 'average_attractiveness'] = (tmp - tmp.mean()) / tmp.std()
    if ratings:
        these_ratings = ratings_df.loc[df['face_id'], :]
        these_ratings.index = df.index
        these_ratings.columns = ['subject_attractiveness', 'subject_dominance', 'subject_trustworthiness']
        df = pd.concat((df, these_ratings), axis=1)
    else:
        for col in ['subject_attractiveness', 'subject_dominance', 'subject_trustworthiness']:
            df[col] = 'n/a'

    df.index = np.arange(df.shape[0])
    df = df.fillna('n/a')
    save_f = f'../{sub}/{ses}/func/{baselog}'
    if not op.isdir(op.dirname(save_f)):
        print(f"Not saving data for {save_f}")
        continue  # skip if there's no bids datai
    df.to_csv(save_f, sep='\t', index=False)
    """ 
    for exp1 in ['neutral', 'smiling']:
        for exp2 in ['male', 'female']:
            tmp = df.query("expression == @exp1 & face_sex == @exp2").loc[:, ['onset', 'duration']]
            tmp['weight'] = 1
            tmp.loc[:, 'duration'] = tmp.loc[:, 'duration'].round(2)
            tmp.to_csv(
                save_dir.replace('_events.tsv', f'_condition-{exp1}{exp2}_events.txt'),
                header=False, index=False, sep='\t'
            )
    """
    #for hand in ['left', 'right']:
    #    tmp = df.query("response_hand == @hand").loc[:, ['onset', 'duration']]
    #    tmp['weight'] = 1
    #    tmp.to_csv(
    #        save_dir.replace('_events.tsv', f'_condition-response{hand}_events.txt'),
    #        header=False, index=False, sep='\t'
    #    )

    #tmp = df.query("face_sex == 'male' or face_sex == 'female'").loc[:, ['onset', 'duration', 'average_attractiveness']]
    #tmp['weight'] = tmp['average_attractiveness']
    #tmp = tmp.drop('average_attractiveness', axis=1)
    #tmp.to_csv(
    #    save_dir.replace('_events.tsv', f'_condition-attractivenessmodulated_events.txt'),
    #    header=False, index=False, sep='\t'
    #)
    
    #tmp['weight'] = 1
    #tmp.to_csv(
    #    save_dir.replace('_events.tsv', f'_condition-attractivenessunmodulated_events.txt'),
    #    header=False, index=False, sep='\t'
    #)
