import os.path as op
import numpy as np
import pandas as pd
from glob import glob


stim_df = pd.read_csv('exp/faces_SINGLETRIAL.tsv', sep='\t')
face_ids = [str(s).zfill(3) for s in stim_df['face_id'].unique()]

lms = []
files = []
for i, fid in enumerate(face_ids):
    search = f'../stimuli/london_face_dataset/neutral_front/{str(fid).zfill(3)}_*.tem'
    tems = glob(search)
    if len(tems) != 1:
        raise ValueError(f"Could not find tem file with {search}.")

    with open(tems[0], 'r') as f_in:
        n_lm = f_in.readline().strip()

    lms.append(np.loadtxt(tems[0], skiprows=1, max_rows=int(n_lm)).T)
    files.append(tems[0])

cols = [f'lm_{i}' for i in range(189)]
lms = pd.DataFrame(np.vstack(lms), columns=cols)
lms['face_id'] = np.repeat(face_ids, 2)
lms['coord'] = np.tile(['x', 'y'], len(face_ids))
lms['stim_path'] = np.repeat([f"london_face_dataset/neutral_front/{op.basename(f).replace('tem', 'jpg')}" for f in files], 2)
lms = lms.loc[:, ['face_id', 'stim_path', 'coord'] + cols]
lms.to_csv('../stimuli/task-face_landmarks.tsv', sep='\t', index=False) 
