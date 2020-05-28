import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from nilearn import image, masking
from joblib import Parallel, delayed


def run_subject(sub, nibs_dir):
        
    for space in ['T1w', 'MNI152NLin2009cAsym']:
        for ses in [1, 2]:
            imgs = []
            search_str = op.join(nibs_dir, sub, f'ses-{ses}', 'func', f'*task-face*space-{space}*STIM*betaseries.nii.gz')
            files = sorted(glob(search_str))
            stims = np.sort(np.unique([f.split('desc-')[1].split('_')[0][6:] for f in files]))
            print(stims)
            shape = image.load_img(files[0]).shape
            for i, stim in enumerate(tqdm(stims, desc=f'ses-{ses}, space-{space}')):
                imgs.append(image.mean_img([f for f in files if stim in f]))
            
            f_out = f'{sub}_ses-{ses}_task-face_acq-Mb4Mm27Tr700_space-{space}_desc-all_betaseries.nii.gz'
            img = image.concat_imgs(imgs)
            print(img.shape)
            img.to_filename(op.join(nibs_dir, f_out))

if __name__ == '__main__':

    nibs_dir = '../derivatives/nibetaseries_lsa_unn'
    run_subject('sub-02', nibs_dir)