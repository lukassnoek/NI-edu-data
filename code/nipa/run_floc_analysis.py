import os
import os.path as op
import pandas as pd
import pandas as pd
from glob import glob
from nilearn import masking
from joblib import Parallel, delayed
from nistats.first_level_model import FirstLevelModel


def fit_subject(sub, space):

    funcs = sorted(glob(f'../derivatives/fmriprep/{sub}/ses-*/func/*task-flocBLOCKED*space-{space}*desc-preproc_bold.nii.gz'))
    masks = [f.replace('preproc_bold', 'brain_mask') for f in funcs]
    mask = masking.intersect_masks(masks, threshold=0.9)
     
    conf_files = [f.split('space')[0] + 'desc-confounds_regressors.tsv' for f in funcs]
    ccols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confs = [pd.read_csv(c, sep='\t').loc[:, ccols] for c in conf_files]
    events = [f"../{sub}/ses{f.split('ses')[1].split('func')[0]}/func/{op.basename(f).split('desc')[0]}events.tsv"
              for f in conf_files]
    
    flm = FirstLevelModel(
        t_r=0.7,
        slice_time_ref=0.5,
        drift_model='cosine',
        high_pass=0.01,
        mask_img=mask,
        smoothing_fwhm=3.5,
        noise_model='ols',
        verbose=True
    )
    flm.fit(funcs, events, confs)
    con_defs = [
        ('face', '4*face - object - character - body - place'),
        ('place', '4*place - object - face - character - body'),
        ('body', '4*body - object - face - character - place'),
        ('character', '4*character - object - face - place - body')
    ]
    for name, df in con_defs:
        roi = flm.compute_contrast(df)
        f_out = f'../derivatives/floc/{sub}/rois/{sub}_task-flocBLOCKED_space-{space}_desc-{name}_zscore.nii.gz'
        if not op.isdir(op.dirname(f_out)):
            os.makedirs(op.dirname(f_out))

        roi.to_filename(f_out)


if __name__ == '__main__':

    subs = sorted([op.basename(d) for d in glob('../sub-*') if op.isdir(d)])
    _ = [fit_subject(sub, space='T1w') for sub in subs]

