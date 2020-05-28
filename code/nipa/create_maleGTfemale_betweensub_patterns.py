import os
import pandas as pd
import nibabel as nib
from glob import glob
from nilearn import image, masking, datasets
from nistats.first_level_model import FirstLevelModel

ho_atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
idx = ho_atlas['labels'].index('Temporal Occipital Fusiform Cortex')
roi = nib.Nifti1Image(
    (nib.load(ho_atlas['maps']).get_fdata() == idx).astype(int),
    affine=nib.load(ho_atlas['maps']).affine
)

subs = sorted([os.path.basename(s) for s in glob('../derivatives/fmriprep/sub-??')])
R = []
for sub in subs:
    funcs = sorted(glob(f'../derivatives/fmriprep/{sub}/ses-*/func/*task-face*MNI*desc-preproc_bold.nii.gz'))
    confs = sorted(glob(f'../derivatives/fmriprep/{sub}/ses-*/func/*task-face*desc-confounds_regressors.tsv'))
    masks = [f.replace('preproc_bold', 'brain_mask') for f in funcs]
    
    events = sorted(glob(f'../{sub}/ses-*/func/*task-face*events.tsv'))
    cols = ['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']
    confs = [pd.read_csv(c, sep='\t').loc[:, cols] for c in confs]
    events = [pd.read_csv(e, sep='\t').drop('trial_type', axis=1).rename({'expression': 'trial_type'}, axis=1)
              for e in events]
    events = [e.loc[~e['trial_type'].isna(), :] for e in events]
    print(events)
    mask = masking.intersect_masks(masks, threshold=1)
    flm = FirstLevelModel(
        t_r=0.7,
        slice_time_ref=0.5,
        hrf_model='glover',
        drift_model='cosine',
        high_pass=0.01,
        noise_model='ols',
        mask_img=mask,
        verbose=True,
        smoothing_fwhm=4,
        n_jobs=10
    )
    flm.fit(funcs, confounds=confs, events=events)
    con = flm.compute_contrast('smiling - neutral')
    #roi = image.resample_to_img(roi, con, interpolation='nearest')
    con.to_filename(f'../derivatives/{sub}_smilingGTneutral.nii.gz')
    #R.append(con)
    
#R = image.concat_imgs(R)
#R.to_filename('../derivatives/maleGTfemale.nii.gz')
