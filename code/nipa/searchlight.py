import sys
import numpy as np
import os.path as op
import pandas as pd
from glob import glob
from nilearn import image, decoding, masking
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from tqdm import tqdm
from sklearn.metrics import make_scorer, balanced_accuracy_score


def run_searchlight(sub, space='T1w', factor='expression'):

    event_files = sorted(glob(f'../../{sub}/ses-*/func/*task-face*_events.tsv'))
    events = []
    for i, ev in enumerate(event_files):
        ev = pd.read_csv(ev, sep='\t').query("trial_type != 'response' and trial_type != 'rating'")
        ev['run'] = i+1
        ev['face_eth'] = ['asian' if 'sian' in s else s for s in ev['face_eth']]
        events.append(ev)
    
    events = pd.concat(events, axis=0)
    S = events.loc[:, factor]
    lab_enc = LabelEncoder()
    S = lab_enc.fit_transform(S)
    runs = events.loc[:, 'run']
    files = sorted(glob(f'../../derivatives/nibetaseries_lsa_unn/{sub}/ses-*/func/*task-face*space-{space}*STIM*_betaseries.nii.gz'))
    R = []
    pca = PCA(n_components=10)
    for ses in [1, 2]:
        for run in [1, 2, 3, 4, 5, 6]:
            these_f = [f for f in files if f'run-{run}' in f and f'ses-{ses}' in f]
            this_R = image.concat_imgs(these_f)
            brain_mask = image.math_img('img.mean(axis=3) != 0', img=this_R)
            this_R = image.math_img('(img - img.mean(axis=3, keepdims=True)) / img.std(axis=3, keepdims=True)', img=this_R)
            tmp = masking.apply_mask(this_R, brain_mask)
            #tmp_pca = pca.fit_transform(tmp)[:, :6]
            x = np.arange(tmp.shape[0])
            #b = np.polyfit(x, tmp_pca, deg=8)
            b = np.polyfit(x, tmp, deg=8)
            lf = np.polyval(b, x[:, np.newaxis])
            #tmp -= (lf @ np.linalg.lstsq(lf, tmp, rcond=-1)[0])
            tmp = tmp - lf
            this_R = masking.unmask(tmp, brain_mask)
            R.append(this_R)

    R = image.concat_imgs(R)
    R.to_filename('filt.nii.gz')
    mask = image.math_img('img.sum(axis=-1) != 0', img=R)
    process_mask = np.zeros(mask.shape)
    process_mask[:, :, 18:27] = 1
    process_mask = image.new_img_like(mask, process_mask)
    process_mask = masking.intersect_masks([process_mask, mask], threshold=1)

    clf = SVC(kernel='linear', class_weight='balanced')
    cv = LeaveOneGroupOut()
    b_acc = make_scorer(balanced_accuracy_score, adjusted=True)

    sl = decoding.SearchLight(
        mask_img=mask,
        process_mask_img=process_mask,
        radius=8,
        verbose=True,
        estimator=clf,
        cv=cv,
        scoring=b_acc,
        n_jobs=20
    )
    print("Starting fit ...")
    sl.fit(R, S, runs)
    scores = image.new_img_like(mask, sl.scores_)
    scores.to_filename(f'{sub}_factor-{factor}_balancedaccuracyFILT.nii.gz')




if __name__ == '__main__':

    factor = sys.argv[1]
    subs = [op.basename(s) for s in sorted(glob('../../sub-*'))]
    for sub in subs[1:2]:
        print(sub)
        run_searchlight(sub, factor=factor)
