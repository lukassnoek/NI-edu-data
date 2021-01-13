import os
import os.path as op
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix


def run_subject(sub, out_dir):
    """ Runs pattern estimation for a single subject. """
    print(f"INFO: Processing {sub}")

    # Define in- and output directories
    bids_dir = op.join(f'../{sub}')
    fprep_dir = op.join(f'../derivatives/fmriprep/{sub}')
    out_dir = op.join(out_dir, sub)

    funcs = sorted(glob(fprep_dir + '/ses-?/func/*task-face*space-MNI*desc-preproc_bold.nii.gz'))
    for func in funcs:
        t_r = nib.load(func).header['pixdim'][4]
        conf = func.split('_space')[0] + '_desc-confounds_regressors.tsv'
        mask = func.replace('preproc_bold', 'brain_mask')
        events = bids_dir + func.split(fprep_dir)[1].split('_space')[0] + '_events.tsv'

        flm = FirstLevelModel(
            t_r=t_r, slice_time_ref=0.5, hrf_model='glover', drift_model='cosine', high_pass=0.01,
            mask_img=mask, smoothing_fwhm=5, noise_model='ar1', n_jobs=1, minimize_memory=False
        )

        # Select confounds
        conf = pd.read_csv(conf, sep='\t').loc[:, 
            ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']
        ]

        # Redefine output dir
        this_out_dir = op.join(out_dir, func.split(fprep_dir)[1].split('/')[1])
        for d in ['patterns', 'model', 'figures']:
            if not op.isdir(op.join(this_out_dir, d)):
                os.makedirs(op.join(this_out_dir, d), exist_ok=True)
        
        # Fit model
        flm.fit(run_imgs=func, events=events, confounds=conf)

        # Save some stuff!
        f_base = op.basename(func).split('preproc')[0]
        rsq_out = op.join(this_out_dir, 'model', f_base + 'model_r2.nii.gz')
        flm.r_square[0].to_filename(rsq_out)

        dm = flm.design_matrices_[0]
        dm_out = op.join(this_out_dir, 'model', f_base + 'design_matrix.tsv')
        dm.to_csv(dm_out, sep='\t', index=False)

        dmfig_out = op.join(this_out_dir, 'figures', f_base + 'design_matrix.png')
        plot_design_matrix(dm, output_file=dmfig_out)

        dmcorrfig_out = op.join(this_out_dir, 'figures', f_base + 'design_corr.png')
        labels = dm.columns.tolist()[:-1]
        ax = plot_design_matrix(dm.drop('constant', axis=1).corr())
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        plt.savefig(dmcorrfig_out)
        plt.close()

        resids_out = op.join(this_out_dir, 'model', f_base + 'model_residuals.nii.gz')
        flm.residuals[0].to_filename(resids_out)

        trials = [l for l in labels if 'STIM' in l]
        b, vb = [], []
        for trial in trials:
            dat = flm.compute_contrast(trial, stat_type='t', output_type='all')
            b.append(dat['effect_size'])
            vb.append(dat['effect_variance'])

        beta_out = op.join(this_out_dir, 'patterns', f_base + 'trial_beta.nii.gz')
        image.concat_imgs(b).to_filename(beta_out)
        varbeta_out = op.join(this_out_dir, 'patterns', f_base + 'trial_varbeta.nii.gz')
        image.concat_imgs(vb).to_filename(varbeta_out)

if __name__ == '__main__':

    from joblib import Parallel, delayed
    from glob import glob

    # Define where results should be saved
    out_dir = '../derivatives/pattern_estimation'

    # Gather subjects
    sub_dirs = sorted(glob('../sub-??'))

    # Run parallel
    Parallel(n_jobs=5)(delayed(run_subject)(
        op.basename(sub_dir), out_dir) for sub_dir in sub_dirs
    )

