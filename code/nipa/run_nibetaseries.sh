bids_dir=$(realpath ..)
deriv_dir=${bids_dir}/derivatives/fmriprep
output_dir=${bids_dir}/derivatives

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

nibs ${bids_dir} ${deriv_dir} ${output_dir} participant \
    --estimator lsa \
    --confounds trans_x trans_y trans_z rot_x rot_y rot_z white_matter csf \
    --hrf-model glover \
    --participant-label 01 02 03 04 05 06 07 08 09 10 11 12 13 14 17 \
    --high-pass 0.01 \
    --smoothing-kernel 0 \
    --task-label flocBLOCKED \
    --return-residuals \
    --space-label MNI152NLin2009cAsym \
    --normalize-betas \
    --nthreads 10
