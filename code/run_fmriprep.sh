bids_dir=`realpath ../`
out_dir=$bids_dir/derivatives
subs=`ls -d1 $bids_dir/sub-??`
# Run subjects one at the time as to avoid memory issues
i=0
n_cores=8
for sub in $subs; do

    base_sub=`basename $sub`
    if [ -f $out_dir/fmriprep/${base_sub}.html ]; then
	echo "SKIPPING SUBJECT ${base_sub}!"
	continue
    else
        echo "RUNNING SUBJECT ${base_sub}!"
    fi
    label=${base_sub//sub-/}
    fmriprep-docker $bids_dir $out_dir \
        --image nipreps/fmriprep:20.2.1 \
        --participant-label $label \
        --nthreads 2 \
        --omp-nthreads 2 \
	--no-tty \
	--work ../../fmriprep_work \
        --ignore slicetiming \
        --output-space T1w MNI152NLin2009cAsym \
        --fs-license-file /usr/local/freesurfer/license.txt &
    i=$(($i + 1))
    if [[ $(( i % $n_cores )) == 0 ]]; then
        wait
    fi
done
wait
