import os
import math
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psychopy.contrib.mseq import mseq
from nideconv import ResponseFitter
from nideconv.utils import double_gamma_with_d
from tqdm import tqdm


def optimize_design(N, P, cond_names=None, use_mseq=True, isi_type='pseudo_exponential',
                    min_isi=0.1, stim_dur=1, wait2start=2, TR=1.8, search_attempts=1000,
                    noise_level=1, true_betas=None, output_dir=None, output_str=None, **rf_args):
    """ Brute-force design optimization routine.
    N : int
        Number of trials in total
    P : int
        Number of conditions
    cond_names : list
        List with names of conditions (default: range(0, P))
    use_mseq : bool
        Whether to use m-sequences
    isi_type : array-like
        Can be a list/array with predefined ISIs or 'pseudo_exponential'
    min_isi : float/int
        Minimum ISI (added to all ISIs); only relevant for 'pseudo_exponential' ISIs
    stim_dur : float
        Duration of stimulus
    wait2start : float
        Time to wait after starting experiment ('baseline period')
    TR : float
        Time to repetition (1 / sampling rate)
    search_attempts : int
        Number to iterate search routine
    noise_level : float
        Noise-level of simulated signal (sigma of normal noise dist)
    true_betas : array-like
        True betas (parameters) for simulated signal
    output_dir : str
        Path to output-dir (default: None, i.e., nothing is saved)
    rf_args : dict
        Arguments for ResponseFitter, which will deconvolve the simulated signal.
    """

    if true_betas is None:
        true_betas = np.ones(P + 1)

    if len(true_betas) == P:
        true_betas = np.append(1, true_betas)

    if cond_names is None:
        cond_names = [str(cn) for cn in range(P)]

    if output_str is None:
        output_str = 'HrfMapper'

    if 'basis_set' not in rf_args.keys():
        rf_args['basis_set'] = 'fourier'

    if 'n_regressors' not in rf_args.keys():
        rf_args['n_regressors'] = 11

    if 'interval' not in rf_args['interval']:
        rf_args['interval'] = [0, TR*10]

    best_eff = 0
    for attempt in tqdm(range(search_attempts)):

        # DEFINE SEQUENCE
        if use_mseq:
            exponent = math.log(N+1, P)
            if exponent % 1 != 0:
                raise ValueError(f"Cannot do mseq with {N} trials and {P} conditions!")
            
            seq = mseq(P, int(exponent))
        else:
            trials_per_con = N / P
            if trials_per_con % 1 != 0:
                raise ValueError(f"Cannot create equal partitions for {N} "
                                f"trials and {P} conditions!")

            seq = np.concatenate([np.ones(int(trials_per_con))*i for i in range(P)])
            seq = np.random.permutation(seq.astype(int))

        # DEFINE ISIs
        if isinstance(isi_type, (list, tuple, np.ndarray)):
            if len(isi_type) == N:
                isis = np.random.permutation(isi_type)
            else:
                isis = np.random.choice(isi_type, size=N)
        elif isi_type == 'pseudo_exponential':
            isis = []
            i = 0
            while len(isis) < N:
                this_n = np.ceil((N - len(isis)) / 2)
                these_isis = min_isi + np.ones(int(this_n)) * i
                isis.extend(these_isis.tolist())
                i += 1
 
            isis = np.array(isis)[:N]
            #halves = int(np.ceil(math.log(N, P)))
            #isis = min_isi + np.concatenate([int((N+1) / 2**(i+1)) * [(i)] for i in range(halves)])
        else:
            pass

        isis = np.random.permutation(isis)
    
        # DEFINE EVENTS
        events = dict(onset=[], duration=[], trial_type=[], isi=[])
        t = wait2start
        for i, (cond, isi) in enumerate(zip(seq, isis)):
            events['onset'].append(t)
            events['duration'].append(stim_dur)
            events['trial_type'].append(cond_names[cond])
            events['isi'].append(isi)
            t += (stim_dur + isi)

        events = pd.DataFrame(events)
        n_scans = int(np.ceil(t / TR))
        rf = ResponseFitter(
            # input_signal is just a dummy signal
            input_signal=np.random.normal(0, 1, size=n_scans),
            sample_rate=1/TR
        )

        # Create canonical HRF design
        for cond in cond_names:
            onsets = events.loc[events.trial_type == cond, 'onset']
            rf.add_event(cond, onsets, interval=rf_args['interval'], basis_set='canonical_hrf')

        X_canon = rf.X
        noise = np.random.normal(0, noise_level, X_canon.shape[0])
        signal = X_canon.dot(true_betas) + noise

        rf = ResponseFitter(
            input_signal=signal,
            sample_rate=1/TR
        )
        
        # Create 'real' ResponseFitter
        for cond in cond_names:
            onsets = events.loc[events.trial_type == cond, 'onset']
            rf.add_event(cond, onsets, **rf_args)

        # Calculate efficiency (for shape estimation)
        X = rf.X
        c = np.eye(X.shape[1])
        eff = 1 / np.trace(c.dot(np.linalg.pinv(X.T.dot(X))).dot(c.T))

        if eff > best_eff:  # keep track of best one so far
            best_eff = eff
            best_events = events

            # Also calculate detection efficiency for comparison
            cd = np.append(np.ones(X_canon.shape[1] -1), 0)
            eff_d = 1 / cd.dot(np.linalg.pinv(X_canon.T.dot(X_canon))).dot(cd.T)
            best_rf = rf
            total_dur = np.sum(isis) + stim_dur * N

    print(f"Efficiency shape estimation: {best_eff:.3f}, detection: {eff_d:.3f}")
    t_min = int(np.floor(total_dur / 60))
    t_sec = (total_dur / 60 - t_min) * 60
    isi_stats = events.loc[:, 'isi'].describe()
    print(f"Mean ISI: {isi_stats['mean']:.2f}, "
          f"std ISI: {isi_stats['std']:.2f}, "
          f"min ISI: {isi_stats['min']:.2f}, "
          f"max ISI: {isi_stats['max']:.2f}"
          )
    print(f"Total duration for {N} trials: {t_min} min and {t_sec:.1f} seconds\n")

    if output_dir is not None:
        best_rf.regress()
        best_rf.plot_timecourses(legend=False)

        t = np.linspace(*rf_args['interval'])
        plt.plot(t, double_gamma_with_d(t), ls='--', c='gray')
        [plt.axhline(b, ls='--', lw=0.25, c='gray') for b in true_betas]
        fig = plt.gcf()
        fig.savefig(os.path.join(output_dir, f"{output_str}_simulation_deconv_result.png"))
        plt.close(fig)

        print(f"\nTotal duration experiment with {N} trials: {total_dur:.2f}")
        
        onset_vals = np.zeros((int(total_dur*1000), P))
        name2idx = {cond: i for i, cond in enumerate(cond_names)}
        for i in range(best_events.shape[0]):
            ons = best_events.loc[i, 'onset']
            dur = best_events.loc[i, 'duration']
            cond = name2idx[best_events.loc[i, 'trial_type']]
            onset_vals[int(ons*1000):int((ons+dur)*1000), cond] = 1

        plt.imshow(onset_vals, aspect='auto')
        [plt.text(i, -3000, cond) for i, cond in enumerate(cond_names)]
        plt.xticks([])
        fig = plt.gcf()
        fig.savefig(os.path.join(output_dir, f'{output_str}_onsets.png'))
        plt.close(fig)

        plt.imshow(best_rf.X, aspect='auto')
        plt.axis('off')
        fig = plt.gcf()
        fig.savefig(os.path.join(output_dir, f'{output_str}_design.png'))
        plt.close(fig)

        best_events.to_csv(os.path.join(output_dir, f'{output_str}_stims.tsv'), sep='\t')

    return best_events, best_eff