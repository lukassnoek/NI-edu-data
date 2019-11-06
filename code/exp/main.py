import os
import sys
sys.path.append('exptools2')

import os.path as op
from psychopy.gui import DlgFromDict
from session import FLocSession, FaceSession

inp = {
    'sub_id': '07',  # '01', '02', etc  (zero-padded!)
    'session': '2',  # '1' or '2'
    'task': '',  # 'floc' or 'face'
    'design': '',  # 'BLOCKED' or 'ER'
    'run': '',  # for floc, choose '1', '2', '3', '4', '5', or '6'
    'language': 'NL'  # 'NL' or 'EN'
}

db = DlgFromDict(inp, order=['sub_id', 'session', 'task', 'design', 'run', 'language'], title='Settings')
if not db.OK:
    print("User cancelled!")
    sys.exit()
    
sub, ses, task, design, run, lang = db.data
if sub not in [str(i).zfill(2) for i in range(1, 21)]:
    raise ValueError("Please provide a (zero-padded!) subject-nr (e.g., '02', '03', or '12')!")
    
if ses not in ['1', '2']:
    raise ValueError("Session should be '1' or '2'!")

if task not in ['floc', 'face', 'restingstate']:
    raise ValueError("Task should be 'floc', 'face', or 'restingstate'!")

if design not in ['BLOCKED', 'ER'] and task == 'floc':
    raise ValueError("Design should be 'BLOCKED' or 'ER' when running 'floc' task!")

if run not in ['1', '2', '3', '4', '5', '6'] and task == 'face':
    raise ValueError("Run should be 1-6 when running 'face' task!")

if not run and task == 'floc':
    run = 1

if lang not in ['NL', 'EN']:
    raise ValueError("Language should be 'NL' or 'EN'!")

ses = int(ses)
run = int(run)

if task == 'floc':

    stim_file = f'fLoc_{design}.tsv'

    fLoc_session = FLocSession(
        sub=sub,
        run= run if ses == 1 else run + 2,
        output_str=f'sub-{sub}_ses-{ses}_task-floc{design}_acq-Mb4Mm27Tr700',
        stim_file=stim_file,
        settings_file='settings.yml',
        stim_dir=op.join('..', 'data', 'fLoc'),
        dummies=0,
        scrambled=False
    )

    fLoc_session.run(lang)
    fLoc_session.quit()
    
elif task == 'face':

    face_session = FaceSession(
        sub=sub,
        run=run,
        output_str=f'sub-{sub}_ses-{ses}_task-face_acq-Mb4Mm27Tr700_run-{run}',
        stim_file=op.join(op.dirname(__file__), 'faces_SINGLETRIAL.tsv'),
        settings_file='settings.yml'
    )
    
    face_session.create_trials(sub='sub-' + sub, ses=ses, run=run)
    face_session.run(lang)
    face_session.quit()