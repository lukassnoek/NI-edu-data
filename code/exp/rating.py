import os
import sys
sys.path.append('exptools2')

import os.path as op
from psychopy.gui import DlgFromDict
from session import RatingSession

inp = {
    'sub_id': '17',
    'session': '1',
    'language': 'EN'  # 'NL' or 'EN'
}

db = DlgFromDict(inp, order=['sub_id', 'session', 'language'], title='Settings')
if not db.OK:
    print("User cancelled!")
    sys.exit()
    
sub, ses, lang = db.data
if sub not in [str(i).zfill(2) for i in range(1, 21)]:
    raise ValueError("Please provide a (zero-padded!) subject-nr (e.g., '02', '03', or '12')!")

if ses not in ['1', '2']:
    raise ValueError("Please select session (either '1' or '2')")

if lang not in ['NL', 'EN']:
    raise ValueError("Language should be 'NL' or 'EN'!")

rating_session = RatingSession(
    output_str=f'sub-{sub}_ses-{ses}_task-rating',
    settings_file='settings_cubicle.yml',
    language=lang,
    session=ses
)

rating_session.create_trials()
rating_session.run(lang)
rating_session.quit()