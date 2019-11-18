import sys
sys.path.append('exptools2')

import os.path as op
import pandas as pd
import numpy as np
from glob import glob
from psychopy.visual import ImageStim, TextStim
from exptools2.core import Session, Trial
from utils import ButtonBoxRating, make_custom_unidim_scale


class BaselineTrial(Trial):
    
    def draw(self):
        self.session.default_fix.draw()


class FLocTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, pic=None, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)

        if pic == 'baseline':
            self.to_draw = self.session.default_fix
        else:
            spath = op.join(self.session.stim_dir, 'stimuli', pic.split('-')[0], pic)
            self.session.current_stim.setImage(spath)
            self.to_draw = self.session.current_stim

    def draw(self):
        """ Draws stimuli """

        if self.phase == 0:
            self.to_draw.draw()
        else:
            if isinstance(self.to_draw, ImageStim):
                if self.phase_durations[1] > 0.1:
                    self.session.default_fix.draw()
            else:
                self.session.default_fix.draw()


class FLocSession(Session):
    """ Simple session with x trials. """
    def __init__(self, sub, run, output_str, stim_dir, scrambled, dummies, stim_file,
                 ntrials=None, rt_cutoff=1, output_dir=None, settings_file=None):
        """ Initializes TestSession object. """

        msg = ("When using this localizer, please acknowledge the original "
               "creators of the task (Stigliani et al.); for more info "
               "about how to cite the original authors, check "
               "http://vpnl.stanford.edu/fLoc\n")
        print(msg)

        if not op.isdir(stim_dir):
            msg = (f"Directory {stim_dir} does not exist!\n"
                   f"To get the stimuli, simply run the following:\n"
                   f"git clone https://github.com/FEED-UvA/fLoc.git")
            raise OSError(msg)
            
        self.stim_dir = stim_dir
        self.scrambled = scrambled
        self.dummies = dummies
        self.ntrials = ntrials
        self.rt_cutoff = rt_cutoff
        self.stim_file = stim_file

        df = pd.read_csv(stim_file, sep='\t')
        sub_id = f'sub-{sub}'
        self.stim_df = df.query('sub_id == @sub_id & run == @run')
        
        if self.ntrials is not None:  # just for debugging
            self.stim_df = self.stim_df.iloc[:self.ntrials, :]

        self.stim_df.index = np.arange(0, len(self.stim_df), dtype=int)
        self.trials = []
        self.current_trial = None

        super().__init__(output_str=output_str, settings_file=settings_file,
                         output_dir=output_dir)

        self.current_stim = ImageStim(self.win, image=None)
        self.type2condition = dict(child='face', adult='face',
                                   body='body', limb='body',
                                   corridor='place', house='place',
                                   word='character', number='character',
                                   instrument='object', car='object',
                                   scrambled='scrambled', scrambled1='scrambled',
                                   scrambled2='scrambled', baseline='')

    def create_trial(self, trial_nr):
        
        if trial_nr == (self.stim_df.shape[0] - 1):  # last trial!
            load_next_during_phase = None
        else:
            load_next_during_phase = 1

        stim_type = self.stim_df.loc[trial_nr, 'trial_type']
        stim_name = self.stim_df.loc[trial_nr, 'stim_name']
        task_probe = self.stim_df.loc[trial_nr, 'task_probe']

        trial = FLocTrial(
            session=self,
            trial_nr=trial_nr,
            phase_durations=(self.stim_df.loc[trial_nr, 'duration'], self.stim_df.loc[trial_nr, 'isi']),
            phase_names=(stim_type, 'ISI'),
            pic=stim_name,
            load_next_during_phase=load_next_during_phase,
            verbose=True,
            timing='seconds',
            parameters={'trial_type': self.type2condition[stim_type],
                        'stim_name': stim_name, 'task_probe': task_probe}
        )

        self.trials.append(trial)

    def run(self, language):
        """ Runs experiment. """

        watching_response = False
        self.create_trial(trial_nr=0)
        
        if language == 'NL':
            txt = ('Deze run duurt ongeveer 4 minuten.\n\n'
                   'Als je precies hetzelfde plaatje twee keer achter elkaar ziet,\n'
                   'druk dan met je rechter wijsvinger op de knop.\n\n'
                   'Blijf zo stil mogelijk liggen tijdens (en na) de scan!')
        else:
            txt = ('This run takes about 4 minutes.\n\n'
                   'When you see exactly the same image twice in a row,\n'
                   'press the button with your right index finger.\n\n'
                   'Please try to move as little as possible during (and after) the scan!')
        
        self.display_text(txt, keys=self.settings['mri'].get('sync', 't'), height=0.5, wrapWidth=500)
        self.start_experiment()

        hits = []
        for trial_nr in range(self.stim_df.shape[0]):

            if self.stim_df.loc[trial_nr, 'task_probe'] == 1:
                watching_response = True
                onset_watching_response = self.clock.getTime()

            self.trials[trial_nr].run()

            if watching_response:

                if self.trials[-2].last_resp is None: # No answer given
                    if (self.clock.getTime() - onset_watching_response) > self.rt_cutoff:
                        hits.append(0)  # too late!
                        watching_response = False
                    else:
                        pass  # keep on watching
                else:  # answer was given
                    rt = self.trials[-2].last_resp_onset - onset_watching_response
                    print(f'Reaction time: {rt:.5f}')
                    if rt > self.rt_cutoff:  # too late!
                        hits.append(0)
                    else:  # on time! (<1 sec after onset 1-back stim)
                        hits.append(1)
                    watching_response = False

        mean_hits = np.mean(hits) * 100 if hits else 0
        #txt = f'{mean_hits:.1f}% correct ({sum(hits)} / {len(hits)})!'
        #self.display_text(txt, duration=1)
        fname = op.join(self.output_dir, self.output_str + '_accuracy.txt')
        with open(fname, 'w') as f_out:
            f_out.write(f'{mean_hits:.3f}')


class FaceTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, img, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.img = img

    def draw(self):
        """ Draws stimuli """
        if self.phase == 0:
            self.img.draw()
        elif self.phase == 1:
            self.session.default_fix.draw()
        elif self.phase == 2:
            if self.last_resp is not None:
                self.session.bb_rating[self.parameters['rating_type']].update_color(self.last_resp)
                self.last_resp = None
            self.session.bb_rating[self.parameters['rating_type']].draw()

        elif self.phase == 3:
            self.session.default_fix.draw()


class FaceSession(Session):
    """ Simple session with x trials. """
    def __init__(self, sub, run, output_str, stim_file,
                 output_dir=None, settings_file=None):
        """ Initializes TestSession object. """

        super().__init__(output_str=output_str, settings_file=settings_file,
                         output_dir=output_dir)
                         
        self.stim_file = stim_file
        self.stim_dir = op.join('..', '..', 'stimuli', 'london_face_dataset')
            
        self.trials = []

    def create_trials(self, sub, ses, run):
        df = pd.read_csv(self.stim_file, sep='\t').query('sub_id == @sub & ses == @ses & run == @run')
        df['face_id'] = [str(s).zfill(3) for s in df['face_id']]
        self.trials = [BaselineTrial(self, trial_nr=-1, phase_durations=(6,), phase_names=('fix',), verbose=True)]
        catches = np.random.permutation(['attractiveness', 'attractiveness', 'dominance', 'dominance', 'trustworthiness', 'trustworthiness'])
        i_catch = 0
        for i in range(df.shape[0]):
            params = df.iloc[i, :]
            stim_dir = op.join(self.stim_dir, f"{params['expression']}_front")
            imgs = glob(op.join(stim_dir, f"{params['face_id']}_*.jpg"))
            if len(imgs) != 1:
                raise ValueError("Could not find stim (or too many).")
            
            img = ImageStim(self.win, imgs[0], size=(10, 10), units='deg')
            if int(params['catch']) == 0:
                phase_durations = [1.25, 3.75]
                phase_names = ['face', 'fix']
            else:
                phase_durations = [1.25, 1.5, 2.5, 3.75]
                phase_names = ['face', 'fix', 'rating', 'fix']
                i_catch += 1
                params['rating_type'] = catches[i_catch - 1]

            if i == 20:
                phase_durations[-1] += 6.25

            trial = FaceTrial(
                session=self,
                trial_nr=i,
                phase_durations=phase_durations,
                img=img,
                phase_names=phase_names,
                parameters=params.to_dict(),
                verbose=True,
                timing='seconds'
            )
            self.trials.append(trial)
            
        self.trials.append(BaselineTrial(self, trial_nr=i+1, phase_durations=(6,), phase_names=('fix',), verbose=True))

    def run(self, language):
        
        if language == 'NL':
            txt = ("Deze run duurt ongeveer 4 minuten.\n\n"
                   "Je gaan een aantal gezichten bekijken,\n"
                   "waarbij je af en toe een beoordeling geeft over\n"
                   "hoe aantrekkelijk/dominant/betrouwbaar jij het gezicht vindt\n\n"
                   "Blijf zo stil mogelijk liggen tijdens (en na) de scan!")            
            lab = ['Helemaal niet', 'Heel erg']
            q = dict(attractiveness="Hoe aantrekkelijk?", dominance="Hoe dominant?", trustworthiness="Hoe betrouwbaar?")
        else:
            lab = ['Not at all', 'Very much']
            q = dict(attractiveness="How attractive?", dominance="How dominant?", trustworthiness="How trustworthy?")
            txt = ("This run takes about 4 minutes.\n\n"
                   "You will view a bunch of faces,\n"
                   "which you will sometimes rate on attractiveness/dominance/trustworthiness.\n\n"
                   "Please try to move as little as possible during (and after) the scan!")            

        self.bb_rating = dict(
            attractiveness=ButtonBoxRating(win=self.win, rating_question=q['attractiveness'], labels=lab, buttons=['d', 'n', 'w', 'e', 'b', 'y', 'g', 'r'], fixation=self.default_fix),
            dominance=ButtonBoxRating(win=self.win, rating_question=q['dominance'], labels=lab, buttons=['d', 'n', 'w', 'e', 'b', 'y', 'g', 'r'], fixation=self.default_fix),
            trustworthiness=ButtonBoxRating(win=self.win, rating_question=q['trustworthiness'], labels=lab, buttons=['d', 'n', 'w', 'e', 'b', 'y', 'g', 'r'], fixation=self.default_fix)
        )

        self.display_text(txt, keys=self.settings['mri'].get('sync', 't'), height=0.5, wrapWidth=500)
        self.start_experiment()

        for trial in self.trials:
            trial.run()
            for rat in self.bb_rating.keys():
                self.bb_rating[rat].reset()
            

class RatingTrial(Trial):
    """ Simple trial with text (trial x) and fixation. """
    def __init__(self, session, trial_nr, phase_durations, face_stim, rating, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.face_stim = face_stim
        self.rating = rating

    def draw(self):
        """ Draws stimuli """
        if self.phase == 0:
            self.face_stim.draw()
            self.session.mouse.setVisible(0)
            self.session.mouse.setPos([0, 0])
        elif self.phase == 1:  # draw rating
            self.session.all_scales[self.rating].draw()        
            self.session.mouse.setVisible(1)

            if not self.session.all_scales[self.rating].noResponse:
                self.session.mouse.setVisible(False)
                self.parameters['rating'] = self.session.all_scales[self.rating].getRating()
                self.session.all_scales[self.rating].reset()
                self.stop_phase()
                

class CueTrial(Trial):
    
    def __init__(self, session, trial_nr, phase_durations, txt, **kwargs):
        super().__init__(session, trial_nr, phase_durations, **kwargs)
        self.txt = TextStim(self.session.win, txt, height=1, wrapWidth=300)
        
    def draw(self):

        if self.phase == 0:
            self.txt.draw()
        else:
            pass

class RatingSession(Session):
    """ Simple session with x trials. """
    def __init__(self, output_str, language, output_dir=None, settings_file=None, session=None,
                 verbose=False):
        """ Initializes TestSession object. """
        super().__init__(output_str, output_dir, settings_file)
        self.language = language
        self.session = session
        self.trials = []
        self.en2nl = dict(attractiveness='aantrekkelijkheid', trustworthiness='betrouwbaarheid', dominance='dominantie')
        
        mapping = dict(
            EN=dict(attractiveness=['Not at all', '', 'Very much'], dominance=['Not at all', '', 'Very'], trustworthiness=['Not at all', '', 'Very']),
            NL=dict(attractiveness=['Helemaal niet', '', 'Heel erg'], dominance=['Helemaal niet', '', 'Heel erg'], trustworthiness=['Helemaal niet', '', 'Heel erg'])
        )

        self.all_scales = dict(
            attractiveness=make_custom_unidim_scale(self.win, labels=mapping[self.language]['attractiveness'], title='Hoe aantrekkelijk?' if language == 'NL' else 'How attractive?'),
            dominance=make_custom_unidim_scale(self.win, labels=mapping[self.language]['dominance'], title='Hoe dominant?' if language == 'NL' else 'How dominant?'),
            trustworthiness=make_custom_unidim_scale(self.win, labels=mapping[self.language]['trustworthiness'], title='Hoe betrouwbaar?' if language == 'NL' else 'How trustworthy?')
        )

    def create_trials(self):
        stim_file = 'faces_SINGLETRIAL.tsv'
        faces = pd.read_csv(stim_file, sep='\t').loc[:, 'face_id'].unique()
        print(faces)
        trial_nr = 0
        
        for rep in [1, 2]:
            for rating in np.random.permutation(['dominance', 'attractiveness', 'trustworthiness']):
                if self.language == 'NL':
                    cue_txt = f"Je gaat nu gezichten op {self.en2nl[rating]} beoordelen!"
                else:
                    cue_txt = f"You're now going to rate faces on {rating}!"
                    
                cue_trial = CueTrial(
                    self,
                    trial_nr=trial_nr,
                    phase_durations=(4, 1),
                    phase_names=('cue', 'fix'),
                    txt=cue_txt
                )
                self.trials.append(cue_trial)
                for face in np.random.permutation(faces):
                    face = str(face).zfill(3)
                    stim = op.join('..', '..', 'data', 'london_face_dataset', 'neutral_front', f'{face}_03.jpg')
                    self.display_text(f'Loading {trial_nr} / 240 ...', duration=0.001)
                    trial = RatingTrial(
                        session=self,
                        trial_nr=trial_nr,
                        face_stim=ImageStim(self.win, stim, size=(10, 10), units='deg'),
                        phase_durations=(1.25, np.inf, 0.5),
                        phase_names=('face', 'rating', 'fix'),
                        rating=rating,
                        parameters=dict(rating_type=rating, face_id=face, repetition=rep, session=self.session)
                    )
                    self.trials.append(trial)
                    trial_nr += 1

            if rep == 1:
                if self.language == 'NL':
                    cue_txt = f"Je bent op de helft!\nEven 30 seconden pauze; daarna gaan we weer beginnen."
                else:
                    cue_txt = f"You're halfway!\nYou have a 30 second break; after that, we'll start again."
    
                pause_trial = CueTrial(
                    self,
                    trial_nr=trial_nr,
                    phase_durations=(30, 1),
                    phase_names=('cue', 'fix'),
                    txt=cue_txt
                )
                self.trials.append(pause_trial)
        
    def run(self, language):
        """ Runs experiment. """

        if language == 'NL':
            txt = ("Je gaat nu de gezichten beoordelen op drie eigenschappen:\n"
                   "aantrekkelijkheid, dominantie, en betrouwbaarheid,\n"
                   "op een schaal van 'Helemaal niet' tot 'Heel erg'.\n\n"
                   "Dit doe je door met de muis te klikken op de schaal.\n"
                   "Er is geen tijdsdruk voor je beoordeling.\n\n"
                   "(Druk op enter om te beginnen.)")
        else:
            txt = ("You're going to rate the faces on three attributes:\n"
                   "attractiveness, dominance, and trustworthiness,\n"
                   "from 'Not at all' to 'Very much'.\n\n"
                   "You do this by clicking on the rating scale with the mouse.\n"
                   "There is no time pressure to respond.\n\n"
                   "(Press enter to start.)")
        
        height = 0.8
        self.display_text(txt, keys=['return'], height=height, wrapWidth=30)
        self.start_experiment()

        for i, trial in enumerate(self.trials):
            trial.run()

        self.close()

