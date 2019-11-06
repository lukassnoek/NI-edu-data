from psychopy.visual import Circle, TextStim, GratingStim, RatingScale
import numpy as np
from psychopy.core import getTime


def make_custom_unidim_scale(win, labels, title=None, **kwargs):

    return RatingScale(
        win,
        precision=100,
        labels=labels,
        scale=title,
        low=-4,
        high=4,
        textSize=0.7,
        tickMarks=(-4, 0, 4),
        textColor=(1, 1, 1),
        stretch=2,
        pos=(0, 0),
        noMouse=False,
        mouseOnly=True,
        singleClick=True,
        markerStart=0,
        minTime=0.3,
        markerColor='white',
        showAccept=False,
        marker='slider',
        **kwargs
        )


class Rating:

    def __init__(self, win, labels=None, label_height=45, anchor_size=15,
                 anchor_color='lightblue', anchor_color_chosen='orange',
                 offset_label=50, time_until_disappear=0.3, fixation=None):
        """ Initializes ButtonBoxRating object. """

        self.win = win
        self.labels = ['0', '100'] if labels is None else labels
        self.label_height = label_height
        self.anchor_size = anchor_size
        self.anchor_color = anchor_color
        self.anchor_color_chosen = anchor_color_chosen
        self.offset_label = offset_label
        self.time_until_disappear = time_until_disappear

        if fixation is None:
            self.fixation = GratingStim(
                win,
                tex='sin',
                mask='raisedCos',
                size=10,
                texRes=512,
                color='white',
                sf=0,
                maskParams={'fringeWidth': 0.4}
            )

        self.elements = []
        self.rating_anchors = []
        self.label_texts = []
        self.resp = None
        self.time_resp = np.inf

    def create(self):
        raise NotImplementedError

    def update_marker(self):

        raise NotImplementedError

    def draw(self):
        """ Draws the rating. """
        
        this_time = getTime()
        if (this_time - self.time_resp) < self.time_until_disappear:
            for element in self.elements:
                element.draw()
        else:
            self.fixation.draw()


class KeyboardRating(Rating):

    def __init__(self, leftKey='f', rightKey='j', **kwargs):
        super(KeyboardRating, self).__init__(**kwargs)
        self.leftKey = leftKey
        self.rightKey = rightKey

    def create(self):

        pass


class ButtonBoxRating:
    """ Class to draw and record button box responses in the MRI scanner.

    Parameters
    ----------
    win : Psychopy window
        Window/screen of current session
    rating_question : str
        Question to show
    labels : list
        List of labels corresponding to response options
    coords : list[tuple]
        List of x-y coordinates of anchors
    buttons : list
        List of possible button (response) values
    label_height : int
        Size of letters of labels
    anchor_size : int
        Size of anchor (in pix)
    anchor_color : str
        Color of anchor
    anchor_color_chosen : str
        Color of anchor after it's chosen
    offset_label : int
        Offset of label relative to anchor
    time_until_disappear : float
        After the response, how long it takes for the rating to disappear
        (set to inf if it shouldn't disappear)
    fixation : GratingStim/Circle
        Fixation cross to draw (should have a draw method)
    """

    def __init__(self, win, rating_question=None, labels=None, coords=None, buttons=None,
                 label_height=45, anchor_size=15, anchor_color='lightblue',
                 anchor_color_chosen='orange', offset_label=50, time_until_disappear=0.3,
                 fixation=None):
        """ Initializes ButtonBoxRating object. """

        if rating_question is None:
            rating_question = ''

        if len(labels) == 2 and coords is None:
            labels = [labels[0], ' ', ' ', ' ', ' ', ' ', ' ', labels[1]]
            offset_label = -1 * offset_label

        if coords is None:
            coords = [
                (-600, -300), (-450, -200), (-300, -150), (-150, -100),
                (150, -100), (300, -150), (450, -200), (600, -300)
            ]

        if buttons is None:
            # ONLY TEST VERSION (MRI should be 'b', 'y', 'g' etc)
            buttons = [str(i+1) for i in range(8)]

        self.rating_question = rating_question
        self.labels = labels
        self.coords = coords
        self.buttons = buttons
        self.actual_buttons = [b for i, b in enumerate(buttons) if self.labels[i]]
        self.label_height = label_height
        self.anchor_size = anchor_size
        self.anchor_color = anchor_color
        self.anchor_color_chosen = anchor_color_chosen
        self.offset_label = offset_label
        self.time_until_disappear = time_until_disappear

        if fixation is None:
            self.fixation = GratingStim(
                win,
                tex='sin',
                mask='raisedCos',
                size=10,
                texRes=512,
                color='white',
                sf=0,
                maskParams={'fringeWidth': 0.4}
            )

        self.fixation = fixation
        self.rating_anchors = []
        self.label_texts = []
        self.resp = None
        self.time_resp = np.inf

        for i in range(len(self.coords)):
            
            if not self.labels[i]:
                continue  # remove anchor for empty label

            rating_anchor = Circle(
                win=win,
                pos=coords[i],
                radius=self.anchor_size,
                edges=200,
                units='pix',
                fillColor=self.anchor_color,
            )

            #rating_anchor.setColor(self.anchor_color)
            label_text = TextStim(
                win=win,
                pos=(self.coords[i][0], self.coords[i][1] + self.offset_label),
                height=self.label_height,
                units='pix',
                text=self.labels[i]
            )

            self.rating_anchors.append(rating_anchor)
            self.label_texts.append(label_text)

        self.rating_text = TextStim(
            win=win,
            pos=(0, max([c[1] for c in self.coords]) + 200),
            height=self.label_height*1.5,
            units='pix',
            text=self.rating_question,
            wrapWidth=max([c[0] for c in self.coords]) - min([c[0] for c in self.coords])
        )

    def update_color(self, resp=None):
        """ Updates the color of the chosen label.

        Parameters
        ----------
        resp : str
            Response as recorded by the button box.
        """

        if resp is None or not resp:
            raise ValueError("You called update color, but no resp was given!")

        if resp in self.buttons:
            if resp in self.actual_buttons:
                resp_idx = self.actual_buttons.index(resp)
            else:
                return None
        else:
            return None

        for i, anchor in enumerate(self.rating_anchors):

            if resp_idx == i:
                anchor.setFillColor(self.anchor_color_chosen)
            else:
                anchor.setFillColor(self.anchor_color)

        self.time_resp = getTime()
        self.resp = resp
        
    def draw(self):
        """ Draws the rating. """
        
        this_time = getTime()
        if (this_time - self.time_resp) < self.time_until_disappear:

            for anchor, label in zip(self.rating_anchors, self.label_texts):
                anchor.draw()
                label.draw()
            self.rating_text.draw()
        else:
            self.fixation.draw()

    def reset(self):
        if self.resp is not None:
            resp_idx = self.actual_buttons.index(self.resp)
            self.rating_anchors[resp_idx].setFillColor(self.anchor_color) 
        
        self.resp = None
        self.time_resp = np.inf

