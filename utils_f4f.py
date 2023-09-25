from __future__ import division
import math
import re

import pandas as pd
import numpy as np
import matplotlib as mpl
import seaborn as sns

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM 
from keras.callbacks import Callback


#------------------------------------------------
#%% Heat Maps 
#------------------------------------------------

# Counting function
aa_alphabet = [
    'A','C','D','E','F','G','H','I','K','L',
    'M','N','P','Q','R','S','T','V','W','Y'
]

n_pos = 7

def aa_to_matrix(aa_seqs, normalize=True):
    
    """
    
    aa_to_matrix calculates amino acid frequencies in aa_seqs 
    
    """
    
    seq_mat = pd.DataFrame.from_records(aa_seqs.apply(list))
    mat = pd.DataFrame(np.zeros((len(aa_alphabet), n_pos)), index=aa_alphabet)
    
    for i in range(n_pos):
        mat[i] = seq_mat.iloc[:, i].value_counts().sort_index()
    
    mat = mat.fillna(0)
    
    if normalize:
        mat = mat / len(aa_seqs)
    
    return mat.values


# Heatmap function 
def heatmap(aa_seqs, ax, title,saveFileName, vmin, cmap, vmax):
    
    """
    
    heatmap visualizes a heatmap of the amino acid frequencies, aa_seq, 
    for a list of sequences obtained from aa_to_matrix function. 
    
    """
    
    crnt_heatmap = aa_to_matrix(aa_seqs) - (1/len(aa_alphabet))
    hm = sns.heatmap(crnt_heatmap,
        vmin=vmin, center=0, vmax=vmax,
        cmap=cmap,
        square=True,
        cbar=False
    )
    ax.set_xticks(np.arange(0, n_pos) + 0.5)
    ax.set_xticklabels([str(x) for x in np.arange(0, n_pos) + 1], fontsize=5)

    ax.set_yticks(np.arange(0, len(aa_alphabet)) + 0.5)
    ax.set_yticklabels(aa_alphabet, fontsize=5, fontfamily='monospace')
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        
    ax.tick_params(axis='x', pad=-1)
    ax.tick_params(axis='y', pad=-1)
    
    ax.set_title(title, y=0.95, fontsize=7)
    
    # Save heatmap 
    df_heatmap = pd.DataFrame(crnt_heatmap)
    df_heatmap['AA'] = aa_alphabet
    df_heatmap.to_csv(saveFileName)

    
    return hm



#------------------------------------------------
#%% Modeling 
#------------------------------------------------

# Hot Encoding function 
def AA_hotencoding(variant):
    
    """
    
    AA_hotencoding takes an amino acid sequence 'variant' of an arbitrary length, 
    and returns a 20xlength one-hot encoding matrix 'onehot_encoded'.   
    
    """
       
    AAs = 'ARNDCQEGHILKMFPSTWYV'
    encoding_length = len(AAs)
    variant_length = len(variant)

    # Define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(AAs))
    int_to_char = dict((i, c) for i, c in enumerate(AAs))

    # Encode input data 
    integer_encoded = [char_to_int[char] for i, char in enumerate(variant) if i <variant_length]
    
    # Start one-hot-encoding
    onehot_encoded = list()
    
    for value in integer_encoded:
        letter = [0 for _ in range(encoding_length)]
        letter[value] = 1
        onehot_encoded.append(letter)
                
    return onehot_encoded




# Custom early stopping 
class CustomEarlyStopping(Callback):
    
    """
    
    Modified from:
    https://stackoverflow.com/questions/42470604/keras-callback-earlystopping-comparing-training-and-validation-loss
    Distribution of this class is subject to the licenses enforced by stackoverflow.com 
    
    'CustomEarlyStopping' enforces early stop of ML training process when a user-defined condition is met. 
    
    """
    def __init__(self, ratio=0.0,
                 patience=0, verbose=0, restore_best_weights = True):
        #super(EarlyStopping, self).__init__()
        super(CustomEarlyStopping, self).__init__()

        self.ratio = ratio
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.greater
        self.restore_best_weights = True

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used

    def on_epoch_end(self, epoch, logs=None):
        current_val = logs.get('val_loss')
        current_train = logs.get('loss')
        if current_val is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        # If ratio current_loss / current_val_loss > self.ratio
        if self.monitor_op(np.divide(current_train,current_val),self.ratio):
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

            

            
# Define master fitness learning model 

def parent_model(L1=140, L2=20):
    
    """
    
    parent_model builds an LSTM model with paramters that work accross all functional fitness models in the Fit4Function study. 
    L1 and L2 define the sizes of the model two layers. 
    
    """
    model = Sequential()
    model.add(LSTM(L1, return_sequences=True, input_shape=(7, 20)))
    model.add(LSTM(L2, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    return model



#------------------------------------------------
#%% Spiderplots
#------------------------------------------------



def spiderplot(ax, df, dfa, AA, assays, assay_label_colors):
    
    """
    
    spiderplot is specific for visualization the spiderplots of multiple on-target and off-target variants for capsid variant measurements. 
    
    """
    
    assay_cols, assay_names, assay_bounds, assay_thetas = assays
    
    control_labels=['AAV9rep']
    aa_df = pd.concat([
        df.loc[(df['Label'].isin(control_labels))],
        df.loc[(df['SeqID'] == AA)]
    ], axis=0)
    aa_dfa = pd.concat([
        dfa.loc[(dfa['Label'].isin(control_labels))],
        dfa.loc[(dfa['SeqID'] == AA)]
    ], axis=0)
    
    #theta_inc = (np.pi*2)/len(assay_cols) # Angle between assays
    #theta = np.arange(0, np.pi*2, theta_inc)
    theta = np.array(assay_thetas) * np.pi * 2

    # POLAR COORDINATES
    # X = theta
    # Y = r (radius)    
    c_aa = (0.12156862745098039, 0.4666666666666667, 0.7058823529411765) # Blue
    c_control = (0.3, 0.3, 0.3) # lightish grey
    
    for i, row in aa_df.iterrows():
        r = []
        for k, col in enumerate(assay_cols):
            bounds = assay_bounds[k]
            x = np.clip(row[col], bounds[0], bounds[1])
            # zero_offset = np.abs(bounds[0]) / (bounds[1] - bounds[0])
            x = (x - bounds[0]) / (bounds[1] - bounds[0])
            r.append(x)
            
        if row['Label'] in control_labels:
            c = c_control
        else:
            c = c_aa
        
        ec = tuple([x for x in c] + [1.0])
        
        ax.scatter(
            theta, r, 
            facecolor='none', edgecolor=ec, s=5, marker='o', linewidths=0.5, alpha=0.6,
            #facecolor=ec, s=20, marker='x',
            zorder=1000 # Plot over polygon patches
        )
        
    for i, row in aa_dfa.iterrows():
        r = []
        for k, col in enumerate(assay_cols):
            bounds = assay_bounds[k]
            x = np.clip(row[col], bounds[0], bounds[1])
            # zero_offset = np.abs(bounds[0]) / (bounds[1] - bounds[0])
            x = (x - bounds[0]) / (bounds[1] - bounds[0])
            r.append(x)
            
        
        if row['Label'] in control_labels:
            facealpha = 0.1
            c = c_control
        else:
            facealpha = 0.3
            c = c_aa
        
        # Reduce alpha of facecolor
        fc = tuple([x for x in c] + [facealpha])
        ec = tuple([x for x in c] + [1.0])
        
        ax.add_patch(mpl.patches.Polygon(
            np.vstack([theta, r]).T,
            facecolor=fc, edgecolor=ec, linewidth=0.5
        ))
        
    ax.set_xticks(theta)
    ax.set_xticklabels(assay_names)
    # Place labels on the outside of the plot
    # Check for left/right side of the circle
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_fontsize(8)
        tick.set_color(assay_label_colors[i])
        if tick._x > (np.pi/2) and tick._x < (3 * (np.pi/2)):
            tick.set_ha('right')
        else:
            tick.set_ha('left')
    ax.tick_params(axis='x', pad=-4, length=1, 
               grid_color='#CCC', grid_linewidth=0.5)
    
    rticks = np.arange(0, 1.1, 0.2)
    ax.set_rlim([0, 1])
    ax.set_rticks(rticks)
    ax.set_yticklabels([])
    
    rgridlines, rgridlabels = ax.set_rgrids(rticks)
    for line in rgridlines:
        line.set_linewidth(0.5)
        line.set_color('#EEE')
            
    # Radius Axis Label
    rlabel_pos = -np.pi/2
    ax.set_rlabel_position(np.rad2deg(rlabel_pos))

    title_x = 0.5
    title_y = 1.26
    ax.text(title_x, title_y, AA, 
            ha='center', va='center', transform=ax.transAxes,
           fontsize=8, fontfamily='monospace')
    
    ax.add_artist(mpl.lines.Line2D([np.pi, np.pi], [0, 1], clip_on=False, color='#AAA', linewidth=0.5))
    ax.add_artist(mpl.lines.Line2D([0,0], [0, 1], clip_on=False, color='#AAA', linewidth=0.5))
    
    return ax



#------------------------------------------------
#%% si-prefix
#------------------------------------------------


# https://github.com/cfobel/si-prefix

# from ._version import get_versions
# __version__ = get_versions()['version']
# del get_versions

# Print a floating-point number in engineering notation.
# Ported from [C version][1] written by
# Jukka “Yucca” Korpela <jkorpela@cs.tut.fi>.
#
# [1]: http://www.cs.tut.fi/~jkorpela/c/eng.html

#: .. versionchanged:: 1.0
#:     Define as unicode string and use µ (i.e., ``\N{MICRO SIGN}``, ``\x0b5``)
#:     to denote micro (not u).
#:
#:     .. seealso::
#:
#:         `Issue #4`_.
#:
#:         `Forum post`_ discussing unicode using µ as an example.
#:
#:         `The International System of Units (SI) report`_ from the Bureau
#:         International des Poids et Mesures
#:
#: .. _`Issue #4`: https://github.com/cfobel/si-prefix/issues/4
#: .. _`Forum post`: https://mail.python.org/pipermail/python-list/2009-February/525913.html
#: .. _`The International System of Units (SI) report`: https://www.bipm.org/utils/common/pdf/si_brochure_8_en.pdf
SI_PREFIX_UNITS = u"yzafpnµm kMGTPEZY"
#: .. versionchanged:: 1.0
#:     Use unicode string for SI unit to support micro (i.e., µ) character.
#:
#:     .. seealso::
#:
#:         `Issue #4`_.
#:
#: .. _`Issue #4`: https://github.com/cfobel/si-prefix/issues/4
CRE_SI_NUMBER = re.compile(r'\s*(?P<sign>[\+\-])?'
                           r'(?P<integer>\d+)'
                           r'(?P<fraction>.\d+)?\s*'
                           u'(?P<si_unit>[%s])?\s*' % SI_PREFIX_UNITS)


def split(value, precision=1):
    '''
    Split `value` into value and "exponent-of-10", where "exponent-of-10" is a
    multiple of 3.  This corresponds to SI prefixes.

    Returns tuple, where the second value is the "exponent-of-10" and the first
    value is `value` divided by the "exponent-of-10".

    Args
    ----
    value : int, float
        Input value.
    precision : int
        Number of digits after decimal place to include.

    Returns
    -------
    tuple
        The second value is the "exponent-of-10" and the first value is `value`
        divided by the "exponent-of-10".

    Examples
    --------

    .. code-block:: python

        si_prefix.split(0.04781)   ->  (47.8, -3)
        si_prefix.split(4781.123)  ->  (4.8, 3)

    See :func:`si_format` for more examples.
    '''
    negative = False
    digits = precision + 1

    if value < 0.:
        value = -value
        negative = True
    elif value == 0.:
        return 0., 0

    expof10 = int(math.log10(value))
    if expof10 > 0:
        expof10 = (expof10 // 3) * 3
    else:
        expof10 = (-expof10 + 3) // 3 * (-3)

    value *= 10 ** (-expof10)

    if value >= 1000.:
        value /= 1000.0
        expof10 += 3
    elif value >= 100.0:
        digits -= 2
    elif value >= 10.0:
        digits -= 1

    if negative:
        value *= -1

    return value, int(expof10)


def prefix(expof10):
    '''
    Args:

        expof10 : Exponent of a power of 10 associated with a SI unit
            character.

    Returns:

        str : One of the characters in "yzafpnum kMGTPEZY".
    '''
    prefix_levels = (len(SI_PREFIX_UNITS) - 1) // 2
    si_level = expof10 // 3

    if abs(si_level) > prefix_levels:
        raise ValueError("Exponent out range of available prefixes.")
    return SI_PREFIX_UNITS[si_level + prefix_levels]


def si_format(value, precision=1, format_str=u'{value} {prefix}',
              exp_format_str=u'{value}e{expof10}'):
    '''
    Format value to string with SI prefix, using the specified precision.

    Parameters
    ----------
    value : int, float
        Input value.
    precision : int
        Number of digits after decimal place to include.
    format_str : str or unicode
        Format string where ``{prefix}`` and ``{value}`` represent the SI
        prefix and the value (scaled according to the prefix), respectively.
        The default format matches the `SI prefix style`_ format.
    exp_str : str or unicode
        Format string where ``{expof10}`` and ``{value}`` represent the
        exponent of 10 and the value (scaled according to the exponent of 10),
        respectively.  This format is used if the absolute exponent of 10 value
        is greater than 24.

    Returns
    -------
    unicode
        :data:`value` formatted according to the `SI prefix style`_.

    Examples
    --------

    For example, with `precision=2`:

    .. code-block:: python

        1e-27 --> 1.00e-27
        1.764e-24 --> 1.76 y
        7.4088e-23 --> 74.09 y
        3.1117e-21 --> 3.11 z
        1.30691e-19 --> 130.69 z
        5.48903e-18 --> 5.49 a
        2.30539e-16 --> 230.54 a
        9.68265e-15 --> 9.68 f
        4.06671e-13 --> 406.67 f
        1.70802e-11 --> 17.08 p
        7.17368e-10 --> 717.37 p
        3.01295e-08 --> 30.13 n
        1.26544e-06 --> 1.27 u
        5.31484e-05 --> 53.15 u
        0.00223223 --> 2.23 m
        0.0937537 --> 93.75 m
        3.93766 --> 3.94
        165.382 --> 165.38
        6946.03 --> 6.95 k
        291733 --> 291.73 k
        1.22528e+07 --> 12.25 M
        5.14617e+08 --> 514.62 M
        2.16139e+10 --> 21.61 G
        9.07785e+11 --> 907.78 G
        3.8127e+13 --> 38.13 T
        1.60133e+15 --> 1.60 P
        6.7256e+16 --> 67.26 P
        2.82475e+18 --> 2.82 E
        1.1864e+20 --> 118.64 E
        4.98286e+21 --> 4.98 Z
        2.0928e+23 --> 209.28 Z
        8.78977e+24 --> 8.79 Y
        3.6917e+26 --> 369.17 Y
        1.55051e+28 --> 15.51e+27
        6.51216e+29 --> 651.22e+27

    .. versionchanged:: 1.0
        Use unicode string for :data:`format_str` and SI value format string to
        support micro (i.e., µ) characte, and change return type to unicode
        string.

        .. seealso::

            `Issue #4`_.

    .. _`Issue #4`: https://github.com/cfobel/si-prefix/issues/4
    .. _SI prefix style:
        http://physics.nist.gov/cuu/Units/checklist.html
    '''
    svalue, expof10 = split(value, precision)
    value_format = u'%%.%df' % precision
    value_str = value_format % svalue
    try:
        return format_str.format(value=value_str,
                                 prefix=prefix(expof10).strip())
    except ValueError:
        sign = ''
        if expof10 > 0:
            sign = "+"
        return exp_format_str.format(value=value_str,
                                     expof10=''.join([sign, str(expof10)]))


def si_parse(value):
    '''
    Parse a value expressed using SI prefix units to a floating point number.

    Parameters
    ----------
    value : str or unicode
        Value expressed using SI prefix units (as returned by :func:`si_format`
        function).


    .. versionchanged:: 1.0
        Use unicode string for SI unit to support micro (i.e., µ) character.

        .. seealso::

            `Issue #4`_.

    .. _`Issue #4`: https://github.com/cfobel/si-prefix/issues/4
    '''
    CRE_10E_NUMBER = re.compile(r'^\s*(?P<integer>[\+\-]?\d+)?'
                                r'(?P<fraction>.\d+)?\s*([eE]\s*'
                                r'(?P<expof10>[\+\-]?\d+))?$')
    CRE_SI_NUMBER = re.compile(r'^\s*(?P<number>(?P<integer>[\+\-]?\d+)?'
                               r'(?P<fraction>.\d+)?)\s*'
                               u'(?P<si_unit>[%s])?\s*$' % SI_PREFIX_UNITS)
    match = CRE_10E_NUMBER.match(value)
    if match:
        # Can be parse using `float`.
        assert(match.group('integer') is not None or
               match.group('fraction') is not None)
        return float(value)
    match = CRE_SI_NUMBER.match(value)
    assert(match.group('integer') is not None or
           match.group('fraction') is not None)
    d = match.groupdict()
    si_unit = d['si_unit'] if d['si_unit'] else ' '
    prefix_levels = (len(SI_PREFIX_UNITS) - 1) // 2
    scale = 10 ** (3 * (SI_PREFIX_UNITS.index(si_unit) - prefix_levels))
    return float(d['number']) * scale


def si_prefix_scale(si_unit):
    '''
    Parameters
    ----------
    si_unit : str
        SI unit character, i.e., one of "yzafpnµm kMGTPEZY".

    Returns
    -------
    int
        Multiple associated with `si_unit`, e.g., 1000 for `si_unit=k`.
    '''
    return 10 ** si_prefix_expof10(si_unit)


def si_prefix_expof10(si_unit):
    '''
    Parameters
    ----------
    si_unit : str
        SI unit character, i.e., one of "yzafpnµm kMGTPEZY".

    Returns
    -------
    int
        Exponent of the power of ten associated with `si_unit`, e.g., 3 for
        `si_unit=k` and -6 for `si_unit=µ`.
    '''
    prefix_levels = (len(SI_PREFIX_UNITS) - 1) // 2
    return (3 * (SI_PREFIX_UNITS.index(si_unit) - prefix_levels))