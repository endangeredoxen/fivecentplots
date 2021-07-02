import importlib
import os, sys
import pdb
import numpy as np
import pandas as pd
import scipy.stats as ss
import importlib
#import ctypes
from matplotlib.font_manager import FontProperties, findfont
try:
    from PIL import ImageFont
except:
    pass
try:
    import cv2
except:
    pass
db = pdb.set_trace

# Get user default file
user_dir = os.path.expanduser('~')
default_path = os.path.join(user_dir, '.fivecentplots')
if os.path.exists(default_path) and default_path not in sys.path:
    sys.path = [default_path] + sys.path
    from defaults import *


class RepeatedList:
    def __init__(self, values, name):
        """
        Set a default list of items and loop through it beyond the maximum
        index value

        Args:
            values (list): user-defined list of values
            name (str): label to describe contents of class
        """

        self.values = validate_list(values)
        self.shift = 0

        if type(self.values) is not list and len(self.values) < 1:
            raise(ValueError, 'RepeatedList for "%s" must contain an actual '
                              'list with more at least one element')

    def __len__(self):
        return len(self.values)

    def get(self, idx):

        # can we make this a next??

        return self.values[(idx + self.shift) % len(self.values)]


class PlatformError(Exception):
    def __init__(self):
        super().__init__('Image tests currently require Windows 10 installation to run')


def ci(data, coeff=0.95):
    """
    Compute the confidence interval

    Args:
        data (pd.Series): raw data for computation

    Returns:
        lower confidence interval, upper confidence interval
        returns np.nan if standard err < 0

    """

    sem = data.sem()
    size = len(data.dropna()) - 1

    if sem > 0:
        return ss.t.interval(coeff, size, loc=data.mean(), scale=sem)

    else:
        return np.nan, np.nan


def dfkwarg(args, kwargs):
    """
    Add the DataFrame to kwargs

    Args:
        args (tuple):  *args sent to plot
        kwargs (dict):  **kwargs sent to plot

    Returns:
        updated kwargs

    """

    if len(args) > 0:
        kwargs['df'] = args[0]

    return kwargs


def df_filter(df, filt_orig, drop_cols=False):
    """
    Filter the DataFrame

    Due to limitations in pd.query, column names must not have spaces.  This
    function will temporarily replace spaces in the column names with
    underscores, but the supplied query string must contain column names
    without any spaces

    Args:
        df (pd.DataFrame):  DataFrame to filter
        filt_orig (str):  query expression for filtering
        drop_cols (bool): drop filtered columns from results

    Returns:
        filtered DataFrame
    """

    def special_chars(text, skip=[]):
        """
        Replace special characters in a text string
        Args:
            text (str): input string
            skip (list): characters to skip
        Returns:
            formatted string
        """

        chars = {' ': '_', '.': 'dot', '[': '',']': '', '(': '', ')': '',
                    '-': '_', '^': '', '>': '', '<': '', '/': '_', '@': 'at',
                    '%': 'percent', '*': '_', ':': 'sc'}
        for sk in skip:
            chars.pop(sk)
        for k, v in chars.items():
            text = text.replace(k, v).lstrip(' ').rstrip(' ')
        return text

    # Parse the filter string
    filt = get_current_values(df, filt_orig)

    # Rename the columns to remove special characters
    cols_orig = [f for f in df.columns]
    cols_new = ['fCp%s' % f for f in cols_orig.copy()]
    cols_new = [special_chars(f) for f in cols_new]
    cols_used = []

    df.columns = cols_new

    # Reformat the filter string for compatibility with pd.query
    operators = ['==', '<', '>', '!=']
    ands = [f.lstrip().rstrip() for f in filt.split('&')]
    for ia, aa in enumerate(ands):
        if 'not in [' in aa:
            key, val = aa.split('not in ')
            key = key.rstrip(' ')
            key = 'fCp%s' % special_chars(key)
            vals = val.replace('[', '').replace(']', '').split(',')
            for iv, vv in enumerate(vals):
                vals[iv] = vv.lstrip().rstrip()
            key2 = '&' + key + '!='
            val_str = key2.join(vals)
            ands[ia] = '(%s%s)' % (key + '!=', key2.join(vals))
            continue
        elif 'in [' in aa:
            key, val = aa.split('in ')
            key = key.rstrip(' ')
            key = 'fCp%s' % special_chars(key)
            vals = val.replace('[', '').replace(']', '').split(',')
            for iv, vv in enumerate(vals):
                vals[iv] = vv.lstrip().rstrip()
            key2 = '|' + key + '=='
            val_str = key2.join(vals)
            ands[ia] = '(%s%s)' % (key + '==', key2.join(vals))
            continue
        ors = [f.lstrip() for f in aa.split('|')]
        for io, oo in enumerate(ors):
            # Temporarily remove any parentheses
            param_start = False
            param_end = False
            if oo[0] == '(':
                oo = oo[1:]
                param_start = True
            if oo[-1] == ')':
                oo = oo[0:-1]
                param_end = True
            for op in operators:
                if op not in oo:
                    continue
                vals = oo.split(op)
                vals[0] = vals[0].rstrip()
                cols_used += [vals[0]]
                vals[1] = vals[1].lstrip()
                if vals[1] == vals[0]:
                    vals[1] = 'fCp%s' % special_chars(vals[1])
                vals[0] = 'fCp%s' % special_chars(vals[0])
                ors[io] = op.join(vals)
                if param_start:
                    ors[io] = '(' + ors[io]
                if param_end:
                    ors[io] = ors[io] + ')'
        if len(ors) > 1:
            ands[ia] = '|'.join(ors)
        else:
            ands[ia] = ors[0]
    if len(ands) > 1:
        filt = '&'.join(ands)
    else:
        filt = ands[0]

    # Apply the filter
    try:
        df = df.query(filt)
        df.columns = cols_orig

        if drop_cols:
            cols_used = list(set(cols_used))
            for col in cols_used:
                del df[col]

        return df

    except:
        print('Could not filter data!\n   Original filter string: %s\n   '
              'Modified filter string: %s' % (filt_orig, filt))

        df.columns = cols_orig

        return df


def df_summary(df, columns=[], exclude=[], multiple=False):
    """
    Return a summary table of unique conditions in a DataFrame

    If more than one value exists, replace with "Multiple" or a
    string list of the values

    Args:
        df (pd.DataFrame): input DataFrame
        columns (list): list of column names to include; if empty,
            include all columns
        exclude(list): list of column names to ignore
        multiple (bool): toggle between showing discrete values or
            "Multiple" when a column contains more than one value

    """

    # Filter the DataFrame
    if columns:
        df = df[columns]
    if exclude:
        for ex in exclude:
            del df[ex]

    # Build the new DataFrame
    summ = pd.DataFrame(index=[0])
    for col in df.columns:
        uni = df[col].unique()
        if len(uni) == 1:
            summ[col] = uni[0]
        elif multiple:
            summ[col] = '; '.join([str(f) for f in uni])
        else:
            summ[col] = 'Multiple'

    return summ


def df_unique(df):
    """
    Get column names with one unique value

    Args:
        df (pd.DataFrame): data to check

    Returns:
        dict of col names and unique values

    """

    unq = {}

    for col in df.columns:
        val = df[col].unique()
        if len(val) == 1:
            unq[col] = val[0]

    return unq


def get_current_values(df, text, key='@'):
    """
    Parse a string looking for text enclosed by 'key' and replace with the
    current value from the DataFrame

    Args:
        df (pd.DataFrame): DataFrame containing values we are looking for
        text (str): string to parse
        key (str): matching chars that enclose the value to replace from df

    Returns:
        updated string
    """

    if key in text:
        rr = re.search(r'\%s(.*)\%s' % (key, key), text)
        if rr is not None:
            val = rr.group(1)
            pos = rr.span()
            if val in df.columns:
                val = df[val].iloc[0]
            else:
                val = ''
            text = text[0:pos[0]] + str(val) + text[pos[1]:]

    return text


def get_decimals(value, max_places=4):
    """
    Get the number of decimal places of a float number excluding
    rounding errors

    Args:
        value (int|float): value to check
        max_places (int): maximum number of decimal places to check

    Returns:
        number of decimal places
    """

    last = np.nan
    for i in range(0, max_places+1):
        current = round(value, i)
        if current == last and current > 0:
            break
        else:
            last = current

    return i - 1


def get_mpl_version_dir():
    """
    Get the matplotlib version directory for test images based
    on the current version of mpl
    """

    from distutils.version import LooseVersion
    import matplotlib as mpl
    version = LooseVersion(mpl.__version__)

    if version > LooseVersion('1') and version < LooseVersion('3'):
        return 'mpl_v2'

    else:
        return 'mpl_v3'


def get_text_dimensions(text, **kwargs):
    # class SIZE(ctypes.Structure):
    #     _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]

    # hdc = ctypes.windll.user32.GetDC(0)
    # hfont = ctypes.windll.gdi32.CreateFontW(-points, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, font)
    # hfont_old = ctypes.windll.gdi32.SelectObject(hdc, hfont)
    # size = SIZE(0, 0)
    # ctypes.windll.gdi32.GetTextExtentPoint32W(hdc, text, len(text), ctypes.byref(size))
    # ctypes.windll.gdi32.SelectObject(hdc, hfont_old)
    # ctypes.windll.gdi32.DeleteObject(hfont)

    # return (size.cx, size.cy)

    try:
        ImageFont
    except:
        print('get_text_dimensions requires pillow which was not found.  Please '
              'run pip install pillow and try again.')
        return False

    font = FontProperties()
    font.set_family(kwargs['font'])
    font.set_style(kwargs['font_style'])
    font.set_weight(kwargs['font_weight'])
    fontfile = findfont(font, fallback_to_default=True)

    size = ImageFont.truetype(fontfile , kwargs['font_size']).getsize(text)

    return size[0]*1.125, size[1]*1.125  # no idea why it is off


def kwget(dict1, dict2, val, default):
    """
    Augmented kwargs.get function

    Args:
        dict1 (dict): first dictionary to check for the value
        dict2 (dict): second dictionary to check for the value
        val (str): value to look for
        default (multiple): default value if not found in
            dict1 or dict2 keys

    Returns:
        value to use
    """

    if val in dict1.keys():
        return dict1[val]
    elif val in dict2.keys():
        return dict2[val]
    else:
        return default


def img_compare(img1, img2, show=False):
    """
    Read two images and compare for difference by pixel
    This is an optional utility used only for developer tests, so
    the function will only work if opencv is installed
    Args:
        img1 (str): path to file #1
        img2 (str): path to file #2
        show (bool): display the difference
    Returns:
        True/False of existence of differences
    """

    try:
        cv2
    except:
        print('img_compare requires opencv which was not found.  Please '
              'run pip install opencv-python and try again.')
        return False

    # read images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # compare
    if img1 is None or img2 is None or img1.shape != img2.shape:
        is_diff = True
        if show and img1.shape != img2.shape:
            print('image sizes do not match')

    else:
        difference = cv2.subtract(img1, img2)
        is_diff = np.any(difference)
        if show and is_diff:
            cv2.imwrite('difference.png', 10 * difference)
            os.startfile('difference.png')

    return is_diff


def nq(data, column='Value', **kwargs):
    """
    Normal quantile calculation
    """

    # Defaults
    sig = kwargs.get('sigma', None)
    tail = kwargs.get('tail', 3)
    step_tail = kwargs.get('step_tail', 0.2)
    step_inner = kwargs.get('step_inner', 0.5)

    # Flatten the DataFrame to an array
    data = data.values.flatten()

    # Get sigma
    if not sig:
        sig = sigma(data)
    index = np.concatenate([np.arange(-sig, -tail, step_tail),
                            np.arange(-tail, tail, step_tail),
                            np.arange(tail, sig + 1e-9, step_tail)])

    # Get the sigma value
    values = np.zeros(len(index))
    for ii, idx in enumerate(index):
        values[ii] = np.percentile(data, ss.norm.cdf(idx) * 100)

    return pd.DataFrame({'Sigma': index, column: values})


def plot_num(ir, ic, ncol):
    """
    Get the plot number based on grid location
    """

    return (ic + ir * ncol + 1)


def rectangle_overlap(r1, r2):
    """
    Determine if the bounds of two rectangles overlap

    Args:
        r1 (list|tuple): width, height, center point (x, y) of left rectangle
        r2 (list|tuple): width, height, center point (x, y) of right rectangle

    Returns:
        True if overlap | False if not
    """

    # Get bounds
    b1 = [r1[2][0] - r1[0]/2, r1[2][0] + r1[0]/2,
          r1[2][1] + r1[1]/2, r1[2][1] - r1[1]/2]
    b2 = [r2[2][0] - r2[0]/2, r2[2][0] + r2[0]/2,
          r2[2][1] + r2[1]/2, r2[2][1] - r2[1]/2]

    if b1[0] < b2[1] and b1[1] > b2[0] \
            and b1[2] > b2[3] and b1[3] < b2[2]:
        return True
    else:
        return False


def reload_defaults(theme=None):
    """
    Reload the fcp params
    """

    theme_dir = os.path.join(os.path.dirname(__file__), 'themes')
    reset_path = False
    err_msg = 'Requested theme not found; using default'

    if theme is not None and os.path.exists(theme):
        # full filename case
        theme_dir = os.sep.join(theme.split(os.sep)[0:-1])
        theme = theme.split(os.sep)[-1]
        sys.path = [theme_dir] + sys.path
        reset_path = True
        try:
            defaults = importlib.import_module(theme.replace('.py', ''), theme_dir)
            importlib.reload(defaults)
        except:
            print(err_msg)
            import defaults
            importlib.reload(defaults)

    elif theme is not None and \
            (theme in os.listdir(theme_dir) or theme+'.py' in os.listdir(theme_dir)):
        sys.path = [theme_dir] + sys.path
        reset_path = True
        try:
            defaults = importlib.import_module(theme.replace('.py', ''), theme_dir)
            importlib.reload(defaults)
        except:
            print(err_msg)
            import defaults
            importlib.reload(defaults)
    else:
        # use default theme
        import defaults
        importlib.reload(defaults)

    fcp_params = defaults.fcp_params
    colors = defaults.colors if hasattr(defaults, 'colors') else None
    markers = defaults.markers if hasattr(defaults, 'markers') else None

    if reset_path:
        sys.path = sys.path [1:]

    return fcp_params, colors, markers


def set_save_filename(df, ifig, fig_item, fig_cols, layout, kwargs):
    """
    Build a filename based on the plot setup parameters

    Args:
        df (pd.DataFrame): input data used for the plot
        fig_item (None | str): unique figure item
        fig_cols (None | str | list): names of columns used to generate fi
        layout (obj): plot layout object
        kwargs (dict): input keyword args

    Returns:
        str filename
    """

    # Use provided filename
    if 'filename' in kwargs.keys() and type(kwargs['filename']) is str:
        filename = kwargs['filename']
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            ext = kwargs.get('save_ext')
        filename = filename.replace(ext, '')
        fig = ''
        if fig_item is not None:
            fig_items = validate_list(fig_item)
            for icol, col in enumerate(fig_cols):
                if icol > 0:
                    fig += ' and'
                fig += ' where %s=%s' % (col, fig_items[icol])
        return filename + fig + ext

    # Build a filename
    if 'z' in kwargs.keys():
        z = layout.label_z.text + ' vs '
    else:
        z = ''
    y = ' and '.join(validate_list(layout.label_y.text))
    if layout.axes.twin_x:
        y += ' + ' + layout.label_y2.text
    if 'x' in kwargs.keys():
        y += ' vs '
        x = ' and '.join(validate_list(layout.label_x.text))
    else:
        x = ''
    if layout.axes.twin_y:
        x += ' + ' + layout.label_x2.text
    row, col, wrap, groups, fig = '', '', '', '', ''
    if layout.label_row.text is not None:
        row = ' by ' + layout.label_row.text
    if layout.label_col.text is not None:
        col = ' by ' + layout.label_col.text
    if layout.title_wrap.text is not None:
        wrap = ' by ' + layout.title_wrap.text
    # this one may need to change with box plot
    if kwargs.get('groups', False):
        groups = ' by ' + ' + '.join(validate_list(kwargs['groups']))
    if fig_item is not None:
        fig_items = validate_list(fig_item)
        for icol, col in enumerate(fig_cols):
            if icol > 0:
                fig += ' and'
            fig += ' where %s=%s' % (col, fig_items[icol])

    filename = '%s%s%s%s%s%s%s%s' % (z, y, x, row, col, wrap, groups, fig)

    # Cleanup forbidden symbols
    bad_char = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for bad in bad_char:
        filename = filename.replace(bad, '_')

    # Make sure extension has a '.'
    if kwargs.get('save_ext')[0] != '.':
        kwargs['save_ext'] = '.' + kwargs['save_ext']

    return filename + kwargs.get('save_ext')


def sigma(x):
    """
    Calculate the sigma range for a data set

    Args:
        x (pd.Series | np.array): data set

    Returns:
        int value to use for +/- max sigma

    """

    return np.round(np.trunc(10 * abs(ss.norm.ppf(1 / len(x)))) / 10)


def validate_list(items):
    """
    Make sure a list variable is actually a list and not a single string

    Args:
        items (str|list): values to check dtype

    Return:
        items as a list
    """

    if items is None:
        return None
    if type(items) is tuple:
        return list(items)
    elif type(items) is not list:
        return [items]
    else:
        return items
