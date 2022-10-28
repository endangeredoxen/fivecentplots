import importlib
import os
import sys
import pdb
import numpy as np
import pandas as pd
import scipy.stats as ss
import datetime
import subprocess
import pathlib
import re
import shlex
import inspect
from matplotlib.font_manager import FontProperties, findfont
try:
    from PIL import ImageFont  # used only for bokeh font size calculations
except ImportError:
    pass
try:
    import cv2  # required for testing
except (ImportError, ModuleNotFoundError):
    pass
db = pdb.set_trace

# Get user default file
user_dir = pathlib.Path.home()
default_path = user_dir / '.fivecentplots'
if default_path.exists() and default_path not in sys.path:
    sys.path = [str(default_path)] + sys.path
    try:
        from defaults import *  # noqa
    except ModuleNotFoundError:
        from . themes.gray import *  # noqa


# Convenience kwargs
HIST = {'ax_scale': 'logy', 'markers': False, 'line_width': 2, 'preset': 'HIST'}


class RepeatedList:
    def __init__(self, values: list, name: str, override: dict = {}):
        """Set a default list of items and loop through it beyond the maximum
        index value.

        Args:
            values: user-defined list of values
            name: label to describe contents of class
            override: override the RepeatedList value based on the legend value for this item
        """
        self.values = validate_list(values)
        self.shift = 0
        self.override = override

        if not isinstance(self.values, list) or len(self.values) < 1:
            raise ValueError('RepeatedList must contain an actual list with more at least one element')

    def __len__(self):
        """Custom length."""
        return len(self.values)

    def __getitem__(self, idx):
        """Custom __get__ to cycle back to start of list for index value > length of the array."""
        if isinstance(idx, tuple):  # use a tuple when overriding
            idx, key = idx
        else:
            key = None

        val = self.values[(idx + self.shift) % len(self.values)]

        if len(list(self.override.keys())) == 0 or key not in self.override.keys():
            return val
        else:
            return self.override[key]


class CFAError(Exception):
    def __init__(self, *args, **kwargs):
        """Exception class for CFA definition issues."""
        Exception.__init__(self, *args, **kwargs)


class Timer:
    def __init__(self, print: bool = True, start: bool = False, units: str = 's'):
        """Quick timer class to optimize/debug speed of various functions.

        Args:
            print (optional): enable/disable print messages. Defaults to True.
            start (optional): start timer on class init. Defaults to False.
            units (optional): convert timedelta into "s" or "ms". Defaults to 's'.
        """
        self.print = print
        self.init = None
        self.units = units
        self._total = 0
        if start:
            self.start()
        if units not in ['s', 'ms']:
            raise ValueError('timer only supports "s" and "ms" time units')

    @property
    def now(self):
        """Current datetime."""
        return datetime.datetime.now()

    @property
    def total(self):
        """Total time."""
        return self._total

    @total.setter
    def total(self, val):
        """Add to the running total."""
        self._total += val

    def get(self, label: str = '', restart: bool = True, stop: bool = False):
        """Get the current timer value.

        Args:
            label (optional): prepend the time with a string label. Defaults to ''.
            reset (optional): restart the timer on get call. Defaults to True.
            stop (optional): ends the timer. Defaults to False.
        """
        if not self.init:
            print('timer has not been started')
            return

        delta = self.now - self.init
        if self.units == 'ms':
            delta = delta.seconds * 1000 + delta.microseconds / 1000
        else:
            delta = delta.seconds + delta.microseconds / 1E6
        self.total = delta

        if label != '':
            label += ': '

        if self.print is True:
            print(label + str(delta) + ' [%s]' % self.units)

        if restart is True:
            self.start()

        if stop is True:
            self.stop()

    def get_total(self):
        """Return the total time."""
        if self.print:
            print(f'Total time: {self.total} [{self.units}]')

    def start(self):
        """Start the timer."""
        self.init = self.now

    def stop(self):
        """Stop the timer."""
        self.init = None


def ci(data: pd.Series, coeff: float = 0.95) -> [float, float]:
    """Compute a confidence interval.

    Args:
        data: raw data column for computation
        coeff (optional): the confidence value. Defaults to 0.95.

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


def dfkwarg(args: tuple, kwargs: dict) -> dict:
    """Add the DataFrame to kwargs.

    Args:
        args:  *args sent to plot
        kwargs:  **kwargs sent to plot

    Returns:
        updated kwargs
    """
    if isinstance(args, pd.DataFrame) or isinstance(args, np.ndarray):
        kwargs['df'] = args
    else:
        kwargs['df'] = None

    return kwargs


def df_filter(df: pd.DataFrame, filt_orig: str, drop_cols: bool = False,
              keep_filtered: bool = False) -> pd.DataFrame:
    """Filter the DataFrame.

    Due to limitations in pd.query, column names must not have spaces.  This function will temporarily replace
    spaces in the column names with underscores, but the supplied query string must contain column names
    without any spaces

    Args:
        df:  DataFrame to filter
        filt_orig:  query expression for filtering
        drop_cols (optional): drop filtered columns from results. Defaults to False.
        keep_filtered (optional): make a copy of the original data. Defaults to False.

    Returns:
        filtered DataFrame
    """
    def special_chars(text: str, skip: list = []) -> str:
        """Replace special characters in a text string.

        Args:
            text: input string
            skip: characters to skip

        Returns:
            formatted string
        """
        chars = {' ': '_', '.': 'dot', '[': '', ']': '', '(': '', ')': '',
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
            ands[ia] = '(%s%s)' % (key + '!=', key2.join(vals))
            continue
        elif 'in [' in aa:
            key, val = aa.split('in ')
            key = key.rstrip(' ')
            key = 'fCp%s' % special_chars(key)
            vals = val.replace('[', '').replace(']', '')
            vals = shlex.split(vals, ',', posix=False)
            vals = [f for f in vals if f != ',']  # remove commas
            for iv, vv in enumerate(vals):
                vals[iv] = vv.lstrip().rstrip()
            key2 = '|' + key + '=='
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
        if keep_filtered:
            df_orig = df.copy()

        df = df.query(filt)

        if keep_filtered:
            dropped = df_orig.copy().loc[~df_orig.index.isin(df.index)]
            cols_filt = [f for f in cols_new if f in filt]
            dropped.loc[:, cols_filt] = np.nan
            df = pd.concat([df, dropped])

        df.columns = cols_orig

        if drop_cols:
            cols_used = list(set(cols_used))
            for col in cols_used:
                del df[col]

        return df

    except:  # noqa
        print('Could not filter data!\n   Original filter string: %s\n   '
              'Modified filter string: %s' % (filt_orig, filt))

        df.columns = cols_orig

        return df


def df_from_array2d(arr: np.ndarray) -> pd.DataFrame:
    """Convert 2D numpy array to DataFrame.

    Args:
        arr: input numpy array

    Returns:
        DataFrame of arr or original DataFrame if arr already of this dtype
    """
    if isinstance(arr, np.ndarray) and len(arr.shape) == 2:
        return pd.DataFrame(arr)
    elif isinstance(arr, pd.DataFrame):
        return arr
    else:
        raise ValueError('input data must be a numpy array or a pandas DataFrame')


def df_from_array3d(arr: np.ndarray, labels: [list, np.ndarray] = [],
                    name: str = 'Item', verbose: bool = True) -> pd.DataFrame:
    """Convert a 3d numpy array to a DataFrame.

    Args:
        arr: input data to stack into a DataFrame
        label (optional): list of labels to designate each
            sub array in the 3d input array; added to DataFrame in column
            named ``label``. Defaults to [].
        name (optional): name of label column. Defaults to 'Item'.
        verbose (optional): toggle error print statements. Defaults to True.

    Returns:
        pd.DataFrame
    """
    shape = arr.shape
    if len(shape) != 3:
        raise ValueError('Input numpy array is not 3d')

    # Regroup into a stacked DataFrame
    m, n, r = shape
    arr = np.column_stack((np.repeat(np.arange(m), n),
                           arr.reshape(m * n, -1)))
    df = pd.DataFrame(arr)

    # Add a label column
    if len(labels) > 0:
        df[name] = np.repeat(labels, n)
    else:
        df[name] = np.floor(df.index / r).astype(int)

    return df


def df_int_cols(df: pd.DataFrame, non_int: bool = False) -> list:
    """Return column names that are integers (or not).

    Args:
        df: input DataFrame
        non_int (optional): if False, return column names that are integers;
            if True, return column names that are not integers

    Returns:
        list of column names
    """
    int_types = [np.int32, np.int64, int]

    if non_int:
        return [f for f in df.columns if type(f) not in int_types]
    else:
        return [f for f in df.columns if type(f) in int_types]


def df_int_cols_convert(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Convert integer column names to int type.
        - in case column names are ints but cast as string (after read_csv, etc).

    Args:
        df: input DataFrame
        force (optional): force the conversion even if int cols are initially
            found. Defaults to False.

    Returns:
        DataFrame with updated column names
    """
    int_cols = df_int_cols(df)

    if len(int_cols) == 0 or force:
        cols = [int(f) if f.isdigit() else f for f in df.columns]
        df.columns = cols

    return df


def df_summary(df: pd.DataFrame, columns: list = [], exclude: list = [],
               multiple: bool = False) -> pd.DataFrame:
    """Return a summary table of unique conditions in a DataFrame.

    If more than one value exists, replace with "Multiple" or a
    string list of the values.

    Args:
        df: input DataFrame
        columns (optional): list of column names to include; if empty,
            include all columns. Defaults to [].
        exclude(optional): list of column names to ignore. Defaults to [].
        multiple (optional): toggle between showing discrete values or
            "Multiple" when a column contains more than one value. Defaults
            to False.

    Returns:
        summary DataFrame
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


def df_unique(df: pd.DataFrame) -> dict:
    """Get column names with one unique value.

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


def get_current_values(df: pd.DataFrame, text: str, key: str = '@') -> str:
    """Parse a string looking for text enclosed by 'key' and replace with the current value from the DataFrame.
    This function is used by df_filter (but honestly can't remember why!)

    Args:
        df: DataFrame containing values we are looking for
        text: string to parse
        key (optional): matching chars that enclose the value to replace from df. Defaults to '@'

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


def get_decimals(value: [int, float], max_places: int = 4):
    """Get the number of decimal places of a float number excluding rounding errors.

    Args:
        value: value to check
        max_places (optional): maximum number of decimal places to check. Defaults to 4

    Returns:
        number of decimal places
    """
    last = np.nan
    value = abs(value)
    for i in range(0, max_places + 1):
        current = round(value, i)
        if current == last and current > 0:
            break
        else:
            last = current

    return i - 1


def get_text_dimensions(text: str, font: str, font_size: int, font_style: str, font_weight: str, **kwargs) -> tuple:
    """Use pillow to try and figure out actual dimensions of text.

    Args:
        text: the text from which to calculate size
        font: name of the font family or the font itself
        font_size: font size
        font_style: normal vs italic
        font_weight: normal vs bold
        kwargs: in place to

    Returns:
        size tuple
    """
    try:
        ImageFont
    except NameError:
        print('get_text_dimensions requires pillow which was not found.  Please '
              'run pip install pillow and try again.')
        return False

    fp = FontProperties()
    fp.set_family(font)
    fp.set_style(font_style)
    fp.set_weight(font_weight)
    fontfile = findfont(fp, fallback_to_default=True)
    size = ImageFont.truetype(fontfile, font_size).getsize(text)

    return size[0] * 1.125, size[1] * 1.125  # no idea why it is off


def kwget(dict1: dict, dict2: dict, vals: [str, list], default: [list, dict]):
    """Augmented kwargs.get function.

    Args:
        dict1: first dictionary to check for the value
        dict2: second dictionary to check for the value
        vals: value(s) to look for
        default: default value if not found in dict1 or dict2 keys

    Returns:
        value to use (dtype varies)
    """
    vals = validate_list(vals)

    for val in vals:
        if val in dict1.keys():
            return dict1[val]
    for val in vals:
        if val in dict2.keys():
            return dict2[val]

    return default


def img_compare(img1: str, img2: str, show: bool = False) -> bool:
    """Read two images and compare for difference by pixel. This is an optional
    utility used only for developer tests, so the function will only work if
    opencv is installed.

    Args:
        img1: path to file #1
        img2: path to file #2
        show: display the difference

    Returns:
        True/False of existence of differences
    """
    try:
        cv2
    except NameError:
        print('img_compare requires opencv which was not found.  Please '
              'run pip install opencv-python and try again.')
        return False

    # read images
    img1 = cv2.imread(str(img1))
    img2 = cv2.imread(str(img2))

    # compare
    if img1 is None:
        is_diff = True
        print('master image not available')

    elif img2 is None:
        is_diff = True
        print('test image not availble')

    else:
        if img1.shape != img2.shape:
            print(f'image sizes do not match [img1={img1.shape} | img2={img2.shape}')
            nrows = max(img1.shape[0], img2.shape[0])
            ncols = max(img1.shape[1], img2.shape[1])
            img1b = np.zeros((nrows, ncols, 3)).astype('uint8')
            img2b = np.zeros((nrows, ncols, 3)).astype('uint8')
            img1b[:img1.shape[0], :img1.shape[1]] = img1
            img2b[:img2.shape[0], :img2.shape[1]] = img2
            difference = cv2.subtract(img1b, img2b)
            is_diff = True
        else:
            difference = cv2.subtract(img1, img2)
            is_diff = np.any(np.where(difference > 1))  # 1 gives buffere for slight aliasing differences
            if not is_diff:
                # no negative pixels so now double check the other way
                difference = cv2.subtract(img2, img1)
                is_diff = np.any(np.where(difference > 1))
        if show and is_diff:
            cv2.imwrite('difference.png', 10 * difference)
            show_file('difference.png')

    return is_diff


def img_grayscale(img: np.ndarray) -> pd.DataFrame:
    """Convert an RGB image to grayscale.

    Args:
        img: 3D array of image data

    Returns:
        DataFrame with grayscale pixel values
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    return pd.DataFrame(0.2989 * r + 0.5870 * g + 0.1140 * b)


def nq(data, column: str = 'Value', **kwargs) -> pd.DataFrame:
    """Normal quantile calculation.

    Args:
        columns (optional): new name for the output column in the nq DataFrame

    Returns:
        DataFrame with normal quantile calculated values
    """
    # Defaults
    sig = kwargs.get('sigma', None)  # number of sigma to include for the analysis (assuming enough data exist)
    tail = abs(kwargs.get('tail', 3))  # start of the tail region
    step_tail = kwargs.get('step_tail', 0.2)  # step size of the tail region
    step_inner = kwargs.get('step_inner', 0.5)  # step size of the non-tail region

    # Flatten the DataFrame to an array
    data = data.values.flatten()

    # Get sigma
    if not sig:
        sig = sigma(data)
    else:
        sig = abs(sig)
    tail = min(tail, sig)  # tail must be <= full sigma range
    index = np.concatenate([np.arange(-sig, -tail, step_tail),
                            np.arange(-tail, tail, step_inner),
                            np.arange(tail, sig + 1e-9, step_tail)])
    # Get the sigma value
    values = np.zeros(len(index))
    for ii, idx in enumerate(index):
        values[ii] = np.percentile(data, ss.norm.cdf(idx) * 100)

    return pd.DataFrame({'Sigma': index, column: values})


def pie_wedge_labels(x: np.ndarray, y: np.ndarray, start_angle: float) -> [float, float]:
    """Identify the wedge labels that extend furthest on the horizontal axis.

    Args:
        x: pie wedge labels
        y: pie wedge values
        start_angle: the starting angle (0 = right | 90 = top, etc)

    Returns:
        indices of the longest labels on the horizontal axis
    """
    yper = y / sum(y)
    yperrad = yper * 2 * np.pi
    csr = yperrad.cumsum()
    start_angle = start_angle * np.pi / 180
    left = np.where((csr - (np.pi - start_angle)) > 0)[0][0]
    right = np.where((csr - (2 * np.pi - start_angle)) > 0)
    if len(right[0]) > 0:
        right = right[0][0]
    else:
        right = (csr - (2 * np.pi - start_angle)).argmax()

    return left, right


def plot_num(ir: int, ic: int, ncol: int) -> int:
    """Get the subplot index number based on grid location.

    Args:
        ir: axes row index
        ic: axes column index
        ncol: number of columns in the row x column grid

    Return:
        index number of the subplot
    """
    return (ic + ir * ncol + 1)


def rgb2bayer(img: np.ndarray, cfa: str = 'rggb',
              bit_depth: int = np.uint8) -> pd.DataFrame:
    """Approximate a Bayer raw image using an RGB image.

    Args:
        img: 3D numpy array of RGB pixel values
        cfa (optional):  Bayer pattern. Defaults to 'rggb'.
        bit_depth (optional): bit depth of the input data. Defaults to 8-bit

    Returns:
        DataFrame containing Bayer pixel values
    """
    raw = np.zeros((img.shape[0], img.shape[1]), dtype=bit_depth)
    cp = list(cfa)
    channel = {'r': 0, 'g': 1, 'b': 2}

    for cpp in list(set(cp)):
        idx = [i for i, f in enumerate(cp) if f == cpp]
        for ii in idx:
            row = 0 if ii < 2 else 1
            col = 0 if ii % 2 == 0 else 1
            raw[row::2, col::2] = img[row::2, col::2, channel[cpp]]

    return pd.DataFrame(raw)


def rectangle_overlap(r1: [list, tuple], r2: [list, tuple]) -> bool:
    """Determine if the bounds of two rectangles overlap.

    Args:
        r1: width, height, center point (x, y) of left rectangle
        r2: width, height, center point (x, y) of right rectangle

    Returns:
        True if overlap | False if not
    """
    # Get bounds
    b1 = [r1[2][0] - r1[0] / 2, r1[2][0] + r1[0] / 2,
          r1[2][1] + r1[1] / 2, r1[2][1] - r1[1] / 2]
    b2 = [r2[2][0] - r2[0] / 2, r2[2][0] + r2[0] / 2,
          r2[2][1] + r2[1] / 2, r2[2][1] - r2[1] / 2]

    if b1[0] < b2[1] and b1[1] > b2[0] \
            and b1[2] > b2[3] and b1[3] < b2[2]:
        return True
    else:
        return False


def reload_defaults(theme: [str, None] = None, verbose: bool = False):
    """Reload the fcp params.

    Args:
        theme (optional): name of the theme file to load. Defaults to None.
        verbose (bool): optional print more info flag
    """
    theme_dir = pathlib.Path(__file__).parent / 'themes'
    reset_path = False
    err_msg = 'Requested theme not found; using default'
    user_dir = pathlib.Path.home()
    success = True

    if theme is not None and os.path.exists(theme):
        # full filename case
        case = 1
        theme = str(theme)
        theme_dir = os.sep.join(theme.split(os.sep)[0:-1])
        theme = theme.split(os.sep)[-1]
        sys.path = [str(theme_dir)] + sys.path
        reset_path = True
        try:
            defaults = importlib.import_module(theme.replace('.py', ''), theme_dir)
            importlib.reload(defaults)
        except (TypeError, NameError, ImportError):
            print(err_msg)
            import defaults
            importlib.reload(defaults)
            success = False
    elif theme is not None and (theme in os.listdir(theme_dir) or theme + '.py' in os.listdir(theme_dir)):
        case = 2
        sys.path = [str(theme_dir)] + sys.path
        reset_path = True
        try:
            defaults = importlib.import_module(theme.replace('.py', ''), theme_dir)
            importlib.reload(defaults)
        except (TypeError, NameError, ImportError):
            print(err_msg)
            import defaults
            importlib.reload(defaults)
            success = False
    elif (user_dir / '.fivecentplots' / 'defaults.py').exists():
        # use default theme
        case = 3
        sys.path = [str(user_dir / '.fivecentplots')] + sys.path
        defaults = importlib.import_module('defaults', str(user_dir / '.fivecentplots'))
        importlib.reload(defaults)
    else:
        case = 4
        sys.path = [str(theme_dir)] + sys.path
        import gray as defaults
        importlib.reload(defaults)

    fcp_params = defaults.fcp_params if hasattr(defaults, 'fcp_params') else {}
    colors = defaults.colors if hasattr(defaults, 'colors') else []
    markers = defaults.markers if hasattr(defaults, 'markers') else []
    rcParams = defaults.rcParams if hasattr(defaults, 'rcParams') else {}

    if verbose:
        print(f'sys.path[0]: {sys.path[0]}')
        print(f'theme: {theme}\ntheme exists: {os.path.exists(str(theme))}')
        print(f'theme_dir: {theme_dir}\nuser_dir: {user_dir}')
        print(f'case: {case}\nsuccess: {success}')

    if reset_path:
        sys.path = sys.path[1:]

    return fcp_params, colors, markers, rcParams  # could convert to dict in future


def see(obj) -> pd.DataFrame:
    """Prints a readable list of class attributes.

    Args:
        some object (Data, Layout, etc)

    Returns:
        DataFrame of all the attributes of the object
    """
    df = pd.DataFrame({'Attribute': list(obj.__dict__.copy().keys()),
                       'Name': [str(f) for f in obj.__dict__.copy().values()]})
    df = df.sort_values(by='Attribute').reset_index(drop=True)

    return df


def set_save_filename(df: pd.DataFrame, ifig: int, fig_item: [None, str],
                      fig_cols: [None, str], layout: 'engines.Layout',  # noqa
                      kwargs: dict) -> str:
    """Build a filename based on the plot setup parameters.

    Args:
        df: input data used for the plot
        ifig: current figure number
        fig_item: unique figure item
        fig_cols: names of columns used to generate fi
        layout: plot layout object
        kwargs: user-defined keyword args

    Returns:
        str filename
    """
    # Use provided filename
    if 'filename' in kwargs.keys() and isinstance(kwargs['filename'], pathlib.PosixPath):
        kwargs['filename'] = str(kwargs['filename'])
    if 'filename' in kwargs.keys() and isinstance(kwargs['filename'], str):
        filename = kwargs['filename']
        ext = os.path.splitext(filename)[-1]
        if ext == '':
            ext = f'.{kwargs.get("save_ext")}'
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
        if layout.label_col.text is None:
            col = ''

    filename = '%s%s%s%s%s%s%s%s' % (z, y, x, row, col, wrap, groups, fig)

    # Cleanup forbidden symbols
    bad_char = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for bad in bad_char:
        filename = filename.replace(bad, '_')

    # Make sure extension has a '.'
    if kwargs.get('save_ext')[0] != '.':
        kwargs['save_ext'] = '.' + kwargs['save_ext']

    return filename + kwargs.get('save_ext')


def sigma(x: [pd.Series, np.ndarray]) -> int:
    """Calculate the sigma range for a data set.

    Args:
        x: data set

    Returns:
        int value to use for +/- max sigma
    """
    return np.round(np.trunc(10 * abs(ss.norm.ppf(1 / len(x)))) / 10)


def show_file(filename: str):
    """Platform independent show saved plot func.

    Args:
        filename: path to image
    """
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, str(filename)])


def split_color_planes(img: pd.DataFrame, cfa: str = 'rggb',
                       as_dict: bool = False) -> pd.DataFrame:
    """Split image data into respective color planes.

    Args:
        img: image data
        cfa (optional): four-character cfa pattern. Defaults to 'rggb'.
        as_dict (optional): return each plane DataFrame as a value in a dict.
            Defaults to False.

    Returns:
        updated DataFrame
    """
    # Break the cfa code to list
    cfa = cfa.lower()  # force lowercase for now
    cp = list(cfa)
    if len(cp) != 4:
        raise CFAError('Only CFAs with a 2x2 grid of colors is supported')

    # Check for a repeated cfa
    counts = {i: cp.count(i) for i in cp}
    doubles = [k for k, v in counts.items() if v == 2]
    triples = [k for k, v in counts.items() if v == 3]

    if len(doubles) > 0:
        # cfa has one color plane repeated 2x (rggb, rccb)
        idx = [i for i, f in enumerate(cp) if f == doubles[0]]
        for ii in idx:
            if ii % 2 == 0:
                cp[ii] = f'{cp[ii]}{cp[ii + 1]}'
            else:
                cp[ii] = f'{cp[ii]}{cp[ii - 1]}'
    elif len(triples) > 0:
        # cfa has one color plane repeated 3x (rccc)
        idx = [i for i, f in enumerate(cp) if f == triples[0]]
        for ii in idx:
            cp[ii] = f'{cp[ii]}{ii}'

    # Separate planes
    if as_dict:
        img2 = {}
    else:
        img2 = pd.DataFrame()

    for ic, cc in enumerate(cp):
        temp = img.loc[ic // 2::2, (ic % 2)::2]
        if as_dict:
            img2[cc] = temp
        else:
            temp['Plane'] = cc
            img2 = pd.concat([img2, temp])

    if as_dict:
        return img2

    cols = df_int_cols(img2) + ['Plane']
    return img2[cols]


def test_checker(module) -> list:
    """For test files with benchmarking, find any missing "plt_" functions without a matching "test_" func

    Args:
        module: imported module

    Returns:
        list of names of plt_ funcs without a test_ func
    """

    funcs = inspect.getmembers(module, inspect.isfunction)
    tests = [f[0].replace('test_', '') for f in funcs if 'test_' in f[0]]
    plts = [f[0].replace('plt_', '') for f in funcs if 'plt_' in f[0]]

    return [f for f in plts if f not in tests]


def validate_list(items: [str, int, float, list]) -> list:
    """Make sure a list variable is actually a list and not a single item.
    Excludes None.

    Args:
        items: value(s) that will be returned as a list

    Return:
        items as a list unless items is None
    """
    if items is None:
        return None
    if isinstance(items, tuple) or isinstance(items, np.ndarray):
        return list(items)
    elif not isinstance(items, list):
        return [items]
    else:
        return items
