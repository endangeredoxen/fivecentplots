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
import ast
import operator
from matplotlib.font_manager import FontProperties, findfont
import matplotlib.dates as mdates
from pathlib import Path
from typing import Any, Union, Tuple, Dict, List
from . import data
import numpy.typing as npt
from PIL import ImageFont, Image, ImageDraw
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

# Read the package version file
with open(Path(__file__).parent / 'version.txt', 'r') as fid:
    __version__ = fid.readlines()[0].replace('\n', '')


# Convenience kwargs
HIST = {'ax_scale': 'logy', 'markers': False, 'line_width': 2, 'preset': 'HIST'}
NQ = {'markers': False, 'line_width': 2, 'preset': 'NQ'}


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
            raise ValueError('RepeatedList must contain an actual list with at least one element')

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

    def max(self):
        """Return the maximum value of the RepeatedList."""
        # TODO:  address NaN?
        return max(self.values)


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
            print(label + str(delta) + f' [{self.units}]')

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


class CustomWarning(Warning):
    pass


def arithmetic_eval(s):
    s = s.replace(' ', '')
    s = s.replace('--', '+')
    s = s.replace('++', '+')
    node = ast.parse(s, mode='eval')

    def _eval(node):
        binOps = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod
        }

        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))
        else:
            raise Exception('Unsupported type {}'.format(node))

    return _eval(node.body)


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


def close_preview_windows_macos(filenames: Union[str, list]):
    """
    MACOS only, close open Preview windows for image files

    Args:
        filenames: name of single file or list of filenames that are open

    """
    if sys.platform != 'darwin':
        return

    try:
        import osascript  # noqa
    except ModuleNotFoundError:
        return

    filenames = validate_list(filenames)

    for ff in filenames:
        script = """
                 tell application "Preview"
                     activate
                     repeat with w in windows
                         if name of w contains "%s" then
                             close w
                             exit repeat
                         end if
                     end repeat
                 end tell
                 """ % ff
        osascript.run(script)


def date_to_pixels(date: Union[datetime.datetime, float], vmin: float, vmax: float,
                   pxmin: float, pxmax: float) -> float:
    """Convert a matplotlib date to pixels.

    Args:
        date: datetime or matplotlib date float to convert to pixels
        vmin: minimum date or matplotlib date float
        vmax: maximum date or matplotlib date float
        pxmin: minimum pixel value of the axes range
        pxmax: maximum pixel value of the axes range

    Returns:
        date value in pixels
    """
    if isinstance(date, datetime.datetime):
        date = mdates.date2num(date)

    return (date - vmin) * (pxmax - pxmin) / (vmax - vmin) + pxmin


def date_vals(date_num: Union[datetime.datetime, float]) -> Tuple[int, int, int]:
    """Convert a date number to a datetime object.

    Args:
        date_num: matplotlib date number

    Returns:
        day, month, year as integers
    """
    if isinstance(date_num, datetime.datetime):
        return date_num.day, date_num.month, date_num.year
    else:
        return mdates.num2date(date_num).day, mdates.num2date(date_num).month, mdates.num2date(date_num).year


def dfkwarg(args: tuple, kwargs: dict, plotter: object) -> dict:
    """Add the DataFrame to kwargs.

    Args:
        args:  *args sent to plot
        kwargs:  **kwargs sent to plot

    Returns:
        updated kwargs
    """
    if 'df' in kwargs.keys() and isinstance(kwargs['df'], pd.DataFrame):
        return kwargs
    elif isinstance(args, pd.DataFrame):
        kwargs['df'] = args
        return kwargs
    elif isinstance(args, np.ndarray) and plotter.name in ['hist', 'imshow', 'nq']:
        # Certain plots can accept a numpy array of 2D - 4D
        if len(args.shape) < 2 or len(args.shape) > 4:
            raise data.DataError('Data source has valid data type but invalid array shape!'
                                 '\n\nPlease consult the docs for '
                                 f'"{plotter.name}" plot '
                                 'which describe the requirement for 2D, 3D, or 4D image array shapes'
                                 f'\n{doc_url(plotter.url)}')
        kwargs['df'] = args
        return kwargs
    elif (isinstance(args, np.ndarray) and (len(args.shape) == 2 or len(args.shape) == 3)) or \
            (isinstance(args, list) and (len(args) == 2 or len(args) == 3)) \
            and len(args[0]) == len(args[1]) and len(args[0]) == len(args[-1]) and plotter.name in ['xy']:
        if len(args) == 3:
            kwargs['df'] = pd.DataFrame({'x': args[0], 'y': args[1], 'groups': args[2]})
        else:
            kwargs['df'] = pd.DataFrame({'x': args[0], 'y': args[1]})
        kwargs['x'] = 'x'
        kwargs['y'] = 'y'
        return kwargs
    else:
        raise data.DataError('Data source has invalid data type!  '
                             '\n\nPlease consult the docs for more information on which data types are allowed for a '
                             f'"{plotter.name}" plot.\n{doc_url(plotter.url)}')


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
    cols_new = [f'fCp{f}' for f in cols_orig.copy()]
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
            key = f'fCp{special_chars(key)}'
            vals = val.replace('[', '').replace(']', '').split(',')
            for iv, vv in enumerate(vals):
                vals[iv] = vv.lstrip().rstrip()
            key2 = '&' + key + '!='
            ands[ia] = f'({key} != {key2.join(vals)})'
            continue
        elif 'in [' in aa:
            key, val = aa.split('in ')
            key = key.rstrip(' ')
            key = f'fCp{special_chars(key)}'
            vals = val.replace('[', '').replace(']', '')
            vals = shlex.split(vals, '(', posix=False)  # to ignore within parentheses
            vals = [f.rstrip(',').lstrip().rstrip() for f in vals]  # remove trailing commas or ws
            vals = [f for f in vals if f != '']
            key2 = '|' + key + '=='
            ands[ia] = f'({key} == {key2.join(vals)})'
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
                    vals[1] = f'fCp{special_chars(vals[1])}'
                vals[0] = f'fCp{special_chars(vals[0])}'
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
        print(f'Could not filter data!\n   Original filter string: {filt_orig}\n   '
              f'Modified filter string: {filt}')

        df.columns = cols_orig

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


def doc_url(paths=[]) -> str:
    """Return the documentation url.

    Args:
        paths: add a subpath to the url

    Returns:
        string html path
    """

    url = f'https://endangeredoxen.github.io/fivecentplots/{__version__}'
    if not isinstance(paths, list):
        paths = [paths]

    for path in paths:
        url += f'/{path}'

    return url


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


def get_decimals(value: [int, float], max_places: int = 4, exponential: bool = False) -> int:
    """Get the number of decimal places of a float number excluding rounding errors.

    Args:
        value: value to check
        max_places (optional): maximum number of decimal places to check. Defaults to 4 which is equal to 3 decimals
        exponential (optional): if True, assume exponential notation. Defaults to False.
    Returns:
        number of decimal places
    """
    if value == 0:
        return 0

    # Ignore negative values and track the original value
    value = abs(value)
    original = value

    # Store the previous loop value
    previous = np.nan

    # Scale up small values < 1 to the 1E-1 range to avoid rounding errors
    if value < 0.1:
        power = -int(np.log10(value))
        value *= 10 ** power

    # Round with an increasing number of decimal places until the value is stable
    for i in range(0, max_places + 1):
        current = round(value, i)
        if (current == previous and current > 0):
            break
        else:
            previous = current

    # Check for repeating decimals
    if '.' in str(current):
        repeats, repeat_pattern = get_repeating_decimal_end(current)
        if repeats > 0:
            i = int(str(current).split('.')[1].index(repeat_pattern)) + 2
        elif i > 0:
            i -= 1

    # Return the number of decimal places
    if original < 0.1 and exponential is not True:
        # For non-exponential values, add back the leading decimal places
        return min(i + power, max_places)
    else:
        return min(i, max_places)


def get_nested_files(path: Union[Path, str], pattern: Union[str, None] = None, exclude: list = []) -> list:
    """Get a list of files within a nested directory.

    Args:
        path: top-level directory
        pattern: substring to filter the list of files; defaults to None = no filtering

    Returns:
        list of file paths
    """
    if not isinstance(path, Path):
        path = Path(path)

    files = []
    for entry in path.iterdir():
        if entry.is_file():
            if pattern and pattern in str(entry):
                skip = False
                for ex in exclude:
                    if ex in str(entry):
                        skip = True
                if not skip:
                    files += [entry]
            elif not pattern:
                files += [entry]
        elif entry.is_dir():
            files += get_nested_files(entry, pattern, exclude)

    return files


def get_repeating_decimal_end(value: float) -> int:
    """
    Counts the number of repeated digits at the end of a decimal string,
    accounting for potential rounding up of the last digit.

    Args:
        number_str: The string representation of the decimal number.

    Returns:
        The number of repeated digits at the end, or 0 if no repetition is found.
    """
    number_str = str(value)

    if '.' not in number_str:
        return 0, ''

    integer_part, fractional_part = number_str.split('.')
    n = len(fractional_part)
    if n < 3:
        return 0, ''

    last_digit = fractional_part[-1]
    pattern = fractional_part[-1]
    repeated_count = 0

    # Check for exact repetition
    for i in range(n - 2, -1, -1):
        if fractional_part[i] == last_digit:
            repeated_count += 1
            pattern += fractional_part[i]
        else:
            break

    if repeated_count > 0:
        return repeated_count + 1, pattern   # Include the last digit itself

    # Check for potential rounding up (last digit is one greater than the preceding)
    pattern = []
    if last_digit.isdigit():
        rounded_down_digit = str(int(last_digit) - 1)
        if rounded_down_digit >= '0':
            rounded_up_count = 0
            for i in range(n - 2, -1, -1):
                if fractional_part[i] == rounded_down_digit:
                    rounded_up_count += 1
                    if len(pattern) == 0:
                        pattern = [last_digit]
                    else:
                        pattern = [fractional_part[i]] + pattern
                else:
                    break
            if rounded_up_count > 0:
                pattern = ''.join([fractional_part[i]] + pattern)
                if pattern not in number_str:
                    return 0, ''
                else:
                    return rounded_up_count + 1, pattern

    return 0, ''


def get_text_dimensions(text: str, font: str, font_size: int, font_style: str, font_weight: str, rotation: float = 0,
                        dpi: int = 100, ignore_html: bool = True, show=False, **kwargs) -> tuple:
    """Use pillow to try and figure out actual dimensions of text.

    Args:
        text: the text from which to calculate size
        font: name of the font family or the font itself
        font_size: font size
        font_style: normal vs italic
        font_weight: normal vs bold
        rotation: text rotation
        dpi: dots per inch, mpl uses px for font and pillow uses pt so need to convert
        ignore_html: strip html tags

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
    fp.set_name(font)
    fp.set_style(font_style)
    fp.set_weight(font_weight)
    fontfile = findfont(fp, fallback_to_default=True)
    font = ImageFont.truetype(fontfile, int(np.ceil(font_size * dpi / 72)))

    if ignore_html:
        text = strip_html(text)

    size = font.getbbox(text)[2:]

    if rotation != 0:
        w = size[0] * np.abs(np.cos(rotation * np.pi / 180)) \
            + size[1] * np.abs(np.sin(rotation * np.pi / 180))
        h = size[1] * np.abs(np.cos(rotation * np.pi / 180)) \
            + size[0] * np.abs(np.sin(rotation * np.pi / 180))
    else:
        w, h = size

    if show:
        image = Image.new('RGB', (w, h), color='white')
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, font=font, fill='black')
        image.save("utl_font_size.png")
        show_file("utl_font_size.png")

    return w, h


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


def img_array_from_df(df, shape, dtype=int):
    cols = [f for f in df.columns if f in ['Value', 'R', 'G', 'B', 'A']]
    return df[cols].to_numpy().reshape(shape).astype(int)


def img_checker(rows, cols, num_x, num_y, high, low):
    """Make a checker pattern array

    Args:
        rows: number rows in single checker square
        cols: number cols in single checker square
        num_x: number of horizontal squares
        num_y: number of vertical squares
        high: value of the higher checker squares
        low: value of the lower checker squares

    Returns:
        2D numpy array
    """
    grid = [[high, low] * num_x, [low, high] * num_x] * num_y
    square = np.ones([rows, cols])
    total_rows = int(rows * num_y)
    total_cols = int(cols * num_x)

    return np.kron(grid, square)[:total_rows, :total_cols]


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
            # macos only convenience - close last opened files
            close_preview_windows_macos('difference.png')

            cv2.imwrite('difference.png', 10 * difference)
            show_file('difference.png')

    return is_diff


def img_data_format(kwargs: dict) -> Tuple[pd.DataFrame, dict]:
    """Format image data into the required format for fivecentplots

    Args:
        kwargs: user input

    Returns:
        grouping info DataFrame
        dict of numpy array image data
    """
    shape_cols = ['rows', 'cols', 'channels']
    df = kwargs['df']
    imgs = kwargs.get('imgs', {})

    # Find image data source (could be in df as 2D DataFrame or in images as numpy array)
    if len(imgs) == 0 and isinstance(df, pd.DataFrame):
        # Case 1: image data is in df
        grouping_cols = [f for f in df_int_cols(df, True) if f not in shape_cols]
        int_cols = df_int_cols(df)

        if len(int_cols) == 0:
            raise ValueError('image DataFrame does not contain 2D image data')

        if len(grouping_cols) == 0:
            imgs[0] = df[int_cols].to_numpy()
            shape = imgs[0].shape
            df_groups = pd.DataFrame({'rows': shape[0],
                                      'cols': shape[1],
                                      'channels': shape[2] if len(shape) > 2 else 1}, index=[0])
        else:
            df_groups = pd.DataFrame()
            for ii, (nn, gg) in enumerate(df.groupby(grouping_cols, dropna=False)):
                nn = validate_list(nn)
                imgs[ii] = gg[int_cols].to_numpy()
                shape = imgs[0].shape
                temp = pd.DataFrame({'rows': shape[0], 'cols': shape[1], 'channels': shape[2] if len(shape) > 2 else 1},
                                    index=[ii])
                for igroup, group in enumerate(grouping_cols):
                    temp[group] = nn[igroup]
                df_groups = pd.concat([df_groups, temp])
    elif len(imgs) == 0 and isinstance(df, np.ndarray):
        imgs[0] = df
        shape = imgs[0].shape
        df_groups = pd.DataFrame({'rows': shape[0],
                                  'cols': shape[1],
                                  'channels': shape[2] if len(shape) > 2 else 1}, index=[0])

    elif isinstance(imgs, dict) and isinstance(df, pd.DataFrame):
        # Case 2: image data already in correct format
        if df.index.has_duplicates:
            raise data.DataError('image dataframe has duplicate index values so it cannot be matched with image dict')

        df_groups = df.copy()
        df_groups['rows'] = -1
        df_groups['cols'] = -1
        df_groups['channels'] = -1
        for k, v in imgs.items():
            if k not in df.index:
                continue  # raise KeyError(f'image array item "{k}" not found in grouping DataFrame')
            if not isinstance(v, np.ndarray):
                raise TypeError(f'image array item "{k}" must be a numpy array')
            shape = v.shape
            df_groups.loc[k, 'rows'] = shape[0]
            df_groups.loc[k, 'cols'] = shape[1]
            df_groups.loc[k, 'channels'] = shape[2] if len(shape) > 2 else 1

    else:
        raise TypeError('image data must be either a numpy array or a dict of numpy arrays with a DataFrame '
                        'containing grouping information')

    return df_groups, imgs


def img_df_transform(data: Union[pd.DataFrame, np.ndarray, Tuple[pd.DataFrame, dict]]) -> Tuple[pd.DataFrame, dict]:
    """
    Transform a numpy array or a DataFrame into the image DataFrame format used in fcp.  Image arrays can be very
    large and memory-intensive (especially with string grouping labels) so this format is used to improve speed.
    Support is provided for both 2D (Bayer-like) and 3/4D (RGB[A]) arrays.

    Format:
        1) df_groups = smaller DataFrame containing all the non-integer grouping columns without repeats and an index
           value that corresponds to the key of item 2 below
        2) dict of DataFrames with the actual image data

    Args:
        data: input data to convert into a DataFrame
        label (optional): list of labels to designate each
            sub array in the 3d input array; added to DataFrame in column
            named ``label``. Defaults to [].
        name (optional): name of label column. Defaults to 'Item'.
        verbose (optional): toggle error print statements. Defaults to True.

    Returns:
        pd.DataFrame, dict
    """
    # Verify input type
    if isinstance(data, tuple) or isinstance(data, list) \
            and len(data) == 2 and isinstance(data[0], pd.DataFrame) and isinstance(data[1], dict):
        # Check grouping DataFrame indexes match image data dict keys
        if not list(data[0].index) == list(data[1].keys()):
            raise ValueError('Cannot associate image grouping table with image data dict because indexes and dict keys '
                             'do not match!')
        return data
    elif isinstance(data, pd.DataFrame) or isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError('Cannot create fcp image DataFrame.  Input data must be a numpy array, pandas DataFrame, '
                        'or tuple of a grouping DataFrame and a dict of fcp-formatted image DataFrames')

    # Output containers
    imgs = {}

    # Case 1: input DataFrame already in semi-correct format
    if isinstance(data, pd.DataFrame) and 'Row' in data.columns and 'Column' in data.columns:
        group_cols = [f for f in data.columns if f not in ['Row', 'Column', 'Value', 'R', 'G', 'B', 'A']]
        channels = max(1, len([f for f in data.columns if f in ['R', 'G', 'B', 'A']]))

        if len(group_cols) == 0:
            imgs[0] = data

            # Make the grouping DataFrame
            df_groups = pd.DataFrame({'rows': len(data.Row.unique()),
                                      'cols': len(data.Column.unique()),
                                      'channels': channels},
                                     index=[0])

        else:
            temp_groups = []
            for idx, (nn, gg) in enumerate(data.groupby(group_cols)):
                imgs[idx] = gg[[f for f in gg.columns if f not in group_cols]]
                ss = (len(gg.Row.unique()), len(gg.Column.unique()))
                nn = validate_list(nn)

                # Make the grouping DataFrame
                temp = pd.DataFrame({'rows': ss[0], 'cols': ss[1], 'channels': channels}, index=[idx])
                for igroup, group in enumerate(group_cols):
                    temp[group] = nn[igroup]
                temp_groups += [temp]

            df_groups = pd.concat(temp_groups)

    # Case 2: input DataFrame in 2D format
    elif isinstance(data, pd.DataFrame) and len(data.shape) == 2:
        # Separate the input columns by integer or string (i.e., group labels) type
        group_cols = df_int_cols(data, True)
        int_cols = df_int_cols(data)

        if len(int_cols) == 0:
            raise TypeError('image DataFrame has incorrect format; expecting integer-labeled columns')

        # Case 2a: no groups
        if len(group_cols) == 0:
            # Convert subset to numpy array and reshape
            data = data[int_cols].to_numpy()
            ss = data.shape

            # Make the img DataFrame
            imgs[0] = pd.DataFrame(data.reshape(ss[0] * ss[1], -1), columns=['Value'])
            imgs[0]['Row'] = imgs[0].index // ss[1]
            imgs[0]['Column'] = np.tile(np.arange(ss[1]), ss[0])

            # Make the grouping DataFrame
            df_groups = pd.DataFrame({'rows': ss[0], 'cols': ss[1], 'channels': 1}, index=[0])

        # Case 2b: DataFrame has grouping columns
        else:
            temp_groups = []
            for idx, (nn, gg) in enumerate(data.groupby(group_cols, dropna=False)):
                gg = gg[int_cols].to_numpy()
                nn = validate_list(nn)
                ss = gg.shape

                # Make the img DataFrame
                imgs[idx] = pd.DataFrame(gg.reshape(ss[0] * ss[1], -1), columns=['Value'])
                imgs[idx]['Row'] = imgs[idx].index // ss[1]
                imgs[idx]['Column'] = np.tile(np.arange(ss[1]), ss[0])

                # Make the grouping DataFrame
                temp = pd.DataFrame({'rows': ss[0], 'cols': ss[1], 'channels': 1}, index=[idx])
                for igroup, group in enumerate(group_cols):
                    temp[group] = nn[igroup]
                temp_groups += [temp]

            df_groups = pd.concat(temp_groups)

    # Case 3: input numpy array (2D, 3D, or 4D)
    else:
        # Check valid dtype
        try:
            data.astype(float)
        except ValueError:
            raise ValueError('Image array can only contain numeric data')

        # Input numpy array
        ss = data.shape

        # Get the correct column names
        if len(ss) == 2:
            cols = ['Value']
        else:
            cols = ['R', 'G', 'B', 'A'][0:ss[-1]]

        # Make the img DataFrame
        imgs[0] = pd.DataFrame(data.reshape(ss[0] * ss[1], -1), columns=cols)
        imgs[0]['Row'] = imgs[0].index // ss[1]
        imgs[0]['Column'] = np.tile(np.arange(ss[1]), ss[0])

        # Make the grouping DataFrame
        df_groups = pd.DataFrame({'rows': ss[0], 'cols': ss[1], 'channels': len(cols)}, index=[0])

    return df_groups, imgs


def img_df_transform_from_dict(groups: pd.DataFrame, imgs: Dict[int, npt.NDArray]) -> Tuple[pd.DataFrame, dict]:
    groups = groups.copy()

    # Update DataFrame
    cols = ['rows', 'cols', 'channels']
    for cc in cols:
        if cc not in groups.columns:
            groups[cc] = -1

    # Format the numpy arrays
    imgs_new = {}
    for k, v in imgs.items():
        if k not in groups.index:
            continue
        if not isinstance(v, np.ndarray):
            raise TypeError('Image dictionary must contain numpy arrays')
        if len(v.shape) == 2:
            rows, cols, channels = v.shape[0], v.shape[1], 1
        else:
            rows, cols, channels = v.shape

        # Transform the image to an fcp DataFrame
        imgs_new[k] = img_df_transform(v)[1][0]

        # Update the groups (I DON"T THINK THIS WORKS)
        if groups.loc[k, cc] == -1:
            groups.loc[k, 'rows'] = rows
        if groups.loc[k, cc] == -1:
            groups.loc[k, cc] == -1
        if groups.loc[k, cc] == -1:
            groups.loc[k, 'channels'] = channels

    return groups, imgs_new


def img_grayscale(img: np.ndarray, bit_depth: int = 16, as_df: bool = False) -> Union[pd.DataFrame, npt.NDArray]:
    """Convert an RGB image to grayscale and convert to a 2D DataFrame

    Args:
        img: 3D array of RGB image data
        bit_depth: scale to this bit depth
        as_df: if True, return image as pd.DataFrame

    Returns:
        DataFrame with grayscale pixel values
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    raw = (0.2989 * r + 0.5870 * g + 0.1140 * b) * (2**(bit_depth) - 1) / (2**8 - 1)
    if bit_depth > 16:
        raw = raw.astype(np.uint32)
    else:
        raw = raw.astype(np.uint16)

    if as_df:
        return pd.DataFrame(raw)

    else:
        return raw


def img_grayscale_deprecated(img: np.ndarray, as_array: bool = False) -> Union[pd.DataFrame, npt.NDArray]:
    """Old function for converting an RGB image to grayscale and convert to a 2D DataFrame; keeping for tests

    Args:
        img: 3D array of image data

    Returns:
        DataFrame with grayscale pixel values
    """
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if not as_array:
        return pd.DataFrame(0.2989 * r + 0.5870 * g + 0.1140 * b)

    else:
        return 0.2989 * r + 0.5870 * g + 0.1140 * b


def img_rgb_to_df(data):
    return img_df_transform(data)[1][0]


def nq(data: Union[npt.NDArray, pd.DataFrame], column: str = 'Value', **kwargs) -> pd.DataFrame:
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
    percentiles_on = kwargs.get('percentiles', False)  # display y-axis as percentiles instead of sigma

    # Flatten the DataFrame/2D array to a 1D array
    if isinstance(data, pd.DataFrame) and column not in data.columns:
        data = data.values.flatten()
    elif isinstance(data, pd.DataFrame):
        data = data[column]
    elif isinstance(data, np.ndarray) and len(data.shape) > 1:
        data = data.flatten()

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
    percentiles = ss.norm.cdf(index) * 100
    values = np.percentile(data, percentiles)

    if percentiles_on:
        return pd.DataFrame({'Percent': percentiles, column: values})
    else:
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


def qt(last: Union[None, datetime.datetime] = None, label='timer') -> datetime.datetime:
    """
    Quick timer for speed debugging; time print out is in ms

    Args:
        last: previous timer
        label: optional string label

    Return:
        new timer start
    """

    if last is None:
        return datetime.datetime.now()

    else:
        print(f'{label}: {(datetime.datetime.now() - last).total_seconds() * 1000} [ms]')
        return datetime.datetime.now()


def rgb2bayer(img: np.ndarray, cfa: str = 'rggb', bit_depth: int = np.uint8) -> pd.DataFrame:
    """Crudely approximate a Bayer raw image using an RGB image.

    Args:
        img: 3D numpy array of RGB pixel values
        cfa (optional):  Bayer pattern. Defaults to 'rggb'.
        bit_depth (optional): bit depth of the input data. Defaults to 8-bit

    Returns:
        np.array containing Bayer-like RAW pixel values
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

    return raw


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


def remove_duplicates_list_preserve_order(seq: List[Any]) -> List[Any]:
    """
    Remove duplicates from a list while preserving original order

    Args:
        seq: original list

    Returns:
        updated list
    """
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


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


def set_save_filename(df: pd.DataFrame, ifig: int, fig_item: [None, str], fig_cols: [None, str],
                      layout: 'engines.Layout', kwargs: dict) -> str:  # noqa
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
    if kwargs.get('filename') and isinstance(kwargs['filename'], pathlib.PosixPath):
        kwargs['filename'] = str(kwargs['filename'])
    if kwargs.get('filename') and isinstance(kwargs['filename'], str):
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
                fig += f' where {col}={fig_items[icol]}'
        return filename + fig + ext

    # Build a filename
    if 'z' in kwargs.keys():
        z = strip_html(layout.label_z.text) + ' vs '
    else:
        z = ''
    y = ' and '.join(validate_list(strip_html(layout.label_y.text)))
    if layout.axes.twin_x:
        y += ' + ' + strip_html(layout.label_y2.text)
    if 'x' in kwargs.keys():
        y += ' vs '
        x = ' and '.join(validate_list(strip_html(layout.label_x.text)))
    else:
        x = ''
    if layout.axes.twin_y:
        x += ' + ' + strip_html(layout.label_x2.text)
    row, col, wrap, groups, fig = '', '', '', '', ''
    if layout.label_row.text is not None:
        row = ' by ' + strip_html(layout.label_row.text)
    if layout.label_col.text is not None:
        col = ' by ' + strip_html(layout.label_col.text)
    if layout.title_wrap.text is not None:
        wrap = ' by ' + strip_html(layout.title_wrap.text)
    # this one may need to change with box plot
    if kwargs.get('groups', False):
        groups = ' by ' + ' + '.join(validate_list(kwargs['groups']))
    if fig_item is not None:
        fig_items = validate_list(fig_item)
        for ifig, fig_col in enumerate(fig_cols):
            if ifig > 0:
                fig += ' and'
            fig += f' where {fig_col}={fig_items[ifig]}'
        if strip_html(layout.label_col.text) is None:
            col = ''

    filename = f'{z}{y}{x}{row}{col}{wrap}{groups}{fig}'

    # Cleanup forbidden symbols
    bad_char = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for bad in bad_char:
        filename = filename.replace(bad, '_')

    # Make sure extension has a '.'
    if not kwargs.get('save_ext'):
        kwargs['save_ext'] = '.png'  # set a default
    if kwargs.get('save_ext')[0] != '.':
        kwargs['save_ext'] = '.' + kwargs['save_ext']

    if len(filename + kwargs.get('save_ext')) > 255:
        filename = 'filename_too_long'
        print('Warning!  Default filename is too long...saving with non-unique name')

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


def split_color_planes(img: pd.DataFrame, cfa: str = 'rggb', as_dict: bool = True) -> pd.DataFrame:
    """Split image data into respective color planes.  dtype changed to float

    Args:
        img: image data
        cfa (optional): four-character cfa pattern. Defaults to 'rggb'.
        as_dict (optional): return each plane DataFrame as a value in a dict. Defaults to True.

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

    # DataFrame input
    if isinstance(img, pd.DataFrame):
        if len(img.shape) > 2:
            raise ValueError('split_color_planes only supports 2D DataFrames')

        # Separate planes
        planes = []
        int_cols = df_int_cols(img)
        other_cols = df_int_cols(img, True)
        for ic, cc in enumerate(cp):
            idx = [f for f in img.index if f % 2 == ic // 2]
            cols = [f for f in int_cols if f % 2 == ic % 2]
            sub = img.loc[idx, cols].copy()
            for oc in other_cols:
                sub[oc] = img.loc[idx[0], oc]
            if not as_dict:
                sub['Plane'] = cc
            planes += [sub]

        if not as_dict:
            return pd.concat(planes)

        if as_dict:
            return dict(zip(cp, planes))

    # Numpy array input
    else:
        img2 = {}
        for ic, cc in enumerate(cp):
            img2[cc] = img[ic // 2::2, (ic % 2)::2]

        if not as_dict:
            df = []
            for k, v in img2.items():
                temp = pd.DataFrame(v)
                temp['Plane'] = k
                df += [temp]
            return pd.concat(df)

        else:
            return img2


def split_color_planes_wrapper(df_groups: pd.DataFrame,
                               imgs: Dict[int, pd.DataFrame],
                               cfa: str) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Wrapper function for Data classes using img data that need to split planes

    Args:
        df_groups: grouping DataFrame from img_df_transform
        imgs: dict of img DataFrames
        cfa: cfa-type name

    Returns:
        updated df_groups
        updated imgs dict
    """
    imgs_ = {}
    for i, (k, v) in enumerate(imgs.items()):
        planes = split_color_planes(v, cfa, as_dict=True)
        if i == 0:
            df_groups_ = pd.merge(df_groups.reset_index(),
                                  pd.DataFrame({'Plane': list(planes.keys())}), how='cross')
            df_groups_.rows //= 2
            df_groups_.cols //= 2
        for j, (pk, pv) in enumerate(planes.items()):
            imgs_[len(planes.keys()) * i + j] = pv

    return df_groups_, imgs_


def strip_html(text: str):
    """Remove text in between HTML tags, if present.

    Args:
        text: input string

    Returns:
        cleaned up text
    """
    if not isinstance(text, str):
        return text
    if '<' in text and '>' in text:
        return re.sub('<[^>]*>', '', text)
    else:
        return text


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


def tuple_list_index(tuple_list: List[Tuple[Any]], search_value: Any, idx: int = 0) -> int:
    """
    Find the index of a search value that matches the value of a tuple at a specific tuple index in a list of tuples.

    Args:
        tuple_list: list of tuples; tuples can have any len
        search_value:  the value to find
        idx: the index within a specific tuple to search

    Return:
        index or -1 if not found
    """
    return next((i for i, t in enumerate(tuple_list) if t[0] == search_value), -1)


def unit_test_get_img_name(name: str, make_reference: bool, reference_path: pathlib.Path) -> pathlib.Path:
    """
    Make the image path for a unit test

    Args:
        name: image name
        make_reference: flag for making the reference images
        reference_path:  path to the reference images
    Returns:
        image_path
    """
    path = reference_path / f'{name}_reference' if make_reference else Path(name)
    return path.with_suffix('.png')


def unit_test_options(make_reference: bool, show: Union[bool, int], img_path: pathlib.Path,
                      reference_path: pathlib.Path):
    """
    Complete a unit test with one of four options:
        1. Show the generated image only
        2. Show the generated image, the reference image, and the difference image
        3. Assert generated image and reference image are the same
        4. Do nothing (if making new reference images)

    Args:
        make_reference:  make new references images (bypasses this function)
        show:  if -1, show just the new generated image; if True, show the generated image, the reference image, and
            the difference image
        img_path: path to the generated image (or reference image if make_reference==True)
        reference_path:  path to the reference image

    Returns:
        none
    """
    if make_reference:
        return

    reference_path = (reference_path / f'{img_path.stem}_reference').with_suffix('.png')
    if show == -1:
        show_file(img_path)
        unit_test_debug_margins(img_path)
        return img_path
    elif show:
        show_file(reference_path)
        show_file(img_path)
        compare = img_compare(img_path, reference_path, show=True)
        return img_path, reference_path
    else:
        compare = img_compare(img_path, reference_path)
        os.remove(img_path)

        assert not compare


def unit_test_measure_axes(img_path: pathlib.Path, row: Union[int, str, None], col: Union[int, None],
                           width: int = 0, height: int = 0, channel: int = 1, skip: int = 0, alias=True):
    """
    Get axes + axes edge width size from pixel values.  Only works if surrounding border pixels are the same color but
    different values than the axes area.

    Args:
        img_path: path the test image
        row: row index on which to measure (should be clear of labels, legends, ticks etc); if None, skip
        col: col index on which to measure (should be clear of labels, title, ticks, etc); if None, skip
        width: target width; if None skip test
        height: target height; if None skip test
        channel: which color channel to use (RGB image only)
        skip: skip some number of pixels
        alias: skip up to 1 pixel due to axes edge aliasing
    """
    img = cv2.imread(str(img_path))
    if len(img.shape) == 3:
        img = img[:, :, channel]

    if row:
        if row == 'c':
            row = int(img.shape[0] / 2)
        row = img[row, skip:]
        x0 = (np.diff(row) != 0).argmax(axis=0) + 1  # add 1 for diff
        assert (row[x0:] == 255).argmax(axis=0) == width + (2 if alias else 0), \
               f'expected width: {width} | actual: {(row[x0:] == 255).argmax(axis=0) - (2 if alias else 0)}'

    if col:
        if col == 'c':
            col = int(img.shape[1] / 2)
        col = img[skip:, col]
        y0 = (np.diff(col) != 0).argmax(axis=0) + 1
        assert (col[y0:] == 255).argmax(axis=0) == height + (2 if alias else 0), \
               f'expected height: {height} | actual: {(col[y0:] == 255).argmax(axis=0) - (2 if alias else 0)}'


def unit_test_measure_axes_cols(img_path: pathlib.Path, row: Union[int, str, None], width: int, num_cols: int,
                                target_pixel_value: int = 255, alias=True, cbar: bool = False, ws=False):
    """
    Get margin sizes from pixel values.  Only works if surrounding border pixels are the same color but different
    values than the axes area.

    Args:
        img_path: path the test image
        row: row index on which to measure (should be clear of labels, legends, ticks etc)
        width: expected axes width for each column
        num_cols: number of subplot columns
        target_pixel_value: pixel value to look for (whitespace 255 typically)
        alias: skip up to 1 pixel due to axes edge aliasing
        cbar: skip the cbars if enabled
        ws: check the ws between columns

        Note: for np.diff statements, need to subtract 1
    """
    # Test width of all column images
    img = cv2.imread(str(img_path))
    if row == 'c':
        row = int(img.shape[0] / 2)
    dd = np.diff(np.concatenate(([False], np.all(img[row] == target_pixel_value, axis=-1), [False])).astype(int))
    trans1 = np.argwhere(dd == -1).T[0]
    trans2 = np.argwhere(dd == 1).T[0]
    if not cbar:
        widths = trans2[1:] - trans1[:-1] - width - (2 if alias else 0)
    else:
        widths = (trans2[1:] - trans1[:-1])[::2] - width - (2 if alias else 0)
    assert len(widths[widths == 0]) == num_cols, \
        f'expected {num_cols} columns with axes width {width} | detected: {widths + width}'

    if ws:
        ws_meas = (trans1 - trans2)[1:-1]
        assert np.all(ws_meas == ws), \
            f'expected ws between columns {ws} | detected: {ws_meas[1]}'


def unit_test_measure_axes_rows(img_path: pathlib.Path, col: Union[int, str, None], height: int, num_rows: int,
                                target_pixel_value: int = 255, alias=True):
    """
    Get margin sizes from pixel values.  Only works if surrounding border pixels are the same color but different
    values than the axes area.

    Args:
        img_path: path the test image
        col: column index on which to measure (should be clear of labels, legends, ticks etc)
        height: expected axes height for each column
        num_rows: number of subplot rows
        target_pixel_value: pixel value to look for (whitespace 255 typically)
        alias: skip up to 1 pixel due to axes edge aliasing

        Note: for np.diff statements, need to subtract 1
    """
    # Test width of all row images
    img = cv2.imread(str(img_path))
    if col == 'c':
        col = int(img.shape[1] / 2)
    dd = np.diff(np.concatenate(([False], np.all(img[:, col] == target_pixel_value, axis=-1), [False])).astype(int))
    trans1 = np.argwhere(dd == -1).T[0]
    trans2 = np.argwhere(dd == 1).T[0]
    heights = (trans2[1:] - trans1[:-1]) - height - (2 if alias else 0)
    assert len(heights[heights == 0]) == num_rows, \
        f'expected {num_rows} rows with axes height {height} | detected: {heights + height}'


def unit_test_debug_margins(img_path: pathlib.Path):
    """Try to measure all the margins."""
    img = cv2.imread(str(img_path))[:, :, 1]
    if not (img[0, 0] == img[-1, -1] == img[0, -1] == img[-1, 0]):
        print('could not determine border color and detect margins')

    border_color = img[0, 0]

    # Side margins
    left, right = [], []
    for row in range(0, img.shape[0]):
        left += [np.argmax(img[row, :] != border_color)]
        right += [np.argmax(img[row, :][::-1] != border_color)]
    left = np.argmax(np.bincount(left))
    right = np.argmax(np.bincount(right))
    width = img.shape[1] - left - right

    # Top/bottom margins
    top, bottom = [], []
    for col in range(0, img.shape[1]):
        top += [np.argmax(img[:, col] != border_color)]
        bottom += [np.argmax(img[:, col][::-1] != border_color)]
    top = np.argmax(np.bincount(top))
    bottom = np.argmax(np.bincount(bottom))
    height = img.shape[0] - top - bottom

    print('Axes area and borders:')
    print(f'axes: {width} x {height}\nleft: {left}\nright: {right}\ntop: {top}\nbottom: {bottom}')
    print('Note: axes size includes subplots + whitespace + 2 * edge_width + 2 pixels for aliasing (if edge_width > 0)')


def unit_test_measure_margin(img_path: pathlib.Path, row: Union[int, str, None], col: Union[int, None],
                             left: Union[int, None] = None, right: Union[int, None] = None,
                             top: Union[int, None] = None, bottom: Union[int, None] = None, alias=True,
                             target_pixel_value=255):
    """
    Get margin sizes from pixel values.  Only works if surrounding border pixels are the same color but different
    values than the axes area.

    Args:
        img_path: path the test image
        row: row index on which to measure (should be clear of labels, legends, ticks etc)
        col: col index on which to measure (should be clear of labels, title, ticks, etc)
        left: target left margin; if None skip measurement
        right: target right margin; if None skip measurement
        top: target top margin; if None skip measurement
        bottom: target bottom margin; if None skip measurement
        alias: skip up to 1 pixel due to axes edge aliasing

        Note: for np.diff statements, need to subtract 1
    """
    img = cv2.imread(str(img_path))

    if row:
        if row == 'c':
            row = int(img.shape[0] / 2)

        # Find the transitions from the target pixel to something else
        tps = np.where(~(img[row, :] == target_pixel_value).all(axis=1))[0]

        if len(tps) == 0:
            raise ValueError('all pixels in specified row match the target pixel')

        # Get column indexes are not preceeded by same value
        if left:
            assert tps[0] == left - (1 if alias else 0), \
               f'expected left margin: {left} | actual: {tps[0] + (1 if alias else 0)}'
        if right:
            assert img.shape[1] - (tps[-1] + 1) == right - (1 if alias else 0), \
               f'expected right margin: {right} | ' + \
               f'actual: {img.shape[1] - (tps[-1] + 1) + (1 if alias else 0)}'
    if col:
        if col == 'c':
            col = int(img.shape[1] / 2)

        # Find the transitions from the target pixel to something else
        tps = np.where(~(img[:, col] == target_pixel_value).all(axis=1))[0]

        if len(tps) == 0:
            raise ValueError('all pixels in specified column match the target pixel')

        if top:
            assert tps[0] == top - (1 if alias else 0), \
               f'expected top margin: {top} | actual: {tps[0] + (1 if alias else 0)}'
        if bottom:
            assert img.shape[0] - (tps[-1] + 1) == bottom - (1 if alias else 0), \
               f'expected bottom: {bottom} | actual: {img.shape[0] - (tps[-1] + 1) + (1 if alias else 0)}'


def unit_test_make_all(reference: pathlib.Path, name: str, start: Union[str, None] = None,
                       stop: Union[str, None] = None):
    """Remake all reference test images.

    Args:
        reference: path to the reference test images
        name: sys.modules[__name__] for the test file
        start: name of file to start with
        stop: name of file to stop
    """
    if not reference.exists():
        os.makedirs(reference)
    members_ = inspect.getmembers(name)
    members = sorted([f for f in members_ if 'plt_' in f[0]])
    if len(members) == 0:
        members = sorted([f for f in members_ if 'test_' in f[0]])
    if len(members) == 0:
        print('no test functions found')
        return
    if start is not None:
        idx_found = [i for (i, f) in enumerate(members) if start in f[0]]
        if len(idx_found) > 0:
            members = members[idx_found[0]:]
    for member in members:
        if stop and stop in member[0]:
            print('stopping!')
            return
        print(f'Running {member[0]}...', end='')
        member[1](make_reference=True)
        print('done!')


def unit_test_show_all(only_fails: bool, reference: pathlib.Path, name: str, start: Union[str, None] = None):
    """Show all unit test plots.

    Args:
        only_fails: only show the test plots that fail the unit test
        reference: path to the reference test images
        name: sys.modules[__name__] for the test file
        start: search string to start the show_all loop with a specific plot
    """
    if not reference.exists():
        os.makedirs(reference)
    members_ = inspect.getmembers(name)
    members = sorted([f for f in members_ if 'plt_' in f[0]])
    if len(members) == 0:
        members = sorted([f for f in members_ if 'test_' in f[0]])
    if len(members) == 0:
        print('no test functions found')
        return
    if start is not None:
        idx_found = [i for (i, f) in enumerate(members) if start in f[0]]
        if len(idx_found) > 0:
            members = members[idx_found[0]:]
    for member in members:
        print(f'Running {member[0]}...', end='')
        paths = []
        if only_fails:
            try:
                member[1]()
            except AssertionError:
                paths = member[1](show=True)
                db()
        else:
            paths = member[1](show=True)
            db()

        # macos only convenience - close last opened files
        if paths is not None and len(paths) > 0:
            close_preview_windows_macos([f.name for f in paths] + ['difference.png'])


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
