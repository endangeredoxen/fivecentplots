import pandas as pd
import pdb
import scipy.stats
import numpy as np
import copy
import math
import logging
import datetime
from typing import Callable, Dict, List, Tuple, Union
import numpy.typing as npt
from fivecentplots.utilities import RepeatedList
import fivecentplots.utilities as utl
from packaging import version
from . layout import LOGX, LOGY, SYMLOGX, SYMLOGY, LOGITX, LOGITY, LOG_ALLX, LOG_ALLY, BaseLayout, Element  # noqa
import warnings
import matplotlib as mpl
import matplotlib.pyplot as mplp
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import ConnectionPatch
import matplotlib.dates as mdates
from itertools import groupby

warnings.filterwarnings('ignore', category=UserWarning)


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
# weird error in boxplot with no groups
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")

# MPL fonts
logging.getLogger('matplotlib.font_manager').disabled = True  # disable font logger
FONT_LIST = font_manager.get_font_names()

db = pdb.set_trace
TICK_OVL_MAX = 1.15  # maximum allowed overlap for tick labels in float pixels
ROTATE_90_OFFSET_X = 0  # spacing offset for 90 deg rotated axes labels
DEFAULT_MARKERS = ['o', '+', 's', 'x', 'd', 'Z', '^', 'Y', 'v', r'\infty',
                   r'\#', r'<', u'\u2B21', u'\u263A', '>', u'\u29C6', r'\$',
                   u'\u2B14', u'\u2B1A', u'\u25A6', u'\u229E', u'\u22A0',
                   u'\u22A1', u'\u20DF', r'\gamma', r'\sigma', r'\star', ]


def approx_gte(x: float, y: float):
    """Check if a float is approximately ?= to another float.
    Stolen from stack overflow answer from Barmar

    Args:
        x: float #1
        y: float #2

    Returns:
        bool
    """
    atol = 1e-8
    if y > 0:
        # set tolerance for very small numbers
        expo = round(np.log10(y))
        atol = expo - 3 if expo < 0 else 1e-8
    return x >= y or np.isclose(x, y, atol=atol)


def approx_lte(x: float, y: float):
    """Check if a float is approximately <= to another float.
    Stolen from stack overflow answer from Barmar

    Args:
        x: float #1
        y: float #2

    Returns:
        bool
    """
    atol = 1e-8
    if y > 0:
        # set tolerance for very small numbers
        expo = round(np.log10(y))
        atol = expo - 3 if expo < 0 else 1e-8
    return x <= y or np.isclose(x, y, atol=atol)


def cbar_ticks(cbar: 'Colorbar_Object', zmin: float, zmax: float):  # noqa: F821
    """Format cbar ticks and labels.

    Args:
        cbar: colorbar
        zmin: min z-value (vmin)
        zmax: max z-value (vmax)

    Returns:
        new tick labels
    """
    num_ticks = len(cbar.get_ticks())
    new_ticks = np.linspace(zmin, zmax, num_ticks)
    decimals = [utl.get_decimals(f) for f in new_ticks]
    for it, nt in enumerate(new_ticks[0:-1]):
        new_ticks[it] = '{num:.{width}f}'.format(num=nt, width=max(decimals))
    return new_ticks


def df_tick(ticks: 'Element', ticks_size: 'np.array', ax: str) -> pd.DataFrame:
    """Create a DataFrame of tick extents.  Used to look for overlapping ticks.

    Args:
        ticks: Element class for tick lables
        ticks_size: Bounding box arrays of all tick labels
        ax: axes name (x or y)

    Returns:
        pd.DataFrame of tick labels with start and stop pixel locations
    """

    idx0 = ticks.size_cols.index(f'{ax}0')
    idx1 = ticks.size_cols.index(f'{ax}1')

    # Convert the tick size array to DataFrame denoting the start and stop pixel locations
    tt = pd.DataFrame(ticks_size[:, [idx0, idx1]], columns=['start', 'stop'])

    # Check ascending vs descending
    if len(tt) > 1 and tt['start'].diff()[1] < 0:
        tt = tt.iloc[::-1]

    return tt


def df_tick_update(tt: pd.DataFrame) -> pd.DataFrame:
    """Calculate the next round of overlaps in the ticks DataFrame

    Args:
        tt: ticks DataFrame

    Returns:
        updated ticks DataFrame
    """
    tt = tt.copy()

    # Shift the start column to line up the stop coordinate of one tick with the start of the next
    tt['next start'] = tt['start'].shift(-1)

    # Check the difference between stop coordinate of one tick and start of next; positive means overlap
    tt['delta'] = tt['stop'] - tt['next start']
    tt['ol'] = tt['delta'] > TICK_OVL_MAX    # allow a litle bit of overlap (otherwise would choose > 0)
    if 'visible' not in tt.columns:
        tt['visible'] = True

    return tt


def get_non_overlapping_rectangles(df):
    """
    Select maximum number of non-overlapping rectangles from a DataFrame
    using dynamic programming. When multiple options exist with the same count,
    select the set with rectangles closest to the center of the overall range.

    Parameters:
    df (pd.DataFrame): DataFrame with 'start' and 'stop' columns

    Returns:
    pd.DataFrame: Subset of original DataFrame with non-overlapping rectangles
    """
    if len(df) <= 1:
        return df

    # Calculate overall center for prioritization
    min_x = df['start'].min()
    max_x = df['stop'].max()
    overall_center = (min_x + max_x) / 2

    # Add center and distance to center for each rectangle
    df_work = df.copy()
    df_work['center'] = (df['start'] + df['stop']) / 2
    df_work['distance'] = np.abs(df_work['center'] - overall_center)

    # Sort intervals by end time (crucial for the dynamic programming approach)
    df_sorted = df_work.sort_values('stop')
    n = len(df_sorted)

    # dp[i] = max number of non-overlapping intervals up to i
    dp = [1] * n
    # prev[i] = previous interval in optimal solution ending at i
    prev = [-1] * n
    # result[i] = list of indexes in optimal solution ending at i
    result = [[i] for i in range(n)]

    for i in range(1, n):
        # Find all compatible previous intervals
        compatible_solutions = []
        for j in range(i):
            if df_sorted.iloc[j]['stop'] <= df_sorted.iloc[i]['start']:
                compatible_solutions.append((dp[j] + 1, j))

        # If we found any compatible solutions, use the best one
        if compatible_solutions:
            best_count, best_j = max(compatible_solutions)
            if best_count > dp[i]:
                dp[i] = best_count
                prev[i] = best_j
                result[i] = result[best_j] + [i]

    # Find solutions with maximum count
    max_count = max(dp)
    candidates = [i for i, count in enumerate(dp) if count == max_count]

    # If multiple solutions exist with the same count, find the one closest to center
    if len(candidates) > 1:
        best_distance = float('inf')
        best_candidate = None

        for idx in candidates:
            # Calculate average distance for this solution
            solution_indices = result[idx]
            solution_df = df_sorted.iloc[solution_indices]
            avg_distance = solution_df['distance'].mean()

            if avg_distance < best_distance:
                best_distance = avg_distance
                best_candidate = idx

        best_solution = result[best_candidate]
    else:
        best_solution = result[candidates[0]]

    # Map back to original indices
    original_indices = df_sorted.index.values[best_solution]
    return df.loc[original_indices]


def hide_overlaps(ticks: 'Element', tt: pd.DataFrame, ir: int, ic: int) -> pd.DataFrame:
    """Find and hide any overlapping tick marks on the same axis.

    Args:
        ticks: Element class for tick lables
        tt: tick DataFrame created by df_ticks
        ir: current axes row index
        ic: current axes column index

    Returns:
        pd.DataFrame of tick label visibility status
    """
    tt = df_tick_update(tt)
    tt_orig = tt.copy()
    tt = tt[tt['visible']]
    tt = df_tick_update(tt)

    # While overlaps exist, cycle through and disable the next tick in line
    while True in tt['ol'].values:
        tt.loc[tt.index[-1], 'ol'] = True  # modify the last line of the table
        for i in range(0, len(tt[:-1])):
            if tt.iloc[i]['ol']:
                jj = tt.iloc[i + 1].name
                ticks.obj[ir, ic][jj].set_visible(False)  # turn off the tick visibility
                tt.loc[jj, 'ol'] = False
                tt_orig.loc[jj, 'visible'] = False

        # Rebuild the overlap table to reflect ticks that have already been disabled
        tt = tt[tt['ol']].copy()
        tt = df_tick_update(tt)

    return tt_orig


def is_in_range(n: float, tbl: np.ndarray):
    """Convenience function for tick marks to check whether particular number
    (n) is in the range between borders defined by each row in a 2-column
    table (tbl).

    From: https://stackoverflow.com/a/71380003

    Args:
        n: number to check if it is in range
        tbl: 2-column array defining borders in which to check if n is in range

    Returns:
        bool True if in range else False
    """
    ol = np.apply_along_axis(lambda row: row[0] <= n <= row[1], 1, tbl).tolist()
    if True in ol:
        return True
    return False


def iterticks(ax: mplp.Axes, minor: bool = False):
    """Replication of iterticks function which was deprecated in later versions
    of mpl but used in fcp, so just copying it here to avoid warnings or future
    removal.

    Args:
        ax: MPL axes object
        minor: True = minor ticks, False = major ticks. Defaults to False.

    Yields:
       ticks
    """
    if version.Version(mpl.__version__) >= version.Version('3.1'):
        major_locs = ax.get_majorticklocs()
        major_labels = ax.major.formatter.format_ticks(major_locs)
        major_ticks = ax.get_major_ticks(len(major_locs))
        yield from zip(major_ticks, major_locs, major_labels)
        # the minor ticks are a major slow down so only do when needed
        if minor:
            minor_locs = ax.get_minorticklocs()
            minor_labels = ax.minor.formatter.format_ticks(minor_locs)
            minor_ticks = ax.get_minor_ticks(len(minor_locs))
            yield from zip(minor_ticks, minor_locs, minor_labels)
    else:
        yield from getattr(ax, 'iter_ticks')()


def mplc_to_hex(color: tuple, alpha: bool = True):
    """Convert mpl color to hex.

    Args:
        color: matplotlib style color code
        alpha: include or exclude the alpha value

    Returns:
        hex string
    """
    hexc = '#'
    for ic, cc in enumerate(color):
        if not alpha and ic == 3:
            continue
        hexc += f'{hex(int(cc * 255))[2:].zfill(2)}'

    return hexc


def mpl_get_ticks(ax: mplp.Axes, xon: bool = True, yon: bool = True,
                  minor: bool = False):
    """Divine a bunch of tick and label parameters for mpl layouts.

    Args:
        ax: axes object
        xon: True = get x-axis ticks. Defaults to True.
        yon: True = get y-axis ticks. Defaults to True.
        minor: True = minor ticks, False = major ticks. Defaults to False.

    Returns:
        dict of x and y ax tick parameters
    """
    tp = {}
    xy = []
    if xon:
        xy += ['x']
    if yon:
        xy += ['y']

    for vv in xy:
        tp[vv] = {}
        lim = getattr(ax, f'get_{vv}lim')()
        tp[vv]['min'] = min(lim)
        tp[vv]['max'] = max(lim)
        tp[vv]['ticks'] = getattr(ax, f'get_{vv}ticks')()
        tp[vv]['labels'] = [f for f in iterticks(getattr(ax, f'{vv}axis'), minor)]
        tp[vv]['label_vals'] = [f[1] for f in tp[vv]['labels']]
        tp[vv]['label_text'] = [f[2] for f in tp[vv]['labels']]
        try:
            tp[vv]['first'] = [i for i, f in enumerate(tp[vv]['labels'])
                               if f[1] >= tp[vv]['min'] and f[2] != ''][0]
        except IndexError:
            tp[vv]['first'] = -999
        try:
            tp[vv]['last'] = [i for i, f in enumerate(tp[vv]['labels'])
                              if f[1] <= tp[vv]['max'] and f[2] != ''][-1]
        except IndexError:
            tp[vv]['last'] = -999

    missing = [f for f in ['x', 'y'] if f not in tp.keys()]
    for mm in missing:
        tp[mm] = {}
        tp[mm]['ticks'] = []
        tp[mm]['labels'] = []
        tp[mm]['label_text'] = []
        tp[mm]['first'] = -999
        tp[mm]['last'] = -999

    return tp


def rendered_edge_width(value: Union[int, float]) -> Callable:
    """
    Scale the nominal edge width to account for rendering differences; empirically determined by trial and error.

    Args:
        param: name of the Element attribute to scale

    Return:
        scaled value
    """

    return (value + 0.3323) / 1.4059


def select_minor_ticks(majors, minors):
    minors['weight'] = 0
    for imaj, maj in majors[:-1].iterrows():
        minor = minors[(minors['start'] > maj['stop'] - TICK_OVL_MAX) &
                       (minors['stop'] < maj['next start'] + TICK_OVL_MAX)].copy()
        minor['start'] += TICK_OVL_MAX
        best_ticks = get_non_overlapping_rectangles(minor)
        minors.loc[best_ticks.index, 'visible'] = True

    return minors


class Layout(BaseLayout):
    DEFAULT_MARKERS = ['o', '+', 's', 'x', 'd', 'Z', '^', 'Y', 'v', r'\infty', r'\#', r'<', u'\u2B21', u'\u263A', '>',
                       u'\u29C6', r'\$', u'\u2B14', u'\u2B1A', u'\u25A6', u'\u229E', u'\u22A0', u'\u22A1', u'\u20DF',
                       r'\gamma', r'\sigma', r'\star', ]

    def __init__(self, data: 'Data', defaults: list = [], **kwargs):  # noqa: F821
        """Layout attributes and methods for matplotlib Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        self.engine = 'mpl'
        self.font_warning = []
        self.default_box_marker = 'o'

        # Set tick style to classic if using fcp tick_cleanup (default)
        if kwargs.get('tick_cleanup', True):
            mplp.style.use('classic')
        else:
            mplp.style.use('default')

        # Unless specified, close previous plots
        if not kwargs.get('hold', False):
            mplp.close('all')

        # Rendering correction equations
        # edge_width doesn't render as specified (aliasing?) so adjust it to render as specified
        kwargs['corrections'] = {
            'edge_width': lambda self: 0 if getattr(self, 'edge_width') == 0
            else (getattr(self, 'edge_width') + 0.3323) / 1.4059}

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)

        # Apply rcParams
        if len(defaults) == 4:
            mpl.rcParams.update(defaults[3])

        # Initialize other class variables
        self.label_col_height = 0
        self.label_row_left = 0
        self.label_row_width = 0
        self.title_wrap_bottom = 0

        # Weird spacing defaults out of our control
        self.legend_top_offset = 8
        self.legend_border = 3
        self.fig_legend_border = 8
        self.x_tick_xs = 0
        self.ws_ticks_ax_adj = self.ws_ticks_ax + 0.373 * self.axes.edge_width
        self.axes.edge_width_size = 0 if self.axes.edge_width == 1 else self.axes.edge_width

        # Other
        self._set_colormap(data)

        # Update kwargs
        if not kwargs.get('save_ext'):
            kwargs['save_ext'] = '.png'
        self.kwargs = kwargs

    @property
    def _bottom(self) -> int:
        """
        Space below the bottom of the axes. Round up fractional pixels to match figure output.
        """
        val = np.ceil(self.ws_fig_label) \
            + np.ceil(self.box_labels) \
            + np.ceil(self._legy) \
            + np.ceil(self.pie.xs_bottom) \
            + np.ceil(self._labtick_x) \
            + np.ceil(self.legend.overflow)

        if self.gantt.on and self.gantt.date_location == 'top' and self.gantt.today.on:
            val += np.ceil(self.gantt.today.size[1] + self.gantt.today.edge_width)

        return int(val)

    @property
    def _cbar(self) -> float:
        """Width of all the cbars and cbar ticks/labels and z-labels."""
        if not self.cbar.on:
            return 0

        val = \
            (self.ws_ax_cbar + self.cbar.size[0] + self.tick_labels_major_z.size[0]) \
            * (self.ncol if not self.cbar.shared else 1) \
            + (self.label_z.size[0] * (self.ncol if self.separate_labels else 1)
               + self.ws_ticks_ax * (self.ncol - 1 if not self.cbar.shared else 0)  # btwn z-ticks and the next axes
               + self.ws_ticks_ax * self.label_z.on)  # this is between the z-ticks and label_z
        return val

    @property
    def _labtick_x(self) -> float:
        """Height of the x label + x tick labels + related whitespace."""
        if self.name == 'gantt' and self.gantt.date_location == 'top':
            return 0

        val = self.label_x.size[1] * self.label_x.on \
            + self.ws_label_tick * ((self.tick_labels_major_x.on | self.tick_labels_minor_x.on) & self.label_x.on) \
            + self._tick_x
        if self.label_x.on and self._tick_x == 0:
            val += self.ws_label_tick

        return val

    @property
    def _labtick_x2(self) -> float:
        """Height of the secondary x label + x tick labels + related whitespace."""
        if self.name == 'gantt' and self.gantt.date_location == 'top':
            # Flip tick and label to the top for some gantt cases
            val = self.label_x.size[1] * self.label_x.on \
                + self.ws_label_tick * ((self.tick_labels_major_x.on | self.tick_labels_minor_x.on) & self.label_x.on) \

            # Add the ticks (no edges)
            if any(f in self.gantt.date_type for f in self.gantt.DATE_TYPES):
                val += len(self.gantt.date_type) * self._tick_x2

            # Correction since month-year is two lines and others are one line
            if 'month-year' in self.gantt.date_type:
                val += self.tick_labels_major_x.size[1]

            # Add edge widths
            edge = np.ceil(self.grid_major_y.width[0]) - 1
            val += edge * (len(self.gantt.date_type) - 1)

            return np.ceil(val) - 1

        if not self.axes.twin_y:
            return 0

        val = self.label_x2.size[1] * self.label_x2.on \
            + self.ws_label_tick * ((self.tick_labels_major_x2.on | self.tick_labels_minor_x2.on) & self.label_x2.on) \
            + self._tick_x2
        if self.label_x2.on and self._tick_x2 == 0:
            val += self.ws_label_tick

        return val

    @property
    def _labtick_y(self) -> float:
        """Width of the y label + y tick labels + related whitespace."""
        if self.pie.on:
            return 0

        val = self.label_y.size[0] * self.label_y.on \
            + self.ws_label_tick * ((self.tick_labels_major_y.on | self.tick_labels_minor_y.on) & self.label_y.on) \
            + self._tick_y
        if self.label_y.on and self._tick_y == 0:
            val += self.ws_label_tick

        if self.gantt.on and self.gantt.labels_as_yticks:
            val += self.gantt.box_padding_x - np.ceil(self.grid_minor_y.width[0] / 2)

        return np.ceil(val)

    @property
    def _labtick_y2(self) -> float:
        """Width of the secondary y label + y tick labels + related whitespace."""
        if not self.axes.twin_x:
            return 0

        val = self.label_y2.size[0] * self.label_y2.on \
            + self.ws_label_tick * ((self.tick_labels_major_y2.on | self.tick_labels_minor_y2.on) & self.label_y2.on) \
            + self._tick_y2
        if self.label_y2.on and self._tick_y2 == 0:
            val += self.ws_label_tick
        return val

    @property
    def _left(self) -> int:
        """
        Width of the space to the left of the axes object. Round up fractional pixels to match figure output.
        """
        left = np.ceil(self.ws_fig_label) + np.ceil(self._labtick_y)

        title_xs_left = np.ceil(self.title.size[0] / 2) + np.ceil(self.ws_fig_ax if self.title.on else 0) \
            - (left + np.ceil(self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2)
        if title_xs_left < 0:
            title_xs_left = 0
        left += title_xs_left

        # pie labels
        left += np.ceil(self.pie.xs_left)

        # gantt workstream labels
        if self.gantt.on and self.gantt.workstreams.on and self.gantt.workstreams.location == 'left':
            left += self.gantt.workstreams.size[0] + np.ceil(self.gantt.workstreams.edge_width / 2)
        if self.gantt.on and self.gantt.workstreams_title.on and self.gantt.workstreams.location == 'left':
            left += self.gantt.workstreams_title.size[0] + np.ceil(self.gantt.workstreams_title.edge_width / 2)

        return int(left)

    @property
    def _legx(self) -> float:
        """Legend whitespace x if location == 0."""
        if self.legend.location == 0 and self.legend._on:
            return self.legend.size[0] + self.ws_ax_leg + self.ws_leg_fig + self.fig_legend_border \
                   + np.ceil(self.legend.edge_width / 2) - (self.fig_legend_border if self.legend._on else 0)
        else:
            return 0

    @property
    def _legy(self) -> float:
        """Legend whitespace y is location == 11."""
        if self.legend.location == 11 and self.legend._on:
            return self.legend.size[1]
        else:
            return 0

    @property
    def _row_label_width(self) -> float:
        """Width of an rc label with whitespace."""
        return (self.label_row.size[0] + self.ws_label_row
                + 2 * self.label_row.edge_width) * self.label_row.on

    @property
    def _right(self) -> int:
        """
        Width of the space to the right of the axes object (ignores cbar [bar and tick labels] and legend).
        Round up fractional pixels to match figure output.
        """
        # axis to fig right side ws with or without legend
        ws_ax_fig = (self.ws_ax_fig if not self.legend._on or self.legend.location != 0 else 0)

        # sum all parts
        right = np.ceil(ws_ax_fig) \
            + np.ceil(self._labtick_y2) \
            + np.ceil(self._row_label_width) \
            + np.ceil(self.x_tick_xs)

        # box title excess
        if self.box_group_title.on and (self.ws_ax_box_title + self.box_title) > \
                self._legx + (self.fig_legend_border if self.legend._on else 0):
            right = np.ceil(self.ws_ax_box_title) + np.ceil(self.box_title) \
                 + np.ceil((self.ws_ax_fig if not self.legend.on else 0))
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            right += np.ceil(self.box_title)

        # Main figure title excess size
        title_xs_right = np.ceil(self.title.size[0] / 2) \
            - np.ceil((right + (self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2)) \
            - np.ceil(self.legend.size[0])
        if title_xs_right < 0:
            title_xs_right = 0

        right += title_xs_right

        # pie labels
        right += np.ceil(self.pie.xs_right)

        # gantt workstream labels
        if self.gantt.on and self.gantt.workstreams.on and self.gantt.workstreams.location == 'right':
            right += np.ceil(self.gantt.workstreams.size[0] + self.gantt.workstreams.edge_width)
        if self.gantt.on and self.gantt.workstreams_title.on and self.gantt.workstreams.location == 'right':
            right += np.ceil(self.gantt.workstreams_title.size[0] + self.gantt.workstreams_title.edge_width)

        return int(right)

    @property
    def _tick_x(self) -> float:
        """Height of the primary x ticks and whitespace."""
        if self.tick_labels_major_x.size[1] > self.tick_labels_minor_x.size[1]:
            tick = self.tick_labels_major_x
        else:
            tick = self.tick_labels_minor_x

        val = (tick.size[1] + tick.edge_width + self.ws_ticks_ax) * tick.on

        return val

    @property
    def _tick_x2(self) -> float:
        """Height of the secondary x ticks and whitespace."""
        if self.name == 'gantt' and self.gantt.label_boxes and self.gantt.date_location == 'top':
            if self.tick_labels_major_x.size[1] > self.tick_labels_minor_x.size[1]:
                tick = self.tick_labels_major_x
            else:
                tick = self.tick_labels_minor_x
            val = (tick.size[1] + self.ws_ticks_ax) * tick.on
            return np.ceil(val)

        if self.tick_labels_major_x2.size[1] > self.tick_labels_minor_x2.size[1]:
            tick = self.tick_labels_major_x2
        else:
            tick = self.tick_labels_minor_x2

        return (tick.size[1] + tick.edge_width + self.ws_ticks_ax) * tick.on

    @property
    def _tick_y(self) -> float:
        """Width of the primary y ticks and whitespace."""
        if self.tick_labels_major_y.size[0] > self.tick_labels_minor_y.size[0]:
            tick = self.tick_labels_major_y
        else:
            tick = self.tick_labels_minor_y
        return (tick.size[0] + tick.edge_width + self.ws_ticks_ax) * tick.on

    @property
    def _tick_y2(self) -> float:
        """Width of the secondary y ticks and whitespace."""
        if self.tick_labels_major_y2.size[0] > self.tick_labels_minor_y2.size[0]:
            tick = self.tick_labels_major_y2
        else:
            tick = self.tick_labels_minor_y2

        return (tick.size[0] + tick.edge_width + self.ws_ticks_ax) * tick.on

    @property
    def _top(self) -> int:
        """
        Space above the top of the axes. Round up fractional pixels to match figure output.
        """
        val = np.ceil(self._ws_title) \
            + np.ceil(self.title_wrap.size[1] + 2 * self.title_wrap.edge_width * self.title_wrap.on) \
            + np.ceil(self.label_wrap.size[1] + 2 * self.label_wrap.edge_width * self.label_wrap.on) \
            + np.ceil(self.label_col.size[1] + 2 * self.label_col.edge_width * self.label_col.on) \
            + np.ceil(self.ws_label_col * self.label_col.on) \
            + np.ceil(self._labtick_x2)

        tick_top_xs = 0
        if np.ceil(self.tick_y_top_xs + self.tick_labels_major_y.padding) >= val and self.tick_labels_major_y.on:
            tick_top_xs = np.ceil(self.tick_y_top_xs + 2 * self.tick_labels_major_y.padding - val)

        if np.ceil(self.tick_z_top_xs + self.tick_labels_major_z.padding >= val) and self.tick_labels_major_z.on:
            tick_top_xs = max(tick_top_xs, np.ceil(self.tick_z_top_xs + 2 * self.tick_labels_major_z.padding - val))
        val += tick_top_xs

        val += np.ceil(self.pie.xs_top)

        if self.gantt.on and self.gantt.date_location == 'bottom' and self.gantt.today.on:
            val += np.ceil(self.gantt.today.size[1] + self.gantt.today.edge_width)

        return int(val)

    @property
    def _ws_title(self) -> float:
        """Get ws in the title region depending on title visibility."""
        if self.title.on:
            val = self.ws_fig_title + self.title.size[1] + self.ws_title_ax
        else:
            val = self.ws_fig_ax

        return val

    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
            data: fcp Data object
        """
        num_cols = len(data.changes.columns)

        # Set up the label/title arrays to reflect the groups
        max_labels = int(data.changes.sum().max())
        self.box_group_label.obj[ir, ic] = np.array([[None] * max_labels] * num_cols)
        self.box_group_label.obj_bg[ir, ic] = np.array([[None] * max_labels] * num_cols)
        self.box_group_title.obj[ir, ic] = np.array([[None]] * num_cols)
        self.box_group_title.obj_bg[ir, ic] = np.array([[None]] * num_cols)

        # Create the labels
        for ii in range(0, num_cols):
            k = num_cols - 1 - ii
            sub = data.changes[num_cols - 1 - ii][data.changes[num_cols - 1 - ii] == 1]
            if len(sub) == 0:
                sub = data.changes[num_cols - 1 - ii]

            # Group labels
            if self.box_group_label.on:
                # This array structure just makes one big list of all the labels
                # can we use it with multiple groups or do we need to reshape??
                # Probably need a 2D but that will mess up size_all indexing
                for jj in range(0, len(sub)):
                    # set the width now since it is a factor of the axis size
                    if jj == len(sub) - 1:
                        width = len(data.changes) - sub.index[jj]
                    else:
                        width = sub.index[jj + 1] - sub.index[jj]
                    width = width * (self.axes.size[0]) / len(data.changes)
                    label = data.indices.loc[sub.index[jj], num_cols - 1 - ii]
                    self.box_group_label.size = [np.ceil(width), self.box_group_label.height]
                    self.box_group_label.obj[ir, ic][ii, jj], self.box_group_label.obj_bg[ir, ic][ii, jj] = \
                        self.add_label(ir, ic, self.box_group_label, label)
                    self.box_group_label.obj[ir, ic][ii, jj].width = sub.index[jj] / len(data.changes)
                    self.box_group_label.obj_bg[ir, ic][ii, jj].width = sub.index[jj] / len(data.changes)

            # Group titles
            if self.box_group_title.on and ic == data.ncol - 1:
                self.box_group_title.obj[ir, ic][ii, 0], self.box_group_title.obj_bg[ir, ic][ii, 0] = \
                    self.add_label(ir, ic, self.box_group_title, data.groups[k])

    def add_cbar(self, ax: mplp.Axes, contour: 'MPL_Contour_Plot_Object') -> 'MPL_Colorbar_Object':  # noqa: F821
        """Add a color bar.

        Args:
            ax: current axes object
            contour: current contour plot obj

        Returns:
            reference to the colorbar object
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        self.cbar.divider = make_axes_locatable(ax)
        size = f'{100 * (self.cbar.size[0] / self.axes.size[0])}%'
        if self.axes.edge_width > 1:
            ew = self.axes.edge_width / 2
        else:
            ew = 1
        pad = (self.ws_ax_cbar + ew) / 100
        self.cbar.cax = self.cbar.divider.append_axes('right', size=size, pad=pad)

        # Add the colorbar
        cbar = mplp.colorbar(contour, cax=self.cbar.cax)
        cbar.outline.set_edgecolor(self.cbar.edge_color[0])
        cbar.outline.set_linewidth(self.cbar.edge_width)

        # Style tick labels
        self._check_font(getattr(self, 'tick_labels_major_z').font)
        ticks_font = \
            font_manager.FontProperties(family=getattr(self, 'tick_labels_major_z').font,
                                        size=getattr(self, 'tick_labels_major_z').font_size,
                                        style=getattr(self, 'tick_labels_major_z').font_style,
                                        weight=getattr(self, 'tick_labels_major_z').font_weight)

        for text in self.cbar.cax.get_yticklabels():
            if getattr(self, 'tick_labels_major_z').rotation != 0:
                text.set_rotation(getattr(self, 'tick_labels_major_z').rotation)
            text.set_fontproperties(ticks_font)
            text.set_bbox(dict(edgecolor=getattr(self, 'tick_labels_major_z').edge_color[0],
                               facecolor=getattr(self, 'tick_labels_major_z').fill_color[0],
                               linewidth=getattr(self, 'tick_labels_major_z').edge_width))

        return cbar

    def add_fills(self, ir: int, ic: int, df: pd.DataFrame, data: 'Data'):  # noqa: F821
        """Add rectangular fills to the plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data
            data: fcp Data object

        """
        if not self.fills.on:
            return

        for ii in range(0, len(self.fills.x0)):
            # Skip subplots if rows and cols defined
            if not ((self.fills.rows[ii] is None or self.fills.rows[ii] == ir)
                    and (self.fills.cols[ii] is None or self.fills.cols[ii] == ic)):
                continue

            # Get the axis
            ax = self.axes.obj[ir, ic]

            # Get the x and y locations
            x = [self.fills.x0[ii] if self.fills.x0[ii] is not None else data.ranges['xmin'][ir, ic],
                 self.fills.x1[ii] if self.fills.x1[ii] is not None else data.ranges['xmax'][ir, ic]]
            y0 = self.fills.y0[ii] if self.fills.y0[ii] is not None else data.ranges['ymin'][ir, ic]
            y1 = self.fills.y1[ii] if self.fills.y1[ii] is not None else data.ranges['ymax'][ir, ic]

            # Make the fill
            fill = ax.fill_between(x, y0, y1,
                                   facecolor=self.fills.colors[ii],
                                   edgecolor=self.fills.colors[ii],
                                   alpha=self.fills.alpha[ii],
                                   label=self.fills.labels[ii]
                                   )

            if self.fills.labels[ii] is not None:
                self.legend.add_value(self.fills.labels[ii], [fill], 'fill')

    def add_hvlines(self, ir: int, ic: int, df: [pd.DataFrame, None] = None,
                    elements=['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']):
        """Add horizontal/vertical lines.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data. Defaults to None.
            elements: names of the elements to plot
        """
        # Set default line attributes
        for axline in elements:
            ll = getattr(self, axline)
            ax = self.axes2 if axline[0:3] == 'ax2' else self.axes
            if ax.obj[ir, ic] is None:
                continue
            func = self.axes.obj[ir, ic].axhline if 'hline' in axline \
                else self.axes.obj[ir, ic].axvline
            if ll.on:
                lines = []
                if hasattr(ll, 'by_plot') and ll.by_plot:
                    num_plots = self.axes.obj.size
                    num_lines = len(ll.values)
                    # crude assumption that you have the same number for each plot; fix later
                    lines_per_plot = int(num_lines / num_plots)
                    plot_num = utl.plot_num(ir, ic, self.ncol) - 1
                    vals = range(plot_num * lines_per_plot, plot_num * lines_per_plot + lines_per_plot)
                    for ival in vals:
                        if ival < len(ll.values):
                            line = func(ll.values[ival],
                                        color=ll.color[ival],
                                        linestyle=ll.style[ival],
                                        linewidth=ll.width[ival],
                                        zorder=ll.zorder)
                            lines += [line]
                            if isinstance(ll.text, list) and ll.text[ival] is not None:
                                self.legend.add_value(ll.text[ival], [line], 'ref_line')

                else:
                    for ival, val in enumerate(ll.values):
                        if isinstance(val, str) and isinstance(df, pd.DataFrame):
                            val = df[val].iloc[0]
                        line = func(val,
                                    color=ll.color[ival],
                                    linestyle=ll.style[ival],
                                    linewidth=ll.width[ival],
                                    zorder=ll.zorder)
                        lines += [line]
                        if isinstance(ll.text, list) and ll.text[ival] is not None:
                            self.legend.add_value(ll.text[ival], [line], 'ref_line')

                ll.obj[ir, ic] = lines

    def add_label(self, ir: int, ic: int, label: Element,
                  text_str: Union[None, str] = None, max_dimension = None) -> ['Text_Object']:  # noqa
        """
        Add a label to the plot with figure coordinates.

        Args:
            ir: subplot row index
            ic: subplot column index
            label:  fcp Element object
            text_str: override for Element.text value
            max_dimension: maximum dimension for scaling font size

        Keyword Args:
            dict of value overrides

        Returns:
            reference to the text box object
        """
        # Label background is either a bbox patch of the Text class or a separate rect patch
        if hasattr(label, 'edge_style') and label.edge_style is not None:
            boxstyle = dict(edgecolor=label.edge_color[0], facecolor=label.fill_color[0],
                            linewidth=label.edge_width_adj, alpha=label.fill_alpha,
                            boxstyle=f'{label.edge_style.lower()},pad={label.padding}')
            rect = None
        else:
            boxstyle = None
            rect = patches.Rectangle((0, 0), label.size[0] / self.fig.size[0], label.size[1] / self.fig.size[1],
                                     fill=True, transform=self.fig.obj.transFigure,
                                     facecolor=label.fill_color[0], edgecolor=label.edge_color[0],
                                     lw=label.edge_width_adj, clip_on=False, zorder=2)
            self.axes.obj[ir, ic].add_patch(rect)

        # Create the label text
        if not text_str:
            text_str = label.text
        self._check_font(label.font)
        text = self.axes.obj[ir, ic].text(0, 0, text_str, bbox=boxstyle, transform=self.fig.obj.transFigure,
                                          horizontalalignment='center', verticalalignment='center',
                                          rotation=label.rotation, color=label.font_color, fontname=label.font,
                                          style=label.font_style, weight=label.font_weight, size=label.font_size)

        # Optionally shrink font size for long labels (rotation = 0 or rotation = 90 only)
        # Maybe turn this off for axes labels??
        dont_scale = ['title']
        if self.scale_font_size and text.get_rotation() in [0, 90] and label.name not in dont_scale:
            if text.get_rotation() == 0:
                ax = 0
                param = 'width'
            elif text.get_rotation() == 90:
                ax = 1
                param = 'height'
            if label.name in ['title_wrap', 'title']:
                max_dimension = self.axes.size[ax] * self.ncol
            elif max_dimension is None:
                max_dimension = self.axes.size[ax]
            text_dimension = getattr(text.get_window_extent(), param)
            if text_dimension > max_dimension:
                scale_factor = (max_dimension - 4) / text_dimension  # 2 pixel padding each side
                if '.' in label.name:
                    names = label.name.split('.')
                    text.set_fontsize(scale_factor * getattr(getattr(self, f'{names[0]}'), f'{names[1]}').font_size)
                else:
                    text.set_fontsize(scale_factor * getattr(self, f'{label.name}').font_size)

        # Auto-adjust the background height to match oversized text unless background size explicitly defined in kwargs
        if f'{label.name}_size' not in self.kwargs \
                and text is not None and rect is not None \
                and text.get_window_extent().height > rect.get_window_extent().height:
            height = text.get_window_extent().height + 4  # 2 pixels
            rect.set_height(height / self.fig.size[1])
            if '.' in label.name:
                names = label.name.split('.')
                getattr(getattr(self, f'{names[0]}'), f'{names[1]}').size[1] = height
            else:
                label.size[1] = height

        return text, rect

    def add_legend(self, leg_vals: pd.DataFrame):
        """Add a legend to a figure.

        Args:
            data.legend_vals, used to ensure proper sorting
        """
        # TODO: add separate_label support to have a unique legend per subplot

        def format_legend(self, leg: mpl.legend.Legend):
            """Format the legend object based on the Legend Element object attributes.

            Args:
                leg: legend object
            """
            for itext, text in enumerate(leg.get_texts()):
                text.set_color(self.legend.font_color)
                text.set_fontsize(self.legend.font_size)
                if self.name not in ['hist', 'bar', 'pie', 'gantt']:
                    if hasattr(leg, 'legendHandles'):
                        leg_handle = leg.legendHandles[itext]
                    else:
                        leg_handle = leg.legend_handles[itext]
                    if isinstance(leg_handle, mpl.patches.Rectangle):
                        continue
                    # Set legend point color and alpha
                    leg_handle._sizes = \
                        np.ones(len(self.legend.values) + 1) * self.legend.marker_size**2
                    if not self.markers.on and self.legend.marker_alpha is not None:
                        if hasattr(leg_handle, '_legmarker'):
                            leg_handle._legmarker.set_alpha(self.legend.marker_alpha)
                        else:
                            # required for mpl 3.5
                            alpha = str(hex(int(self.legend.marker_alpha * 255)))[-2:].replace('x', '0')
                            base_color = self.markers.edge_color[itext][0:7] + alpha
                            leg_handle._markeredgecolor = base_color
                            leg_handle._markerfillcolor = base_color
                    elif self.legend.marker_alpha is not None:
                        leg_handle.set_alpha(self.legend.marker_alpha)

            font = self.legend.font if self.legend.font != 'sans-serif' else 'sans'
            try:
                leg.get_title().set_font(font)
            except AttributeError:
                # Needed for earlier versions of mpl
                leg.get_title().set_fontfamily(font)
            leg.get_title().set_fontsize(self.legend.title.font_size)
            leg.get_title().set_color(self.legend.title.font_color)
            leg.get_title().set_style(self.legend.title.font_style)
            leg.get_title().set_weight(self.legend.title.font_weight)
            leg.get_frame().set_facecolor(self.legend.fill_color[0])
            leg.get_frame().set_alpha(self.legend.fill_alpha)
            leg.get_frame().set_edgecolor(self.legend.edge_color[0])
            leg.get_frame().set_linewidth(self.legend.edge_width_adj)

        if self.legend.on and len(self.legend.values) > 0:

            # Remove nan values
            if 'NaN' in self.legend.values['Key'].values:
                self.legend.del_value('NaN')

            # Ensure sort order matches order from data class (may be different depending on the order they are
            # added to the plot, like in the case of multiple subplots where the first plot is missing the first
            # sorted value in data.legend_vals)
            if leg_vals is not None \
                    and len(self.legend.values) == len(leg_vals) \
                    and isinstance(self.legend.values, pd.DataFrame) \
                    and isinstance(leg_vals, pd.DataFrame) \
                    and not np.array_equal(self.legend.values['Key'].values, leg_vals['names'].values):
                leg_vals['names'] = leg_vals['names'].map(str)
                leg_vals = self.legend.values.set_index('Key').loc[leg_vals['names'].values].reset_index()
            else:
                leg_vals = self.legend.values

            # Set the font properties
            fontp = {}
            fontp['family'] = self.legend.font
            fontp['size'] = self.legend.font_size
            fontp['style'] = self.legend.font_style
            fontp['weight'] = self.legend.font_weight

            keys = list(leg_vals['Key'])
            lines = list(leg_vals['Curve'])

            if self.legend.location == 0:
                self.legend.obj = \
                    self.fig.obj.legend(lines, keys, loc='upper right',
                                        title=self.legend.text if self.legend is not True else '',
                                        bbox_to_anchor=(self.legend.position[1],
                                                        self.legend.position[2]),
                                        numpoints=self.legend.points,
                                        prop=fontp,
                                        scatterpoints=self.legend.points)
                format_legend(self, self.legend.obj)
            elif self.legend.location == 11:
                self.legend.obj = \
                    self.fig.obj.legend(lines, keys, loc='lower center',
                                        title=self.legend.text if self.legend is not True else '',
                                        bbox_to_anchor=(self.legend.position[0],
                                                        self.legend.position[2]),
                                        numpoints=self.legend.points,
                                        prop=fontp,
                                        scatterpoints=self.legend.points)
                format_legend(self, self.legend.obj)
            else:
                for irow, row in enumerate(self.axes.obj):
                    for icol, col in enumerate(row):
                        if self.legend.nleg == 1 and not (irow == 0 and icol == self.ncol - 1):
                            continue
                        self.legend.obj = \
                            col.legend(lines, keys, loc=self.legend.location,
                                       title=self.legend.text if self.legend is not True else '',
                                       numpoints=self.legend.points,
                                       prop=fontp,
                                       scatterpoints=self.legend.points)
                        self.legend.obj.set_zorder(102)
                        format_legend(self, self.legend.obj)

    def add_text(self, ir: int, ic: int, text: [str, None] = None,
                 element: Union[str, None, 'layout.Element'] = None,  # noqa: F821
                 coord: Union['mpl_coordinate', None] = None,  # noqa: F821
                 position: Union[None, List[List[float]], List[float]] = None,
                 offsetx: int = 0, offsety: int = 0,
                 units: [str, None] = None, track_element: bool = True, **kwargs):
        """Add a text box.

        Args:
            ir: subplot row index
            ic: subplot column index
            text (optional): text str to add. Defaults to None.
            element (optional): name of or reference to an Element object. Defaults to None.
            coord (optional): MPL coordinate type. Defaults to None.
            position: position of the text box. Defaults to None.
            units (optional): pixel or inches. Defaults to None which is 'pixel'.
            track_element (optional): keep a reference to the text object to move later

        Returns:
            if not track_element, a single text object or a list of text objects
        """
        # Shortcuts
        ax = self.axes.obj[ir, ic]
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1

        # Determine what element is being used
        if element is None:
            # If no element name or object is provided, assume we are using the layout.text element
            el = self.text
        elif isinstance(element, str):
            # Get element by name
            el = getattr(self, element)
        else:
            # Element object passed
            el = element

        # Determine what text to use
        if text is not None:
            # Text can be a string or list, but make sure this reference is a list
            text = utl.validate_list(text)
        elif hasattr(el, 'text') and hasattr(el.text, 'values'):
            # Check if text is in a RepeatedList within the "text" attribute
            text = el.text.values
        else:
            # Text attribute is not a RepeatedList
            text = utl.validate_list(el.text)

        # Set the coordinate transform
        if not coord:
            coord = None if not hasattr(el, 'coordinate') else el.coordinate.lower()
        if coord == 'figure':
            transform = self.fig.obj.transFigure
        elif coord == 'data':
            transform = ax.transData
        else:
            transform = ax.transAxes

        # Add each text box
        text_objs = []
        for itext, txt in enumerate(text):
            if isinstance(txt, dict):
                if plot_num in txt.keys():
                    txt = txt[plot_num]
                else:
                    continue

            # Get/set position (this may be updated in set_figure_final_layout)
            if isinstance(position, list) and (isinstance(position[0], list) or isinstance(position[0], tuple)):
                pos = position[itext]
            elif position is not None:
                pos = position
            elif hasattr(el, 'position') and str(type(el.position)) == str(RepeatedList):
                pos = copy.copy(el.position[itext])
            elif hasattr(el, 'position'):
                pos = copy.copy(el.position)
            else:
                pos = [0.01, 0]  # 0.01 prevents a weird bug

            # Set the units
            if not units:
                units = 'pixel' if not hasattr(el, 'units') else el.units
            if units == 'pixel' and coord == 'figure':
                if isinstance(pos[0], str):
                    pos[0] = 0.01
                else:
                    pos[0] /= self.fig.size[0]
                offsetx /= self.fig.size[0]
                if isinstance(pos[1], str):
                    pos[1] = 0
                else:
                    pos[1] /= self.fig.size[1]
                offsety /= self.fig.size[1]
            elif units == 'pixel' and coord != 'data':
                if isinstance(pos[0], str):
                    pos[0] = 0.01
                else:  # can have some equations that evaluated in final figure layout
                    pos[0] /= self.axes.size[0]
                offsetx /= self.axes.size[0]
                if isinstance(pos[1], str):
                    pos[1] = 0
                else:
                    pos[1] /= self.axes.size[1]
                offsety /= self.axes.size[1]
            if isinstance(pos[0], datetime.datetime) and isinstance(offsetx, int):
                offsetx = datetime.timedelta(offsetx)

            # Set style attributes
            kw = {}
            attrs = ['rotation', 'font_color', 'font', 'fill_color', 'edge_color', 'font_style', 'font_weight',
                     'font_size', 'padding']
            for attr in attrs:
                if attr in kwargs.keys():
                    kw[attr] = kwargs[attr]
                elif hasattr(el, attr) and isinstance(getattr(el, attr), RepeatedList):
                    kw[attr] = getattr(el, attr)[itext]
                elif hasattr(el, attr) and str(type(getattr(el, attr))) == str(RepeatedList):
                    # isinstance fails on python3.6, so hack this way
                    kw[attr] = getattr(el, attr)[itext]
                elif hasattr(el, attr):
                    kw[attr] = getattr(el, attr)

            # Replace newlines with \n
            txt = txt.replace('\\n', '\n')

            # Make the text and store in a temp variable
            text_objs += [ax.text(pos[0] + offsetx,
                                  pos[1] + offsety,
                                  txt,
                                  transform=transform,
                                  rotation=kw['rotation'],
                                  color=kw['font_color'],
                                  fontname=kw['font'],
                                  style=kw['font_style'],
                                  weight=kw['font_weight'],
                                  size=kw['font_size'],
                                  bbox=dict(facecolor=kw['fill_color'],
                                            edgecolor=kw['edge_color'],
                                            pad=kw['padding'],
                                            ),
                                  zorder=45)]

        # Handle result
        if not track_element:
            if len(text_objs) == 1:
                return text_objs[0]
            return text_objs

        # Update element object with the text object
        if el.obj is None:
            # Element does not yet have an object attribute
            el.obj = text_objs
        elif isinstance(el.obj, mpl.text.Text):
            # Element has an object attribute but it is a singular text object
            el.obj = utl.validate_list(el.obj)
            el.obj += text_objs
        elif isinstance(el.obj, list):
            # Element has an object attribute for common for all subplots
            el.obj = utl.validate_list(el.obj)
            el.obj += text_objs
        elif isinstance(el.obj, np.ndarray) and isinstance(el.obj[ir, ic], mpl.text.Text):
            # Element has an object attribute for each subplot and the current subplot has a text object in it
            el.obj[ir, ic] = utl.validate_list(el.obj)
            el.obj[ir, ic] += text_objs
        elif el.obj[ir, ic] is None:
            # Element has an object attribute for each subplot and the current subplot has no text object in it
            el.obj[ir, ic] = text_objs
        else:
            el.obj[ir, ic] += text_objs

    def _check_font(self, font):
        """Check if font exists and warn once if not."""
        if f'font.{font}' in mpl.rcParams:
            fonts = mpl.rcParams[f'font.{font}']
        else:
            fonts = [font]

        if len(set(fonts) & set(FONT_LIST)) == 0:
            if len(set(fonts) & set(self.font_warning)) == 0:
                print('Font Warning!  Requested fonts are not installed; using defaults')
            self.font_warning += fonts

    def close(self):
        """Close an inline plot window."""
        mplp.close('all')

    def fill_between_lines(self, ir: int, ic: int, iline: int, x: [np.ndarray, pd.Index],
                           lcl: [np.ndarray, pd.Series], ucl: [np.ndarray, pd.Series], element: str,
                           leg_name: [str, None] = None, twin: bool = False):
        """Shade a region between two curves.

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            x: x-axis values
            lcl: y-axis values for the lower bound of the fill
            ucl: y-axis values for the upper bound of the fill
            element: name of the Element object associated with this fill
            leg_name (optional): legend value name if legend enabled.
                Defaults to None.
            twin (optional): denotes if twin axis is enabled or not.
                Defaults to False.
        """
        if twin:
            ax = self.axes2.obj[ir, ic]
        else:
            ax = self.axes.obj[ir, ic]
        element = getattr(self, element)
        fc = element.fill_color
        ec = element.edge_color
        fill = ax.fill_between(x, lcl, ucl,
                               facecolor=fc[iline] if str(type(fc)) == str(RepeatedList) else fc,
                               edgecolor=ec[iline] if str(type(ec)) == str(RepeatedList) else ec,
                               linestyle=element.edge_style,
                               linewidth=element.edge_width,
                               label='hi')

        # Add a reference to the line to self.lines
        if leg_name is not None:
            self.legend.add_value(leg_name, fill, 'fill')

    def _edge_offset(self, label: str) -> float:
        """Get the offset between the axes edge and a label's edge widths and account for aliasing weirdness."""
        edge_diff = (getattr(self, label).edge_width - self.axes.edge_width) / 2
        if edge_diff < 0:
            sign = -1
        else:
            sign = 1
        return sign * np.ceil(np.abs(edge_diff))

    def _edge_width(self, element_name: str, up=True) -> int:
        """
        Compute the effective axes edge width in terms of its impact on element spacing (1/2 of the width
        rounded either up or down depending on which side of the object.

        Args:
            element_name: the Element class name to reference the correct object
            up:  if True, np.ceil; if False, np.floor

        Returns:
            effective edge width
        """
        val = {0: 0, 1: 1}.get(getattr(self, element_name), getattr(self, element_name).edge_width / 2)

        if up:
            return int(np.ceil(val))
        else:
            return int(np.floor(val))

    def _get_axes_label_position(self):
        """
        Get the position of the axes label background rectangles.

        self.label_@.position --> [left, right, top, bottom]
        """
        # x-label
        self.label_x.position[0] = self.axes.obj[0, 0].get_position().x0 + self.axes.size[0] / 2 / self.fig.size[0]
        self.label_x.position[3] = (self.label_x.size[1] / 2
                                    + self.ws_fig_label
                                    + (1 if self.label_x.edge_width % 2 == 1 else 0)) / self.fig.size[1]

        # x2-label
        self.label_x2.position[0] = self.label_x.position[0]
        self.label_x2.position[3] = \
            1 \
            - (self.label_x2.size[1] / 2
               + self.ws_fig_label
               + (1 if self.label_y.edge_width % 2 == 1 else 0)) \
            / self.fig.size[1]

        # y-label
        title_xs = self._left - np.ceil(self.ws_fig_label) - np.ceil(self._labtick_y)
        self.label_y.position[0] = (self.label_y.size[0] / 2
                                    + self.ws_fig_label
                                    + title_xs
                                    - (1 if self.label_y.edge_width % 2 == 1 else 0)) / self.fig.size[0]
        self.label_y.position[3] = self.axes.obj[0, 0].get_position().y0 + self.axes.size[1] / 2 / self.fig.size[1]

        # y2-label
        self.label_y2.position[0] = \
            1 \
            - (self.label_y2.size[0] / 2
               + (self.ws_fig_label if self._legx == 0 else 0)
               + self._legx
               + self._row_label_width
               + (1 if self.label_y.edge_width % 2 == 1 else 0)) \
            / self.fig.size[0]
        self.label_y2.position[3] = self.label_y.position[3]

        # z-label
        self.label_z.position[0] = \
            1 \
            - (self.label_z.size[0] / 2
               + self.ws_fig_label
               - (1 if self.label_z.edge_width % 2 == 1 else 0)
               + (self.label_row.size[0] + self.ws_label_row) * self.label_row.on) \
            / self.fig.size[0]
        self.label_z.position[3] = self.axes.obj[0, 0].get_position().y0 + self.axes.size[1] / 2 / self.fig.size[1]

    @property
    def _box_label_heights(self):
        """Calculate the box label height."""
        lab = self.box_group_label
        labt = self.box_group_title
        if len(lab.size_all) == 0:
            return np.array(0)

        # Determine the box group label row heights and account for edge overlaps
        heights = lab.size_all_bg.groupby('ii').max()['height']  # contains edge width

        # Determine the box group title heights
        heightst = labt.size_all_bg.groupby('ii').max()['height']  # contains edge width

        # Get the largest of labels and titles
        return np.maximum(heights, heightst)

    def _get_element_sizes(self, data: 'Data'):  # noqa: F821
        """Calculate the actual rendered size of select elements by pre-plotting
        them.  This is needed to correctly adjust the figure dimensions.

        Args:
            data: fcp Data object

        Returns:
            updated version of `data`
        """
        # Render the figure to extract true dimensions of various elements
        self.fig.obj.canvas.draw()

        # labels
        for label in data.axs:
            lab = getattr(self, f'label_{label}')
            if not lab.on or np.all(lab.obj is None):
                continue
            for ir, ic in np.ndindex(lab.obj.shape):
                if lab.obj[ir, ic] is None:
                    continue

                # text label size
                bbox = lab.obj[ir, ic]._bbox_patch.get_window_extent()
                width = bbox.width + lab.edge_width
                height = bbox.height + lab.edge_width
                lab.size_all = (ir, ic, 0, 0, width, height, bbox.x0, bbox.x1, bbox.y0, bbox.y1, np.nan)

            lab.size = lab.size_all.width.max(), lab.size_all.height.max()

        # titles
        if self.title.on:
            self.title.size = self.title.obj.get_window_extent().width, self.title.obj.get_window_extent().height
        if self.title_wrap.on:
            # this prevents title_wrap font from being bigger than the rectangle but needs work!
            bbox = self.title_wrap.obj.get_window_extent()
            height = bbox.height

        # legend
        if self.legend.on and self.legend.location in [0, 11]:
            self.legend.size = \
                [self.legend.obj.get_window_extent().width - 1e-12 + np.ceil(self.legend.edge_width / 2),
                 self.legend.obj.get_window_extent().height - 1e-12 + np.ceil(self.legend.edge_width / 2)]

        # tick labels
        self._get_tick_label_sizes()

        # box labels and titles
        if self.box_group_label.on:
            lab = self.box_group_label
            lens_all = pd.DataFrame(columns=['ir', 'ic', 'ii', 'vals'])
            for ir, ic in np.ndindex(lab.obj.shape):
                divider = self.axes.size[0] / len(data.changes)
                if lab.obj[ir, ic] is None:
                    continue
                ii_last = -1
                for ii, jj in np.ndindex(lab.obj[ir, ic].shape):
                    # get the available length for a label in units of box groups
                    # changes = data.changes[len(data.changes.columns) - ii - 1].append(
                    #     pd.Series([1], index=[len(data.changes)]))
                    if ii != ii_last:
                        changes = pd.concat([data.changes[len(data.changes.columns) - ii - 1],
                                            pd.Series([1], index=[len(data.changes)])])
                        lens = np.diff(changes[changes == 1].index)
                        ii_last = ii
                        if len(lens_all.loc[(lens_all.ir == ir) & (lens_all.ic == ic) & (lens_all.ii == ii)]) == 0:
                            lens_all = pd.concat(
                                [lens_all, pd.DataFrame({'ir': ir, 'ic': ic, 'ii': ii, 'vals': [lens]}, index=[0])])

                    # skip any nones
                    if lab.obj[ir, ic][ii, jj] is None:
                        continue

                    # get the label dimensions and the max size available for this label
                    bbox = lab.obj[ir, ic][ii, jj].get_window_extent()
                    bbox_obj = lab.obj_bg[ir, ic][ii, jj].get_window_extent()
                    if ii < len(lens):
                        label_max_width = lens[jj] * divider
                    else:
                        label_max_width = bbox.width + 1

                    # rotate labels that are longer than the box axis size
                    if bbox.width > label_max_width \
                            and not (self.box_scale == 'auto') \
                            and utl.kwget(self.kwargs, self.fcpp, 'box_group_label_rotation', None) is None:
                        lab.obj[ir, ic][ii, jj].set_rotation(90)
                        bbox = lab.obj[ir, ic][ii, jj].get_window_extent()

                    # Account for edge width
                    width = bbox_obj.width + 2 * np.floor(lab.edge_width)
                    height = bbox_obj.height + 2 * np.floor(lab.edge_width)

                    # update the size_all DataFrame
                    lab.size_all = (ir, ic, ii, jj, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1,
                                    lab.obj[ir, ic][ii, jj].get_rotation())
                    lab.size_all_bg = (ir, ic, ii, jj, width, height, bbox.x0, bbox.x1, bbox.y0, bbox.y1, np.nan)

            # Adjust background height for any rotated labels
            resize = lab.size_all.groupby(['ir', 'ic', 'ii']).max()['height'] \
                > lab.size_all_bg.groupby(['ir', 'ic', 'ii']).max()['height']
            if len(resize) > 0:
                bg = lab.size_all_bg.set_index(['ir', 'ic', 'ii'])
                for irow, row in resize.items():
                    if not row:
                        continue
                    new_height = lab.size_all.set_index(['ir', 'ic', 'ii']).loc[irow, 'height'].max()
                    bg.loc[irow, 'height'] = new_height + 2 * np.floor(lab.edge_width) + 12  # padding
                lab._size_all_bg = bg.reset_index()

            # Auto resizing via ax_size = 'auto'
            if self.box_scale == 'auto':
                # Get the max label for each grouping label row
                maxes = lab.size_all.groupby(['ir', 'ic', 'ii']).max()
                margin = 4 + self.box_group_label.edge_width

                # Compare every two rows to make sure widths line up
                chg = data.changes.copy()
                cols = chg.columns
                chg = chg[list(reversed([f for f in chg.columns]))]
                chg.columns = cols
                wchg = (maxes.loc[(ir, ic), 'width'].values + margin) * chg  # 4 is for margins

                num_label_rows = len(wchg.columns)
                for irow in range(0, num_label_rows - 1):
                    # Get two adjacent label rows
                    row_top = num_label_rows - irow - 2
                    row_bot = num_label_rows - irow - 1
                    sub = wchg[[row_top, row_bot]].copy()

                    # Set float dtype
                    sub[row_top] = sub[row_top].astype(np.float32)
                    sub[row_bot] = sub[row_bot].astype(np.float32)

                    # Determine the if row_bot labels can fit in row_top width
                    sub['group'] = sub[row_bot].ne(sub[row_bot].shift()).cumsum()
                    sub.loc[sub['group'] % 2 == 1, 'group'] += 1
                    sub['cumsum'] = sub.groupby('group')[row_top].cumsum()
                    row_bot_sum = sub.loc[sub[row_bot] != 0, row_bot].values
                    row_top_sum = sub.loc[sub.group % 2 == 0].groupby('group')['cumsum'].max().values

                    # Calculate the required bottom row width to fit the top row data
                    if len(row_bot_sum) == len(row_top_sum) and any(row_bot_sum > row_top_sum):
                        bot_row_width_max = np.maximum(row_top_sum, row_bot_sum).sum() / len(row_bot_sum)
                        maxes.loc[(ir, ic, num_label_rows - irow - 1), 'width'] = bot_row_width_max

                # Calculate that axes width
                size0 = (maxes['jj'] + 1) * maxes['width'] + divider
                self.axes.size[0] = size0.max() + margin * maxes['jj'].max()

                # Reset the horizontal label widths and rotated box group label heights
                bg = lab.size_all_bg.set_index(['ir', 'ic', 'ii'])
                for ir, ic in np.ndindex(lab.obj.shape):
                    if self.label_y.obj_bg[ir, ic] is not None:
                        self.label_y.obj_bg[ir, ic].set_width(
                            self.label_y.obj_bg[ir, ic].get_window_extent().width / self.axes.size[0])
                    if self.label_row.obj_bg[ir, ic]:
                        self.label_row.obj_bg[ir, ic].set_width(
                            self.label_row.obj_bg[ir, ic].get_window_extent().width / self.axes.size[0])
                    for ii, jj in np.ndindex(lab.obj[ir, ic].shape):
                        row_height = lab.size_all_bg.groupby(['ir', 'ic', 'ii']).max()['height'].loc[ir, ic, ii].max()
                        if lab.obj_bg[ir, ic][ii, jj] is not None:
                            bg.loc[(ir, ic, ii), 'height'] = row_height
                lab._size_all_bg = bg.reset_index()

        if self.box_group_title.on:
            lab = self.box_group_title
            sz = self.box_group_label.size_all_bg
            for ir, ic in np.ndindex(lab.obj.shape):
                group_label = self.box_group_label.size_all_bg.set_index(['ir', 'ic', 'ii'])
                if lab.obj[ir, ic] is None:
                    continue
                for ii in range(0, len(lab.obj[ir, ic])):
                    # text label size
                    if lab.obj[ir, ic][ii, 0] is None:
                        continue
                    bbox = lab.obj[ir, ic][ii, 0].get_window_extent()
                    lab.size_all = (ir, ic, ii, 0, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1, np.nan)

                    # text label rect background size
                    bbox = lab.obj_bg[ir, ic][ii, 0].get_window_extent()
                    lab.size_all_bg = (ir, ic, ii, 0, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1,
                                       np.nan)

                    # if box_group_title height is larger than box_group_label height, resize box_group_labels
                    if bbox.height > group_label.loc[ir, ic, ii].height.mean():
                        group_label.loc[(ir, ic, ii), 'height'] = bbox.height
                        sz.loc[(sz.ir == ir) & (sz.ic == ic) & (sz.ii == ii), 'height'] = bbox.height

            width = lab.size_all.width.max()
            height = lab.size_all.height.max()
            width_bg = lab.size_all_bg.width.max()
            height_bg = lab.size_all_bg.height.max()
            lab.size = [max(width, width_bg), max(height, height_bg)]

        # pie labels
        if self.pie.on:
            for ir, ic in np.ndindex(lab.obj.shape):
                bboxes = [f.get_window_extent() for f in self.pie.obj[1]]
                ax_bbox = self.axes.obj[ir, ic].get_window_extent()
                for ibox, bbox in enumerate(bboxes):
                    if self.pie.obj[1][ibox].get_text() == '':
                        continue
                    self.pie.size_all = (ir, ic, ibox, 0, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1,
                                         np.nan)

                if len(self.pie.size_all) > 0:
                    left = self.pie.size_all['x0'].min() - ax_bbox.x0
                    self.pie.xs_left = max(-left if left < 0 else 0, self.pie.xs_left)

                    right = self.pie.size_all['x1'].max() - ax_bbox.x1
                    self.pie.xs_right = max(right if right > 0 else 0, self.pie.xs_right)

                    bottom = self.pie.size_all['y0'].min() - ax_bbox.y0
                    self.pie.xs_bottom = max(-bottom if bottom < 0 else 0, self.pie.xs_bottom)

                    top = self.pie.size_all['y1'].max() - ax_bbox.y1
                    self.pie.xs_top = max(top if top > 0 else 0, self.pie.xs_top)

            # # someday move labels to edge and draw a line from wedges to label
            # theta1 = self.pie.obj[0][0].theta1
            # theta2 = self.pie.obj[0][0].theta2
            # r = self.pie.obj[0][0].r
            # delta = (theta2 - theta1) / 2
            # x1, y1 = -1., 1.
            # x2 = -0.95 * r * np.sin(np.pi / 180 * delta)
            # y2 = 0.95 * r * np.cos(np.pi / 180 * delta)
            # self.axes.obj[0, 0].plot([x1, x2], [y1, y2], ".")
            # self.axes.obj[0, 0].annotate("",
            #             xy=(x1, y1), xycoords='data',
            #             xytext=(x2, y2), textcoords='data',
            #             arrowprops=dict(arrowstyle="-", color="0.5",
            #                             shrinkA=5, shrinkB=5,
            #                             patchA=None, patchB=None,
            #                             connectionstyle="arc,angleA=-90,angleB=0,armA=0,armB=40,rad=0",
            #                             ),
            #             )

        # Gantt today labels
        if self.gantt.on and self.gantt.today.on:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                if self.gantt.today.obj[ir, ic] is None:
                    continue
                txt = self.gantt.today.obj[ir, ic][0]._bbox_patch.get_window_extent()
                self.gantt.today.size = txt.width, txt.height

        return data

    def _get_figure_size(self, data: 'Data', temp=False, **kwargs):  # noqa: F821
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            temp: first fig size calc is a dummy calc to get a rough size; don't resize user parameters
            kwargs: user-defined keyword args
        """
        # Set some values for convenience
        self.ws_row = self.ws_row_def
        self.ws_col = self.ws_col_def
        self.ws_ax_leg = max(0, self.ws_ax_leg - self._labtick_y2) if self.legend.location == 0 else 0
        self.ws_leg_fig = self.ws_leg_fig if self.legend.location == 0 else 0
        self.fig_legend_border = self.fig_legend_border if self.legend.location == 0 else 0

        self.box_labels = self._box_label_heights.sum()
        self.box_labels -= self.box_group_label.edge_width * self.box_group_label.size_all['ii'].max() \
            if len(self.box_group_label.size_all) > 0 else 0
        if self.axes.edge_width == self.box_group_label.edge_width \
                and self.axes.edge_color[0] == self.box_group_label.edge_color[0] \
                and self.box_group_label.on:
            self.box_labels -= 1
        self.box_labels = np.round(self.box_labels)

        self.box_title = 0
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            self.box_title = self.box_group_title.size[0] + self.ws_ax_box_title
        elif self.box_group_title.on and self.box_group_title.size != [0, 0] and \
                self.box_group_title.size[0] > self.legend.size[0]:
            self.box_title = self.box_group_title.size[0] - self.legend.size[0]  # + self.ws_ax_box_title

        # Adjust the column and row whitespace
        if self.cbar.on and utl.kwget(kwargs, self.fcpp, 'ws_col', -999) == -999 and not self.cbar.shared:
            self.ws_col = 0

        if self.nrow == 1:
            self.ws_row = 0
        if self.ncol == 1:
            self.ws_col = 0

        # separate ticks and labels
        if (self.separate_ticks or self.axes.share_y is False) and not self.cbar.on:
            self.ws_col = max(self._tick_y + self.ws_fig_label, max(self.ws_col, self.ws_col_def))
        elif (self.separate_ticks or self.axes.share_y is False) and self.cbar.on and not temp:
            self.ws_col += self._tick_y
        if self.axes2.on and (self.separate_ticks or self.axes2.share_y is False):
            if self.ws_col < self.ws_col + self._tick_y2 + self.ws_fig_label:
                self.ws_col += self._tick_y2 + self.ws_fig_label

        if self.separate_ticks or (self.axes.share_x is False and self.box.on is False) and not temp:
            self.ws_row = max(self._tick_x + self.ws_fig_label, max(self.ws_row, self.ws_row_def))
        elif self.axes2.on \
                and (self.separate_ticks or self.axes2.share_x is False) \
                and self.box.on is False \
                and not temp:
            if self.ws_row < self.ws_row + self._tick_x2 + self.ws_fig_label:
                self.ws_row += self._tick_x2 + self.ws_fig_label

        if self.separate_labels:
            self.ws_col = \
                max(self._labtick_y - self._tick_y + self._labtick_y2 - self._tick_y2 + self.ws_col, self.ws_col)
            if self.cbar.on and not temp:
                self.ws_col += self.label_z.size[0] / 2
            self.ws_row = \
                max(self._labtick_x - self._tick_x + self._labtick_x2 - self._tick_x2 + self.ws_row, self.ws_row)

        if self.label_wrap.on and 'ws_row' not in kwargs.keys() and self.ws_row == self.ws_row_def:
            self.ws_row += self.label_wrap.size[1] + 2 * self.label_wrap.edge_width - self.ws_row_def
        elif self.label_wrap.on and 'ws_row' not in kwargs.keys():
            self.ws_row += self.label_wrap.size[1] + 2 * self.label_wrap.edge_width
        if self.box_group_label.on and 'ws_row' not in kwargs.keys():
            self.ws_row += self.box_labels
            if self.label_wrap.on:
                self.ws_row += self.label_wrap.size[1] + 2 * self.label_wrap.edge_width - self.ws_row_def
        elif not temp and self.name == 'box':
            self.ws_row += self.box_labels - self.ws_row_def

        self.ws_col = np.ceil(self.ws_col)  # round up to nearest whole pixel
        self.ws_row = np.ceil(self.ws_row)  # round up to nearest whole pixel

        if self.name == 'heatmap' and self.heatmap.cell_size is not None and data.num_x is not None:
            self.axes.size = [self.heatmap.cell_size * data.num_x, self.heatmap.cell_size * data.num_y]
            self.label_col.size[0] = self.axes.size[0]
            self.label_row.size[1] = self.axes.size[1]
            self.label_wrap.size[1] = self.axes.size[0]

        # imshow ax adjustment
        if self.name == 'imshow' and getattr(data, 'wh_ratio'):
            if data.wh_ratio >= 1:
                self.axes.size[1] = np.round(self.axes.size[0] / data.wh_ratio)
                self.label_row.size[1] = self.axes.size[1]
            else:
                self.axes.size[0] = np.round(self.axes.size[1] * data.wh_ratio)
                self.label_col.size[0] = self.axes.size[0]
                self.label_wrap.size[0] = self.axes.size[0]

        # Compute the sum of axes edge thicknesses
        if self.ws_col == 0 and not self.cbar.on and self.ncol > 1:
            col_edge_width = self.axes.edge_width * (self.ncol + 1)
        else:
            col_edge_width = 2 * self.axes.edge_width * self.ncol
        if self.ws_row == 0 and not self.label_wrap.on and self.nrow > 1:
            row_edge_height = self.axes.edge_width * (self.nrow + 1)
        else:
            row_edge_height = 2 * self.axes.edge_width * self.nrow

        # Set figure width
        self.fig.size[0] = \
            self._left \
            + self.axes.size[0] * self.ncol \
            + col_edge_width \
            + self._right \
            + self._legx \
            + self.ws_col * (self.ncol - 1) \
            + self._cbar

        # Figure height
        self.fig.size[1] = \
            self._top \
            + self.axes.size[1] * self.nrow \
            + row_edge_height \
            + self.ws_row * (self.nrow - 1) \
            + self._bottom

        # Account for legends longer than the figure
        header = self._ws_title + \
            (self.label_col.size[1] + self.ws_label_col) * self.label_col.on + \
            self.title_wrap.size[1] + self.label_wrap.size[1] + self._labtick_x2

        if self.legend.size[1] + header > self.fig.size[1]:
            self.legend.overflow = self.legend.size[1] + header - (self.fig.size[1] - self.ws_fig_label)
        self.fig.size[1] += self.legend.overflow

    @staticmethod
    def _get_grid_visibility_kwarg(visible: bool) -> Dict:
        """Handle visibility kwarg for different mpl versions (changed at v3.5)

        Args:
            visible: flag to show or hide grid

        Returns:
            single-key dict with the correctly-named bool for showing/hiding grids
        """
        if version.Version(mpl.__version__) < version.Version('3.5'):
            return {'b': visible}
        else:
            return {'visible': visible}

    def _get_legend_position(self):
        """Get legend position."""
        if self.legend.location == 0:
            title_xs = max(0, (self.title.size[0] - self.axes.size[0]) / 2 - self.legend.size[0])
            if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
                self.legend.position[1] = 1 + (self.fig_legend_border - self.ws_leg_fig - title_xs) / self.fig.size[0]
            elif self.box_group_title.on and str(self.legend.column) != 'True':
                self.legend.position[1] = 1 + (self.fig_legend_border - self.ws_leg_fig - self.ws_ax_box_title
                                               - self.box_title + self.ws_ax_fig - title_xs) / self.fig.size[0]
            else:
                self.legend.position[1] = \
                    1 \
                    + (self.fig_legend_border - np.floor(self.legend.edge_width / 2) - self.ws_leg_fig - title_xs) \
                    / self.fig.size_int[0]

            self.legend.position[2] = \
                self.axes.position[2] \
                + (self.fig_legend_border - np.floor(self.legend.edge_width / 2)) / self.fig.size_int[1]

        if self.legend.location == 11:
            self.legend.position[1] = 0.5
            self.legend.position[2] = 0

    def _get_tick_label_size(self, ax: mplp.Axes, tick: str, tick_num: str, which: str, pad: Tuple = (0, 0)):
        """Get the size of the tick labels on a specific axes (plot must be already rendered).

        Args:
            ax: the axes object for the labels of interest
            tick: name of the tick axes
            tick_num: '' or '2' for secondary
            which: 'major' or 'minor'
        """
        if tick == 'x':
            idx = 0
        else:
            idx = 1

        for ir, ic in np.ndindex(ax.obj.shape):
            # Get the tick label Element
            minor = True if which == 'minor' else False
            tt = getattr(self, f'tick_labels_{which}_{tick}{tick_num}')

            if tt.obj is None:
                continue
            if not tt.on:
                return

            # Get the VISIBLE tick labels and add references to the Element object
            if tick == 'z' and hasattr(ax.obj[ir, ic], 'ax'):
                tlabs = getattr(ax.obj[ir, ic].ax, 'get_yticklabels')(minor=minor)
                vmin, vmax = getattr(ax.obj[ir, ic].ax, 'get_ylim')()
            elif tick != 'z':
                tlabs = getattr(ax.obj[ir, ic], f'get_{tick}ticklabels')(minor=minor)
                vmin, vmax = getattr(ax.obj[ir, ic], f'get_{tick}lim')()
            elif tick == 'z':
                continue

            tt.limits[ir, ic] = [vmin, vmax]
            tlabs = [f for f in tlabs if approx_gte(f.get_position()[idx], min(vmin, vmax))
                     and approx_lte(f.get_position()[idx], max(vmin, vmax))]
            tt.obj[ir, ic] = tlabs
            if len(tlabs) == 0:
                continue

            # Get the label sizes (text and background) and store sizes as 2D array
            bboxes = [t.get_window_extent() for t in tlabs]
            bboxes_bkg = []
            for t in tlabs:
                if hasattr(t, '_bbox_patch') and t._bbox_patch is not None:
                    bboxes_bkg += [t._bbox_patch.get_window_extent()]
                else:
                    bboxes_bkg += [t.get_window_extent()]

            iir, iic, ii, jj, width, height, x0, x1, y0, y1, rotation = [], [], [], [], [], [], [], [], [], [], []
            for ib, bbox in enumerate(bboxes):
                delta_width = (bboxes_bkg[ib].width - bbox.width) / 2
                delta_height = (bboxes_bkg[ib].height - bbox.height) / 2
                iir += [ir]
                iic += [ic]
                ii += [ib]
                jj += [-1]
                width += [bboxes_bkg[ib].width + tt.edge_width]
                height += [bboxes_bkg[ib].height + tt.edge_width]
                x0 += [bbox.x0 - delta_width]
                x1 += [bbox.x1 + delta_width]
                y0 += [bbox.y0 - delta_height]
                y1 += [bbox.y1 + delta_height]
                rotation += [tlabs[ib].get_rotation()]
            tt.size_all = (iir, iic, ii, jj, width, height, x0, x1, y0, y1, rotation)

            # Calculate xs label width extending beyond the top of the plot
            if tick == 'y' and ir == 0 and ic == 0 and not self.title.on and not self.label_col.on:
                # Padding for top y-tick label that extends beyond top of axes
                # doesn't capture every possible case yet
                ax_y1 = ax.obj[0, 0].get_window_extent().y1
                self.tick_y_top_xs = max(0, self.tick_y_top_xs, tt.size_all['y1'].max() - ax_y1)

            if tick == 'z' and not self.label_col.on:
                # Padding for top z-tick in cbar
                ax_y1 = self.axes.obj[0, 0].get_window_extent().y1
                self.tick_z_top_xs = tt.size_all['y1'].max() - ax_y1

            # Set the padding from the axes
            if tick != 'z':
                for tlab in tlabs:
                    tlab.set_x(pad[0])
                    tlab.set_y(pad[1])

        if len(tt.size_all) == 0:
            return

        tt.size = [tt.size_all.width.max(), tt.size_all.height.max()]

    def _get_tick_label_sizes(self):
        """Get the tick label sizes for each axis."""

        # primary x-axis
        getattr(self, 'tick_labels_major_x').size_all_reset()
        pad_x_major = 0, -(self.tick_labels_major_x.padding + self.tick_labels_major_x.edge_width + self.ws_ticks_ax
                           + self._edge_width('axes')) / self.axes.size[1]
        self._get_tick_label_size(self.axes, 'x', '', 'major', pad_x_major)

        getattr(self, 'tick_labels_minor_x').size_all_reset()
        pad_x_minor = 0, -(self.tick_labels_minor_x.padding + self.tick_labels_minor_x.edge_width + self.ws_ticks_ax
                           + self._edge_width('axes')) / self.axes.size[1]
        self._get_tick_label_size(self.axes, 'x', '', 'minor', pad_x_minor)

        # secondary x-axis
        if self.axes2.on and self.axes.twin_y:
            getattr(self, 'tick_labels_major_x2').size_all_reset()
            pad_x2_major = 0, -(self.tick_labels_major_x2.padding + self.tick_labels_major_x2.edge_width
                                + self.ws_ticks_ax + self._edge_width('axes')) / self.axes.size[1]
            pad = pad_x2_major[0], 1 - pad_x2_major[1]
            self._get_tick_label_size(self.axes2, 'x', '2', 'major', pad)

            getattr(self, 'tick_labels_minor_x2').size_all_reset()
            pad_x2_minor = 0, -(self.tick_labels_minor_x2.padding + self.tick_labels_minor_x2.edge_width
                                + self.ws_ticks_ax + self._edge_width('axes')) / self.axes.size[1]
            pad = pad_x2_minor[0], 1 - pad_x2_minor[1]
            self._get_tick_label_size(self.axes2, 'x', '2', 'minor', pad)

        # primary y-axis
        getattr(self, 'tick_labels_major_y').size_all_reset()
        pad_y_major = -(self.tick_labels_major_y.padding + self.tick_labels_major_y.edge_width + self.ws_ticks_ax
                        + self._edge_width('axes')) / self.axes.size[0], 0
        self._get_tick_label_size(self.axes, 'y', '', 'major', pad_y_major)

        getattr(self, 'tick_labels_minor_y').size_all_reset()
        pad_y_minor = -(self.tick_labels_minor_y.padding + self.tick_labels_minor_y.edge_width + self.ws_ticks_ax
                        + self._edge_width('axes')) / self.axes.size[0], 0
        self._get_tick_label_size(self.axes, 'y', '', 'minor', pad_y_minor)

        # secondary y-axis
        if self.axes2.on and self.axes.twin_x:
            getattr(self, 'tick_labels_major_y2').size_all_reset()
            pad_y2_major = -(self.tick_labels_major_y2.padding + self.tick_labels_major_y2.edge_width
                             + self.ws_ticks_ax + self._edge_width('axes')) / self.axes.size[0], 0
            pad = 1 - pad_y2_major[0], pad_y2_major[1]
            self._get_tick_label_size(self.axes2, 'y', '2', 'major', pad)

            getattr(self, 'tick_labels_minor_y2').size_all_reset()
            pad_y2_minor = -(self.tick_labels_minor_y2.padding + self.tick_labels_minor_y2.edge_width + self.ws_ticks_ax
                             + self._edge_width('axes')) / self.axes.size[0], 0
            pad = 1 - pad_y2_minor[0], pad_y2_minor[1]
            self._get_tick_label_size(self.axes2, 'y', '2', 'minor', pad)

        # z-axis (major only)
        if self.tick_labels_major_z.on:
            getattr(self, 'tick_labels_major_z').size_all_reset()
            self._get_tick_label_size(self.cbar, 'z', '', 'major')

    def _get_tick_overlaps(self, axis: str = ''):
        """Deal with overlapping labels and out of range ticks.

        Args:
            axis: str name of an axis to check for overlaps (Ex: 'x', 'y', etc)

        """
        if not self.tick_cleanup:
            return

        if not getattr(self, f'axes{axis}').on:
            return

        # define some shortcuts
        xticks = getattr(self, f'tick_labels_major_x{axis}')
        yticks = getattr(self, f'tick_labels_major_y{axis}')
        xticksm = getattr(self, f'tick_labels_minor_x{axis}')
        yticksm = getattr(self, f'tick_labels_minor_y{axis}')
        sf = yticks.scale_factor  # scale factor for tick font size

        for ir, ic in np.ndindex(self.axes.obj.shape):
            # size_all by idx: ir, ic, width, height, x0, x1, y0, y1
            # major
            if len(xticks.size_all) > 0:
                xticks_size_all = xticks.size_all[(xticks.size_all.ir == ir) & (xticks.size_all.ic == ic)]
                xticks_size_all = np.array(xticks_size_all)
            else:
                xticks_size_all = []
            if len(yticks.size_all) > 0:
                yticks_size_all = yticks.size_all[(yticks.size_all.ir == ir) & (yticks.size_all.ic == ic)]
                yticks_size_all = np.array(yticks_size_all)
            else:
                yticks_size_all = []

            # minor
            if len(xticksm.size_all) > 0:
                xticksm_size_all = xticksm.size_all[(xticksm.size_all.ir == ir) & (xticksm.size_all.ic == ic)]
                xticksm_size_all = np.array(xticksm_size_all)
            else:
                xticksm_size_all = []
            if len(yticksm.size_all) > 0:
                yticksm_size_all = yticksm.size_all[(yticksm.size_all.ir == ir) & (yticksm.size_all.ic == ic)]
                yticksm_size_all = np.array(yticksm_size_all)
            else:
                yticksm_size_all = []

            # Shrink/remove overlapping ticks x-y origin
            if len(xticks_size_all) > 0 and len(yticks_size_all) > 0:
                # If tick label edges are disabled, be a bit generous on the overlap
                if self.tick_labels_major_x.edge_width > 0 or self.tick_labels_major_y.edge_width > 0:
                    xs = 0
                else:
                    xs = 2  # this might not be enough to do anything
                idx = xticks.size_cols.index('width')
                xw, xh, xx0, xx1, xy0, xy1 = xticks_size_all[0][idx:-1]
                xx0 += xs
                xx1 -= xs
                xc = (xx0 + (xx1 - xx0) / 2, xy0 + (xy0 - xy1) / 2)
                yw, yh, yx0, yx1, yy0, yy1 = yticks_size_all[0][idx:-1]
                yy0 += xs
                yy1 -= xs
                yc = (yx0 + (yx1 - yx0) / 2, yy0 + (yy0 - yy1) / 2)
                if utl.rectangle_overlap((xw, xh, xc), (yw, yh, yc)):
                    # Because these are vertically offset we don't need to be as strict on overlap so use a scale factor
                    skew = 1.4
                    if self.tick_cleanup == 'shrink' and \
                            not utl.rectangle_overlap((xw / sf / skew, xh, xc), (yw / sf / skew, yh, yc)):
                        xticks.obj[ir, ic][0].set_size(xticks.font_size / sf)
                        yticks.obj[ir, ic][0].set_size(yticks.font_size / sf)
                    else:
                        # If self.tick_cleanup == 'remove' and the resized tick label would still overlap, remove it
                        yticks.obj[ir, ic][0].set_visible(False)

            # Shrink/remove overlapping ticks in column/wrap plots at x-origin
            if ic > 0 and len(xticks_size_all) > 0:
                xxticks = xticks.size_all.set_index(['ir', 'ic', 'ii'])
                if (ir, ic - 1) in xxticks.index:
                    _, xwl, xhl, x0l, x1l, y0l, y1l, _ = xxticks.loc[ir, ic - 1].iloc[-1].values
                    xcl = (x0l + (x1l - x0l) / 2, y0l + (y0l - y1l) / 2)
                    _, xwf, xhf, x0f, x1f, y0f, y1f, _ = xxticks.loc[ir, ic].iloc[0].values
                    xcf = (x0f + (x1f - x0f) / 2, y0f + (y0f - y1f) / 2)
                    if xcl[0] == xcf[0]:
                        # if ticks are on the identical x-axis location, force remove
                        xticks.obj[ir, ic - 1][-1].set_visible(False)
                    if utl.rectangle_overlap((xwl, xhl, xcl), (xwf, xhf, xcf)):
                        if self.tick_cleanup == 'shrink' and \
                                not utl.rectangle_overlap((xwl / sf, xhl, xcl), (xwf, xhf, xcf)):
                            xticks.obj[ir, ic - 1][-1].set_size(xticks.font_size / sf)
                        else:
                            # If self.tick_cleanup == 'remove' and the resized tick label would still overlap, remove it
                            xticks.obj[ir, ic - 1][-1].set_visible(False)

            # First and last x ticks that may fall under a wrap label
            if len(xticks_size_all) > 0 and ir != self.nrow - 1:
                xxticks = xticks.size_all.set_index(['ir', 'ic', 'ii'])
                if ic > 0 and self.label_wrap.on and len(xticks.obj[ir, ic]) > 0:
                    x0_left_edge = xxticks.loc[ir, ic, 0]['x0'] - xxticks.loc[ir, ic, 0]['width'] / 2
                    prev_ax_right_edge = self.axes.obj[ir, ic - 1].get_position().x1 * self.fig.size_int[0]
                    if prev_ax_right_edge > x0_left_edge:
                        xticks.obj[ir, ic][0].set_visible(False)

                if ic < self.ncol - 1 and self.label_wrap.on and len(xticks.obj[ir, ic]) > 0:
                    x1_right_edge = xxticks.loc[ir, ic].iloc[-1]['x1'] + xxticks.loc[ir, ic].iloc[-1]['width'] / 2
                    next_ax_left_edge = self.axes.obj[ir, ic + 1].get_position().x0 * self.fig.size_int[0] + self.ws_col
                    if x1_right_edge > next_ax_left_edge:
                        xticks.obj[ir, ic][-1].set_visible(False)

            # TODO: Shrink/remove overlapping ticks in grid plots at y-origin

            # Remove overlapping ticks on same axis
            if len(xticks_size_all) > 0:
                xbbox = df_tick(xticks, xticks_size_all, 'x')
                xbbox = hide_overlaps(xticks, xbbox, ir, ic)
            if len(xticksm_size_all) > 0:
                xbboxm = df_tick_update(df_tick(xticksm, xticksm_size_all, 'x'))
                xbboxm['visible'] = False
                xbboxm = select_minor_ticks(xbbox, xbboxm)
                for irow, row, in xbboxm.iterrows():
                    xticksm.obj[ir, ic][irow].set_visible(row.visible)

            # Leave overlappint yticks for gantt workstreams
            if len(yticks_size_all) > 0 and \
                    not (self.gantt.on and self.gantt.workstreams.on and self.gantt.labels_as_yticks):
                ybbox = df_tick(yticks, yticks_size_all, 'y')
                ybbox = hide_overlaps(yticks, ybbox, ir, ic)
            if len(yticksm_size_all) > 0:
                ybboxm = df_tick_update(df_tick(yticksm, yticksm_size_all, 'y'))
                ybboxm['visible'] = False
                ybboxm = select_minor_ticks(ybbox, ybboxm)
                for irow, row, in ybboxm.iterrows():
                    yticksm.obj[ir, ic][irow].set_visible(row.visible)

    def _get_tick_xs(self):
        """Calculate extra whitespace at the edge of the plot for the last tick."""
        xticks = self.tick_labels_major_x
        yticks = self.tick_labels_major_y

        # TODO:: minor overlaps
        # TODO:: self.axes2

        for ir, ic in np.ndindex(self.axes.obj.shape):
            # size_all by idx:
            #   ir, ic, width, height, x0, x1, y0, y1
            if len(xticks.size_all) > 0:
                xticks_size_all = xticks.size_all[(xticks.size_all.ir == ir) & (xticks.size_all.ic == ic)]
                xticks_size_all = np.array(xticks_size_all)
            else:
                xticks_size_all = []
            if len(yticks.size_all) > 0:
                yticks_size_all = yticks.size_all[(yticks.size_all.ir == ir) & (yticks.size_all.ic == ic)]
                yticks_size_all = np.array(yticks_size_all)
            else:
                yticks_size_all = []

            if len(xticks_size_all) > 0:
                if xticks.rotation == 90:
                    idx = xticks.size_cols.index('y1')
                elif xticks.rotation == 270:
                    idx = xticks.size_cols.index('y0')
                else:
                    idx = xticks.size_cols.index('x1')
                xxs = self.axes.obj[ir, ic].get_window_extent().x1 + self._right + self.legend.size[0] \
                    - xticks_size_all[-1][idx]
                # this may fail with long legend
                if xxs < 0 and not self.gantt.on:
                    self.x_tick_xs = -int(np.floor(xxs)) + 1

            if len(yticks_size_all) > 0:
                if yticks.rotation == 90:
                    idx = xticks.size_cols.index('x1')
                elif yticks.rotation == 270:
                    idx = xticks.size_cols.index('x0')
                else:
                    idx = xticks.size_cols.index('y0')
                yxs = self.axes.obj[ir, ic].get_window_extent().y1 + self._top - yticks_size_all[-1][idx]
                if yxs < 0:
                    self.y_tick_xs = -int(np.floor(yxs)) + 1  # not currently used

            # Prevent single tick label axis by adding a text label at one or more range limits
            if len(xticks_size_all) <= 1 \
                    and self.name not in ['box', 'bar', 'pie'] \
                    and (not self.axes.share_x or len(self.axes.obj.flatten()) == 1) \
                    and self.tick_labels_major_x.on:
                kw = {}
                kw['rotation'] = xticks.rotation
                kw['font_color'] = xticks.font_color
                kw['font'] = xticks.font
                kw['fill_color'] = xticks.fill_color[0]
                kw['edge_color'] = xticks.edge_color[0]
                kw['font_style'] = xticks.font_style
                kw['font_weight'] = xticks.font_weight
                kw['font_size'] = xticks.font_size / xticks.scale_factor
                if len(xticks_size_all) == 0:
                    # NEEDS WORK and may need similar for y-axis?
                    # this case can only happen if the xlimits are so tight there are no gridlines present
                    precision = utl.get_decimals(xticks.limits[ir, ic][0], 8)
                    txt = f'{xticks.limits[ir, ic][0]:.{precision}f}'
                    x = -kw['font_size']
                    y = -kw['font_size'] - self.ws_ticks_ax
                    position = [x, y]
                    self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', track_element=False,
                                  position=position, **kw)
                    x += self.axes.size[0]
                else:
                    x = self.axes.size[0] - xticks.size_all.loc[0, 'width'] / 2
                    y = - yticks.size_all.loc[0, 'height'] / xticks.scale_factor - self.tick_labels_major_x.padding
                precision = utl.get_decimals(xticks.limits[ir, ic][1], 8)
                txt = f'{xticks.limits[ir, ic][1]:.{precision}f}'
                kw['position'] = [x, y]
                tick = \
                    self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', track_element=False, **kw)
                tick_x1 = tick.get_window_extent().x1
                xxs = self.axes.obj[ir, ic].get_window_extent().x1 - tick_x1
                if xxs < 0 and (not self.legend.on or (self.legend.on and self.legend.size[1] > self.axes.size[1])):
                    self.x_tick_xs = -int(np.floor(xxs)) + 1

            if len(yticks_size_all) <= 1 \
                    and self.name not in ['box', 'bar', 'pie', 'gantt'] \
                    and (not self.axes.share_y or len(self.axes.obj.flatten()) == 1) \
                    and yticks.limits[ir, ic]:
                kw = {}
                kw['rotation'] = yticks.rotation
                kw['font_color'] = yticks.font_color
                kw['font'] = yticks.font
                kw['fill_color'] = yticks.fill_color[0]
                kw['edge_color'] = yticks.edge_color[0]
                kw['font_style'] = yticks.font_style
                kw['font_weight'] = yticks.font_weight
                kw['font_size'] = yticks.font_size / yticks.scale_factor
                if len(yticks_size_all) == 0:
                    # this case can only happen if the ylimits are so tight that there are no gridlines present
                    precision = utl.get_decimals(yticks.limits[ir, ic][0], 8)
                    txt = f'{yticks.limits[ir, ic][0]:.{precision}f}'
                    x = -kw['font_size'] * len(txt.replace('.', ''))
                    y = -kw['font_size'] / 2
                    kw['position'] = [x, y]
                    tick = self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel',
                                         track_element=False, **kw)
                    y += self.axes.size[1]
                    limit = 1
                else:
                    # first figure out where the single tick is
                    pos = (yticks.size_all.iloc[0].y1 - yticks.size_all.iloc[0].y0) / 2 + yticks.size_all.iloc[0].y0
                    x = -yticks.size_all.loc[0, 'width']
                    if abs(pos - self.axes.size[1]) < pos:
                        # tick is closer to top, add new label to bottom
                        y = 0
                        limit = 0
                    else:
                        # tick is closer to bottom, add new label to top
                        y = self.axes.size[1] - yticks.size_all.loc[0, 'height'] / 2 / yticks.scale_factor
                        limit = 1
                # is this criterion sufficient?
                if yticks.limits[ir, ic][1] - yticks.limits[ir, ic][0] < 1:
                    precision = utl.get_decimals(yticks.limits[ir, ic][1], 8)
                else:
                    precision = utl.get_decimals(yticks.limits[ir, ic][1], 2)
                txt = f'{yticks.limits[ir, ic][limit]:.{precision}f}'
                x = -kw['font_size'] * len(txt.replace('.', ''))
                kw['position'] = [x, y]
                tick = self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', track_element=False,
                                     **kw)
                tick_w = tick.get_window_extent().width
                self.tick_labels_major_y.size[0] = max(self.tick_labels_major_y.size[0], tick_w)

    def _get_title_position(self):
        """Calculate the title position.

        self.title.position --> [left, right, top, bottom]
        """
        col_label = (self.label_col.size[1] + self.ws_label_col * self.label_col.on) \
            + (self.label_wrap.size[1] + self.ws_label_col * self.label_wrap.on)
        cbar = (self.ws_col + self.tick_labels_major_z.size[0] + self.ws_ax_cbar) * self.ncol / self.axes.size[0]
        self.title.position[0] = (self.ncol + cbar * self.cbar.on) / 2
        if self.label_wrap.size[1] > 0:
            title_height = 0
        else:
            title_height = self.title.size[1] / 2 - 2  # can't recall why this offset is needed
        self.title.position[3] = 1 + (self.ws_title_ax
                                      + col_label + title_height
                                      + self.label_wrap.size[1]) / self.axes.size[1]
        self.title.position[2] = \
            self.title.position[3] + cbar * self.cbar.on + (self.ws_title_ax + self.title.size[1]) / self.axes.size[1]

    def make_figure(self, data: 'Data', **kwargs):  # noqa: F821
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        # Create the subplots
        #   Note we don't have the actual element sizes until rendereing
        #   so we use an approximate size based upon what is known
        self._get_figure_size(data, temp=True, **kwargs)
        self.fig.obj, self.axes.obj = mplp.subplots(data.nrow, data.ncol,
                                                    figsize=[self.fig.size_inches[0], self.fig.size_inches[1]],
                                                    sharex=self.axes.share_x,
                                                    sharey=self.axes.share_y,
                                                    dpi=self.fig.dpi,
                                                    facecolor=self.fig.fill_color[0],
                                                    edgecolor=self.fig.edge_color[0],
                                                    linewidth=self.fig.edge_width,
                                                    )
        self._subplots_adjust_x0y0()

        # Reformat the axes variable if it is only one plot
        if not isinstance(self.axes.obj, np.ndarray):
            self.axes.obj = np.array([self.axes.obj])
        if len(self.axes.obj.shape) == 1:
            if data.nrow == 1:
                self.axes.obj = np.reshape(self.axes.obj, (1, -1))
            else:
                self.axes.obj = np.reshape(self.axes.obj, (-1, 1))

        # Twinning
        self.axes2.obj = self.obj_array
        if self.axes.twin_x:
            for ir, ic in np.ndindex(self.axes2.obj.shape):
                self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twinx()
        elif self.axes.twin_y:
            for ir, ic in np.ndindex(self.axes2.obj.shape):
                self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twiny()

        return data

    def plot_bar(self, ir: int, ic: int, iline: int, df: pd.DataFrame,
                 leg_name: str, data: 'Data', ngroups: int, stacked: bool,  # noqa: F821
                 std: [None, float], xvals: np.ndarray, inst: pd.Series,
                 total: pd.Series) -> 'Data':  # noqa: F821
        """Plot bar graph.

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            df: summed column "y" values grouped by x-column -->
                df.groupby(x).sum()[y]
            leg_name: legend value name if legend enabled
            data: Data object
            ngroups: total number of groups in the full data set based on
                data.get_plot_data
            stacked: enables stacked histograms if True
            std: std dev to create error bars if not None
            xvals: sorted array of x-column unique values
            inst: instance value to get the correct alignment of a group
                in the plot when legending
            total: number of instances of x-column when grouped by the legend

        Returns:
            updated Data Object with new axes ranges
        """
        ax = self.axes.obj[ir, ic]
        idx = np.where(np.isin(xvals, df.index))[0]
        ixvals = list(range(0, len(xvals)))
        kwargs = {}

        # Orientation
        if self.bar.horizontal:
            bar = ax.barh
            axx = 'y'
            if self.bar.stacked:
                kwargs['height'] = self.bar.width
                if iline > 0:
                    if isinstance(stacked, pd.Series):
                        stacked = stacked.loc[xvals[idx]].values
                    kwargs['left'] = stacked
            else:
                kwargs['height'] = self.bar.width / ngroups
                idx = [f + inst[i] * kwargs['height'] for i, f in enumerate(idx)]
                init_off = (total - 1) / 2 * kwargs['height']
                idx = list((idx - init_off).values)
        else:
            bar = ax.bar
            axx = 'x'
            if self.bar.stacked:
                kwargs['width'] = self.bar.width
                if len(stacked) > 0:
                    if isinstance(stacked, pd.Series):
                        stacked = stacked.loc[xvals[idx]].values
                    kwargs['bottom'] = stacked
            else:
                kwargs['width'] = self.bar.width / ngroups
                idx = [f + inst[i] * kwargs['width'] for i, f in enumerate(idx)]
                init_off = (total - 1) / 2 * kwargs['width']
                idx = list((idx - init_off).values)

        if self.bar.color_by == 'bar':
            edgecolor = [self.bar.edge_color[i] for i, f in enumerate(df.index)]
            fillcolor = [self.bar.fill_color[i] for i, f in enumerate(df.index)]
        else:
            edgecolor = self.bar.edge_color[(iline, leg_name)]
            fillcolor = self.bar.fill_color[(iline, leg_name)]

        # Error bars
        if std is not None and self.bar.horizontal:
            kwargs['xerr'] = std
        elif std is not None:
            kwargs['yerr'] = std

        # Plot
        bar(idx, df.values, align=self.bar.align,  linewidth=self.bar.edge_width,
            edgecolor=edgecolor, color=fillcolor, ecolor=self.bar.error_color, **kwargs)

        # Set ticks
        try:
            # Special case of datetime where you don't want to show all tick labels
            df.index.astype('datetime64[ns]')
        except (TypeError, pd._libs.tslibs.parsing.DateParseError):
            # Show all labels
            getattr(ax, f'set_{axx}ticks')(np.arange(0, len(xvals), 1))
        allowed_ticks = getattr(ax, f'get_{axx}ticks')()  # mpl selected ticks
        allowed_ticks = list(set([int(f) for f in allowed_ticks if f >= 0 and f < len(xvals)]))
        getattr(ax, f'set_{axx}ticks')(np.array(ixvals)[allowed_ticks])
        getattr(ax, f'set_{axx}ticklabels')(xvals[allowed_ticks])

        if iline == 0:
            # Update ranges
            new_ticks = getattr(ax, f'get_{axx}ticks')()
            tick_off = [f for f in new_ticks if f >= 0][0]
            if self.bar.horizontal:
                axx = 'y'
                data.ranges['xmin'][ir, ic] = data.ranges['ymin'][ir, ic]
                data.ranges['xmax'][ir, ic] = data.ranges['ymax'][ir, ic]
                data.ranges['ymin'][ir, ic] = None
                data.ranges['ymax'][ir, ic] = None
            else:
                axx = 'x'
            xoff = 3 * self.bar.width / 4
            if data.ranges[f'{axx}min'][ir, ic] is None:
                data.ranges[f'{axx}min'][ir, ic] = -xoff + tick_off
            else:
                data.ranges[f'{axx}min'][ir, ic] += tick_off
            if data.ranges[f'{axx}max'][ir, ic] is None:
                data.ranges[f'{axx}max'][ir, ic] = len(xvals) - 1 + xoff + tick_off
            else:
                data.ranges[f'{axx}max'][ir, ic] += tick_off

        # Legend
        if leg_name is not None:
            handle = [patches.Rectangle((0, 0), 1, 1, color=self.bar.fill_color[(iline, leg_name)])]
            self.legend.add_value(leg_name, handle, 'lines')

        return data

    def plot_box(self, ir: int, ic: int, data: 'Data', **kwargs) -> 'MPL_Boxplot_Object':  # noqa: F821
        """Plot boxplot.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: Data object
            kwargs: keyword args

        Returns:
            box plot MPL object
        """
        bp = None

        if self.violin.on:
            bp = self.axes.obj[ir, ic].violinplot(data,
                                                  showmeans=False,
                                                  showextrema=False,
                                                  showmedians=False,
                                                  )
            for ipatch, patch in enumerate(bp['bodies']):
                patch.set_facecolor(self.violin.fill_color[ipatch])
                patch.set_edgecolor(self.violin.edge_color[ipatch])
                patch.set_alpha(self.violin.fill_alpha)
                patch.set_zorder(2)
                patch.set_lw(self.violin.edge_width)
                if self.violin.box_on:
                    q25 = np.percentile(data[ipatch], 25)
                    med = np.percentile(data[ipatch], 50)
                    q75 = np.percentile(data[ipatch], 75)
                    iqr = q75 - q25
                    offset = 0.05 * len(data) / 7
                    bb = mtransforms.Bbox([[1 + ipatch - offset, q25],
                                           [1 + ipatch + offset, q75]])
                    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                            abs(bb.width), abs(bb.height),
                                            boxstyle="round,pad=0, rounding_size=0.05",
                                            ec="none", fc=self.violin.box_color, zorder=12)
                    self.axes.obj[ir, ic].add_patch(p_bbox)
                    whisker_max = min(max(data[ipatch]), q75 + 1.5 * iqr)
                    whisker_min = max(min(data[ipatch]), q25 - 1.5 * iqr)
                    self.axes.obj[ir, ic].plot([ipatch + 1, ipatch + 1],
                                               [whisker_min, whisker_max],
                                               linestyle=self.violin.whisker_style,
                                               color=self.violin.whisker_color,
                                               linewidth=self.violin.whisker_width)
                    self.axes.obj[ir, ic].plot([ipatch + 1], [med],
                                               marker=self.violin.median_marker,
                                               color=self.violin.median_color,
                                               markersize=self.violin.median_size,
                                               markeredgecolor=self.violin.median_color,
                                               zorder=13)

        elif self.box.on and not self.violin.on:
            bp = self.axes.obj[ir, ic].boxplot(data,
                                               labels=[''] * len(data),
                                               showfliers=False,
                                               medianprops={'color': self.box.median_color},
                                               notch=self.box.notch,
                                               patch_artist=True,
                                               zorder=3)
            for ipatch, patch in enumerate(bp['boxes']):
                patch.set_edgecolor(self.box.edge_color[ipatch])
                patch.set_facecolor(self.box.fill_color[ipatch])
                patch.set_alpha(self.box.fill_alpha)
                patch.set_lw(self.box.edge_width)
                patch.set_ls(self.box.style[ipatch])
            for ipatch, patch in enumerate(bp['whiskers']):
                patch.set_color(self.box_whisker.color[int(ipatch / 2)])
                patch.set_lw(self.box_whisker.width[ipatch])
                patch.set_ls(self.box_whisker.style[ipatch])
            for ipatch, patch in enumerate(bp['caps']):
                patch.set_color(self.box_whisker.color[int(ipatch / 2)])
                patch.set_lw(self.box_whisker.width[ipatch])
                patch.set_ls(self.box_whisker.style[ipatch])

        ll = ['' for f in self.axes.obj[ir, ic].get_xticklabels()]
        self.axes.obj[ir, ic].set_xticklabels(ll)
        self.axes.obj[ir, ic].set_xlim(0.5, len(data) + 0.5)

        return bp

    def plot_contour(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, z: str, data: 'Data'):  # noqa: F821
        """Plot a contour plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            x: x-axis column name
            y: y-axis column name
            z: z-column name
            data: Data object
        """
        ax = self.axes.obj[ir, ic]

        # Convert data type
        xx = np.array(df[x])
        yy = np.array(df[y])
        zz = np.array(df[z])

        # Make the grid
        xi = np.linspace(min(xx), max(xx))
        yi = np.linspace(min(yy), max(yy))
        zi = scipy.interpolate.griddata((xx, yy), zz, (xi[None, :], yi[:, None]), method=self.contour.interp)

        # Deal with out of range values
        zi = np.clip(zi, data.ranges['zmin'][ir, ic], data.ranges['zmax'][ir, ic])

        # Set the contours
        levels = np.linspace(data.ranges['zmin'][ir, ic] * 0.999, data.ranges['zmax'][ir, ic] * 1.001,
                             self.contour.levels)

        # Plot
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        if self.contour.filled:
            self.contour.obj[ir, ic] = ax.contourf(xi, yi, zi, levels, cmap=self.cmap[plot_num], zorder=2)
        else:
            self.contour.obj[ir, ic] = ax.contour(xi, yi, zi, levels, linewidths=self.contour.width.values,
                                                  cmap=self.cmap[plot_num], zorder=2)

        # Add a colorbar
        if self.cbar.on and (not self.cbar.shared or ic == self.ncol - 1):
            self.cbar.obj[ir, ic] = self.add_cbar(ax, self.contour.obj[ir, ic])
            new_ticks = cbar_ticks(self.cbar.obj[ir, ic], data.ranges['zmin'][ir, ic], data.ranges['zmax'][ir, ic])
            self.cbar.obj[ir, ic].set_ticks(new_ticks)

        # Show points
        if self.contour.show_points:
            ax.scatter(xx, yy, marker=self.markers.type[0],
                       c=self.markers.fill_color[0]
                       if self.markers.filled else 'none',
                       edgecolors=self.markers.edge_color[0],
                       linewidths=self.markers.edge_width[0],
                       s=self.markers.size[0],
                       zorder=40)

    def _pixel_to_mdate(self, ax_width, xmin, xmax, px):
        """
        Convert pixel value to the correspoding matplotlib date value

        Args:
            ax_width: width of the axes in pixels
            xmin: minimum x-axis range value
            xmax: maximum x-axis range value
            px: pixel value

        Returns:
            Converted matplotlib date value
        """
        return px * (xmax - xmin) / (ax_width - px)

    def plot_gantt(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str, y: str,
                   leg_name: str, xvals: npt.NDArray, yvals: list, bar_labels: list, ngroups: int,
                   data: 'Data'):  # noqa: F821
        """Plot gantt graph.

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            df: input data
            x: x-axis column name
            y: y-axis column name
            leg_name: legend value name if legend enabled
            xvals: list of tuples of dates
            yvals: list of tuples of groupling column values
            bar_labels: list of bar labels
            ngroups: total number of groups in the full data set based on data.get_plot_data
        """
        ax = self.axes.obj[ir, ic]
        bar = ax.broken_barh
        new_xmax = mdates.date2num(data.ranges['xmax'][ir, ic])  # current xmax axes range in matplotlib date float

        # Set the color values
        if self.gantt.color_by == 'bar':
            # Every bar gets a different color
            edgecolor = [self.gantt.edge_color[i] for i, f in enumerate(df.index)]
            fillcolor = [self.gantt.fill_color[i] for i, f in enumerate(df.index)]
        elif self.gantt.workstreams.on and self.legend._on and self.gantt.workstreams.column != self.legend.column:
            # Workstreams define the bar positioning but the legend grouping defines the color
            edgecolor = []
            fillcolor = []
            for irow, row in df.iterrows():
                leg_val = row[self.legend.column]
                idx = int(data.legend_vals.loc[data.legend_vals.Leg == leg_val].index[0])
                edgecolor += [self.gantt.edge_color[idx]]
                fillcolor += [self.gantt.fill_color[idx]]
                handle = [patches.Rectangle((0, 0), 1, 1, color=self.gantt.fill_color[idx])]
                self.legend.add_value(leg_val, handle, 'lines')
        else:
            # Use grouping scheme
            edgecolor = [self.gantt.edge_color[(iline, leg_name)] for i, f in enumerate(df.index)]
            fillcolor = [self.gantt.fill_color[(iline, leg_name)] for i, f in enumerate(df.index)]

        # Plot the bars
        for ii, (irow, row) in enumerate(df.iterrows()):
            if leg_name is not None:
                yi = yvals.index((row[y], leg_name))
            else:
                yi = yvals.index((row[y],))

            # Milestone
            if row[x[0]] == row[x[1]]:
                self.gantt.milestone_marker_size = self.axes.size[1] / (len(yvals) - 1) * self.gantt.height / 2 - 2
                ax.plot([row[x[0]]], [yi], self.gantt.milestone_marker, markersize=self.gantt.milestone_marker_size,
                        mfc=fillcolor[ii], mec='#ffffff')

                # Add xs x width if milestone marker is at xmax
                if row[x[0]] == data.ranges['xmax'][ir, ic]:
                    xmin, xmax = ax.get_xlim()
                    w_px = self.gantt.milestone_marker_size
                    w_mdate = self._pixel_to_mdate(self.axes.size[0], xmin, xmax, w_px)
                    new_xmax = max(new_xmax, mdates.date2num(row[x[1]]) + w_mdate)

                # Add milestone text
                if self.gantt.milestone in row and str(row[self.gantt.milestone]) not in \
                        [None, np.nan, 'nan', pd.NaT, 'NaT', 'N/A', 'n/a', 'Nan', 'NAN', 'NaN', '', 'None'] \
                        and self.gantt.milestone_text.on:

                    # Check if milestone fits in the date range
                    milestone_date = row[data.x[0]]
                    if milestone_date >= data.ranges['xmin'][ir, ic] and milestone_date <= data.ranges['xmax'][ir, ic]:
                        # Add text and position to the milestone text element object
                        self.gantt.milestone_text.text += [row[self.gantt.milestone]]
                        self.gantt.milestone_text.position += [(row[x[0]], yi)]

                        # Predict the milestone label size (no current support for multiple font types)
                        txt_size = utl.get_text_dimensions(row[self.gantt.milestone],  # units == pixels
                                                           self.gantt.milestone_text.font,
                                                           self.gantt.milestone_text.font_size,
                                                           self.gantt.milestone_text.font_style,
                                                           self.gantt.milestone_text.font_weight)

                        # Update xmax range value for labels that go beyond the axes range
                        xmin, xmax = ax.get_xlim()
                        if self.gantt.milestone_text.location == 'top':
                            txt_xs_px = txt_size[0] / 2
                            w_px = 0
                        else:
                            txt_xs_px = txt_size[0]
                            w_px = self.gantt.milestone_marker_size  # units == pixels
                        if not self.gantt.auto_expand:
                            txt_xs = self._pixel_to_mdate(self.axes.size[0], xmin, xmax, txt_xs_px + w_px)
                            if (mdates.date2num(row[x[1]]) + txt_xs) > xmax:
                                xmax_xs = (mdates.date2num(row[x[1]]) + txt_xs)
                            else:
                                xmax_xs = 0
                            new_xmax = max(new_xmax, xmax_xs)
                        elif self.gantt.auto_expand and data.xmax[utl.plot_num(ir, ic, self.ncol)] is None:
                            self.axes.size[0] += txt_xs_px
                            txt_xs = self._pixel_to_mdate(self.axes.size[0], xmin, xmax, txt_xs_px)
                            if (mdates.date2num(row[x[0]]) + txt_xs) > xmax:
                                xmax_xs = (mdates.date2num(row[x[1]]) + txt_xs)
                            else:
                                xmax_xs = 0
                            new_xmax = max(new_xmax, xmax_xs)

            # Workstream bracket
            elif self.gantt.workstreams.location == 'inline' and row[self.gantt.workstreams.column] == row[data.y[0]]:
                xmin, xmax = ax.get_xlim()

                # Highlight the workstream title row
                if self.gantt.workstreams.highlight_row:
                    # Because the axes width is indeterminate, we need to use a long bar to ensure the highlight
                    # continues after resizing
                    ax.fill_between([xmin, 100000], yi - 0.5, yi + 0.5,
                                    facecolor=self.gantt.workstreams_title.fill_color[0],
                                    edgecolor=self.gantt.workstreams_title.edge_color[0],
                                    alpha=self.gantt.workstreams_title.fill_alpha)
                    ax.set_xlim((xmin, xmax))

                # Draw the workstream brackets (not yet customizable)
                if self.gantt.workstreams.brackets:
                    align = (xmax - xmin) / self.axes.size[0]  # 1 pixel shift for better alignment (sometimes)
                    ax.plot([mdates.date2num(row[x[0]]) + align, mdates.date2num(row[x[0]]) + align],
                            [yi - 0.3, yi + 0.1], linewidth=1.7, color=edgecolor[ii])
                    ax.plot([mdates.date2num(row[x[0]]) + align, mdates.date2num(row[x[1]]) - align],
                            [yi + 0.1, yi + 0.1], linewidth=1.7, color=edgecolor[ii])
                    ax.plot([mdates.date2num(row[x[1]]) - align, mdates.date2num(row[x[1]]) - align],
                            [yi + 0.1, yi - 0.3], linewidth=1.7, color=edgecolor[ii])

            # Placeholder only (for wrap index or share_y)
            elif str(row[x[1]]) == 'NaT':
                continue

            # Bar
            else:
                bar([(row[x[0]], row[x[1]] - row[x[0]])],
                    (yi - self.gantt.height / 2, self.gantt.height),
                    facecolor=fillcolor[ii],
                    edgecolor=edgecolor[ii],
                    linewidth=self.gantt.edge_width)

        # Convert the numeric tick labels into task labels
        if iline + 1 == ngroups and self.gantt.labels_as_yticks:
            yvals = [f[0] for f in yvals]
            ax.set_yticks(range(-1, len(yvals)))
            if self.gantt.workstreams.on and self.gantt.workstreams.location == 'inline':
                # indent non-workstream items
                inline_workstreams = data.df_rc[data.workstreams].unique()
                yvals = [f'  - {yval}' if yval not in inline_workstreams else yval for yval in yvals]
            ax.set_yticklabels([''] + list(yvals))

        # Add bar labels strings to the right of the bars
        if iline + 1 == ngroups and self.gantt.bar_labels is not None:
            self.gantt.bar_labels.text = bar_labels
            for i in range(len(yvals)):
                position = [(xvals[i][1], i) for i in range(len(yvals))]

            # Filter out any labels that are outside of the range (xmax handled below)
            for ipos, pos in enumerate(position):
                if data.ranges['xmin'][ir, ic] is not None and pos[0] < data.ranges['xmin'][ir, ic]:
                    self.gantt.bar_labels.text[ipos] = ''

            # Add the labels
            self.add_text(ir, ic, element=self.gantt.bar_labels, position=position,
                          offsetx=0, offsety=0)
            if not self.gantt.labels_as_yticks:
                ax.set_yticks(range(-1, len(yvals)))
                ax.set_yticklabels([''] + ['' for f in yvals])

            # Update xmax range value for labels that go beyond the axes range
            txt_xs_px = 0  # units == pixel
            xmin = mdates.date2num(data.ranges['xmin'][ir, ic])
            xmax = mdates.date2num(data.ranges['xmax'][ir, ic])
            for itxt, txt in enumerate(self.gantt.bar_labels.obj[ir, ic]):
                txt_size = [self.gantt.bar_labels.obj[ir, ic][itxt].get_window_extent().width,
                            self.gantt.bar_labels.obj[ir, ic][itxt].get_window_extent().height]

                # Tweak position
                pos = txt.get_position()
                pixel_2_mdate = (xmax - xmin) / self.axes.size[0]
                xoffset = 4 * pixel_2_mdate
                yoffset = txt_size[1] / 4 / self.axes.size[1] * ax.get_ylim()[1]
                if xvals[itxt][1] == xvals[itxt][0]:
                    # Discrete milestone
                    xoffset += self.gantt.milestone_marker_size * pixel_2_mdate
                if self.gantt.milestone in data.df_rc.columns and \
                        len(data.df_rc.loc[data.df_rc[data.y[0]] == yvals[itxt][0], self.gantt.milestone].dropna()) > 0:
                    # Milestone on a bar
                    xoffset += self.gantt.milestone_marker_size * pixel_2_mdate
                txt.set_position((mdates.num2date(mdates.date2num(pos[0]) + xoffset), pos[1] - yoffset))

                # Compute how much label exceeds the axes range
                x, y = txt.get_position()
                x = mdates.date2num(x)
                loc = ax.transData.transform((x, y))
                txt_xs_px = max(txt_xs_px, loc[0] + txt_size[0] - self.axes.obj[ir, ic].get_window_extent().width)

            # Adjust ranges to avoid cutting off labels
            if txt_xs_px > 0 and data.xmax[utl.plot_num(ir, ic, self.ncol)] is None:
                xmin, xmax = ax.get_xlim()
                if not self.gantt.auto_expand:
                    txt_xs = self._pixel_to_mdate(self.axes.size[0], xmin, xmax, txt_xs_px)
                    new_xmax = max(new_xmax, xmax + txt_xs)
                elif self.gantt.auto_expand and data.xmax[utl.plot_num(ir, ic, self.ncol)] is None:
                    self.axes.size[0] += txt_xs_px
                    txt_xs = self._pixel_to_mdate(self.axes.size[0], xmin, xmax, txt_xs_px)
                    new_xmax = max(new_xmax, xmax + txt_xs)

        # Add the milestone labels
        if iline + 1 == ngroups and self.gantt.milestone_text.on and len(self.gantt.milestone_text.text) > 0:
            # Add the labels
            self.gantt.milestone_text.obj[ir, ic] = []
            self.add_text(ir, ic, element=self.gantt.milestone_text, position=self.gantt.milestone_text.position,
                          offsetx=np.timedelta64(datetime.timedelta(days=1), 'D'), offsety=0)

        # Legend
        if leg_name is not None:
            if self.gantt.workstreams.on and self.legend._on:
                # Handled above
                pass
            else:
                handle = [patches.Rectangle((0, 0), 1, 1, color=self.gantt.fill_color[(iline, leg_name)])]
                self.legend.add_value(leg_name, handle, 'lines')

        # Boxes around the labels need left justification
        if self.gantt.label_boxes:
            mplp.setp(self.axes.obj[ir, ic].get_yticklabels(), ha='left')

        return mdates.num2date(new_xmax)

    def plot_gantt_dependencies(self, ir, ic, start, end, min_collision, max_collision,
                                color='gray', linewidth=1, zorder=1,
                                start_offset=5, end_offset=10,
                                is_milestone_start=False, is_milestone_end=False, repeat_dep=False):
        """
        Plot a dependency arrow between two points in a Matplotlib Gantt chart using right angles.

        Args:
            ir: subplot row index
            ic: subplot column index
            start: tuple of (x, y) for start point (end of predecessor task)
            end: tuple of (x, y) for end point (start of dependent task)
            min_collision: earliest date in the range of the connection line to avoid drawing over the bar
            max_collision: latest date in the range of the connection line to avoid drawing over the bar
            color: color of the arrow
            linewidth: width of the arrow line
            zorder: drawing order (higher numbers draw on top)
            start_offset: length of initial horizontal segment in pixels
            end_offset: length of final horizontal segment in pixels
            is_milestone_start: whether the start point is a milestone
            is_milestone_end: whether the end point is a milestone
            repeat_dep: flag for deps that have already been processed to avoid double margin spacing
        """
        ax = self.axes.obj[ir, ic]
        xmin, xmax = ax.get_xlim()
        pixel_2_mdate = (xmax - xmin) / self.axes.size[0]

        start_x, start_y = start
        end_x, end_y = end
        start_offset *= pixel_2_mdate  # units == mdates float
        end_offset *= pixel_2_mdate  # units == mdates float

        # Convert dates to numbers for comparison
        start_x = mdates.date2num(start_x)
        end_x = mdates.date2num(end_x)
        min_collision = mdates.date2num(min_collision)
        max_collision = mdates.date2num(max_collision)
        x_distance = end_x - start_x  # units == mdates float
        sign = -1 if start_y < end_y else 1

        # If the dependent bar starts before the predecessor ends, loop back to the left
        if start_x > end_x or end_x < max_collision:
            points = [
                [start_x, start_y],
                [start_x + start_offset, start_y],  # Initial horizontal segment
                [start_x + start_offset, start_y - (0.5 - 0.15 * is_milestone_end) * sign],  # Go just below the bar
                [min_collision - end_offset, start_y - (0.5 - 0.15 * is_milestone_end) * sign],  # Circle back
                [min_collision - end_offset, end_y],  # Go vertically to the destination bar
                [end_x, end_y]
            ]

        # If the dependent bar starts after the predecessor ends, go to the new row and draw a line up/down
        else:
            if is_milestone_end:
                start_offset = 0
                yoffset = sign * (0.5 - (1 - self.gantt.height) / 2)
            else:
                yoffset = sign * (0.5 - (1 - self.gantt.height) / 3)  # div by 2 would hit top of bar, give some buffer
            if is_milestone_start:
                xs = self.gantt.milestone_marker_size * pixel_2_mdate
                start_x += xs
                x_distance -= xs
            points = [
                [start_x, start_y],
                [start_x + x_distance + start_offset, start_y],  # Initial horizontal segment
                [start_x + x_distance + start_offset, end_y + yoffset]
            ]
            # Check for bar labels and move them if needed to avoid overlapping the arrow
            if self.gantt.bar_labels is not None and not repeat_dep:
                x, y = self.gantt.bar_labels.obj[ir, ic][start_y].get_position()
                x_new = mdates.date2num(x) + x_distance + start_offset
                if is_milestone_start:
                    x_new += xs
                self.gantt.bar_labels.obj[ir, ic][start_y].set_x(mdates.num2date(x_new))  # from plot_gantt

        # Draw the lines
        for i in range(len(points) - 1):
            conn = ConnectionPatch(
                xyA=points[i],
                xyB=points[i + 1],
                coordsA="data",
                coordsB="data",
                axesA=ax,
                axesB=ax,
                color=color,
                linewidth=linewidth,
                linestyle='-',
                zorder=zorder
            )
            ax.add_artist(conn)

        # Add arrow head at the end
        arrow_head = ConnectionPatch(
            xyA=points[-2],
            xyB=points[-1],
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            color=color,
            linewidth=linewidth,
            linestyle='-',
            zorder=zorder,
            arrowstyle='-|>'
        )
        ax.add_artist(arrow_head)

    def plot_gantt_today(self, ir, ic):
        """
        Add a line indicating today's date (or an alternate date specified by the user)

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        if self.gantt.today.on and self.gantt.today.obj[ir, ic] is None:
            if self.gantt.date_location == 'top':
                ymin = -0.01
                ymax = 1 - self.axes.edge_width / self.axes.size[0]
            else:
                ymin = 0
                ymax = 1 + (self.gantt.today.padding + self.axes.edge_width) / self.axes.size[0]
            self.add_text(ir, ic, self.gantt.today.text, self.gantt.today)
            self.axes.obj[ir, ic].axvline(self.gantt.today.date, color=self.gantt.today.color[0],
                                          linestyle=self.gantt.today.style[0], linewidth=self.gantt.today.width[0],
                                          ymin=ymin, ymax=ymax, clip_on=False, zorder=2000)

    def plot_heatmap(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, z: str,
                     data: 'Data') -> 'MPL_imshow_object':  # noqa: F821
        """Plot a heatmap.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            x: x-axis column name
            y: y-axis column name
            z: z-column name
            data: Data object

        Returns:
            MPL imshow plot obj
        """
        ax = self.axes.obj[ir, ic]

        # Make the heatmap (ranges get updated later)
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        im = ax.imshow(df, self.cmap[plot_num], vmin=data.ranges['zmin'][ir, ic], vmax=data.ranges['zmax'][ir, ic],
                       interpolation=self.heatmap.interp)
        im.set_clim(data.ranges['zmin'][ir, ic], data.ranges['zmax'][ir, ic])

        # Adjust the axes and rc label size based on the number of groups
        cols = len(df.columns)
        rows = len(df)
        map_sq = min(self.axes.size[0] / cols, self.axes.size[1] / rows)
        self.axes.size = [map_sq * cols, map_sq * rows]

        # Set the axes
        dtypes = [int, np.int32, np.int64]
        if df.index.dtype not in dtypes:
            ax.set_yticks(np.arange(len(df)))
            ax.set_yticklabels(df.index)
            ax.set_yticks(np.arange(df.shape[0] + 1) - 0.5, minor=True)
        if df.columns.dtype not in dtypes:
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xticks(np.arange(df.shape[1] + 1) - 0.5, minor=True)
        if df.index.dtype not in dtypes or df.columns.dtype not in dtypes:
            ax.grid(which="minor", color=self.heatmap.edge_color[0],
                    linestyle='-', linewidth=self.heatmap.edge_width)
            ax.tick_params(which="minor", bottom=False, left=False)
        if data.ranges['xmin'][ir, ic] is not None and data.ranges['xmin'][ir, ic] > 0:
            xticks = ax.get_xticks()
            ax.set_xticklabels([int(f + data.ranges['xmin'][ir, ic]) for f in xticks])

        # Add the cbar
        if self.cbar.on and (not self.cbar.shared or ic == self.ncol - 1):
            self.cbar.obj[ir, ic] = self.add_cbar(ax, im)

        # Loop over data dimensions and create text annotations
        if self.heatmap.text:
            for iy, yy in enumerate(df.index):
                for ix, xx in enumerate(df.columns):
                    if type(df.loc[yy, xx]) in [float, np.float32, np.float64] and np.isnan(df.loc[yy, xx]):
                        continue
                    if self.heatmap.rounding is None:
                        val = df.loc[yy, xx]
                    else:
                        val = df.loc[yy, xx].round(self.heatmap.rounding)
                    text = ax.text(ix, iy, val,  # noqa
                                   ha="center", va="center",
                                   color=self.heatmap.font_color,
                                   fontsize=self.heatmap.font_size)

        return im

    def plot_hist(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str,
                  y: str, leg_name: str, data: 'Data') -> ['MPL_histogram_object', 'Data']:  # noqa: F821
        """Plot a histogram.

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            df: summed column "y" values grouped by x-column -->
                df.groupby(x).sum()[y]
            x: x-axis column name
            y: y-axis column name
            leg_name: legend value name if legend enabled
            data: Data object

        Returns:
            MPL histogram plot object
            updated Data object
        """
        hist = self.axes.obj[ir, ic].hist(df[x], bins=self.hist.bins, range=data.branges[ir, ic],
                                          color=self.hist.fill_color[iline],
                                          ec=self.hist.edge_color[iline],
                                          lw=self.hist.edge_width,
                                          zorder=3,
                                          align=self.hist.align,
                                          cumulative=self.hist.cumulative,
                                          density=self.hist.normalize,
                                          rwidth=self.hist.rwidth,
                                          orientation='vertical' if not self.hist.horizontal else 'horizontal',
                                          )

        # Add a reference to the line to self.lines
        if leg_name is not None:
            handle = [patches.Rectangle((0, 0), 1, 1, color=self.hist.fill_color[iline])]
            self.legend.add_value(leg_name, handle, 'lines')

        # Add a kde
        if self.kde.on:
            kde = scipy.stats.gaussian_kde(df[x])
            if not self.hist.horizontal:
                x0 = np.linspace(data.ranges['xmin'][ir, ic], data.ranges['xmax'][ir, ic], 1000)
                y0 = kde(x0)
            else:
                y0 = np.linspace(data.ranges['ymin'][ir, ic], data.ranges['ymax'][ir, ic], 1000)
                x0 = kde(y0)
            kwargs = self.make_kw_dict(self.kde)
            kwargs['color'] = RepeatedList(kwargs['color'][iline], 'color')
            kde = self.plot_line(ir, ic, x0, y0, **kwargs)

        return hist, data

    def plot_imshow(self, ir: int, ic: int, df: pd.DataFrame, data: 'Data'):  # noqa: F821
        """Plot an image.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            data: Data object

        Returns:
            MPL imshow plot obj
        """
        ax = self.axes.obj[ir, ic]
        zmin = data.ranges['zmin'][ir, ic]
        zmax = data.ranges['zmax'][ir, ic]

        # Make the imshow plot
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        im = ax.imshow(df, self.cmap[plot_num], vmin=zmin, vmax=zmax, interpolation=self.imshow.interp, aspect='auto')
        im.set_clim(zmin, zmax)

        # Add a cmap
        if self.cbar.on and (not self.cbar.shared or ic == self.ncol - 1):
            self.cbar.obj[ir, ic] = self.add_cbar(ax, im)

        return im

    def plot_line(self, ir: int, ic: int, x0: float, y0: float, x1: float = None,
                  y1: float = None, **kwargs):
        """Plot a simple line.

        Args:
            ir: subplot row index
            ic: subplot column index
            x0: min x coordinate of line
            x1: max x coordinate of line
            y0: min y coordinate of line
            y1: max y coordinate of line
            kwargs: keyword args

        Returns:
            MPL plot object
        """
        if x1 is not None:
            x0 = [x0, x1]
        if y1 is not None:
            y0 = [y0, y1]

        if 'color' not in kwargs.keys():
            kwargs['color'] = RepeatedList('#000000', 'temp')
        if 'style' not in kwargs.keys():
            kwargs['style'] = RepeatedList('-', 'temp')
        if 'width' not in kwargs.keys():
            kwargs['width'] = RepeatedList('-', 'temp')

        line = self.axes.obj[ir, ic].plot(x0, y0,
                                          linestyle=kwargs['style'][0],
                                          linewidth=kwargs['width'][0],
                                          color=kwargs['color'][0],
                                          zorder=kwargs.get('zorder', 1))
        return line

    def plot_pie(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, data: 'Data',  # noqa: F821
                 kwargs) -> 'MPL_pie_chart_object':  # noqa: F821
        """Plot a pie chart.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: input data
            x: x-axis column name
            y: y-axis column name
            data: Data object
            kwargs: keyword args
        """
        # Define various properties
        wedgeprops = {'linewidth': self.pie.edge_width,
                      'alpha': self.pie.alpha,
                      'linestyle': self.pie.edge_style,
                      'edgecolor': self.pie.edge_color[0],
                      'width': self.pie.inner_radius,
                      }
        textprops = {'fontsize': self.pie.font_size,
                     'weight': self.pie.font_weight,
                     'style': self.pie.font_style,
                     'color': self.pie.font_color,
                     }

        if self.pie.explode is not None:
            if self.pie.explode[0] == 'all':
                self.pie.explode = tuple([self.pie.explode[1] for f in y])
            elif len(self.pie.explode) < len(y):
                self.pie.explode = list(self.pie.explode)
                self.pie.explode += [0 for f in range(0, len(y) - len(self.pie.explode))]

        if data.legend is not None:
            xx = [f for f in x]
            x = ['' for f in x]

        self.pie.obj = self.axes.obj[ir, ic].pie(
            y, labels=x, explode=self.pie.explode,  # center=[40,40],
            colors=self.pie.colors, autopct=self.pie.percents,
            counterclock=self.pie.counter_clock,
            labeldistance=self.pie.label_distance,
            pctdistance=self.pie.percents_distance, radius=self.pie.radius,
            rotatelabels=self.pie.rotate_labels, shadow=self.pie.shadow,
            startangle=self.pie.start_angle, wedgeprops=wedgeprops,
            textprops=textprops)

        # change percent font style
        if self.pie.percents is not None:
            for lab in self.pie.obj[2]:
                lab.set_fontsize(self.pie.percents_font_size)
                lab.set_color(self.pie.percents_font_color)
                lab.set_fontweight(self.pie.percents_font_weight)

        self.axes.obj[ir, ic].set_xlim(left=-1)
        self.axes.obj[ir, ic].set_xlim(right=1)
        self.axes.obj[ir, ic].set_ylim(bottom=-1)
        self.axes.obj[ir, ic].set_ylim(top=1)

        if data.legend is not None:
            for i, val in enumerate(x):
                if xx[i] in self.legend.values['Key'].values:
                    continue
                handle = [patches.Rectangle((0, 0), 1, 1, color=self.pie.colors[i])]
                self.legend.add_value(xx[i], handle, 'lines')

        return self.pie.obj

    def plot_polygon(self, ir: int, ic: int, points: list, **kwargs):
        """Plot a polygon.

        Args:
            ir: subplot row index
            ic: subplot column index
            points: list of floats that defint the points on the polygon
            kwargs: keyword args
        """
        if kwargs['fill_color'] is None:
            fill_color = 'none'
        else:
            fill_color = kwargs['fill_color'][0]

        polygon = [patches.Polygon(points, facecolor=fill_color,
                                   edgecolor=kwargs['edge_color'][0],
                                   linewidth=kwargs['edge_width'],
                                   linestyle=kwargs['edge_style'],
                                   alpha=kwargs['alpha'],
                                   )]
        p = PatchCollection(polygon, match_original=True, zorder=kwargs['zorder'])

        self.axes.obj[ir, ic].add_collection(p)

    def plot_xy(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str, y: str,
                leg_name: str, twin: bool, zorder: int = 1, line_type: [str, None] = None,
                marker_disable: bool = False):
        """ Plot xy data

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            df: summed column "y" values grouped by x-column -->
                df.groupby(x).sum()[y]
            x: x-axis column name
            y: y-axis column name
            leg_name: legend value name if legend enabled
            twin: denotes if twin axis is enabled or not
            zorder (optional): z-height of the plot lines. Defaults to 1.
            line_type (optional): set the line type to reference the correct Element.
                Defaults to None.
            marker_disable (optional): flag to disable markers. Defaults to False.
        """
        def format_marker(marker):
            """Format the marker string to mathtext."""
            if marker in ['o', '+', 's', 'x', 'd', '^']:
                return marker
            elif marker is None:
                return 'None'
            else:
                return r'$%s$' % marker

        df = df.copy()

        if not line_type:
            line_type = self.lines
            line_type_name = 'lines'
        else:
            line_type_name = line_type
            line_type = getattr(self, line_type)

        # Select the axes
        if twin:
            ax = self.axes2.obj[ir, ic]
        else:
            ax = self.axes.obj[ir, ic]

        # Make the points
        if x is None:
            dfx = df[utl.df_int_cols(df)].values
        else:
            dfx = df[x]

        points = None
        if self.markers.on and not marker_disable:
            if self.markers.jitter:
                dfx = np.random.normal(df[x], 0.03, size=len(df[y]))
            marker = format_marker(self.markers.type[iline])
            if marker != 'None':
                # use scatter plot for points
                if marker in ['+', 'x']:
                    c = self.markers.edge_color[(iline, leg_name)]
                else:
                    c = self.markers.fill_color[(iline, leg_name)] if self.markers.filled else 'none'
                points = ax.scatter(dfx, df[y],
                                    s=df[self.markers.size]**2 if isinstance(self.markers.size, str)
                                    else self.markers.size[iline]**2,
                                    marker=marker,
                                    c=c,
                                    edgecolors=self.markers.edge_color[(iline, leg_name)],
                                    linewidth=self.markers.edge_width[(iline, leg_name)],
                                    zorder=40
                                    )
            else:
                points = ax.plot(dfx, df[y],
                                 marker=marker,
                                 color=line_type.color[(iline, leg_name)],
                                 linestyle=line_type.style[iline],
                                 linewidth=line_type.width[iline],
                                 zorder=40)

        # Make the line
        lines = None
        if line_type.on:
            # Mask any nans
            try:
                mask = np.isfinite(dfx)
            except TypeError:
                mask = dfx == dfx

            # Plot the line
            lines = ax.plot(dfx[mask], df[y][mask],
                            color=line_type.color[(iline, leg_name)],
                            linestyle=line_type.style[iline],
                            linewidth=line_type.width[iline],
                            )

        # Add a reference to the line to self.lines
        if leg_name is not None:
            if leg_name is not None and str(leg_name) not in list(self.legend.values['Key']):
                self.legend.add_value(str(leg_name), points if points is not None else lines, line_type_name)

    def save(self, filename: str, idx: int = 0):
        """Save a plot window.

        Args:
            filename: name of the file
            idx (optional): figure index in order to set the edge and face color of the
                figure correctly when saving. Defaults to 0.
        """
        kwargs = {'edgecolor': self.fig.edge_color[idx],
                  'facecolor': self.fig.fill_color[idx]}
        if version.Version(mpl.__version__) < version.Version('3.3'):
            kwargs['linewidth'] = self.fig.edge_width
        self.fig.obj.savefig(filename, **kwargs)

    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        axes = self._get_axes()
        try:
            axes[0].obj[ir, ic].set_facecolor(axes[0].fill_color[utl.plot_num(ir, ic, self.ncol)])
        except:  # noqa
            axes[0].obj[ir, ic].set_axis_bgcolor(axes[0].fill_color[utl.plot_num(ir, ic, self.ncol)])
        for f in ['bottom', 'top', 'right', 'left']:
            if len(axes) > 1:
                axes[0].obj[ir, ic].spines[f].set_visible(False)
            if getattr(self.axes, f'spine_{f}'):
                axes[-1].obj[ir, ic].spines[f].set_color(axes[0].edge_color[utl.plot_num(ir, ic, self.ncol)])
            else:
                axes[-1].obj[ir, ic].spines[f].set_color(self.fig.fill_color[0])
            axes[-1].obj[ir, ic].spines[f].set_linewidth(self.axes.edge_width_adj)
            # TODO:  spine style?

    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """
        axes = self._get_axes()

        for ax in axes:
            # Set major grid
            ax.obj[ir, ic].set_axisbelow(True)
            if self.grid_major_x.on:
                kwargs = {'which': 'major', 'color': self.grid_major_x.color[0],
                          'linestyle': self.grid_major_x.style[0], 'linewidth': self.grid_major_x.width[0],
                          'alpha': self.grid_major_x.alpha}
                kwargs.update(self._get_grid_visibility_kwarg(True))
                ax.obj[ir, ic].xaxis.grid(**kwargs)
            else:
                kwargs = {'which': 'major'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].xaxis.grid(**kwargs)

            if self.grid_major_x2 and not ax.primary:
                if self.grid_major_x2.on:
                    kwargs = {'which': 'major', 'color': self.grid_major_x2.color[0],
                              'linestyle': self.grid_major_x2.style[0], 'linewidth': self.grid_major_x2.width[0],
                              'alpha': self.grid_major_x2.alpha}
                    kwargs.update(self._get_grid_visibility_kwarg(True))
                    ax.obj[ir, ic].xaxis.grid(**kwargs)
                else:
                    ax.obj[ir, ic].xaxis.grid(b=False, which='major')
            elif not self.grid_major_x2 and not ax.primary:
                kwargs = {'which': 'major'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].xaxis.grid(**kwargs)

            if self.grid_major_y.on:
                kwargs = {'which': 'major', 'color': self.grid_major_y.color[0],
                          'linestyle': self.grid_major_y.style[0], 'linewidth': self.grid_major_y.width[0],
                          'alpha': self.grid_major_y.alpha}
                kwargs.update(self._get_grid_visibility_kwarg(True))
                ax.obj[ir, ic].yaxis.grid(**kwargs)
            else:
                kwargs = {'which': 'major'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].yaxis.grid(**kwargs)

            if self.grid_major_y2 and not ax.primary:
                if self.grid_major_y2.on:
                    kwargs = {'which': 'major', 'color': self.grid_major_y2.color[0],
                              'linestyle': self.grid_major_y2.style[0], 'linewidth': self.grid_major_y2.width[0],
                              'alpha': self.grid_major_y2.alpha}
                    kwargs.update(self._get_grid_visibility_kwarg(True))
                    ax.obj[ir, ic].yaxis.grid(**kwargs)
                else:
                    kwargs = {'which': 'major'}
                    kwargs.update(self._get_grid_visibility_kwarg(False))
                    ax.obj[ir, ic].yaxis.grid(**kwargs)
            elif not self.grid_major_y2 and not ax.primary:
                kwargs = {'which': 'major'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].yaxis.grid(**kwargs)

            # Set minor grid
            if self.grid_minor_x.on:
                kwargs = {'which': 'minor', 'color': self.grid_minor_x.color[0],
                          'linestyle': self.grid_minor_x.style[0], 'linewidth': self.grid_minor_x.width[0],
                          'alpha': self.grid_minor_x.alpha}
                kwargs.update(self._get_grid_visibility_kwarg(True))
                ax.obj[ir, ic].xaxis.grid(**kwargs)
            if self.grid_minor_y.on:
                kwargs = {'which': 'minor', 'color': self.grid_minor_y.color[0],
                          'linestyle': self.grid_minor_y.style[0], 'linewidth': self.grid_minor_y.width[0],
                          'alpha': self.grid_minor_y.alpha}
                kwargs.update(self._get_grid_visibility_kwarg(True))
                ax.obj[ir, ic].yaxis.grid(**kwargs)
            if self.grid_minor_x2 and not ax.primary:
                if self.grid_minor_x2.on:
                    kwargs = {'which': 'minor', 'color': self.grid_minor_x2.color[0],
                              'linestyle': self.grid_minor_x2.style[0], 'linewidth': self.grid_minor_x2.width[0],
                              'alpha': self.grid_minor_x2.alpha}
                    kwargs.update(self._get_grid_visibility_kwarg(True))
                    ax.obj[ir, ic].xaxis.grid(**kwargs)
                else:
                    kwargs = {'which': 'minor'}
                    kwargs.update(self._get_grid_visibility_kwarg(False))
                    ax.obj[ir, ic].xaxis.grid(**kwargs)
            elif not self.grid_minor_x2 and not ax.primary:
                kwargs = {'which': 'minor'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].xaxis.grid(**kwargs)
            if self.grid_minor_y2 and not ax.primary:
                if self.grid_minor_y2.on:
                    kwargs = {'which': 'minor', 'color': self.grid_minor_y2.color[0],
                              'linestyle': self.grid_minor_y2.style[0], 'linewidth': self.grid_minor_y2.width[0],
                              'alpha': self.grid_minor_y2.alpha}
                    kwargs.update(self._get_grid_visibility_kwarg(False))
                    ax.obj[ir, ic].yaxis.grid(**kwargs)
                else:
                    kwargs = {'which': 'minor'}
                    kwargs.update(self._get_grid_visibility_kwarg(False))
                    ax.obj[ir, ic].xaxis.grid(**kwargs)
            elif not self.grid_minor_y2 and not ax.primary:
                kwargs = {'which': 'minor'}
                kwargs.update(self._get_grid_visibility_kwarg(False))
                ax.obj[ir, ic].yaxis.grid(**kwargs)

    def set_axes_labels(self, ir: int, ic: int, data: 'Data'):  # noqa: F821
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: fcp.data object

        """
        if self.name in ['pie']:
            return

        for ax in data.axs_on:
            label = getattr(self, f'label_{ax}')
            labeltext = None
            if not label.on:
                continue
            if type(label.text) not in [str, list]:
                continue
            if isinstance(label.text, str):
                labeltext = label.text
            if isinstance(label.text, list):
                labeltext = label.text[ic + ir * self.ncol]

            if '2' in ax:
                axes = self.axes2.obj[ir, ic]
                pad = self.ws_label_tick
            else:
                axes = self.axes.obj[ir, ic]  # noqa
                pad = self.ws_label_tick  # noqa

            # Toggle label visibility
            if not self.separate_labels:
                if not self.axes.visible[ir, ic]:
                    continue
                if ax == 'x' and ir != self.nrow - 1 and self.axes.visible[ir + 1, ic]:
                    continue
                if ax == 'x2' and ir != 0:
                    continue
                if ax == 'y' and ic != 0 and self.axes.visible[ir, ic - 1]:
                    continue
                if ax == 'y2' and (ic != self.ncol - 1 and self.axes.visible[ir, ic + 1])\
                        and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue
                if ax == 'z' and ic != self.ncol - 1 \
                        and utl.plot_num(ir, ic, self.ncol) != self.nwrap \
                        and any(self.axes.visible[ir, (ic + 1):]):
                    continue

            # Add the label
            label.obj[ir, ic], label.obj_bg[ir, ic] = self.add_label(ir, ic, label, labeltext)

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        if self.name in ['pie']:  # skip these plot types
            return

        # Set the ranges
        if 'xmin' in ranges and ranges['xmin'][ir, ic] is not None:
            self.axes.obj[ir, ic].set_xlim(left=ranges['xmin'][ir, ic])
        if 'x2min' in ranges and ranges['x2min'][ir, ic] is not None:
            self.axes2.obj[ir, ic].set_xlim(left=ranges['x2min'][ir, ic])
        if 'xmax' in ranges and ranges['xmax'][ir, ic] is not None:
            self.axes.obj[ir, ic].set_xlim(right=ranges['xmax'][ir, ic])
        if 'x2max' in ranges and ranges['x2max'][ir, ic] is not None:
            self.axes2.obj[ir, ic].set_xlim(right=ranges['x2max'][ir, ic])
        if 'ymin' in ranges and ranges['ymin'][ir, ic] is not None:
            self.axes.obj[ir, ic].set_ylim(bottom=ranges['ymin'][ir, ic])
        if 'y2min' in ranges and ranges['y2min'][ir, ic] is not None:
            self.axes2.obj[ir, ic].set_ylim(bottom=ranges['y2min'][ir, ic])
        if 'ymax' in ranges and ranges['ymax'][ir, ic] is not None:
            self.axes.obj[ir, ic].set_ylim(top=ranges['ymax'][ir, ic])
        if 'y2max' in ranges and ranges['y2max'][ir, ic] is not None:
            self.axes2.obj[ir, ic].set_ylim(top=ranges['y2max'][ir, ic])

        # Plot specific adjustments
        if self.name in ['imshow', 'heatmap'] and len(self.axes.obj[ir, ic].get_images()) > 0:
            if 'zmin' in ranges and ranges['zmin'][ir, ic] is not None:
                self.axes.obj[ir, ic].get_images()[0].set_clim(vmin=ranges['zmin'][ir, ic])
            if 'zmax' in ranges and ranges['zmax'][ir, ic] is not None:
                self.axes.obj[ir, ic].get_images()[0].set_clim(vmax=ranges['zmax'][ir, ic])
        elif self.name in ['contour', 'heatmap']:
            if 'zmin' in ranges and ranges['zmin'][ir, ic] is not None:
                getattr(self, self.name).obj[ir, ic].set_clim(vmin=ranges['zmin'][ir, ic])
            if 'zmax' in ranges and ranges['zmax'][ir, ic] is not None:
                getattr(self, self.name).obj[ir, ic].set_clim(vmax=ranges['zmax'][ir, ic])

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Wrap title
        if ir == 0 and ic == 0 and self.title_wrap.on:
            self.title_wrap.obj, self.title_wrap.obj_bg = self.add_label(ir, ic, self.title_wrap)

        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            if not self.label_row.values_only:
                lab = f'{self.label_row.text}={self.label_row.values[ir]}'
            else:
                lab = self.label_row.values[ir]
            self.label_row.obj[ir, ic], self.label_row.obj_bg[ir, ic] = self.add_label(ir, ic, self.label_row, lab)

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            if self.label_wrap.on:
                if not self.label_wrap.values_only:
                    lab = ' | '.join([str(f) for f in utl.validate_list(self.label_wrap.values[ir * self.ncol + ic])
                                      if f != 'wrap_reference_999'])
                else:
                    lab = None
                self.label_wrap.obj[ir, ic], self.label_wrap.obj_bg[ir, ic] = \
                    self.add_label(ir, ic, self.label_wrap, lab)
            else:
                if not self.label_col.values_only:
                    lab = f'{self.label_col.text}={self.label_col.values[ic]}'
                else:
                    lab = self.label_col.values[ic]
                self.label_col.obj[ir, ic], self.label_col.obj_bg[ir, ic] = \
                    self.add_label(ir, ic, self.label_col, lab)

    def set_axes_scale(self, ir: int, ic: int):
        """Set the scale type of the axes.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        axes = self._get_axes()

        for ax in axes:
            if ax.scale is None:
                continue
            else:
                if str(ax.scale).lower() in LOGX:
                    ax.obj[ir, ic].set_xscale('log')
                elif str(ax.scale).lower() in SYMLOGX:
                    ax.obj[ir, ic].set_xscale('symlog')
                elif str(ax.scale).lower() in LOGITX:
                    ax.obj[ir, ic].set_xscale('logit')
                if str(ax.scale).lower() in LOGY:
                    ax.obj[ir, ic].set_yscale('log')
                elif str(ax.scale).lower() in SYMLOGY:
                    ax.obj[ir, ic].set_yscale('symlog')
                elif str(ax.scale).lower() in LOGITY:
                    ax.obj[ir, ic].set_yscale('logit')

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        if self.name in ['pie']:  # skip this plot type
            return

        axes = [f.obj[ir, ic] for f in [self.axes, self.axes2] if f.on]

        # Format ticks
        for ia, aa in enumerate(axes):
            minor_on = max(self.ticks_minor_x.on, self.ticks_minor_y.on)
            tp = mpl_get_ticks(axes[ia], True, True, minor_on)

            if ia == 0:
                lab = ''
            else:
                lab = '2'

            # Skip certain calculations if axes are shared and subplots > 1
            skipx, skipy = False, False
            if hasattr(getattr(self, f'axes{lab}'), f'share_x{lab}') \
                    and getattr(getattr(self, f'axes{lab}'), f'share_x{lab}') is True \
                    and (ir != 0 or ic != 0):
                skipx = False
            if hasattr(getattr(self, f'axes{lab}'), f'share_y{lab}') \
                    and getattr(getattr(self, f'axes{lab}'), f'share_y{lab}') \
                    and (ir != 0 or ic != 0):
                skipy = False

            # Turn off scientific
            if ia == 0:
                if not skipx:
                    self._set_scientific(axes[ia], tp)
            elif self.axes.twin_y or self.axes.twin_x:
                if not skipy:
                    self._set_scientific(axes[ia], tp, 2)

            # Turn off offsets
            if not self.tick_labels_major.offset:
                try:
                    if not skipx:
                        aa.get_xaxis().get_major_formatter().set_useOffset(False)
                except:  # noqa
                    pass
                try:
                    if not skipy:
                        aa.get_yaxis().get_major_formatter().set_useOffset(False)
                except:  # noqa
                    pass

            # General tick params
            if ia == 0:
                if self.ticks_minor_x.on or self.ticks_minor_y.on:
                    axes[0].minorticks_on()
                else:
                    # have to force this sometimes
                    axes[0].minorticks_off()
                ax_edge = 0 if self.axes.edge_width == 1 else self.axes.edge_width_adj / 2
                axes[0].tick_params(axis='x',
                                    which='major',
                                    pad=0,
                                    colors=self.ticks_major_x.color[0],
                                    labelcolor=self.tick_labels_major_x.font_color,
                                    top=False,
                                    bottom=self.ticks_major_x.on,
                                    right=False,
                                    left=False,
                                    length=self.ticks_major_x._size[0] + ax_edge,
                                    width=self.ticks_major_x._size[1],
                                    direction=self.ticks_major_x.direction,
                                    zorder=1000,
                                    )
                axes[0].tick_params(axis='y',
                                    which='major',
                                    pad=0,
                                    colors=self.ticks_major_y.color[0],
                                    labelcolor=self.tick_labels_major_y.font_color,
                                    top=False,
                                    bottom=False,
                                    right=False,
                                    left=self.ticks_major_y.on,
                                    length=self.ticks_major_y._size[0] + ax_edge,
                                    width=self.ticks_major_y._size[1],
                                    direction=self.ticks_major_y.direction,
                                    zorder=100,
                                    )
                axes[0].tick_params(axis='x',
                                    which='minor',
                                    pad=0,
                                    colors=self.ticks_minor_x.color[0],
                                    labelcolor=self.tick_labels_minor_x.font_color,
                                    labelsize=self.tick_labels_minor_x.font_size,
                                    top=False,
                                    bottom=self.ticks_minor_x.on,
                                    right=False,
                                    left=False,
                                    length=self.ticks_minor_x._size[0] + ax_edge,
                                    width=self.ticks_minor_x._size[1],
                                    direction=self.ticks_minor_x.direction,
                                    )
                axes[0].tick_params(axis='y',
                                    which='minor',
                                    pad=0,
                                    colors=self.ticks_minor_y.color[0],
                                    labelcolor=self.tick_labels_minor_y.font_color,
                                    labelsize=self.tick_labels_minor_y.font_size,
                                    top=False,
                                    bottom=False,
                                    right=False if self.axes.twin_x
                                    else self.ticks_minor_y.on,
                                    left=self.ticks_minor_y.on,
                                    length=self.ticks_minor_y._size[0] + ax_edge,
                                    width=self.ticks_minor_y._size[1],
                                    direction=self.ticks_minor_y.direction,
                                    )

                if self.name == 'gantt' and not self.gantt.labels_as_yticks:
                    axes[0].tick_params(left=False)

                if self.axes.twin_x:
                    if self.ticks_minor_y2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=0,
                                        colors=self.ticks_major_y2.color[0],
                                        labelcolor=self.tick_labels_major_y2.font_color,
                                        right=self.ticks_major_y2.on,
                                        length=self.ticks_major_y2.size[0] + ax_edge,
                                        width=self.ticks_major_y2.size[1],
                                        direction=self.ticks_major_y2.direction,
                                        zorder=0,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=0,
                                        colors=self.ticks_minor_y2.color[0],
                                        labelcolor=self.tick_labels_minor_y2.font_color,
                                        right=self.ticks_minor_y2.on,
                                        length=self.ticks_minor_y2._size[0] + ax_edge,
                                        width=self.ticks_minor_y2._size[1],
                                        direction=self.ticks_minor_y2.direction,
                                        zorder=0,
                                        )
                elif self.axes.twin_y:
                    if self.ticks_minor_x2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=0,
                                        colors=self.ticks_major_x2.color[0],
                                        labelcolor=self.tick_labels_major_x2.font_color,
                                        top=self.ticks_major_x2.on,
                                        length=self.ticks_major_x2.size[0] + ax_edge,
                                        width=self.ticks_major_x2.size[1],
                                        direction=self.ticks_major_x2.direction,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=0,
                                        colors=self.ticks_minor.color[0],
                                        labelcolor=self.tick_labels_minor.font_color,
                                        top=self.ticks_minor_x2.on,
                                        length=self.ticks_minor._size[0] + ax_edge,
                                        width=self.ticks_minor._size[1],
                                        direction=self.ticks_minor.direction,
                                        )

            # Set custom tick increment
            redo = True
            xinc = getattr(self, f'ticks_major_x{lab}').increment
            if not skipx and xinc is not None and self.name not in ['bar']:
                xstart = 0 if tp['x']['min'] == 0 else tp['x']['min'] + xinc - tp['x']['min'] % xinc
                axes[ia].set_xticks(np.arange(xstart, tp['x']['max'], xinc))
                redo = True
            yinc = getattr(self, f'ticks_major_y{lab}').increment
            if not skipy and yinc is not None:
                axes[ia].set_yticks(np.arange(tp['y']['min'] + yinc - tp['y']['min'] % yinc, tp['y']['max'], yinc))
                redo = True

            if redo:
                tp = mpl_get_ticks(axes[ia], True, True, minor_on)

            # Force ticks
            if self.separate_ticks or getattr(self, f'axes{lab}').share_x is False:
                if version.Version(mpl.__version__) < version.Version('2.2'):
                    mplp.setp(axes[ia].get_xticklabels(), visible=True)
                else:
                    if self.axes.twin_x and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)
                    elif self.axes.twin_y and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labeltop=True)
                    else:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)

            if self.separate_ticks or getattr(self, f'axes{lab}').share_y is False:
                if version.Version(mpl.__version__) < version.Version('2.2'):
                    mplp.setp(axes[ia].get_yticklabels(), visible=True)
                else:
                    if self.axes.twin_x and ia == 1:
                        axes[ia].yaxis.set_tick_params(which='both', labelright=True)
                    elif self.axes.twin_y and ia == 1:
                        axes[ia].yaxis.set_tick_params(which='both', labelleft=True)
                    else:
                        axes[ia].yaxis.set_tick_params(which='both', labelleft=True)

            if self.nwrap > 0 and (ic + (ir + 1) * self.ncol + 1) > self.nwrap or \
                    (ir < self.nrow - 1 and not self.axes.visible[ir + 1, ic]):
                if version.Version(mpl.__version__) < version.Version('2.2'):
                    mplp.setp(axes[ia].get_xticklabels()[1:], visible=True)
                elif self.axes.twin_y and ia == 1:
                    axes[ia].yaxis.set_tick_params(which='both', labeltop=True)
                else:
                    axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)

            if not self.separate_ticks and not self.axes.visible[ir, ic - 1]:
                if version.Version(mpl.__version__) < version.Version('2.2'):
                    mplp.setp(axes[ia].get_yticklabels(), visible=True)
                elif self.axes.twin_x and ia == 1:
                    axes[ia].yaxis.set_tick_params(which='both', labelright=True)
                else:
                    axes[ia].yaxis.set_tick_params(which='both', labelleft=True)
            elif not self.separate_ticks \
                    and (ic != self.ncol - 1
                         and utl.plot_num(ir, ic, self.ncol) != self.nwrap) \
                    and self.axes.twin_x and ia == 1 \
                    and getattr(self, f'axes{lab}').share_y:
                mplp.setp(axes[ia].get_yticklabels(), visible=False)

            # Disable twinned ticks
            if not self.separate_ticks and ir != 0 and self.axes.twin_y and ia == 1 and self.axes2.share_x:
                mplp.setp(axes[ia].get_xticklabels(), visible=False)

            # Major rotation
            axx = ['x', 'y']
            majmin = ['major', 'minor']
            for ax in axx:
                for mm in majmin:
                    if getattr(self, f'tick_labels_{mm}_{ax}{lab}').on:
                        self._check_font(getattr(self, f'tick_labels_{mm}_{ax}{lab}').font)
                        ticks_font = font_manager.FontProperties(
                            family=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font,
                            size=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_size,
                            style=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_style,
                            weight=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_weight,
                            )
                        style = dict(edgecolor=getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_color[0],
                                     facecolor=getattr(self, f'tick_labels_{mm}_{ax}{lab}').fill_color[0],
                                     linewidth=getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_width,
                                     alpha=max(getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_alpha,
                                               getattr(self, f'tick_labels_{mm}_{ax}{lab}').fill_alpha),
                                     pad=getattr(self, f'tick_labels_{mm}_{ax}{lab}').padding,
                                     )
                        rotation = getattr(self, f'tick_labels_{mm}_{ax}{lab}').rotation
                        for text in getattr(axes[ia], f'get_{ax}ticklabels')(which=mm):
                            text.set_rotation(rotation)
                            text.set_fontproperties(ticks_font)
                            text.set_bbox(style)

                        # Style offset text
                        getattr(axes[ia], f'{ax}axis').get_offset_text().set_fontproperties(ticks_font)

            # Tick label shorthand
            tlmajx = getattr(self, f'tick_labels_major_x{lab}')
            tlmajy = getattr(self, f'tick_labels_major_y{lab}')

            # Turn off major tick labels
            if not tlmajx.on:
                ll = ['' for f in axes[ia].get_xticklabels()]
                axes[ia].set_xticklabels(ll)

            if not tlmajy.on:
                ll = ['' for f in axes[ia].get_yticklabels()]
                axes[ia].set_yticklabels(ll)

            # Turn on minor tick labels
            ax = ['x', 'y']
            sides = {}
            if version.Version(mpl.__version__) < version.Version('2.2'):
                sides['x'] = {'labelbottom': 'off'}
                sides['x2'] = {'labeltop': 'off'}
                sides['y'] = {'labelleft': 'off'}
                sides['y2'] = {'labelright': 'off'}
            else:
                sides['x'] = {'labelbottom': False}
                sides['x2'] = {'labeltop': False}
                sides['y'] = {'labelleft': False}
                sides['y2'] = {'labelright': False}

            tlminon = False  # "tick label min"
            for axx in ax:
                axl = f'{axx}{lab}'
                tlmin = getattr(self, f'ticks_minor_{axl}')

                if ia == 1 and axx == 'x' and self.axes.twin_x:
                    continue

                if ia == 1 and axx == 'y' and self.axes.twin_y:
                    continue

                if getattr(self, f'ticks_minor_{axl}').number is not None and tlmin.on:
                    num_minor = getattr(self, f'ticks_minor_{axl}').number
                    if getattr(self, f'axes{lab}').scale not in (LOG_ALLX if axx == 'x' else LOG_ALLY):
                        loc = None
                        if axl == 'x' and self.name == 'gantt':
                            loc = mdates.AutoDateLocator()
                        else:
                            loc = AutoMinorLocator(num_minor + 1)
                        getattr(axes[ia], f'{axx}axis').set_minor_locator(loc)
                        tp = mpl_get_ticks(axes[ia],
                                           getattr(self, f'ticks_major_x{lab}').on,
                                           getattr(self, f'ticks_major_y{lab}').on,
                                           True)

                if not self.separate_ticks and axl == 'x' and ir != self.nrow - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'y2' and ic != self.ncol - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'x2' and ir != 0 or \
                        not self.separate_ticks and axl == 'y' and ic != 0 or \
                        not self.separate_ticks and axl == 'y2' \
                        and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    axes[ia].tick_params(which='minor', **sides[axl])

                elif tlmin.on:
                    if not getattr(self, f'tick_labels_minor_{axl}').on:
                        continue
                    elif 'x' in axx and skipx:
                        continue
                    elif 'y' in axx and skipy:
                        continue
                    else:
                        tlminon = True  # noqa

                    # Determine the minimum number of decimals needed to display the minor ticks
                    m0 = len(tp[axx]['ticks'])
                    lim = getattr(axes[ia], f'get_{axx}lim')()
                    vals = [f for f in tp[axx]['ticks'] if f > lim[0]]
                    label_vals = [f for f in tp[axx]['label_vals'] if f > lim[0]]
                    minor_ticks = [f[1] for f in tp[axx]['labels']][m0:]
                    inc = label_vals[1] - label_vals[0]

                    # Remove any major tick labels from the list of minor ticks
                    dups = []
                    dups_idx = []
                    for imt, mt in enumerate(minor_ticks):
                        for majt in tp[axx]['ticks']:
                            if math.isclose(mt, majt):
                                dups += [mt]
                                dups_idx += [m0 + imt]
                    minor_ticks2 = [f for f in minor_ticks if f not in dups]
                    number = len([f for f in minor_ticks2 if f > vals[0] and f < vals[1]]) + 1
                    decimals = utl.get_decimals(inc / number, exponential=getattr(self, f'tick_labels_major_{axx}').sci)

                    # Check for minor ticks below the first major tick for log axes
                    if getattr(self, f'axes{lab}').scale in (LOGX if axx == 'x' else LOGY):
                        extra_minor = [f for f in minor_ticks if f < label_vals[0] and f > lim[0]]
                        if len(extra_minor) > 0:
                            decimals += 1

                    # Set the tick decimal format
                    if getattr(self, f'tick_labels_major_{axx}').sci is True:
                        getattr(axes[ia], f'{axx}axis').set_minor_formatter(
                            ticker.FormatStrFormatter('%%.%se' % (decimals)))
                    else:
                        getattr(axes[ia], f'{axx}axis').set_minor_formatter(
                            ticker.FormatStrFormatter('%%.%sf' % (decimals)))

                    # Clean up unecessary zeros at the end of minor ticks
                    tick_labels = [tick.get_text().rstrip('0').rstrip('.')
                                   for tick in getattr(axes[ia], f'get_{axx}ticklabels')(minor=True)]

                    getattr(axes[ia], f'set_{axx}ticklabels')(tick_labels, minor=True)

        if self.cbar.on:
            for ir, ic in np.ndindex(self.cbar.obj.shape):
                if not hasattr(self.cbar.obj[ir, ic], 'ax'):
                    continue
                self._check_font(getattr(self, 'tick_labels_major_z').font)
                ticks_font = font_manager.FontProperties(
                    family=getattr(self, 'tick_labels_major_z').font,
                    size=getattr(self, 'tick_labels_major_z').font_size,
                    style=getattr(self, 'tick_labels_major_z').font_style,
                    weight=getattr(self, 'tick_labels_major_z').font_weight,
                    )
                style = dict(edgecolor=getattr(self, 'tick_labels_major_z').edge_color[0],
                             facecolor=getattr(self, 'tick_labels_major_z').fill_color[0],
                             linewidth=getattr(self, 'tick_labels_major_z').edge_width,
                             alpha=max(getattr(self, 'tick_labels_major_z').edge_alpha,
                                       getattr(self, 'tick_labels_major_z').fill_alpha),
                             pad=getattr(self, 'tick_labels_major_z').padding,
                             )
                text = self.cbar.obj[ir, ic].ax.yaxis.get_offset_text()
                text.set_fontproperties(ticks_font)
                text.set_bbox(style)

    def _set_colormap(self, data: 'Data', **kwargs):  # noqa: F821
        """Replace the color list with discrete values from a colormap.

        Args:
            data: Data object
            kwargs: keyword args
        """
        if not self.cmap[0] or self.name in ['contour', 'heatmap', 'imshow']:
            return

        try:
            # Conver the color map into discrete colors
            cmap = mplp.get_cmap(self.cmap[0])
            color_list = []
            if data.legend_vals is None or len(data.legend_vals) == 0:
                if self.axes.twin_x or self.axes.twin_y:
                    maxx = 2
                else:
                    maxx = 1
            else:
                maxx = len(data.legend_vals)
            for i in range(0, maxx):
                color_list += \
                    [mplc_to_hex(cmap((i + 1) / (maxx + 1)), False)]

            # Reset colors
            # if self.legend.column is None:
            #     # NO IDEA HOW THIS CASE COULD BE, CONSIDER REMOVING
            #     if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
            #         self.label_y.font_color = color_list[0]
            #     if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
            #         self.label_y2.font_color = color_list[1]
            #     if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
            #         self.label_x.font_color = color_list[0]
            #     if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
            #         self.label_x2.font_color = color_list[1]

            self.lines.color.values = copy.copy(color_list)
            self.lines.color_alpha('color', 'alpha')
            self.markers.edge_color.values = copy.copy(color_list)
            self.markers.color_alpha('edge_color', 'edge_alpha')
            self.markers.fill_color.values = copy.copy(color_list)
            self.markers.color_alpha('fill_color', 'fill_alpha')

        except:  # noqa
            print(f'Could not find a colormap called "{self.cmap[0]}". Using default colors...')

    def set_figure_final_layout(self, data: 'Data', **kwargs):  # noqa: F821
        """
        Final adjustment of the figure size and plot spacing.

        Subplots within self.fig.obj are not currently in the right place and do not have the right size.  Before
        checking for overlaps with other elements we temporarily move the subplots to (0, 0) and size properly

        Args:
            data: Data object
            kwargs: keyword args
        """
        # Render dummy figure to get the element sizes
        self._get_element_sizes(data)

        # Determine if extra whitespace is needed at the plot edge for the last tick
        self._get_tick_xs()

        # Resize the figure and get rendered dimensions
        self._get_figure_size(data, **kwargs)
        self.fig.size[0] = np.ceil(self.fig.size[0] - 1E-12)
        self.fig.size[1] = np.ceil(self.fig.size[1] - 1E-12)
        self.fig.obj.set_size_inches((self.fig.size_inches[0], self.fig.size_inches[1]))

        # Double check fig size for rounding errors (small values can change a whole pixel)
        width_err, height_err = 0, 0
        if self.fig.obj.get_window_extent().x1 < self.fig.size[0]:
            width_err = self.fig.size[0] - self.fig.obj.get_window_extent().x1
            self.fig.obj.set_figwidth(self.fig.size_inches[0] + width_err / self.fig.dpi)
            self.fig.size[0] = np.round(self.fig.obj.get_window_extent().x1)
        if self.fig.obj.get_window_extent().y1 < self.fig.size[1]:
            height_err = self.fig.size[1] - self.fig.obj.get_window_extent().y1
            self.fig.obj.set_figheight(self.fig.size_inches[1] + height_err / self.fig.dpi)
            self.fig.size[1] = np.round(self.fig.obj.get_window_extent().y1)

        # Adjust subplot spacing then correct cause god damn mpl
        self._subplots_adjust()

        # Tick overlap cleanup and set location
        self._get_tick_label_sizes()  # update after axes reshape
        self._get_tick_overlaps()
        self._get_tick_overlaps('2')
        self._set_tick_position(data)

        # Update the axes labels
        self._get_axes_label_position()
        for label in data.axs_on:
            lab = getattr(self, f'label_{label}')
            if not lab.on:
                continue
            x, y = getattr(self, f'label_{label}').position_xy   # this is defined in _get_axes_label_position
            for ir, ic in np.ndindex(lab.obj.shape):
                if lab.obj[ir, ic]:
                    # Shift labels to the right subplot
                    xoffset, yoffset = 0, 0
                    if label == 'z':
                        xoffset = \
                            self.axes.obj[0, self.ncol - 1].get_position().x0 - self.axes.obj[ir, ic].get_position().x0
                        yoffset = self.axes.obj[0, 0].get_position().y0 - self.axes.obj[ir, ic].get_position().y0
                    elif label == 'y':
                        xoffset = self.axes.obj[0, 0].get_position().x0 - self.axes.obj[ir, ic].get_position().x0
                        if self.gantt.on and self.gantt.workstreams.on:
                            xoffset += self.gantt.workstreams.size[0] / self.fig.size_int[0]
                            xoffset += self.gantt.workstreams_title.size[0] / self.fig.size_int[0]
                        yoffset = self.axes.obj[0, 0].get_position().y0 - self.axes.obj[ir, ic].get_position().y0
                    elif label == 'x':
                        xoffset = self.axes.obj[0, 0].get_position().x0 - self.axes.obj[ir, ic].get_position().x0
                        yoffset = self.axes.obj[-1, 0].get_position().y0 - self.axes.obj[ir, ic].get_position().y0
                    elif label == 'y2':
                        xoffset = self.axes.obj[0, -1].get_position().x0 - self.axes.obj[ir, ic].get_position().x0
                        yoffset = self.axes.obj[0, 0].get_position().y0 - self.axes.obj[ir, ic].get_position().y0
                    elif label == 'x2':
                        xoffset = self.axes.obj[0, 0].get_position().x0 - self.axes.obj[ir, ic].get_position().x0
                        yoffset = self.axes.obj[0, 0].get_position().y1 - self.axes.obj[ir, ic].get_position().y1
                    lab.obj[ir, ic].set_position((x - xoffset, y - yoffset))

        # Update the rc label positions
        # row
        for ir, ic in np.ndindex(self.label_row.obj.shape):
            if self.label_row.obj[ir, ic]:
                # x-position and width
                if not self.cbar.on:
                    edge = np.ceil(self.axes.edge_width / 2)
                    lab_x = self.axes.obj[ir, ic].get_position().x1 * self.fig.size_int[0] \
                        + edge \
                        + self.ws_label_row \
                        + np.floor(self.label_row.edge_width / 2) \
                        + np.ceil(self._labtick_y2)
                else:
                    lab_x = (1 - (self.ws_ax_fig + self.label_row.size[0] + self.label_row.edge_width)
                             / self.fig.size_int[0]) * self.fig.size_int[0]

                self.label_row.obj_bg[ir, ic].set_x(lab_x / self.fig.size_int[0])
                self.label_row.obj_bg[ir, ic].set_width(
                    (self.label_row.size[0] + self.label_row.edge_width) / self.fig.size_int[0])

                # y-position and height
                bbox = self.axes.obj[ir, ic].get_position()
                lab_y0 = np.round(bbox.y0 * self.fig.size_int[1]) \
                    + np.ceil(self.label_row.edge_width / 2) - np.ceil(self.axes.edge_width / 2)
                lab_y1 = np.round(bbox.y1 * self.fig.size_int[1]) + np.floor(self.axes.edge_width / 2) \
                    - np.floor(self.label_row.edge_width / 2)

                self.label_row.obj_bg[ir, ic].set_y(lab_y0 / self.fig.size_int[1])
                self.label_row.obj_bg[ir, ic].set_height((lab_y1 - lab_y0) / self.fig.size_int[1])

                # Special offset if certain characters are missing (crude; only tested on mac with default font)
                obj = self.label_row.obj[ir, ic]
                if not any(e in ['y', 'j', 'g', 'q', 'p'] for e in self.label_row.obj[ir, ic].get_text()):
                    offsetx = 0.05 * obj.get_font()._size + 2
                elif any(e in ['[', ']'] for e in self.label_row.obj[ir, ic].get_text()):
                    offsetx = 0.125 * obj.get_font()._size
                else:
                    offsetx = 0.125 * obj.get_font()._size
                self.label_row.obj[ir, ic].set_x(
                    (lab_x + self.label_row.size[0] / 2 - offsetx) / self.fig.size[0])
                self.label_row.obj[ir, ic].set_y(
                    (self.axes.obj[ir, ic].get_position().y1 - self.axes.obj[ir, ic].get_position().y0) / 2
                    + self.axes.obj[ir, ic].get_position().y0)

        # col
        for ir, ic in np.ndindex(self.label_col.obj.shape):
            if self.label_col.obj[ir, ic]:
                # bkgd
                bbox = self.axes.obj[ir, ic].get_position()
                lab_x0 = np.round(bbox.x0 * self.fig.size_int[0]) - np.floor(self.axes.edge_width / 2) \
                    + np.floor(self.label_col.edge_width / 2)
                lab_x1 = np.ceil(bbox.x1 * self.fig.size_int[0] + self.axes.edge_width / 2) \
                    - np.ceil(self.label_col.edge_width / 2)
                self.label_col.obj_bg[ir, ic].set_x(lab_x0 / self.fig.size_int[0])
                self.label_col.obj_bg[ir, ic].set_width((lab_x1 - lab_x0) / self.fig.size_int[0])

                lab_y0 = bbox.y1 * self.fig.size_int[1] \
                    + (self.ws_label_col + self._labtick_x2 + np.floor(self.axes.edge_width / 2)
                       + np.ceil(self.label_col.edge_width / 2)) \
                    + np.ceil(self._labtick_x2)
                self.label_col.obj_bg[ir, ic].set_y(lab_y0 / self.fig.size_int[1])

                self.label_col.obj_bg[ir, ic].set_height(
                    (self.label_col.size[1] + self.label_col.edge_width) / self.fig.size_int[1])

                # text
                self.label_col.obj[ir, ic].set_x(
                    (bbox.x1 - bbox.x0) / 2
                    - self.cbar.on * (self.cbar.size[0] + self.ws_ax_cbar) / self.fig.size_int[0] / 2
                    + bbox.x0)
                self.label_col.obj[ir, ic].set_y(
                    bbox.y1 + (self.label_col.size[1] / 2 + self.ws_label_col + self.label_col.edge_width)
                    / self.fig.size_int[1])

        # wrap label
        for ir, ic in np.ndindex(self.label_wrap.obj.shape):
            if self.label_wrap.obj[ir, ic]:
                # Get axes position
                bbox = self.axes.obj[ir, ic].get_position()

                # Move and resize y-dimension
                lab_y0 = \
                    bbox.y1 * self.fig.size_int[1] \
                    + np.floor(self.axes.edge_width / 2) \
                    + np.ceil(self.label_wrap.edge_width / 2)
                height = np.ceil(self.label_wrap.size[1] + self.label_wrap.edge_width) / self.fig.size_int[1]

                self.label_wrap.obj_bg[ir, ic].set_y(lab_y0 / self.fig.size_int[1])
                self.label_wrap.obj_bg[ir, ic].set_height(height)

                bbox = self.axes.obj[ir, ic].get_position()
                lab_x0 = np.round(bbox.x0 * self.fig.size_int[0]) - np.floor(self.axes.edge_width / 2) \
                    + np.floor(self.label_wrap.edge_width / 2)
                lab_x1 = np.round(bbox.x1 * self.fig.size_int[0]) - np.ceil(self.label_wrap.edge_width / 2) \
                    + np.ceil(self.axes.edge_width / 2)
                self.label_wrap.obj_bg[ir, ic].set_x(lab_x0 / self.fig.size_int[0])
                self.label_wrap.obj_bg[ir, ic].set_width((lab_x1 - lab_x0) / self.fig.size_int[0])

                # Move text
                self.label_wrap.obj[ir, ic].set_x(
                    (bbox.x1 - bbox.x0) / 2 + bbox.x0 - (self.cbar.on * (self.cbar.size[0] + self.ws_ax_cbar)) / 2 /
                    self.fig.size_int[0])
                self.label_wrap.obj[ir, ic].set_y(
                    (np.floor(lab_y0) + self.label_wrap.size[1] / 2) / self.fig.size_int[1])

        # wrap title
        if self.title_wrap.on:
            # Get axes position
            bbox = self.axes.obj[0, 0].get_position()

            # Move and size y-dimension
            lab_y0 = \
                bbox.y1 * self.fig.size_int[1] \
                + np.floor(self.axes.edge_width / 2) \
                + 2 * self.label_wrap.edge_width \
                + np.ceil(self.label_wrap.size[1]) \
                + np.ceil(self.title_wrap.edge_width / 2)

            self.title_wrap.obj_bg.set_y(lab_y0 / self.fig.size_int[1])
            self.title_wrap.obj_bg.set_height(
                np.ceil(self.title_wrap.size[1] + self.title_wrap.edge_width) / self.fig.size_int[1])

            # Move and size x-dimension
            lab_x0 = np.round(bbox.x0 * self.fig.size_int[0]) - np.floor(self.axes.edge_width / 2) \
                + np.floor(self.title_wrap.edge_width / 2)
            lab_x1 = np.round(self.axes.obj[0, self.ncol - 1].get_position().x1 * self.fig.size_int[0]) \
                + np.ceil(self.axes.edge_width / 2) - np.ceil(self.title_wrap.edge_width / 2)

            self.title_wrap.obj_bg.set_x(lab_x0 / self.fig.size_int[0])
            self.title_wrap.obj_bg.set_width((lab_x1 - lab_x0) / self.fig.size_int[0])

            # Title wrap text
            self.title_wrap.obj.set_x(
                (self.axes.obj[0, self.ncol - 1].get_position().x1 - bbox.x0) / 2 + bbox.x0
                - (self.cbar.on * (self.cbar.size[0] + self.ws_ax_cbar)) / 2 / self.fig.size_int[0])
            self.title_wrap.obj.set_y((lab_y0 + self.title_wrap.size[1] / 2) / self.fig.size_int[1])

        # Set title position
        if self.title.on:
            self.title.obj_bg.set_y(1 - (self.title.size[1] + self.ws_fig_title) / self.fig.size_int[1])
            if self.title.span == 'fig':
                self.title.obj_bg.set_x(0)
                width = 1
            else:
                self.title.obj_bg.set_x(self.axes.obj[0, 0].get_position().x0
                                        - np.ceil(self.axes.edge_width / 2) / self.fig.size_int[0])
                width = self.axes.obj[0, self.ncol - 1].get_position().x1 - self.axes.obj[0, 0].get_position().x0 \
                    - self.cbar.on * (self.cbar.size[0] + self.ws_ax_cbar) / self.fig.size_int[0] \
                    + 2 * np.ceil(self.axes.edge_width / 2) / self.fig.size_int[0]
            self.title.obj_bg.set_width(width)
            self.title.obj_bg.set_height((self.title.size[1] + 2 * self.title.padding) / self.fig.size_int[1])

            if self.title.span == 'fig':
                self.title.obj.set_x(0.5)
            else:
                self.title.obj.set_x(
                    (self.axes.obj[0, self.ncol - 1].get_position().x1 - self.axes.obj[0, 0].get_position().x0) / 2
                    + self.axes.obj[0, 0].get_position().x0)
            self.title.obj.set_y(1 - np.ceil(self._ws_title / 2 + 2) / self.fig.size_int[1])

        # Fix for shared cbar --> subplots_adjust doesn't work out of the box
        if self.cbar.on and self.cbar.shared:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                # Axes position
                ax = self.axes.obj[ir, ic].get_position()
                ax0 = (self._left + (self.ws_col + self.axes.size[0]) * ic) / self.fig.size_int[0]
                width = self.axes.size[0] / self.fig.size_int[0]
                if ic == self.ncol - 1:
                    width += (self.cbar.size[0] + self.ws_ax_cbar) / self.fig.size_int[0]
                self.axes.obj[ir, ic].set_position([ax0, ax.y0, width, ax.y1 - ax.y0])

                # Column labels
                if self.label_col.obj_bg[ir, ic] is not None:
                    self.label_col.obj_bg[ir, ic].set_x(ax0)
                    self.label_col.obj_bg[ir, ic].set_width(self.axes.size[0] / self.fig.size[0])
                    center_new = self.label_col.obj_bg[ir, ic].get_width() / 2 + self.label_col.obj_bg[ir, ic].get_x()
                    self.label_col.obj[ir, ic].set_x(center_new)

                # Wrap labels
                if self.label_wrap.obj_bg[ir, ic] is not None:
                    self.label_wrap.obj_bg[ir, ic].set_x(ax0)
                    self.label_wrap.obj_bg[ir, ic].set_width(self.axes.size[0] / self.fig.size[0])
                    center_new = self.label_wrap.obj_bg[ir, ic].get_width() / 2 + self.label_wrap.obj_bg[ir, ic].get_x()
                    self.label_wrap.obj[ir, ic].set_x(center_new)

            # Adjust title_wrap
            if self.label_wrap.obj_bg[0, 0] is not None and self.title_wrap.obj_bg is not None:
                # Background
                x0 = self.label_wrap.obj_bg[0, 0]._x0
                x1 = self.label_wrap.obj_bg[0, self.ncol - 1]._x0 + self.label_wrap.obj_bg[0, self.ncol - 1]._width
                self.title_wrap.obj_bg.set_width(x1 - x0)

                # Title text
                lab_pos = self.title_wrap.obj.get_position()
                text_offset = self.axes.obj[0, 0].get_position().width - self.axes.size[0] / self.fig.size[0]
                self.title_wrap.obj.set_x(lab_pos[0] - text_offset)

            # Adjust main figure title
            if self.title.on:
                self.title.obj_bg.set_x(self.axes.obj[0, 0].get_position().x0)

        # Fix for non-shared cbar
        if self.cbar.on and not self.cbar.shared:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                # Force correct size of axes --> cbar can screw this up and we can't guarentee that self.ws_ax_cbar
                # and self.cbar.size[0] render correctly
                ax = self.axes.obj[ir, ic].get_position()
                ax_x0 = np.round(ax.x0 * self.fig.size_int[0])
                width = (self.axes.size[0] + np.ceil(self.axes.edge_width / 2) + self.axes.edge_width
                         + self.ws_ax_cbar + self.cbar.size[0]) / self.fig.size_int[0]
                self.axes.obj[ir, ic].set_position([ax_x0 / self.fig.size_int[0], ax.y0, width, ax.height])

                # Column labels
                if self.label_col.obj_bg[ir, ic] is not None:
                    self.label_col.obj_bg[ir, ic].set_width(
                        width - (self.cbar.size[0] + self.ws_ax_cbar) / self.fig.size_int[0])
                    center_new = self.label_col.obj_bg[ir, ic].get_width() / 2 + self.label_col.obj_bg[ir, ic].get_x()
                    self.label_col.obj[ir, ic].set_x(center_new)

                # Wrap labels
                if self.label_wrap.obj_bg[ir, ic] is not None:
                    bbox = self.axes.obj[ir, ic].get_position()
                    lab_x0 = np.round(bbox.x0 * self.fig.size_int[0]) - np.floor(self.axes.edge_width / 2) \
                        + np.floor(self.label_wrap.edge_width / 2)
                    lab_x1 = np.round(bbox.x1 * self.fig.size_int[0]) - self.ws_ax_cbar - self.cbar.size[0] \
                        - np.ceil(self.label_wrap.edge_width / 2)
                    self.label_wrap.obj_bg[ir, ic].set_x(lab_x0 / self.fig.size_int[0])
                    self.label_wrap.obj_bg[ir, ic].set_width((lab_x1 - lab_x0) / self.fig.size_int[0])

                    center_new = self.label_wrap.obj_bg[ir, ic].get_width() / 2 + self.label_wrap.obj_bg[ir, ic].get_x()
                    self.label_wrap.obj[ir, ic].set_x(center_new)

            # Wrap titles
            if self.title_wrap.obj_bg is not None:
                lab_x0 = self._left + np.floor(self.title_wrap.edge_width / 2)
                ax = self.axes.obj[0, self.ncol - 1].get_position()
                lab_x1 = (ax.x0 + ax.width) * self.fig.size_int[0] - np.ceil(self.title_wrap.edge_width / 2) \
                    - self.ws_ax_cbar - self.cbar.size[0]

                self.title_wrap.obj_bg.set_x(lab_x0 / self.fig.size_int[0])
                self.title_wrap.obj_bg.set_width((lab_x1 - lab_x0) / self.fig.size_int[0])

                center_new = self.title_wrap.obj_bg.get_width() / 2 + self.title_wrap.obj_bg.get_x()
                self.title_wrap.obj.set_x(center_new)

        # Update the legend position
        if self.legend.on and self.legend.location in [0, 11]:
            self._get_legend_position()
            self.legend.obj.set_bbox_to_anchor((self.legend.position[1],
                                                self.legend.position[2]))

        # Update the box labels
        if self.box_group_label.on:
            lab = self.box_group_label
            labt = self.box_group_title
            lab_bg_size = self.box_group_label.size_all_bg.set_index(['ir', 'ic', 'ii', 'jj'])

            # Iterate through labels
            offset_txt = 0.2 * lab.font_size  # to make labels line up better at default font sizes
            if self.axes.edge_width == self.box_group_label.edge_width \
                    and self.axes.edge_color[0] == self.box_group_label.edge_color[0]:
                ax_overlap = self.axes.edge_width
            else:
                ax_overlap = 0
            for ir, ic in np.ndindex(lab.obj.shape):
                bbox = self.axes.obj[ir, ic].get_window_extent()

                if lab.obj[ir, ic] is None:
                    continue
                for ii, jj in np.ndindex(lab.obj[ir, ic].shape):
                    if lab.obj[ir, ic][ii, jj] is None:
                        continue

                    # Position and resize the background boxes
                    if jj == 0:
                        x0 = bbox.x0 - (self.axes.edge_width - self.box_group_label.edge_width) / 2
                    else:
                        x0 = lab.obj_bg[ir, ic][ii, jj - 1].get_window_extent().x1
                    lab.obj_bg[ir, ic][ii, jj].set_x(x0 / self.fig.size_int[0])
                    if jj < lab.obj[ir, ic].shape[1] - 1 and lab.obj_bg[ir, ic][ii, jj + 1] is not None:
                        x1 = x0 + (lab.obj_bg[ir, ic][ii, jj + 1].width
                                   - lab.obj_bg[ir, ic][ii, jj].width) * self.axes.size[0]
                    else:
                        x1 = bbox.x1 + (self.axes.edge_width - self.box_group_label.edge_width) / 2
                    lab.obj_bg[ir, ic][ii, jj].set_width((x1 - x0) / self.fig.size_int[0])

                    # update these for later
                    lab_bg_size.loc[(ir, ic, ii, jj), 'x0'] = x0
                    lab_bg_size.loc[(ir, ic, ii, jj), 'x1'] = x1

                    y0 = np.round(bbox.y0) \
                        - np.ceil(self.axes.edge_width / 2) \
                        + np.ceil(self.box_group_label.edge_width / 2) \
                        - lab_bg_size.loc[ir, ic, :ii, 0].height.sum() \
                        + ax_overlap \
                        + self.box_group_label.edge_width * ii

                    lab.obj_bg[ir, ic][ii, jj].set_y(y0 / self.fig.size_int[1])
                    height = lab_bg_size.loc[ir, ic, ii, jj].height - lab.edge_width
                    lab.obj_bg[ir, ic][ii, jj].set_height(height / self.fig.size_int[1])

                    # Position the label text
                    bk_bbox = lab.obj_bg[ir, ic][ii, jj].get_window_extent()
                    if lab.obj[ir, ic][ii, jj].get_rotation() != 0:
                        lab.obj[ir, ic][ii, jj].set_x(((x1 - x0) / 2 + x0 + offset_txt) / self.fig.size_int[0])
                        lab.obj[ir, ic][ii, jj].set_y(
                            ((bk_bbox.y1 - bk_bbox.y0) / 2 + bk_bbox.y0) / self.fig.size_int[1])
                    else:
                        lab.obj[ir, ic][ii, jj].set_x(((x1 - x0) / 2 + x0) / self.fig.size_int[0])
                        lab.obj[ir, ic][ii, jj].set_y(
                            ((bk_bbox.y1 - bk_bbox.y0) / 2 + bk_bbox.y0 - offset_txt) / self.fig.size_int[1])

                # fix dividers
                if self.box_divider.obj[ir, ic] is not None:
                    dividers = self.box_divider.obj[ir, ic]
                    idx = self.box_group_label.size_all_bg.ii.max()
                    for idiv, div in enumerate(dividers):
                        div.set_transform(self.fig.obj.transFigure)
                        changes = data.changes[data.changes[idx] != 0][1:]
                        if idiv >= len(changes.index):
                            continue
                        div_at = changes.index[idiv]
                        loc = lab_bg_size.loc[ir, ic, idx].iloc[div_at - 1].x1
                        div.set_xdata([loc / self.fig.size_int[0], loc / self.fig.size_int[0]])

                # group title
                labt_size_all = labt.size_all.set_index(['ir', 'ic', 'ii'])
                for ii, jj in np.ndindex(labt.obj[ir, ic].shape):
                    if labt.obj[ir, ic][ii, 0] is None:
                        continue
                    bk_bbox = lab.obj_bg[ir, ic][ii, jj].get_window_extent()
                    labt.obj_bg[ir, ic][ii, 0].set_x(
                        (bbox.x1 + labt.padding) / self.fig.size_int[0])
                    labt.obj_bg[ir, ic][ii, 0].set_width(labt_size_all.loc[ir, ic, ii].width / self.fig.size_int[0])
                    labt.obj_bg[ir, ic][ii, 0].set_y(bk_bbox.y0 / self.fig.size_int[1])

                    labt.obj[ir, ic][ii, 0].set_x((bbox.x1 + np.floor(labt_size_all.loc[ir, ic, ii].width / 2)
                                                   + labt.padding) / self.fig.size_int[0])
                    labt.obj[ir, ic][ii, 0].set_y(
                        ((bk_bbox.y1 - bk_bbox.y0) / 2 + bk_bbox.y0 - offset_txt) / self.fig.size_int[1])

        # Text label adjustments
        if self.text.on:
            self._set_text_position(self.text)
        if self.fit.on and self.fit.eqn:
            self._set_text_position(self.fit)

        # Gantt today text
        if self.gantt.on and self.gantt.today.on:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                if self.gantt.today.obj[ir, ic] is None:
                    continue
                ax = self.axes.obj[ir, ic]
                txt = self.gantt.today.obj[ir, ic][0]
                x, y = txt.get_position()
                x0 = mdates.date2num(x)
                w = txt.get_window_extent().width
                h = txt.get_window_extent().height
                transform = (ax.transData + ax.transAxes.inverted())
                x_axis, y_axis = transform.transform((x0, y))
                if self.gantt.date_location == 'top':
                    y0 = -(h + self.gantt.today.edge_width + self.gantt.today.padding) / self.axes.size[1]
                else:
                    y0 = 1 + (h / 2 + self.gantt.today.edge_width + self.gantt.today.padding) / self.axes.size[1]
                txt.set_position((x_axis - w / 2 / self.axes.size[0], y0))
                txt.set_transform(ax.transAxes)

        # Gantt workstreams labels
        if self.gantt.on and self.gantt.workstreams.on and self.gantt.workstreams.location != 'inline':
            for ir, ic in np.ndindex(self.axes.obj.shape):
                # y-position: line up with the y-gridlines and y-tick labels
                y0 = 0
                valid_ticks = [i for (i, f) in enumerate(self.axes.obj[ir, ic].get_yticklabels())
                               if f.get_position()[1] >= 0]
                ygridlines = [self.axes.obj[ir, ic].get_ygridlines()[i] for i in valid_ticks]
                y_delta = np.diff([f.get_window_extent().y0 for f in ygridlines]).mean()

                # x-position
                edge = np.ceil(self.gantt.workstreams.edge_width / 2)
                if self.gantt.workstreams.location == 'right':
                    # right align
                    lab_x0 = self.axes.obj[ir, ic].get_window_extent().x1
                    lab_x0 += edge
                    lab_x0_text = lab_x0 + self.gantt.workstreams.size[0] / 2
                else:
                    # left align
                    lab_x0 = self.axes.obj[ir, ic].get_window_extent().x0 - self._labtick_y
                    lab_x0 -= edge + self.gantt.workstreams.size[0]
                    lab_x0_text = lab_x0 + self.gantt.workstreams.size[0] / 2

                for iws, ws in enumerate(self.gantt.workstreams.obj[ir, ic]):
                    if self.gantt.workstreams.location.lower() == 'inline':
                        break
                    # x-position and width of background rectangle
                    ws_bg = self.gantt.workstreams.obj_bg[ir, ic][iws]
                    ws_bg.set_x(lab_x0 / self.fig.size_int[0])
                    ws_bg.set_width((self.gantt.workstreams.size[0] + edge) / self.fig.size_int[0])

                    # y-position and height of background rectangle
                    if iws == 0:
                        y0 = ygridlines[0].get_window_extent().y0
                    else:
                        y0_idx = int(sum(self.gantt.workstreams.rows[ir, ic][:iws]))
                        y0 = ygridlines[y0_idx].get_window_extent().y0
                    if iws == len(self.gantt.workstreams.obj[ir, ic]) - 1:
                        y1_idx = int(sum(self.gantt.workstreams.rows[ir, ic][:iws + 1]))
                        y1 = ygridlines[y1_idx - 1].get_window_extent().y0 + y_delta
                    else:
                        y1_idx = int(sum(self.gantt.workstreams.rows[ir, ic][:iws + 1]))
                        y1 = ygridlines[y1_idx].get_window_extent().y0
                    if iws == len(self.gantt.workstreams.obj[ir, ic]) - 1 and \
                            self.grid_major_y.width[0] > self.axes.edge_width:
                        # Correct top label box so it better aligns with axes edge
                        y1 -= np.ceil(self.grid_major_y.width[0] - self.axes.edge_width)

                    ws_bg.set_y((y0 - y_delta / 2) / self.fig.size_int[1])
                    ws_bg.set_height((y1 - y0) / self.fig.size_int[1])

                    # Special offset if certain characters are missing (crude; only tested on mac with default font)
                    if not any(e in ['y', 'j', 'g', 'q', 'p'] for e in ws.get_text()):
                        offsetx = 0.2 * ws.get_font()._size + 2
                    elif any(e in ['[', ']'] for e in ws.get_text()):
                        offsetx = 0.125 * ws.get_font()._size
                    else:
                        offsetx = 0.125 * ws.get_font()._size
                    if self.gantt.workstreams.location == 'right':
                        ws.set_x((lab_x0_text - offsetx) / self.fig.size_int[0])
                    elif self.gantt.workstreams.location == 'left':
                        ws.set_x((lab_x0_text + offsetx) / self.fig.size_int[0])
                    ws.set_y(((y1 + y0 - y_delta) / 2) / self.fig.size_int[1])

        # Gantt workstream title
        if self.gantt.on and self.gantt.workstreams_title.on:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                # y-grid position
                valid_ticks = [i for (i, f) in enumerate(self.axes.obj[ir, ic].get_yticklabels())
                               if f.get_position()[1] >= 0]
                ygridlines = [self.axes.obj[ir, ic].get_ygridlines()[i] for i in valid_ticks]
                y_delta = np.diff([f.get_window_extent().y0 for f in ygridlines]).mean()
                y0 = ygridlines[0].get_window_extent().y0 - y_delta / 2
                y1 = ygridlines[-1].get_window_extent().y0 + y_delta / 2
                if self.grid_major_y.width[0] > self.axes.edge_width:
                    y1 -= np.ceil(self.grid_major_y.width[0] - self.axes.edge_width)

                # x-position
                if self.gantt.workstreams.location.lower() == 'inline':
                    break
                elif self.gantt.workstreams.location == 'right':
                    lab_x0 = self.axes.obj[ir, ic].get_window_extent().x1 + \
                        np.ceil(self.gantt.workstreams_title.edge_width / 2) + \
                        self.gantt.workstreams.size[0]
                else:  # left align
                    lab_x0 = \
                        self._left \
                        - self._labtick_y \
                        - np.ceil(self.gantt.workstreams_title.edge_width / 2) \
                        - self.gantt.workstreams_title.size[0]
                    if self.gantt.workstreams.on:
                        lab_x0 -= self.gantt.workstreams.size[0]
                        lab_x0 -= np.ceil(self.gantt.workstreams.edge_width / 2)

                self.gantt.workstreams_title.obj_bg[ir, ic].set_x(lab_x0 / self.fig.size_int[0])
                self.gantt.workstreams_title.obj_bg[ir, ic].set_width(
                    self.gantt.workstreams_title.size[0] / self.fig.size_int[0])
                self.gantt.workstreams_title.obj_bg[ir, ic].set_y(y0 / self.fig.size_int[1])
                self.gantt.workstreams_title.obj_bg[ir, ic].set_height((y1 - y0) / self.fig.size_int[1])

                # Special offset if certain characters are missing (crude; only tested on mac with default font)
                obj = self.gantt.workstreams_title.obj[ir, ic]
                if not any(e in ['y', 'j', 'g', 'q', 'p'] for e in obj.get_text()):
                    offsetx = 0.05 * obj.get_font()._size - 2
                elif any(e in ['[', ']'] for e in self.gantt.workstreams_title.obj[ir, ic].get_text()):
                    offsetx = 0.125 * obj.get_font()._size
                else:
                    offsetx = 0.125 * obj.get_font()._size
                if self.gantt.workstreams.location == 'right':
                    self.gantt.workstreams_title.obj[ir, ic].set_x(
                        (lab_x0 + self.gantt.workstreams_title.size[0] / 2 + offsetx) / self.fig.size[0])
                elif self.gantt.workstreams.location == 'left':
                    self.gantt.workstreams_title.obj[ir, ic].set_x(
                        (lab_x0 + self.gantt.workstreams_title.size[0] / 2 - offsetx) / self.fig.size[0])
                self.gantt.workstreams_title.obj[ir, ic].set_y(
                    (self.axes.obj[ir, ic].get_position().y1 - self.axes.obj[ir, ic].get_position().y0) / 2
                    + self.axes.obj[ir, ic].get_position().y0)

        # Gantt hide first x-tick mark if it falls on the yticklabel box edge
        if self.gantt.label_boxes and self.gantt.labels_as_yticks and self.gantt.date_location != 'top':
            for ir, ic in np.ndindex(self.axes.obj.shape):
                ax = self.axes.obj[ir, ic]
                if ax.xaxis.get_major_ticks()[0].get_loc() == ax.get_xlim()[0]:
                    ax.xaxis.get_major_ticks()[0].tick1line.set_visible(False)

        # Gantt milestone labels
        if self.gantt.on and self.gantt.milestone_text.on and \
                self.gantt.milestone_text.obj[ir, ic] is not None and len(self.gantt.milestone_text.obj[ir, ic]) > 0:
            for itxt, txt in enumerate(self.gantt.milestone_text.obj[ir, ic]):
                ax = self.axes.obj[ir, ic]
                x, y = txt.get_position()
                x0 = mdates.date2num(x)
                w = txt.get_window_extent().width
                h = txt.get_window_extent().height
                transform = (ax.transData + ax.transAxes.inverted())
                x_axis, y_axis = transform.transform((x0, y))
                if self.gantt.milestone_text.location == 'top':
                    x0 = x_axis - w / 2 / self.axes.size[0]
                    y0 = y_axis + (2 + self.gantt.milestone_marker_size) / (self.axes.size[1])  # 2 == edge_width + 1
                else:
                    x0 = x_axis + (2 + self.gantt.milestone_marker_size) / (self.axes.size[0])  # 2 == edge_width + 1
                    y0 = y_axis - (h / 2 - 1) / self.axes.size[1]
                txt.set_position((x0, y0))
                txt.set_transform(ax.transAxes)

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.on:
            self.title.obj, self.title.obj_bg = self.add_label(0, 0, self.title)

    def _set_scientific(self, ax: mplp.Axes, tp: dict, idx: int = 0) -> mplp.Axes:
        """Turn off scientific notation.

        Args:
            ax: MPL axes object
            tp: tick label, position dict from mpl_get_ticks
            idx: axis number

        Returns:
            updated axis
        """

        def get_sci(ticks, limit):
            """Get the scientific notation format string.

            Args:
                ticks: tp['?']['ticks']
                limit: ax.get_?lim()

            Returns:
                format string
            """

            max_dec = 0
            for itick, tick in enumerate(ticks):
                if tick != 0:
                    if tick < 0:
                        power = np.nan
                    else:
                        power = np.ceil(-np.log10(tick))
                    if np.isnan(power) or tick < limit[0] or tick > limit[1]:
                        continue
                    dec = utl.get_decimals(tick * 10**power)
                    max_dec = max(max_dec, dec)
            dec = '%%.%se' % max_dec
            return dec

        if idx == 0:
            lab = ''
        else:
            lab = '2'

        # Select scientific notation unless specified
        bestx, besty = False, False

        if self.tick_labels_major_x.sci == 'best' and len(tp['x']['ticks']) > 0:
            xrange = tp['x']['ticks'][-1] - tp['x']['ticks'][0]
            nonzero = tp['x']['ticks'][tp['x']['ticks'] != 0]
            xthresh = np.any(np.abs(nonzero) <= self.auto_tick_threshold[0]) or \
                np.any(np.abs(nonzero) >= self.auto_tick_threshold[1])
            if xrange <= self.auto_tick_threshold[0] or \
                    xrange >= self.auto_tick_threshold[1] or xthresh:
                tick_labels_major_x_sci = True
            else:
                tick_labels_major_x_sci = False
            bestx = True
        elif self.tick_labels_major_x.sci == 'best':
            tick_labels_major_x_sci = False
        else:
            tick_labels_major_x_sci = self.tick_labels_major_x.sci

        if self.tick_labels_major_y.sci == 'best' and len(tp['y']['ticks']) > 0:
            yrange = tp['y']['ticks'][-1] - tp['y']['ticks'][0]
            nonzero = tp['y']['ticks'][tp['y']['ticks'] != 0]
            ythresh = np.any(np.abs(nonzero) <= self.auto_tick_threshold[0]) or \
                np.any(np.abs(nonzero) >= self.auto_tick_threshold[1])
            if yrange <= self.auto_tick_threshold[0] or \
                    yrange >= self.auto_tick_threshold[1] or ythresh:
                tick_labels_major_y_sci = True
            else:
                tick_labels_major_y_sci = False
            besty = True
        elif self.tick_labels_major_y.sci == 'best':
            tick_labels_major_y_sci = False
        else:
            tick_labels_major_y_sci = self.tick_labels_major_y.sci

        # Set labels
        logx = getattr(self, f'axes{lab}').scale in LOGX + SYMLOGX + LOGITX
        if self.name in ['hist'] and self.hist.horizontal is True and self.hist.kde is False:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
            self._major_x_formatter = MaxNLocator(integer=True)
        elif not tick_labels_major_x_sci and self.name not in ['box', 'heatmap'] and not logx:
            try:
                ax.get_xaxis().get_major_formatter().set_scientific(False)
            except:  # noqa
                pass
        elif not tick_labels_major_x_sci and self.name not in ['box', 'heatmap']:
            try:
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

                # need to recompute the ticks after changing formatter
                tp = mpl_get_ticks(ax, minor=self.ticks_minor.on)
                for itick, tick in enumerate(tp['x']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['x']['label_text'][itick] = '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_xticklabels(tp['x']['label_text'])
            except:  # noqa
                pass
        elif ((bestx and not logx
                or not bestx and tick_labels_major_x_sci and logx)
                and self.name not in ['box', 'heatmap']) \
                or self.tick_labels_major_x.sci is True:
            xlim = ax.get_xlim()
            dec = get_sci(tp['x']['ticks'], xlim)
            self.ticks_major_x.sci = True
            ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter(dec))

        logy = getattr(self, f'axes{lab}').scale in LOGY + SYMLOGY + LOGITY
        if self.name in ['hist'] and self.hist.horizontal is False and self.hist.kde is False:
            ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        elif not tick_labels_major_y_sci and self.name not in ['heatmap'] and not logy:
            try:
                ax.get_yaxis().get_major_formatter().set_scientific(False)
            except:  # noqa
                pass
        elif not tick_labels_major_y_sci and self.name not in ['heatmap']:
            try:
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())

                # need to recompute the ticks after changing formatter
                tp = mpl_get_ticks(ax, minor=self.ticks_minor.on)
                for itick, tick in enumerate(tp['y']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['y']['label_text'][itick] = '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_yticklabels(tp['y']['label_text'])
            except:  # noqa
                pass
        elif ((besty and not logy
                or not besty and tick_labels_major_y_sci and logy)
                and self.name not in ['heatmap']) \
                or self.tick_labels_major_y.sci is True:
            ylim = ax.get_ylim()
            dec = get_sci(tp['y']['ticks'], ylim)
            self.tick_labels_major_y.sci = True
            ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter(dec))
            self._major_y_formatter = ticker.FormatStrFormatter(dec)

        # Cbar z-axis
        if self.tick_labels_major_z.sci is True and self.cbar.on:
            for ir, ic in np.ndindex(self.axes.obj.shape):
                if not self.cbar.obj[ir, ic]:
                    continue
                ax = self.cbar.obj[ir, ic].ax
                ylim = ax.get_ylim()
                tp = mpl_get_ticks(ax, minor=False)
                dec = get_sci(tp['y']['ticks'], ylim)
                ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter(dec))
        elif self.tick_labels_major_z.sci is False and self.cbar.on:
            try:
                for ir, ic in np.ndindex(self.axes.obj.shape):
                    self.cbar.obj[ir, ic].ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
                    self.cbar.obj[ir, ic].ax.get_yaxis().get_major_formatter().set_scientific(False)
            except:  # noqa
                pass
        return ax

    def _set_text_position(self, obj):
        """Move text labels to the correct location."""
        for ir, ic in np.ndindex(self.axes.obj.shape):
            plot_num = utl.plot_num(ir, ic, self.ncol) - 1
            ax = self.axes.obj[ir, ic]
            offsetx = ir * self.axes.size[0]
            offsety = ic * self.axes.size[1]

            if hasattr(obj.text, 'values'):
                text_vals = obj.text.values
            elif isinstance(obj.obj, np.ndarray) and \
                    (isinstance(obj.obj[ir, ic], mpl.text.Text) or isinstance(obj.obj[ir, ic], list)):
                text_vals = obj.obj[ir, ic]
            else:
                text_vals = obj.obj

            for itext, txt in enumerate(text_vals):
                if isinstance(txt, dict):
                    if plot_num in txt.keys():
                        txt = txt[plot_num]
                    else:
                        continue

                # Set the coordinate so text is anchored to figure, axes, or the current data range
                coord = None if not hasattr(obj, 'coordinate') \
                    else obj.coordinate.lower()
                if coord == 'figure':
                    transform = self.fig.obj.transFigure
                elif coord == 'data':
                    transform = ax.transData
                else:
                    transform = ax.transAxes  # noqa
                units = 'pixel' if not hasattr(obj, 'units') else getattr(obj, 'units')

                # Get position
                if hasattr(obj, 'position') and \
                        str(type(getattr(obj, 'position'))) == str(RepeatedList):
                    position = copy.copy(getattr(obj, 'position')[itext])
                elif hasattr(obj, 'position'):
                    position = copy.copy(getattr(obj, 'position'))
                else:
                    position = txt.get_position()

                # Convert position to correct units
                if units == 'pixel' and coord == 'figure':
                    if isinstance(position[0], str):
                        position[0] = position[0].replace('xmin', 0).replace('xmax', self.fig.size_int[0])
                        position[0] = utl.arithmetic_eval(str(position[0]))
                    if isinstance(position[1], str):
                        position[1] = position[1].replace('ymin', '0').replace('ymax', str(self.fig.size_int[1]))
                        position[1] = utl.arithmetic_eval(str(position[1]))
                    position[0] /= self.fig.size_int[0]
                    offsetx /= self.fig.size_int[0]
                    position[1] /= self.fig.size_int[1]
                    offsety /= self.fig.size_int[1]
                elif units == 'pixel' and coord != 'data':
                    if isinstance(position[0], str):
                        position[0] = position[0].replace('xmin', 0).replace('xmax', self.axes.size[0])
                        position[0] = utl.arithmetic_eval(str(position[0]))
                    if isinstance(position[1], str):
                        position[1] = position[1].replace('ymin', '0').replace('ymax', str(self.axes.size[1]))
                        position[1] = utl.arithmetic_eval(str(position[1]))
                    position[0] = position[0] / self.axes.size[0]
                    position[1] = position[1] / self.axes.size[1]

                # Something goes weird with x = 0 so we need to adjust slightly
                if position[0] == 0:
                    position[0] = 0.01

                # Move text
                if isinstance(txt, mpl.text.Text):
                    txt.set_transform(transform)
                    txt.set_position(position)
                elif obj.obj[ir, ic] is not None:
                    obj.obj[ir, ic][itext].set_transform(transform)
                    obj.obj[ir, ic][itext].set_position(position)

    def _set_tick_position(self, data):
        """Update the tick positions except when tick lines have direction=='in'."""

        # Gantt plot move ticks and labels to top (this doesn't work in set_axes_ticks)
        if self.name == 'gantt' and self.gantt.on:
            self._set_tick_position_gantt(data)

        # No adjustment needed for inward ticks
        if self.ticks_major_x.direction == 'in' and self.ticks_major_y.direction == 'in' and not self.axes.twin_y:
            return

        # Outward ticks
        xticks = self.tick_labels_major_x.size_all.groupby(['ir', 'ic']).mean()
        yticks = self.tick_labels_major_y.size_all.groupby(['ir', 'ic']).mean()
        for ir, ic in np.ndindex(self.axes.obj.shape):
            ax_loc = self.axes.obj[ir, ic].get_window_extent()

            # primary x-axis
            if self.ticks_major_x.direction != 'in':
                ax_y0 = ax_loc.y0
                tt_y1 = xticks.loc[ir, ic]['y1']
                pad = ax_y0 - tt_y1 - xticks.loc[ir, ic]['height'] / 2
                self.axes.obj[ir, ic].tick_params(axis='x', which='both', pad=pad / 2)

            # primary y-axis
            if self.ticks_major_y.direction != 'in':
                ax_x0 = ax_loc.x0
                tt_x1 = yticks.loc[ir, ic]['x1'] - 2
                pad = ax_x0 - tt_x1 - yticks.loc[ir, ic]['width'] / 2
                self.axes.obj[ir, ic].tick_params(axis='y', which='both', pad=pad / 2)

            # secondary x-axis
            if self.axes.twin_y and self.ticks_minor_x.direction != 'in':
                xticks2 = self.tick_labels_major_x2.size_all.groupby(['ir', 'ic']).mean()
                ax_y1 = ax_loc.y1
                tt_y0 = xticks2.loc[ir, ic]['y0']
                pad = tt_y0 - ax_y1 + xticks2.loc[ir, ic]['height'] / 2
                self.axes2.obj[ir, ic].tick_params(axis='x', which='both', pad=pad / 2)

            # secondary y-axis
            if self.axes.twin_x and self.ticks_minor_y.direction != 'in':
                yticks2 = self.tick_labels_major_y2.size_all.groupby(['ir', 'ic']).mean()
                ax_x1 = ax_loc.x1
                tt_x0 = yticks2.loc[ir, ic]['x0'] + 2
                pad = tt_x0 - ax_x1 + yticks2.loc[ir, ic]['width'] / 2
                self.axes2.obj[ir, ic].tick_params(axis='y', which='both', pad=pad / 2)

    def _set_tick_position_gantt(self, data):
        """Tick position specific to gantt plots."""
        # move xticks on top - need to make some mpl adjustments and reset tick params
        if self.gantt.date_location == 'top':
            ticks_font = font_manager.FontProperties(
                family=self.tick_labels_major_x.font,
                size=self.tick_labels_major_x.font_size,
                style=self.tick_labels_major_x.font_style,
                weight=self.tick_labels_major_x.font_weight,
                )
            style = dict(edgecolor=self.tick_labels_major_x.edge_color[0],
                         facecolor=self.tick_labels_major_x.fill_color[0],
                         linewidth=self.tick_labels_major_x.edge_width,
                         alpha=max(self.tick_labels_major_x.edge_alpha,
                                   self.tick_labels_major_x.fill_alpha),
                         pad=self.tick_labels_major_x.padding,
                         )
            rotation = self.tick_labels_major_x.rotation

            for ir, ic in np.ndindex(self.axes.obj.shape):
                ax_y0 = self.axes.obj[ir, ic].get_window_extent().y0
                xticks = self.tick_labels_major_x.size_all.groupby(['ir', 'ic']).mean()
                tt_y1 = xticks.loc[ir, ic]['y1']
                pad = ax_y0 - tt_y1

                self.axes.obj[ir, ic].xaxis.set_ticks_position('top')
                self.axes.obj[ir, ic].xaxis.set_label_position('top')
                self.axes.obj[ir, ic].tick_params(axis='x',
                                                  which='major',
                                                  labelrotation=self.tick_labels_major_x.rotation,
                                                  pad=pad,
                                                  colors=self.ticks_major_x.color[0],
                                                  labelcolor=self.tick_labels_major_x.font_color,
                                                  zorder=1000)
                if self.gantt.date_type is None:
                    self.axes.obj[ir, ic].minorticks_off()
                else:
                    self.axes.obj[ir, ic].tick_params(axis='x',
                                                      which='major',
                                                      width=0,
                                                      length=0)
                    self.axes.obj[ir, ic].tick_params(axis='x',
                                                      which='minor',
                                                      width=0,
                                                      length=0)

                for text in self.axes.obj[ir, ic].get_xticklabels(which='major'):
                    if rotation != 0:
                        text.set_rotation(rotation)
                    text.set_fontproperties(ticks_font)
                    text.set_bbox(style)

                # Style offset text
                self.axes.obj[ir, ic].xaxis.get_offset_text().set_fontproperties(ticks_font)

        # Set custom tick labels (ouch!)
        for ir, ic in np.ndindex(self.axes.obj.shape):
            # Shortest duration becomes actual tick labels while other durations become custom text boxes
            # (i.e., week < month/month-year < quarter/quarter-year < year)
            ax = self.axes.obj[ir, ic]

            if len(self.gantt.date_type) > 0:
                locator = {'year': mdates.YearLocator(),
                           'quarter': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
                           'quarter-year': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
                           'month': mdates.MonthLocator(bymonthday=15),
                           'month-year': mdates.MonthLocator(bymonthday=15),
                           'week': mdates.WeekdayLocator(mdates.TH)
                           }
                minor_locator = {'year': mdates.MonthLocator(bymonth=[1]),
                                 'quarter': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
                                 'quarter-year': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
                                 'month': mdates.MonthLocator(bymonth=range(1, 13)),
                                 'month-year': mdates.MonthLocator(bymonth=range(1, 13)),
                                 'week': mdates.WeekdayLocator(mdates.MO)
                                 }
                fmt = {'year': mdates.DateFormatter('%Y'),
                       'quarter': mdates.DateFormatter('Q%m'),
                       'quarter-year': mdates.DateFormatter('%yQ%m'),
                       'month': mdates.DateFormatter('%b'),
                       'month-year': mdates.DateFormatter('%b\n%y'),
                       'week': mdates.DateFormatter('WW%W')
                       }

                def get_tick_bounds(ax: 'mpl.axes', date: Union[datetime.datetime, float],
                                    date_type: str) -> Dict[str, Tuple]:
                    """
                    Compute the date and pixel location for the previous, current, and next minor tick gridlines

                    Args:
                        ax: the axes object
                        date: current date as a datetime or mpl float
                        date_type: 'year', 'quarter', 'quarter-year', 'month', 'month-year', 'week'

                    Returns:
                        next date as a datetime
                    """
                    if isinstance(date, float):
                        date = mdates.num2date(date)

                    dd, mm, yy = utl.date_vals(date)
                    xmin, xmax = ax.get_xlim()
                    px_x0, px_x1 = ax.get_window_extent().x0, ax.get_window_extent().x1

                    if date_type in ['year']:
                        tick = datetime.datetime(yy, 7, dd)
                        left = datetime.datetime(yy, 1, 1)
                        right = datetime.datetime(yy + 1, 1, 1)
                    elif date_type in ['month', 'month-year']:
                        tick = datetime.datetime(yy, mm, dd)
                        left = datetime.datetime(yy, mm, 1)
                        right = datetime.datetime(yy + (mm + 1) // 13, (mm % 12) + 1, 1)
                    elif date_type in ['quarter', 'quarter-year']:
                        if mm not in [1, 4, 7, 10]:
                            # Use this for ticks in partial boxes
                            tick = datetime.datetime(yy, mm, dd)
                            left = datetime.datetime(yy, mm - 1, 1)
                            right = datetime.datetime(yy + (mm > 10), (((mm - 1) // 3) + 1) % 4 * 3 + 1, 1)
                        elif dd != 15:
                            # Use this if the tick is at the beginning of the quarter
                            tick = datetime.datetime(yy + (mm + 1) // 13, mm + 1, 15)
                            left = datetime.datetime(yy, mm, 1)
                            right = datetime.datetime(yy + (mm > 10), (mm + 2) % 12 + 1, 1)
                        else:
                            # Use this if the tick has already been shifted to mid-quarter
                            tick = datetime.datetime(yy, mm, dd)
                            left = datetime.datetime(yy, mm - 1, 1)
                            right = datetime.datetime(yy + (mm > 10), (mm + 2) % 12 + 1, 1)
                    elif date_type in ['week']:
                        tick = datetime.datetime(yy, mm, dd)
                        left = tick - datetime.timedelta(days=date.weekday())
                        right = tick + datetime.timedelta(days=date.weekday() % 7 + 1)
                        tick = datetime.datetime(yy, mm, dd, 12)  # offset by 12 hours to center
                    else:
                        raise NotImplementedError(f'tick bounds not implemented for {date_type}')

                    if left < mdates.num2date(xmin).replace(tzinfo=None):
                        left = mdates.num2date(xmin)
                    if right > mdates.num2date(xmax).replace(tzinfo=None):
                        right = mdates.num2date(xmax)

                    return {'left': {'date': left.replace(tzinfo=None),
                                     'float': mdates.date2num(left),
                                     'px': utl.date_to_pixels(left, xmin, xmax, px_x0, px_x1)},
                            'tick': {'date': tick.replace(tzinfo=None),
                                     'float': mdates.date2num(tick),
                                     'px': utl.date_to_pixels(tick, xmin, xmax, px_x0, px_x1)},
                            'right': {'date': right.replace(tzinfo=None),
                                      'float': mdates.date2num(right),
                                      'px': utl.date_to_pixels(right, xmin, xmax, px_x0, px_x1)}
                            }

                # Select the major tick date type
                DATE_TYPES = ['week', 'month', 'month-year', 'quarter', 'quarter-year', 'year']
                for tick_type in DATE_TYPES:
                    if tick_type in self.gantt.date_type:
                        primary = tick_type
                        break

                # Set x-major ticks
                ax.xaxis.set_major_locator(locator[primary])
                ax.xaxis.set_major_formatter(fmt[primary])

                # Update some tick labels
                if primary in ['quarter', 'quarter-year']:
                    labels = [f.get_text().replace('04', '2').replace('07', '3').replace('10', '4').replace('01', '1')
                              for f in ax.get_xticklabels()]
                    ax.set_xticklabels(labels)
                else:
                    labels = ax.get_xticklabels()

                # Adjust tick positions to center in tick label boxes
                xmin, xmax = ax.get_xlim()
                xticks = ax.get_xticks()
                partial_start, partial_end = False, False
                tick_font_size = self.tick_labels_major_x.font_size
                height = ax.get_xticklabels()[0].get_window_extent().y1 - ax.get_window_extent().y1 + \
                    2 * self.gantt.box_padding_y

                t0 = get_tick_bounds(ax, xticks[0], primary)
                if t0['left']['date'] != t0['right']['date']:
                    xticks[0] = t0['left']['float'] + (t0['right']['float'] - t0['left']['float']) / 2
                    partial_start = True
                else:
                    xticks[0] = t0['tick']['float']

                for ixt, xt in enumerate(xticks[1:-1]):
                    tt = get_tick_bounds(ax, xt, primary)
                    xticks[ixt + 1] = tt['tick']['float']
                    if ixt == 0:
                        # Check tick label font size and adjust if too big for the tick box (not perfect)
                        label_width = ax.get_xticklabels()[ixt].get_window_extent().width
                        box_width = tt['right']['px'] - tt['left']['px']
                        if box_width < label_width:
                            tick_font_size = self.tick_labels_major_x.font_size * (box_width - 2) / label_width

                t1 = get_tick_bounds(ax, xticks[-1], primary)
                if t1['left']['date'] != t1['right']['date']:
                    xticks[-1] = t1['left']['float'] + (t1['right']['float'] - t1['left']['float']) / 2
                    partial_end = True
                else:
                    xticks[-1] = t1['tick']['float']

                # Add additional ticks if there is space in the partial tick boxes
                if partial_start:
                    label_width = ax.get_xticklabels()[0].get_window_extent().width
                    if t0['left']['px'] - ax.get_window_extent().x0 > label_width:
                        xticks = [xmin + (t0['left']['float'] - xmin) / 2] + list(xticks)
                        if primary in ['quarter', 'quarter-year']:
                            label = mdates.num2date(xticks[0])
                            label = datetime.datetime(label.year, (label.month - 1) // 3 + 1, label.day)
                            label = label.strftime(fmt[primary].fmt).replace('0', '')
                        elif primary == 'week':
                            label = mdates.num2date(xticks[-1]) + datetime.timedelta(days=7)
                            label = label.strftime(fmt[primary].fmt)
                        else:
                            label = mdates.num2date(xticks[0]).strftime(fmt[primary].fmt)
                        labels = [label] + labels
                if partial_end:
                    label_width = ax.get_xticklabels()[-1].get_window_extent().width
                    if ax.get_window_extent().x1 - t1['right']['px'] > label_width:
                        xticks = list(xticks) + [t1['right']['float'] + (xmax - t1['right']['float']) / 2]
                        if primary in ['quarter', 'quarter-year']:
                            label = mdates.num2date(xticks[-1])
                            label = datetime.datetime(label.year, (label.month - 1) // 3 + 1, label.day)
                            label = label.strftime(fmt[primary].fmt).replace('0', '')
                        elif primary == 'week':
                            label = mdates.num2date(xticks[-1]) + datetime.timedelta(days=7)
                            label = label.strftime(fmt[primary].fmt)
                        else:
                            label = mdates.num2date(xticks[-1]).strftime(fmt[primary].fmt)
                        labels = labels + [label]

                # Update the major ticks
                ax.set_xticks(xticks)
                ax.set_xticklabels(labels, fontsize=tick_font_size)
                ax.set_xlim(left=xmin, right=xmax)

                # Minor ticks
                if self.ticks_minor_x.on:
                    ax.xaxis.set_minor_locator(minor_locator[primary])

                # Remove tick labels that get cut off of the axes
                ax = self.axes.obj[ir, ic]
                ax_x0 = ax.get_window_extent().x0
                ax_x1 = ax.get_window_extent().x1
                tlabs = ax.xaxis.get_ticklabels()
                for itt, tt in enumerate(tlabs):
                    # Left side
                    x0 = tt.get_window_extent().x0
                    if x0 <= ax_x0 - 1:  # give a 1 pixel buffer
                        tt.set_visible(False)
                        if self.gantt.quarters[ir, ic] is not None and \
                                self.gantt.quarters[ir, ic][itt] != self.gantt.quarters[ir, ic][itt + 1]:
                            self.gantt.quarters[ir, ic][itt] = ''
                        if self.gantt.years[ir, ic] is not None and \
                                self.gantt.years[ir, ic][itt] != self.gantt.years[ir, ic][itt + 1]:
                            self.gantt.years[ir, ic][itt] = ''
                        continue

                    # Right side
                    x1 = tt.get_window_extent().x1
                    if x1 > ax_x1 + 1:  # give a 1 pixel buffer
                        tt.set_visible(False)
                        if self.gantt.quarters[ir, ic] is not None and \
                                self.gantt.quarters[ir, ic].count(self.gantt.quarters[ir, ic][itt]) == 1:
                            self.gantt.quarters[ir, ic][itt] = ''
                        if self.gantt.years[ir, ic] is not None and \
                                self.gantt.years[ir, ic].count(self.gantt.years[ir, ic][itt]) == 1:
                            self.gantt.years[ir, ic][itt] = ''

            # Gantt tick label boxes
            if self.gantt.label_boxes:
                # ytick labels and boxes
                if self.gantt.labels_as_yticks:
                    # Convenient variables
                    tick_width = self.tick_labels_major_y.size_all.groupby(['ir', 'ic']).max()['width'].iloc[0]
                    ax_x0 = ax.get_window_extent().x0  # right edge of tick boxes
                    yticklabs = ax.get_yticklabels()
                    ygridlines = ax.get_ygridlines()
                    if self.gantt.workstreams.on and self.gantt.workstreams.location == 'inline':
                        inline_workstreams = data.df_rc[data.workstreams].unique()
                    else:
                        inline_workstreams = None

                    # Update workstream title tick marks labels
                    ticks_font = \
                        font_manager.FontProperties(family=self.gantt.workstreams_title.font,
                                                    size=self.gantt.workstreams_title.font_size,
                                                    style=self.gantt.workstreams_title.font_style,
                                                    weight=self.gantt.workstreams_title.font_weight)

                    # Position tick labels (font properties set with self.tick_labels_major_y in set_axes_ticks)
                    for iytl, ytl in enumerate(yticklabs):
                        fill_color = '#ffffff'
                        if inline_workstreams is not None and ytl.get_text() in inline_workstreams:
                            ytl.set_fontproperties(ticks_font)
                            if self.gantt.workstreams.highlight_row:
                                fill_color = '#eeeeee'

                        if ytl.get_text() == '':
                            # Skip if there is no text in the tick label
                            continue

                        # yticklab left position
                        ytl.set_x(ytl.get_position()[0] - (tick_width - self.ws_ticks_ax) / self.axes.size[0])

                        # Left x position is the tick position minus the x-padding
                        x0 = (ax.get_window_extent().x0 - self._labtick_y) / self.fig.size_int[0]

                        # Bottom y position half the distance between grid lines b/c we use minor grid to align boxes;
                        # height = distance between two grid lines
                        grid1 = ygridlines[iytl].get_window_extent().y0
                        grid0 = ygridlines[iytl - 1].get_window_extent().y0
                        y0 = (grid0 + (grid1 - grid0) / 2) / self.fig.size_int[1]
                        height = grid1 - grid0
                        if iytl == len(yticklabs) - 1 and self.grid_major_y.width[0] > self.axes.edge_width:
                            # Correct top y-tick box so it better aligns with axes edge
                            height -= np.ceil(self.grid_major_y.width[0] - self.axes.edge_width)

                        # Add the rectangle
                        rect = patches.Rectangle((x0, y0),
                                                 self._labtick_y / self.fig.size_int[0],
                                                 height / self.fig.size_int[1],
                                                 fill=True, transform=self.fig.obj.transFigure,
                                                 edgecolor=self.axes.edge_color[0],
                                                 lw=self.grid_major_y.width[0], facecolor=fill_color, zorder=-2)
                        self.fig.obj.patches.extend([rect])

                # xtick date boxes (applies to the lowest date level)
                if len(self.gantt.date_type) > 0:
                    if primary in ['month-year']:
                        height = self._tick_x2 + self.tick_labels_major_x.size[1]
                    else:
                        height = self._tick_x2
                    xticks = ax.get_xticks()
                    for xtick in xticks:
                        tt = get_tick_bounds(ax, xtick, primary)
                        self.tick_labels_major_x.obj_bg[ir, ic] = \
                            [patches.Rectangle((tt['left']['px'] / self.fig.size_int[0],
                                                ax.get_window_extent().y1 / self.fig.size_int[1]),
                                               (tt['right']['px'] - tt['left']['px']) / self.fig.size_int[0],
                                               height / self.fig.size_int[1],
                                               fill=True, transform=self.fig.obj.transFigure,
                                               edgecolor=self.axes.edge_color[0],
                                               lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)]
                        self.fig.obj.patches.extend([self.tick_labels_major_x.obj_bg[ir, ic][-1]])

                    # xtick leading and trailing partial boxes
                    t0 = get_tick_bounds(ax, xticks[0], primary)
                    if t0['left']['float'] != xmin:
                        rect = patches.Rectangle((ax.get_window_extent().x0 / self.fig.size_int[0],
                                                 ax.get_window_extent().y1 / self.fig.size_int[1]),
                                                 (t0['left']['px'] - ax.get_window_extent().x0) / self.fig.size_int[0],
                                                 height / self.fig.size_int[1],
                                                 fill=True, transform=self.fig.obj.transFigure,
                                                 edgecolor=self.axes.edge_color[0],
                                                 lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)
                        self.fig.obj.patches.extend([rect])

                    t1 = get_tick_bounds(ax, ax.get_xticks()[-1], primary)
                    if t1['right']['float'] != xmax:
                        rect = patches.Rectangle((t1['right']['px'] / self.fig.size_int[0],
                                                 ax.get_window_extent().y1 / self.fig.size_int[1]),
                                                 (ax.get_window_extent().x1 - t1['right']['px']) / self.fig.size_int[0],
                                                 height / self.fig.size_int[1],
                                                 fill=True, transform=self.fig.obj.transFigure,
                                                 edgecolor=self.axes.edge_color[0],
                                                 lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)
                        self.fig.obj.patches.extend([rect])

                    # secondary date labels (create manually as text boxes using the actual xticks)
                    secondary_dates = [f for f in DATE_TYPES[DATE_TYPES.index(primary) + 1:]]
                    secondary_dates = [f for f in secondary_dates if f in self.gantt.date_type]
                    tick = self.tick_labels_major_x
                    heights, y_rects, y_texts = [], [], []
                    for ii, secondary in enumerate(secondary_dates):
                        if secondary in ['month-year']:
                            heights += [self._tick_x2 + self.tick_labels_major_x.size[1]]
                        else:
                            heights += [self._tick_x2]
                        if ii == 0:
                            y_rects = [ax.get_window_extent().y1 + height]
                        else:
                            y_rects += [y_rects[ii - 1] + heights[ii - 1]]
                        y_texts += [y_rects[ii] + heights[ii] / 2]
                    for ii, secondary in enumerate(secondary_dates):
                        # Get the secondary date values
                        dates = []
                        for ixt, xt in enumerate(xticks):
                            if secondary == 'year':
                                dates += [str(mdates.num2date(xt).year)]
                                attr = 'year'
                            elif secondary == 'quarter':
                                dates += [f'Q{(mdates.num2date(xt).month - 1) // 3 + 1}']
                                attr = 'month'
                            elif secondary == 'quarter-year':
                                dates += \
                                    [f'{str(mdates.num2date(xt).year)[-2:]}Q{(mdates.num2date(xt).month - 1) // 3 + 1}']
                                attr = 'month'
                            else:
                                dates += [mdates.num2date(xt).strftime(fmt[secondary].fmt)]
                                attr = 'month'

                        # Create text boxes for the dates
                        counts = [0] + list(np.cumsum([sum(1 for _ in group) for _, group in groupby(dates)]))
                        label_width = utl.get_text_dimensions(
                            dates[0], tick.font, tick.font_size, tick.font_style, tick.font_weight)[0]

                        for icount, count in enumerate(counts[:-1]):
                            # Determine start and stop point (special care for the partial boxes near the min and max)
                            if secondary in ['quarter', 'quarter-year']:
                                same_date_start = (getattr(mdates.num2date(xmin), attr) - 1) // 3 == \
                                    (getattr(mdates.num2date(xticks[0]), attr) - 1) // 3
                            else:
                                same_date_start = \
                                    getattr(mdates.num2date(xmin), attr) == getattr(mdates.num2date(xticks[0]), attr)
                            if icount == 0 and same_date_start:
                                x0 = ax.get_window_extent().x0
                            else:
                                x0 = get_tick_bounds(ax, xticks[count], primary)['left']['px']

                            if secondary in ['quarter', 'quarter-year']:
                                same_date_end = (getattr(mdates.num2date(xmax), attr) - 1) // 3 == \
                                    (getattr(mdates.num2date(xticks[-1]), attr) - 1) // 3
                            else:
                                same_date_end = \
                                    getattr(mdates.num2date(xmax), attr) == getattr(mdates.num2date(xticks[-1]), attr)
                            if icount + 1 == len(counts) - 1 and same_date_end:
                                x1 = ax.get_window_extent().x1
                            else:
                                x1 = get_tick_bounds(ax, xticks[counts[icount + 1] - 1], primary)['right']['px']
                            rect = patches.Rectangle((x0 / self.fig.size_int[0], y_rects[ii] / self.fig.size_int[1]),
                                                     (x1 - x0) / self.fig.size_int[0],
                                                     heights[ii] / self.fig.size_int[1],
                                                     fill=True, transform=self.fig.obj.transFigure,
                                                     edgecolor=self.axes.edge_color[0],
                                                     lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)
                            self.fig.obj.patches.extend([rect])

                            if x1 - x0 > label_width:
                                self.axes.obj[ir, ic].text(
                                    (x0 + (x1 - x0) / 2) / self.fig.size_int[0], y_texts[ii] / self.fig.size_int[1],
                                    dates[count], transform=self.fig.obj.transFigure,
                                    horizontalalignment='center', verticalalignment='center', rotation=tick.rotation,
                                    color=tick.font_color, fontname=tick.font, style=tick.font_style,
                                    weight=tick.font_weight, size=tick.font_size)

                        # leading and trailing partial boxes
                        t0 = get_tick_bounds(ax, xticks[0], primary)
                        if not same_date_start:
                            rect = patches.Rectangle((ax.get_window_extent().x0 / self.fig.size_int[0],
                                                     y_rects[ii] / self.fig.size_int[1]),
                                                     (t0['left']['px'] - ax.get_window_extent().x0) /
                                                     self.fig.size_int[0],
                                                     heights[ii] / self.fig.size_int[1],
                                                     fill=True, transform=self.fig.obj.transFigure,
                                                     edgecolor=self.axes.edge_color[0],
                                                     lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)
                            self.fig.obj.patches.extend([rect])

                        t1 = get_tick_bounds(ax, xticks[-1], primary)
                        if not same_date_end:
                            rect = patches.Rectangle((t1['right']['px'] / self.fig.size_int[0],
                                                     y_rects[ii] / self.fig.size_int[1]),
                                                     (ax.get_window_extent().x1 - t1['right']['px']) /
                                                     self.fig.size_int[0],
                                                     heights[ii] / self.fig.size_int[1],
                                                     fill=True, transform=self.fig.obj.transFigure,
                                                     edgecolor=self.axes.edge_color[0],
                                                     lw=self.grid_major_y.width[0], facecolor='#ffffff', zorder=-2)
                            self.fig.obj.patches.extend([rect])

    def show(self, *args):
        """Display the plot window."""
        mplp.show(block=False)

    def _subplots_adjust(self):
        """Calculate and apply the subplots_adjust parameters for the axes.  There is some real hocus pocus here
        to fix the way that matplotlib treats axes widths.

        Precise dimensions are subject to weird rounding error; try to check for them to address

        self.axes.position --> [left, right, top, bottom]
        """
        fig_w = self.fig.size_int[0]
        fig_h = self.fig.size_int[1]

        # Right
        if self.cbar.on:
            cbar = np.ceil(self._right + (self.label_z.size[0] + self.ws_ticks_ax + self.tick_labels_major_z.size[0]))
            self.axes.position[1] = \
                1 - cbar / fig_w
        else:
            self.axes.position[1] = \
                1 - (self._right + np.ceil(self._legx) + np.ceil(self.axes.edge_width / 2)) / fig_w

        # Left
        self.axes.position[0] = (self._left + np.floor(self.axes.edge_width / 2)) / fig_w

        # Top
        self.axes.position[2] = 1 - (self._top + np.floor(self.axes.edge_width / 2)) / fig_h

        # Bottom
        self.axes.position[3] = (self._bottom + np.ceil(self.axes.edge_width / 2)) / fig_h

        # Prevent float rounding errors for the positions subtracted from 1
        self.axes.position[1] -= 1E-13
        self.axes.position[2] -= 1E-13

        # wspace
        if self.cbar.on and not self.cbar.shared:
            ws_col = self.ws_col + self.tick_labels_major_z.size[0] - 1  # there is one extra pixel when ws_col = 0
        else:
            if self.axes.edge_width == 0 and self.ws_col > 0:
                h_edge = 0
            elif self.axes.edge_width == 0 or self.ws_col == 0:
                h_edge = -2 if self.ws_row > 0 else 0
            elif self.axes.edge_width == 1:
                h_edge = 1
            else:
                h_edge = 2 * np.floor(self.axes.edge_width / 2)

            ws_col = max(0, self.ws_col + h_edge)

        # hspace
        if self.axes.edge_width == 0 or self.ws_row == 0:
            v_edge = 0
        elif self.axes.edge_width == 1:
            v_edge = 1
        else:
            v_edge = 2 * np.floor(self.axes.edge_width / 2)
        ws_row = max(0, self.ws_row + v_edge)

        # Apply
        self.fig.obj.subplots_adjust(left=self.axes.position[0],
                                     right=self.axes.position[1],
                                     top=self.axes.position[2],
                                     bottom=self.axes.position[3],
                                     hspace=ws_row / self.axes.size[1],
                                     wspace=ws_col / self.axes.size[0],
                                     )

    def _subplots_adjust_x0y0(self):
        """Temporary realigning of subplots to the (0, 0) coordinate of the figure."""
        self.axes.position[0] = 0
        self.axes.position[3] = 0
        self.axes.position[1] = self.axes.size[0] / self.fig.size[0] * self.ncol
        self.axes.position[2] = self.axes.size[1] / self.fig.size[1] * self.nrow
        self.fig.obj.subplots_adjust(left=self.axes.position[0],
                                     right=self.axes.position[1],
                                     top=self.axes.position[2],
                                     bottom=self.axes.position[3],
                                     hspace=self.ws_row / self.axes.size[1],
                                     wspace=self.ws_col / self.axes.size[0],
                                     )
