import pandas as pd
import pdb
import scipy.stats
import numpy as np
import copy
import math
from typing import Dict
from .. utilities import RepeatedList
from .. import utilities as utl
from distutils.version import LooseVersion
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
import matplotlib.mlab as mlab
warnings.filterwarnings('ignore', category=UserWarning)


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
# weird error in boxplot with no groups
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")

db = pdb.set_trace
TICK_OVL_MAX = 0.75  # maximum allowed overlap for tick labels in float pixels


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
    decimals = [f - 1 if f > 0 else f for f in decimals]
    for it, nt in enumerate(new_ticks[0:-1]):
        new_ticks[it] = '{num:.{width}f}'.format(num=nt, width=decimals[it])
    return new_ticks


def df_tick(ticks: 'Element', ticks_size: 'np.array', ax: str) -> pd.DataFrame:
    """Create a dataframe of tick extents.  Used to look for overlapping ticks.

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
    """Calculate the next round of overlaps in the ticks dataframe

    Args:
        tt: ticks dataframe

    Returns:
        updated ticks dataframe
    """
    tt = tt.copy()

    # Shift the start column to line up the stop coordinate of one tick with the start of the next
    tt['next start'] = tt['start'].shift(-1)

    # Check the difference between stop coordinate of one tick and start of next; positive means overlap
    tt['delta'] = tt['stop'] - tt['next start']
    tt['ol'] = tt['delta'] > TICK_OVL_MAX  # allow a litle bit of overlap (otherwise would choose > 0)
    if 'visible' not in tt.columns:
        tt['visible'] = True

    return tt


def hide_overlaps(ticks: 'Element', tt: pd.DataFrame, ir: int, ic: int) -> pd.DataFrame:
    """Find and hide any overlapping tick marks on the same axis.

    Args:
        ticks: Element class for tick lables
        tt: tick dataframe created by df_ticks
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


def hide_overlaps_major_minor(ticks, ticksm, bbox, bboxm, ir, ic) -> pd.DataFrame:
    """Find and hide any overlapping tick marks on the same axis.  Three cases to address:
    1) the starting position of the minor tick falls within the span of the major tick label
    2) the ending position of the minor tick falls within the span of the major tick label
    3) the minor tick completely covers the major tick

    Args:
        ticks: Element class for major tick lables
        ticksm: Element class for major tick lables
        bbox: major tick label visibility status
        bboxm: minor tick label visibility status
        ir: current axes row index
        ic: current axes column index

    Returns:
        updated bboxm table
    """
    bbox = bbox[bbox['visible']]
    if len(bbox) == 0 or len(bboxm) == 0:
        return bboxm
    bboxm_orig = bboxm.copy()

    # Disable minor ticks where starting point falls between the major tick coords
    ol = [is_in_range(tt, bbox[['start', 'stop']].values) for tt in np.nditer(bboxm['start'].values + TICK_OVL_MAX)]
    while True in ol:
        for iol, ool in enumerate(ol):
            if ool:
                idx = bboxm.index[iol]
                ticksm.obj[ir, ic][idx].set_visible(False)
                bboxm.loc[idx, 'ol'] = False
                bboxm.loc[idx, 'visible'] = False
                bboxm_orig.loc[idx, 'visible'] = False
                bboxm_orig.loc[idx, 'label'] = ticksm.obj[ir, ic][idx]._text
        bboxm = bboxm[bboxm['visible']]
        if len(bboxm) == 0:
            break
        ol = [is_in_range(tt, bbox[['start', 'stop']].values) for tt in np.nditer(bboxm['start'].values + TICK_OVL_MAX)]
    if len(bboxm) == 0:
        return bboxm_orig

    # Disable minor ticks where stopping point falls between the major tick coords
    bboxm = bboxm.copy()
    ol = [is_in_range(tt, bbox[['start', 'stop']].values) for tt in np.nditer(bboxm['stop'].values - TICK_OVL_MAX)]
    while True in ol:
        for iol, ool in enumerate(ol):
            if ool:
                idx = bboxm.index[iol]
                ticksm.obj[ir, ic][idx].set_visible(False)
                bboxm.loc[idx, 'ol'] = False
                bboxm.loc[idx, 'visible'] = False
                bboxm_orig.loc[idx, 'visible'] = False
        bboxm = bboxm[bboxm['visible']]
        if len(bboxm) == 0:
            break
        ol = [is_in_range(tt, bbox[['start', 'stop']].values) for tt in np.nditer(bboxm['stop'].values - TICK_OVL_MAX)]

    # Disable minor ticks that completely cover the major tick coords
    for irow, row in bboxm.iterrows():
        covered = bbox[(row['start'] - bbox.start < 0) & (bbox.stop - row['stop'] < 0)]
        if len(covered) > 0:
            ticksm.obj[ir, ic][row.name].set_visible(False)
            bboxm_orig.loc[row.name, 'visible'] = False

    return bboxm_orig


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
    if LooseVersion(mpl.__version__) >= LooseVersion('3.1'):
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
        hexc += '%s' % hex(int(cc * 255))[2:].zfill(2)

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
        lim = getattr(ax, 'get_%slim' % vv)()
        tp[vv]['min'] = min(lim)
        tp[vv]['max'] = max(lim)
        tp[vv]['ticks'] = getattr(ax, 'get_%sticks' % vv)()
        tp[vv]['labels'] = [f for f in iterticks(getattr(ax, '%saxis' % vv), minor)]
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


class Layout(BaseLayout):

    def __init__(self, data: 'Data', defaults: list = [], **kwargs):  # noqa: F821
        """Layout attributes and methods for matplotlib Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        global ENGINE
        ENGINE = 'mpl'

        # Set tick style to classic if using fcp tick_cleanup (default)
        if kwargs.get('tick_cleanup', True):
            mplp.style.use('classic')
        else:
            mplp.style.use('default')

        # Unless specified, close previous plots
        if not kwargs.get('hold', False):
            mplp.close('all')

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
        self.fig_legend_border = 5
        self.x_tick_xs = 0

        # Other
        self.set_colormap(data)

        # Update kwargs
        if not kwargs.get('save_ext'):
            kwargs['save_ext'] = '.png'
        self.kwargs = kwargs

    @property
    def _labtick_x(self) -> float:
        """Height of the x label + x tick labels + related whitespace."""
        val = self.label_x.size[1] \
            + self.ws_label_tick * self.label_x.on \
            + self._tick_x

        return val

    @property
    def _labtick_x2(self) -> float:
        """Height of the secondary x label + x tick labels + related whitespace."""
        if not self.axes.twin_y:
            return 0

        val = self.label_x2.size[1] \
            + self.ws_label_tick * self.label_x2.on \
            + self._tick_x2

        return val

    @property
    def _labtick_y(self) -> float:
        """Width of the y label + y tick labels + related whitespace."""
        if self.pie.on:
            return 0

        val = self.label_y.size[0] \
            + self.ws_label_tick * (self.tick_labels_major_y.on | self.tick_labels_minor_y.on) \
            + self._tick_y

        return val

    @property
    def _labtick_y2(self) -> float:
        """Width of the secondary y label + y tick labels + related whitespace."""
        if not self.axes.twin_x:
            return 0

        val = self.label_y2.size[0] \
            + self.ws_label_tick * (self.tick_labels_major_y.on | self.tick_labels_minor_y.on) \
            + self._tick_y2

        return val

    @property
    def _labtick_z(self) -> float:
        """Width of the z label + z tick labels + related whitespace."""
        val = self.ws_label_tick * self.label_z.on + self.tick_labels_major_z.size[0]

        return val

    @property
    def _left(self) -> float:
        """Width of the space to the left of the axes object."""
        left = self.ws_fig_label + self._labtick_y
        title_xs_left = self.title.size[0] / 2 + (self.ws_fig_ax if self.title.on else 0) \
            - (left + (self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2)
        if title_xs_left < 0:
            title_xs_left = 0
        left += title_xs_left

        # pie labels
        left += self.pie.xs_left

        return left

    @property
    def _legx(self) -> float:
        """Legend whitespace x if location == 0."""
        if self.legend.location == 0 and self.legend._on:
            return self.legend.size[0] + self.ws_ax_leg + self.ws_leg_fig \
                + self.fig_legend_border + self.legend.edge_width
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
    def _rc_label(self) -> float:
        """Width of an rc label with whitespace."""
        return (self.label_row.size[0] + self.ws_label_row
                + 2 * self.label_row.edge_width) * self.label_row.on

    @property
    def _right(self) -> float:
        """Width of the space to the right of the axes object."""
        # axis to fig right side ws with or without legend
        ws_ax_fig = (self.ws_ax_fig if not self.legend._on or self.legend.location != 0 else 0)

        # sum all parts
        right = ws_ax_fig \
            + self._labtick_y2 \
            + self._rc_label \
            + self._labtick_z \
            + self.x_tick_xs \
            + self.label_y2.size[0] \
            + (self.label_z.size[0] * (self.ncol if self.separate_labels else 1) + self.ws_ticks_ax * self.label_z.on)

        # box title excess
        if self.box_group_title.on and (self.ws_ax_box_title + self.box_title) > self._legx:
            right = self.ws_ax_box_title + self.box_title + (self.ws_ax_fig if not self.legend.on else 0)
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            right += self.box_title

        # Main figure title excess size
        title_xs_right = self.title.size[0] / 2 \
            - (right + (self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2) - self.legend.size[0]
        if title_xs_right < 0:
            title_xs_right = 0

        right += title_xs_right

        # pie labels
        right += self.pie.xs_right

        return right

    @property
    def _tick_x(self) -> float:
        """Height of the primary x ticks and whitespace."""
        val = max(self.tick_labels_major_x.size[1], self.tick_labels_minor_x.size[1]) \
            + self.ws_ticks_ax * (self.tick_labels_major_x.on | self.tick_labels_minor_x.on)
        return val

    @property
    def _tick_x2(self) -> float:
        """Height of the secondary x ticks and whitespace."""
        val = max(self.tick_labels_major_x2.size[1], self.tick_labels_minor_x2.size[1]) \
            + self.ws_ticks_ax * (self.tick_labels_major_x.on | self.tick_labels_minor_x.on)
        return val

    @property
    def _tick_y(self) -> float:
        """Width of the primary y ticks and whitespace."""

        # Case of no or solo tick labels due to tight range where we add custom limit text strings
        bonus = 0
        if len(self.tick_labels_major_y.size_all) == 0 and self.tick_labels_major_y.limits[0][0] is not None:
            # not perfect, only use [0, 0] axes obj
            precision = utl.get_decimals(self.tick_labels_major_y.limits[0, 0][0], 8)
            txt0 = f'{self.tick_labels_major_y.limits[0, 0][0]:.{precision}f}'
            precision = utl.get_decimals(self.tick_labels_major_y.limits[0, 0][-1], 8)
            txt1 = f'{self.tick_labels_major_y.limits[0, 0][-1]:.{precision}f}'
            txt = max(txt0, txt1)
            txt = txt.replace('.', '')  # skip "." for sizing
            bonus = self.tick_labels_major_y.font_size / self.tick_labels_major_y.scale_factor * len(txt) * 0.9
            bonus -= self.label_y.size[0]  # - self.ws_label_tick
        elif len(self.tick_labels_major_y.size_all) <= 1 and self.tick_labels_major_y.limits[0][0] is not None:
            # not perfect, only use [0, 0] axes obj
            precision = utl.get_decimals(self.tick_labels_major_y.limits[0, 0][0], 8)
            txt = f'{self.tick_labels_major_y.limits[0, 0][-1]:.{precision}f}'
            bonus = self.tick_labels_major_y.font_size / self.tick_labels_major_y.scale_factor * len(txt) * 0.9
            bonus -= self.label_y.size[0]  # - self.ws_label_tick

        val = max(self.tick_labels_major_y.size[0], self.tick_labels_minor_y.size[0]) + bonus \
            + self.ws_ticks_ax * (self.tick_labels_major_y.on | self.tick_labels_minor_y.on)

        return val

    @property
    def _tick_y2(self) -> float:
        """Width of the secondary y ticks and whitespace."""
        val = max(self.tick_labels_major_y2.size[0], self.tick_labels_minor_y2.size[0]) \
            + self.ws_ticks_ax * (self.tick_labels_major_y2.on | self.tick_labels_minor_y2.on)
        return val

    @property
    def _top(self) -> float:
        """Excess height at the top of the axes.
        Placeholder only
        """
        return 0

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
        for i in range(0, num_cols):
            k = num_cols - 1 - i
            sub = data.changes[num_cols - 1 - i][data.changes[num_cols - 1 - i] == 1]
            if len(sub) == 0:
                sub = data.changes[num_cols - 1 - i]

            # Group labels
            if self.box_group_label.on:
                # This array structure just makes one big list of all the labels
                # can we use it with multiple groups or do we need to reshape??
                # Probably need a 2D but that will mess up size_all indexing
                for j in range(0, len(sub)):

                    # set the width now since it is a factor of the axis size
                    if j == len(sub) - 1:
                        width = len(data.changes) - sub.index[j]
                    else:
                        width = sub.index[j + 1] - sub.index[j]
                    width = width * self.axes.size[0] / len(data.changes)
                    label = data.indices.loc[sub.index[j], num_cols - 1 - i]
                    self.box_group_label.obj[ir, ic][i, j], \
                        self.box_group_label.obj_bg[ir, ic][i, j] = \
                        self.add_label(ir, ic, label,
                                       (sub.index[j] / len(data.changes), 0, 0, 0),
                                       rotation=0, size=[width, 20], offset=True,
                                       **self.make_kw_dict(self.box_group_label, ['size', 'rotation', 'position']))

            # Group titles
            if self.box_group_title.on and ic == data.ncol - 1:
                self.box_group_title.obj[ir, ic][i, 0], \
                    self.box_group_title.obj_bg[ir, ic][i, 0] = \
                    self.add_label(ir, ic, data.groups[k], (1, 0, 0, 0), size=[0, 20],
                                   **self.make_kw_dict(self.box_group_title, ['position', 'size']))

    def add_cbar(self, ax: mplp.Axes, contour: 'MPL_Contour_Plot_Object') -> 'MPL_Colorbar_Object':  # noqa: F821
        """Add a color bar.

        Args:
            ax: current axes object
            contour: current contour plot obj

        Returns:
            reference to the colorbar object
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        size = '%s%%' % (100 * self.cbar.size[0] / self.axes.size[0])
        pad = self.ws_ax_cbar / 100
        cax = divider.append_axes('right', size=size, pad=pad)

        # Add the colorbar
        cbar = mplp.colorbar(contour, cax=cax)
        cbar.outline.set_edgecolor(self.cbar.edge_color[0])
        cbar.outline.set_linewidth(self.cbar.edge_width)

        # Style tick labels
        ticks_font = \
            font_manager.FontProperties(family=getattr(self, 'tick_labels_major_z').font,
                                        size=getattr(self, 'tick_labels_major_z').font_size,
                                        style=getattr(self, 'tick_labels_major_z').font_style,
                                        weight=getattr(self, 'tick_labels_major_z').font_weight)

        for text in cax.get_yticklabels():
            if getattr(self, 'tick_labels_major_z').rotation != 0:
                text.set_rotation(getattr(self, 'tick_labels_major_z').rotation)
            text.set_fontproperties(ticks_font)
            text.set_bbox(dict(edgecolor=getattr(self, 'tick_labels_major_z').edge_color[0],
                               facecolor=getattr(self, 'tick_labels_major_z').fill_color[0],
                               linewidth=getattr(self, 'tick_labels_major_z').edge_width))

        return cbar

    def add_hvlines(self, ir: int, ic: int, df: [pd.DataFrame, None] = None):
        """Add horizontal/vertical lines.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data. Defaults to None.
        """
        # Set default line attributes
        for axline in ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']:
            ll = getattr(self, axline)
            func = self.axes.obj[ir, ic].axhline if 'hline' in axline \
                else self.axes.obj[ir, ic].axvline
            if ll.on:
                if hasattr(ll, 'by_plot') and ll.by_plot:
                    ival = utl.plot_num(ir, ic, self.ncol) - 1
                    if ival < len(ll.values):
                        line = func(ll.values[ival], color=ll.color[ival],
                                    linestyle=ll.style[ival],
                                    linewidth=ll.width[ival],
                                    zorder=ll.zorder)
                        if isinstance(ll.text, list) and ll.text[ival] is not None:
                            self.legend.add_value(ll.text[ival], [line], 'ref_line')

                else:
                    for ival, val in enumerate(ll.values):
                        if isinstance(val, str) and isinstance(df, pd.DataFrame):
                            val = df[val].iloc[0]
                        line = func(val, color=ll.color[ival],
                                    linestyle=ll.style[ival],
                                    linewidth=ll.width[ival],
                                    zorder=ll.zorder)
                        if isinstance(ll.text, list) and ll.text[ival] is not None:
                            self.legend.add_value(ll.text[ival], [line], 'ref_line')

    def add_label(self, ir: int, ic: int, text: str = '', position: [tuple, None] = None,
                  rotation: int = 0, size: [list, None] = None,
                  fill_color: str = '#ffffff', edge_color: str = '#aaaaaa',
                  edge_width: int = 1, font: str = 'sans-serif', font_weight: str = 'normal',
                  font_style: str = 'normal', font_color: str = '#666666', font_size: int = 14,
                  offset: bool = False, **kwargs) -> ['Text_Object', 'Rectangle_Object']:  # noqa: F821
        """Add a label to the plot.

        This function can be used for title labels or for group labels applied
        to rows and columns when plotting facet grid style plots.
        Args:
            ir: subplot row index
            ic: subplot column index
            text:  label text. Defaults to ''.
            position: label position tuple of form (left, right, top, bottom) or None.
                Defaults to None.
            rotation:  degrees of rotation. Defaults to 0.
            size: list of [height, weight] or None. Defaults to None.
            fill_color: hex color code for label fill. Defaults to '#ffffff'.
            edge_color: hex color code for label edge. Defaults to '#aaaaaa'.
            edge_width: width of the label bounding box edge. Defaults to 1.
            font: name of the font for the label. Defaults to 'sans-serif'.
            font_weight: mpl font weight str ('normal', 'bold', etc.). Defaults to 'normal'.
            font_style: mpl font style str ('normal', 'italic', etc.). Defaults to 'normal'.
            font_color:  hex color code for label text. Defaults to '#666666'.
            font_size: label font size (default=14)
            offset: use an offset for positioning the text of the label. Defaults to False.
            kwargs: any other keyword args (they won't be used but a sloppy way to ignore
                any extra keywords that get passed to this function)

        Returns:
            reference to the text box object
            reference to the background rectangle patch object

        """
        # Set slight text offset
        if rotation == 270 and offset:
            offsetx = -2 / self.axes.size[0]  # -font_size/self.axes.size[0]/4
        else:
            offsetx = 0
        if rotation == 0 and offset:
            offsety = -2 / self.axes.size[1]  # -font_size/self.axes.size[1]/4
        else:
            offsety = 0

        # Define the label background
        rect = patches.Rectangle((position[0], position[3]),
                                 size[0] / self.axes.size[0],
                                 size[1] / self.axes.size[1],
                                 fill=True,
                                 transform=self.axes.obj[ir, ic].transAxes,
                                 facecolor=fill_color if isinstance(fill_color, str)
                                 else fill_color[utl.plot_num(ir, ic, self.ncol)],
                                 edgecolor=edge_color if isinstance(edge_color, str)
                                 else edge_color[utl.plot_num(ir, ic, self.ncol)],
                                 lw=edge_width if isinstance(edge_width, int) else 1,
                                 clip_on=False, zorder=1)
        self.axes.obj[ir, ic].add_patch(rect)

        # Add the label text
        text = self.axes.obj[ir, ic].text(
            position[0] + offsetx,
            position[3] + offsety,
            text,
            transform=self.axes.obj[ir, ic].transAxes,
            horizontalalignment='center',  # backgroundcolor='#ff0000',
            verticalalignment='center', rotation=rotation,
            color=font_color, fontname=font, style=font_style,
            weight=font_weight, size=font_size)

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
                if self.name not in ['hist', 'bar', 'pie', 'gantt']:
                    if isinstance(leg.legendHandles[itext], mpl.patches.Rectangle):
                        continue
                    # Set legend point color and alpha
                    leg.legendHandles[itext]._sizes = \
                        np.ones(len(self.legend.values) + 1) * self.legend.marker_size**2
                    if not self.markers.on and self.legend.marker_alpha is not None:
                        if hasattr(leg.legendHandles[itext], '_legmarker'):
                            leg.legendHandles[itext]._legmarker.set_alpha(self.legend.marker_alpha)
                        else:
                            # required for mpl 3.5
                            alpha = str(hex(int(self.legend.marker_alpha * 255)))[-2:].replace('x', '0')
                            base_color = self.markers.edge_color[itext][0:7] + alpha
                            leg.legendHandles[itext]._markeredgecolor = base_color
                            leg.legendHandles[itext]._markerfillcolor = base_color
                    elif self.legend.marker_alpha is not None:
                        leg.legendHandles[itext].set_alpha(self.legend.marker_alpha)

            leg.get_title().set_fontsize(self.legend.font_size)
            leg.get_frame().set_facecolor(self.legend.fill_color[0])
            leg.get_frame().set_alpha(self.legend.fill_alpha)
            leg.get_frame().set_edgecolor(self.legend.edge_color[0])
            leg.get_frame().set_linewidth(self.legend.edge_width)

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
                leg_vals = self.legend.values.set_index('Key').loc[leg_vals['names']].reset_index()
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
                 element: [str, None] = None, offsetx: int = 0,
                 offsety: int = 0, coord: ['mpl_coordinate', None] = None,  # noqa: F821
                 units: [str, None] = None, **kwargs):
        """Add a text box.

        Args:
            ir: subplot row index
            ic: subplot column index
            text (optional): text str to add. Defaults to None.
            element (optional): name of the Element object. Defaults to None.
            offsetx (optional): x-axis shift. Defaults to 0.
            offsety (optional): y-axis shift. Defaults to 0.
            coord (optional): MPL coordinate type. Defaults to None.
            units (optional): pixel or inches. Defaults to None which is 'pixel'.
        """
        # Shortcuts
        ax = self.axes.obj[ir, ic]
        if element is None:
            obj = self.text
        else:
            obj = getattr(self, element)
        text = text if text is not None else obj.text.values
        if isinstance(text, str):
            text = [text]

        # Set the coordinate so text is anchored to figure, axes, or the current
        #    data range, or units
        if not coord:
            coord = None if not hasattr(obj, 'coordinate') else self.text.coordinate.lower()
        if coord == 'figure':
            transform = self.fig.obj.transFigure
        elif coord == 'data':
            transform = ax.transData
        else:
            transform = ax.transAxes
        if not units:
            units = 'pixel' if not hasattr(obj, 'units') else getattr(obj, 'units')

        # Add each text box
        for itext, txt in enumerate(text):
            kw = {}

            # Set style attributes
            attrs = ['rotation', 'font_color', 'font', 'fill_color', 'edge_color',
                     'font_style', 'font_weight', 'font_size']
            for attr in attrs:
                if attr in kwargs.keys():
                    kw[attr] = kwargs[attr]
                elif hasattr(obj, attr) and isinstance(getattr(obj, attr), RepeatedList):
                    kw[attr] = getattr(obj, attr)[itext]
                elif hasattr(obj, attr) and str(type(getattr(obj, attr))) == str(RepeatedList):
                    # isinstance fails on python3.6, so hack this way
                    kw[attr] = getattr(obj, attr)[itext]
                elif hasattr(obj, attr):
                    kw[attr] = getattr(obj, attr)

            if element:
                # Get position
                if 'position' in kwargs.keys():
                    position = copy.copy(kwargs['position'])
                elif hasattr(obj, 'position') and isinstance(getattr(obj, 'position'), RepeatedList):
                    position = copy.copy(getattr(obj, 'position')[itext])
                elif hasattr(obj, 'position'):
                    position = copy.copy(getattr(obj, 'position'))

                # Convert position to correct units
                if units == 'pixel' and coord == 'figure':
                    position[0] /= self.fig.size[0]
                    offsetx /= self.fig.size[0]
                    position[1] /= self.fig.size[1]
                    offsety /= self.fig.size[1]
                elif units == 'pixel' and coord != 'data':
                    position[0] /= self.axes.size[0]
                    offsetx /= self.axes.size[0]
                    position[1] /= self.axes.size[1]
                    offsety /= self.axes.size[1]

                # Something goes weird with x = 0 so we need to adjust slightly
                if position[0] == 0:
                    position[0] = 0.01

                # Add the text
                ax.text(position[0] + offsetx,
                        position[1] + offsety,
                        txt, transform=transform,
                        rotation=kw['rotation'],
                        color=kw['font_color'],
                        fontname=kw['font'],
                        style=kw['font_style'],
                        weight=kw['font_weight'],
                        size=kw['font_size'],
                        bbox=dict(facecolor=kw['fill_color'], edgecolor=kw['edge_color']),
                        zorder=45)
            else:
                obj.obj[ir, ic][itext] = ax.text(0, 0,
                                                 txt, transform=transform,
                                                 rotation=kw['rotation'],
                                                 color=kw['font_color'],
                                                 fontname=kw['font'],
                                                 style=kw['font_style'],
                                                 weight=kw['font_weight'],
                                                 size=kw['font_size'],
                                                 bbox=dict(facecolor=kw['fill_color'], edgecolor=kw['edge_color']),
                                                 zorder=45)

    def close(self):
        """Close an inline plot window."""
        mplp.close('all')

    def fill_between_lines(self, ir: int, ic: int, iline: int,
                           x: [np.ndarray, pd.Index],
                           lcl: [np.ndarray, pd.Series],
                           ucl: [np.ndarray, pd.Series],
                           element: str, leg_name: [str, None] = None,
                           twin: bool = False):
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
                               facecolor=fc[iline] if isinstance(fc, RepeatedList) else fc,
                               edgecolor=ec[iline] if isinstance(ec, RepeatedList) else ec,
                               linestyle=element.edge_style,
                               linewidth=element.edge_width,
                               label='hi')

        # Add a reference to the line to self.lines
        if leg_name is not None:
            self.legend.add_value(leg_name, fill, 'fill')

    def _get_axes_label_position(self):
        """
        Get the position of the axes labels.

        self.label_@.position --> [left, right, top, bottom]
        """
        self.label_x.position[0] = 0.5
        self.label_x.position[3] = \
            (self.label_x.size[1] / 2 - self._labtick_x) / self.axes.size[1]

        self.label_x2.position[0] = 0.5
        self.label_x2.position[3] = \
            1 + (self._labtick_x2 - self.label_x2.size[1] / 2) / self.axes.size[1]

        self.label_y.position[0] = \
            (self.label_y.size[0] / 2 - self._labtick_y) / self.axes.size[0]
        self.label_y.position[3] = 0.5

        self.label_y2.position[0] = 1 + (self._labtick_y2 - self.label_y2.size[0] / 2) / self.axes.size[0]
        self.label_y2.position[3] = 0.5

        self.label_z.position[0] = 1 + (self.ws_ax_cbar + self.cbar.size[0]
                                        + self.tick_labels_major_z.size[0]
                                        + 2 * self.ws_label_tick) \
            / self.axes.size[0]
        self.label_z.position[3] = 0.5

    @property
    def _box_label_heights(self):
        """Calculate the box label height."""
        lab = self.box_group_label
        labt = self.box_group_title

        if len(lab.size_all) == 0:
            return np.array(0)

        # Determine the box group label row heights
        heights = lab.size_all.groupby('ii').max()['height']
        heights *= (1 + 2 * lab.padding / 100)

        # Determine the box group title heights
        heightst = labt.size_all.groupby('ii').max()['height']
        heightst *= (1 + 2 * labt.padding / 100)

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
        for label in ['x', 'x2', 'y', 'y2', 'z', 'row', 'col', 'wrap']:
            lab = getattr(self, f'label_{label}')
            if not lab.on or lab.obj is None:
                continue
            for ir, ic in np.ndindex(lab.obj.shape):
                if lab.obj[ir, ic] is None:
                    continue

                # text label size
                bbox = lab.obj[ir, ic].get_window_extent()
                width = bbox.width
                height = bbox.height
                lab.size_all = (ir, ic, 0, 0, width, height, bbox.x0, bbox.x1, bbox.y0, bbox.y1, np.nan)

                # text label rect background (ax label has to be resized!)
                bbox = lab.obj_bg[ir, ic].get_window_extent()
                lab.size_all_bg = (ir, ic, 0, 0, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1, np.nan)
                if label not in ['row', 'col', 'wrap']:
                    lab.obj_bg[ir, ic].set_width((width + lab.bg_padding * 2) / self.axes.size[0])
                    lab.obj_bg[ir, ic].set_height((height + lab.bg_padding * 2) / self.axes.size[1])

            # set max size
            width = lab.size_all.width.max()
            height = lab.size_all.height.max()
            if label in ['row', 'col', 'wrap']:
                width_bg, height_bg = lab.size
            else:
                width_bg = lab.size_all_bg.width.max()
                height_bg = lab.size_all_bg.height.max()

            lab.size = [max(width, width_bg), max(height, height_bg)]

        # titles
        if self.title.on:
            self.title.size = self.title.obj.get_window_extent().width, self.title.obj.get_window_extent().height

        # legend
        if self.legend.on and self.legend.location in [0, 11]:
            self.legend.size = \
                [self.legend.obj.get_window_extent().width + self.legend_border,
                 self.legend.obj.get_window_extent().height + self.legend_border]

        # tick labels
        self._get_tick_label_sizes()

        if self.cbar.on:
            self._get_tick_label_size(self.cbar, 'z', '', 'major')

        # box labels and titles
        if self.box_group_label.on:
            lab = self.box_group_label
            lens_all = pd.DataFrame(columns=['ir', 'ic', 'ii', 'vals'])
            for ir, ic in np.ndindex(lab.obj.shape):
                data.df_rc = data._subset(ir, ic)
                data.get_box_index_changes()
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
                            lens_all = pd.concat([lens_all,
                                                  pd.DataFrame({'ir': ir, 'ic': ic, 'ii': ii, 'vals': [lens]},
                                                               index=[0])])

                    # skip any nones
                    if lab.obj[ir, ic][ii, jj] is None:
                        continue

                    # get the label dimensions and the max size available for this label
                    bbox = lab.obj[ir, ic][ii, jj].get_window_extent()
                    if ii < len(lens):
                        label_max_width = lens[jj] * divider
                    else:
                        label_max_width = bbox.width + 1

                    # rotate labels that are longer than the box axis size
                    if bbox.width > label_max_width and not (self.box_scale == 'auto' and ii == 0):
                        lab.obj[ir, ic][ii, jj].set_rotation(90)
                        bbox = lab.obj[ir, ic][ii, jj].get_window_extent()

                    # update the size_all dataframe
                    lab.size_all = (ir, ic, ii, jj, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1,
                                    lab.obj[ir, ic][ii, jj].get_rotation())
                    lab.size_all_bg = (ir, ic, ii, jj, bbox.width, bbox.height, bbox.x0, bbox.x1, bbox.y0, bbox.y1,
                                       np.nan)

            if self.box_scale == 'auto':
                # Resize the axes width
                maxes = lab.size_all.groupby(['ir', 'ic', 'ii']).max()
                size0 = (maxes['jj'] + 1) * maxes['width'] + divider
                margin = 4 * maxes['jj'].max()
                self.axes.size[0] = size0.max() + margin

                # Recheck the rotation (inefficient to loop 2x, maybe find a better way)
                lens_all = lens_all.set_index(['ir', 'ic', 'ii'])
                for iii, (nn, gg) in enumerate(lab.size_all.groupby(['ir', 'ic', 'ii'])):
                    gg = gg.copy()
                    if iii == 0:
                        continue
                    gg['num'] = lens_all.loc[nn]['vals']
                    gg['label_max_width'] = self.axes.size[0] / gg['num'].sum() * gg['num']
                    revert = gg.loc[(gg.rotation == 90) & (gg['height'] < gg['label_max_width'])]
                    for irow, row in revert.iterrows():
                        lab.obj[nn[0], nn[1]][nn[2], gg.loc[irow, 'jj']].set_rotation(0)

                # Reset the horizontallabel widths
                for ir, ic in np.ndindex(lab.obj.shape):
                    if self.label_y.obj_bg[ir, ic] is not None:
                        self.label_y.obj_bg[ir, ic].set_width(
                            self.label_y.obj_bg[ir, ic].get_window_extent().width / self.axes.size[0])
                    if self.label_row.obj_bg[ir, ic]:
                        self.label_row.obj_bg[ir, ic].set_width(
                            self.label_row.obj_bg[ir, ic].get_window_extent().width / self.axes.size[0])

            # set max size
            width = lab.size_all.width.max()
            height = lab.size_all.height.max()
            width_bg = lab.size_all_bg.width.max()  # do we want this?
            height_bg = lab.size_all_bg.height.max()
            lab.size = [max(width, width_bg), max(height, height_bg)]

        if self.box_group_title.on:
            lab = self.box_group_title
            for ir, ic in np.ndindex(lab.obj.shape):
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

            # set max size
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

                if self.pie.explode:
                    for iwedge, wedge in enumerate(self.pie.explode):
                        if wedge == 0:
                            continue
                        # this is a bit of a hack and may not hold for all cases
                        # and ignores the orientation of the wedge that is
                        # exploding
                        self.pie.xs_left += wedge * self.axes.size[0] / 4
                        self.pie.xs_right += wedge * self.axes.size[0] / 4
                        self.pie.xs_top += wedge * self.axes.size[0] / 4
                        self.pie.xs_bottom += wedge * self.axes.size[0] / 4

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

        return data

    def _get_figure_size(self, data: 'Data', **kwargs):  # noqa: F821
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            kwargs: user-defined keyword args
        """
        debug = kwargs.get('debug_size', False)

        # Set some values for convenience
        self.ws_ax_leg = max(0, self.ws_ax_leg - self._labtick_y2) if self.legend.location == 0 else 0
        self.ws_leg_fig = self.ws_leg_fig if self.legend.location == 0 else 0
        self.fig_legend_border = self.fig_legend_border if self.legend.location == 0 else 0
        self.box_labels = self._box_label_heights.sum()
        self.box_title = 0
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            self.box_title = self.box_group_title.size[0] + self.ws_ax_box_title
        elif self.box_group_title.on and self.box_group_title.size != [0, 0] and \
                self.box_group_title.size[0] > self.legend.size[0]:
            self.box_title = self.box_group_title.size[0] - self.legend.size[0]  # + self.ws_ax_box_title
        # Adjust the column and row whitespace
        if self.box_group_label.on and self.label_wrap.on and 'ws_row' not in kwargs.keys():
            self.ws_row = self.box_labels + self.title_wrap.size[1]
        else:
            self.ws_row += self.box_labels

        if self.title.on:
            self.ws_title = self.ws_fig_title + self.title.size[1] + self.ws_title_ax
        else:
            self.ws_title = self.ws_fig_ax

        if self.cbar.on and utl.kwget(kwargs, self.fcpp, 'ws_col', -999) == -999:
            self.ws_col = self._labtick_z

        # separate ticks and labels
        if (self.separate_ticks or self.axes.share_y is False) and not self.cbar.on:
            self.ws_col = max(self._tick_y + self.ws_label_tick, self.ws_col_def)
        elif (self.separate_ticks or self.axes.share_y is False) and self.cbar.on:
            self.ws_col += self._tick_y
        if self.axes2.on and (self.separate_ticks or self.axes2.share_y is False) and not self.cbar.on:
            self.ws_col = max(self._tick_y2 + self.ws_label_tick, self.ws_col_def)

        if self.separate_ticks or (self.axes.share_x is False and self.box.on is False):
            self.ws_row += max(self.tick_labels_major_x.size[1], self.tick_labels_minor_x.size[1]) + self.ws_ticks_ax
        elif self.axes2.on and (self.separate_ticks or self.axes2.share_x is False) and self.box.on is False:
            self.ws_row += self._tick_x2
        if self.separate_labels:
            self.ws_col += self._labtick_y - self._tick_y + self.ws_ax_label_xs
            if self.cbar.on:
                self.ws_col += self.ws_label_tick

        if self.name == 'heatmap' and self.heatmap.cell_size is not None and data.num_x is not None:
            self.axes.size = [self.heatmap.cell_size * data.num_x, self.heatmap.cell_size * data.num_y]
            self.label_col.size[0] = self.axes.size[0]
            self.label_row.size[1] = self.axes.size[1]
            self.label_wrap.size[1] = self.axes.size[0]

        # imshow ax adjustment
        if self.name == 'imshow':
            if data.wh_ratio >= 1:
                self.axes.size[1] = self.axes.size[0] / data.wh_ratio
                self.label_row.size[1] = self.axes.size[1]
            else:
                self.axes.size[0] = self.axes.size[1] * data.wh_ratio
                self.label_col.size[0] = self.axes.size[0]
                self.label_wrap.size[0] = self.axes.size[0]

        # Set figure width
        self.fig.size[0] = self._left + self.axes.size[0] * self.ncol \
            + self._right + self._legx + self.ws_col * (self.ncol - 1) \
            - (self.fig_legend_border if self.legend._on else 0) \
            + (self.cbar.size[0] + self.ws_ax_cbar) * self.ncol

        # Figure height
        self.fig.size[1] = int(
            self.ws_title
            + (self.label_col.size[1] + self.ws_label_col) * self.label_col.on
            + self.title_wrap.size[1] + self.label_wrap.size[1]
            + self._labtick_x2
            + self.axes.size[1] * self.nrow
            + self._labtick_x
            + self.ws_fig_label
            + self.ws_row * (self.nrow - 1)
            + self.box_labels) \
            + self._legy \
            + self.pie.xs_top \
            + self.pie.xs_bottom \
            + self.tick_y_top_xs

        # Debug output
        if debug:
            print('self.fig.size[0] = %s' % self.fig.size[0])
            vals = ['ws_fig_label', 'label_y', 'ws_label_tick', 'tick_labels_major_y', 'tick_labels_minor_y',
                    'ws_ticks_ax', 'axes', 'cbar', 'ws_ax_cbar', 'ws_col', 'ws_ax_leg', 'legend', 'ws_leg_fig',
                    'label_y2', 'ws_label_tick', 'ws_ticks_ax', 'tick_labels_major_y2', 'label_row',
                    'ws_label_row', 'label_z', 'tick_labels_major_z', 'box_title',
                    'ncol', '_labtick_y', '_labtick_y2', '_labtick_z']
            for val in vals:
                if isinstance(getattr(self, val), Element):
                    print('   %s.size[0] = %s' % (val, getattr(self, val).size[0]))
                else:
                    print('   %s = %s' % (val, getattr(self, val)))
            print('self.fig.size[1] = %s' % self.fig.size[1])
            vals = ['ws_fig_title', 'title', 'ws_title_ax', 'ws_fig_ax', 'label_col', 'ws_label_col', 'title_wrap',
                    'label_wrap', 'label_x2', 'ws_ticks_ax', 'tick_labels_major_x2', 'axes', 'label_x', 'ws_label_tick',
                    'tick_labels_major_x', 'ws_ticks_ax', 'ws_fig_label', 'ws_row', 'box_labels',
                    'nrow', '_labtick_x', '_labtick_x2', 'ws_title']
            for val in vals:
                if isinstance(getattr(self, val), Element):
                    print('   %s.size[1] = %s' % (val, getattr(self, val).size[1]))
                else:
                    print('   %s = %s' % (val, getattr(self, val)))

        # Account for legends longer than the figure
        header = self.ws_title + \
            (self.label_col.size[1] + self.ws_label_col) * self.label_col.on + \
            self.title_wrap.size[1] + self.label_wrap.size[1] + \
            self._labtick_x2

        if self.legend.size[1] + header > self.fig.size[1]:
            self.legend.overflow = self.legend.size[1] + \
                header - self.fig.size[1]
        self.fig.size[1] += self.legend.overflow

    @staticmethod
    def _get_grid_visibility_kwarg(visible: bool) -> Dict:
        """Handle visibility kwarg for different mpl versions (changed at v3.5)

        Args:
            visible: flag to show or hide grid

        Returns:
            single-key dict with the correctly-named bool for showing/hiding grids
        """
        if LooseVersion(mpl.__version__) < LooseVersion('3.5'):
            return {'b': visible}
        else:
            return {'visible': visible}

    def _get_legend_position(self):
        """Get legend position."""
        if self.legend.location == 0:
            title_xs = max(0, (self.title.size[0] - self.axes.size[0]) / 2 - self.legend.size[0])
            if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
                self.legend.position[1] = 1 + (self.fig_legend_border - self.ws_leg_fig - title_xs) / self.fig.size[0]
            elif self.box_group_title.on:
                self.legend.position[1] = 1 + (self.fig_legend_border - self.ws_leg_fig - self.ws_ax_box_title
                                               - self.box_title + self.ws_ax_fig - title_xs) / self.fig.size[0]
            else:
                self.legend.position[1] = 1 + (self.fig_legend_border - self.ws_leg_fig - title_xs) / self.fig.size[0]
            self.legend.position[2] = self.axes.position[2] + self.legend_top_offset / self.fig.size[1]
        if self.legend.location == 11:
            self.legend.position[1] = 0.5
            self.legend.position[2] = 0

    def _get_rc_label_position(self):
        """Get option group label positions.

        self.label.position --> [left, right, top, bottom]
        """
        self.label_row.position[0] = \
            (self.axes.size[0] + self._labtick_y2 + self.ws_label_row
             + self.label_row.edge_width
             + (self.ws_ax_cbar if self.cbar.on else 0) + self.cbar.size[0]
             + self._labtick_z + self.label_z.size[0]) / self.axes.size[0]

        self.label_col.position[3] = (self.axes.size[1] + self.ws_label_col
                                      + self._labtick_x2) / self.axes.size[1]

        self.label_wrap.position[3] = 1

        self.title_wrap.position[3] = 1 + self.label_wrap.size[1] / self.axes.size[1]

    def _get_subplots_adjust(self):
        """Calculate the subplots_adjust parameters for the axes.

        self.axes.position --> [left, right, top, bottom]
        """
        self.axes.position[0] = int(self._left) / self.fig.size[0]

        # note: if using cbar, self.axes.position[1] = 1 means the edge of the
        # cbar is at the edge of the figure (+2 is a fudge to get the right image size)
        self.axes.position[1] = \
            self.axes.position[0] \
            + int(self.axes.size[0] * self.ncol
                  + self.ws_col * (self.ncol - 1)
                  + self.cbar.on * (self.cbar.size[0] + self.ws_ax_cbar + 2) * (self.ncol)
                  + (self.label_z.size[0] * (self.ncol - 1) if self.separate_labels else 0)) \
            / self.fig.size[0]

        self.axes.position[2] = \
            1 - (self.ws_title + self.title_wrap.size[1]
                 + (self.label_col.size[1] + self.ws_label_col) * self.label_col.on
                 + self.tick_y_top_xs
                 + self.label_wrap.size[1] + self._labtick_x2 + self.pie.xs_top) \
            / self.fig.size[1]

        self.axes.position[3] = \
            (self._labtick_x + self.ws_fig_label + self.box_labels
             + self.legend.overflow + self.pie.xs_bottom
             + (self.legend.size[1] if self.legend.location == 11 else 0)) / self.fig.size[1]

    def _get_tick_label_size(self, ax: mplp.Axes, tick: str, tick_num: str, which: str):
        """Get the size of the tick labels on a specific axes (plot must be already rendered).

        Args:
            ax: the axes object for the labels of interest
            tick: name of the tick axes
            tick_num: '' or '2' for secondary
            which: 'major' or 'minor'
        """
        if tick == 'x':  # what about z??
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
            tt.limits[ir, ic] = [vmin, vmax]
            tlabs = [f for f in tlabs if approx_gte(f.get_position()[idx], min(vmin, vmax))
                     and approx_lte(f.get_position()[idx], max(vmin, vmax))]
            tt.obj[ir, ic] = tlabs

            # Get the label sizes and store sizes as 2D array
            bboxes = [t.get_window_extent() for t in tlabs]
            tt.size_all = ([ir for f in bboxes],
                           [ic for f in bboxes],
                           [0 for f in bboxes],
                           [0 for f in bboxes],
                           [f.width for f in bboxes],
                           [f.height for f in bboxes],
                           [f.x0 for f in bboxes],
                           [f.x1 for f in bboxes],
                           [f.y0 for f in bboxes],
                           [f.y1 for f in bboxes],
                           [np.nan for f in bboxes],
                           )
            if tick == 'y' and ir == 0 and ic == 0 and not self.title.on and not self.label_col.on:
                # Padding for top y-tick label that extends beyond top of axes
                # doesn't capture every possible case yet
                ax_y1 = ax.obj[0, 0].get_window_extent().y1
                self.tick_y_top_xs = max(0, self.tick_y_top_xs, tt.size_all['y1'].max() - ax_y1)

        if len(tt.size_all) == 0:
            return

        tt.size = [tt.size_all.width.max(), tt.size_all.height.max()]

    def _get_tick_label_sizes(self):
        """Get the tick label sizes for each axis."""

        for tick in ['x', 'y']:
            getattr(self, f'tick_labels_major_{tick}').size_all_reset()
            getattr(self, f'tick_labels_minor_{tick}').size_all_reset()
            getattr(self, f'tick_labels_major_{tick}2').size_all_reset()
            getattr(self, f'tick_labels_minor_{tick}2').size_all_reset()

            # primary axes
            self._get_tick_label_size(self.axes, tick, '', 'major')
            self._get_tick_label_size(self.axes, tick, '', 'minor')

            # secondary axes
            if self.axes2.on:
                self._get_tick_label_size(self.axes2, tick, '2', 'major')
                self._get_tick_label_size(self.axes2, tick, '2', 'minor')

    def _get_tick_overlaps(self, axis: str = ''):
        """Deal with overlapping and out of range ticks.

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

            # Prevent single tick label axis by adding a text label at one or more range limits
            if len(xticks_size_all) <= 1 \
                    and self.name not in ['box', 'bar', 'pie'] \
                    and (not self.axes.share_x or len(self.axes.obj.flatten()) == 1) \
                    and axis != '2' \
                    and self.tick_labels_major_x.on:
                kw = {}
                kw['rotation'] = xticks.rotation
                kw['font_color'] = xticks.font_color
                kw['font'] = xticks.font
                kw['fill_color'] = xticks.fill_color[0]
                kw['edge_color'] = xticks.edge_color[0]
                kw['font_style'] = xticks.font_style
                kw['font_weight'] = xticks.font_weight
                kw['font_size'] = xticks.font_size / sf
                kw['position'] = \
                    [self.axes.size[0] - xticks.size_all.loc[0, 'width'] / 2 / sf,
                     -xticks.size_all.loc[0, 'height']]
                precision = utl.get_decimals(xticks.limits[ir, ic][1], 8)
                txt = f'{xticks.limits[ir, ic][1]:.{precision}f}'
                self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', **kw)

            if len(yticks_size_all) <= 1 \
                    and self.name not in ['box', 'bar', 'pie'] \
                    and (not self.axes.share_y or len(self.axes.obj.flatten()) == 1) \
                    and axis != '2' \
                    and yticks.limits[ir, ic]:
                kw = {}
                kw['rotation'] = yticks.rotation
                kw['font_color'] = yticks.font_color
                kw['font'] = yticks.font
                kw['fill_color'] = yticks.fill_color[0]
                kw['edge_color'] = yticks.edge_color[0]
                kw['font_style'] = yticks.font_style
                kw['font_weight'] = yticks.font_weight
                kw['font_size'] = yticks.font_size / sf
                if len(yticks_size_all) == 0:
                    # this case can only happen if the ylimits are so tight that there are no gridlines present
                    precision = utl.get_decimals(yticks.limits[ir, ic][0], 8)
                    txt = f'{yticks.limits[ir, ic][0]:.{precision}f}'
                    x = -kw['font_size'] * len(txt.replace('.', ''))
                    y = -kw['font_size'] / 2
                    kw['position'] = [x, y]
                    self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', **kw)
                    y += self.axes.size[1]
                else:
                    # this case happens if only one gridline is present on the plot
                    x = -yticks.size_all.loc[0, 'width']
                    y = self.axes.size[1] - yticks.size_all.loc[0, 'height'] / 2 / sf
                precision = utl.get_decimals(yticks.limits[ir, ic][1], 8)
                txt = f'{yticks.limits[ir, ic][1]:.{precision}f}'
                x = -kw['font_size'] * len(txt.replace('.', ''))
                kw['position'] = [x, y]
                self.add_text(ir, ic, txt, element='text', coord='axis', units='pixel', **kw)

            # Shrink/remove overlapping ticks x-y origin
            if len(xticks_size_all) > 0 and len(yticks_size_all) > 0:
                idx = xticks.size_cols.index('width')
                xw, xh, xx0, xx1, xy0, xy1 = xticks_size_all[0][idx:-1]
                xc = (xx0 + (xx1 - xx0) / 2, xy0 + (xy0 - xy1) / 2)
                yw, yh, yx0, yx1, yy0, yy1 = yticks_size_all[0][idx:-1]
                yc = (yx0 + (yx1 - yx0) / 2, yy0 + (yy0 - yy1) / 2)
                if utl.rectangle_overlap((xw, xh, xc), (yw, yh, yc)):
                    if self.tick_cleanup == 'remove':
                        yticks.obj[ir, ic][0].set_visible(False)
                    else:
                        xticks.obj[ir, ic][0].set_size(xticks.font_size / sf)
                        yticks.obj[ir, ic][0].set_size(yticks.font_size / sf)

            # Shrink/remove overlapping ticks in grid plots at x-origin
            if ic > 0 and len(xticks_size_all) > 0:
                idx = xticks.size_cols.index('x0')
                ax_x0 = self.axes.obj[ir, ic].get_window_extent().x0
                tick_x0 = xticks_size_all[0][idx]
                if ax_x0 - tick_x0 > self.ws_col:
                    slop = xticks.obj[ir, ic][0].get_window_extent().width / 2
                    if self.tick_cleanup == 'remove' or slop / sf > self.ws_col:
                        xticks.obj[ir, ic][0].set_visible(False)
                    else:
                        xticks.obj[ir, ic][0].set_size(xticks.font_size / sf)

            # TODO: Shrink/remove overlapping ticks in grid plots at y-origin

            # Remove overlapping ticks on same axis
            if len(xticks_size_all) > 0:
                xbbox = df_tick(xticks, xticks_size_all, 'x')
                xbbox = hide_overlaps(xticks, xbbox, ir, ic)
            if len(xticksm_size_all) > 0:
                xbboxm = df_tick_update(df_tick(xticksm, xticksm_size_all, 'x'))
                xbboxm = hide_overlaps_major_minor(xticks, xticksm, xbbox, xbboxm, ir, ic)
                xbboxm = hide_overlaps(xticksm, xbboxm, ir, ic)

            if len(yticks_size_all) > 0:
                ybbox = df_tick(yticks, yticks_size_all, 'y')
                ybbox = hide_overlaps(yticks, ybbox, ir, ic)
            if len(yticksm_size_all) > 0:
                ybboxm = df_tick_update(df_tick(yticksm, yticksm_size_all, 'y'))
                ybboxm = hide_overlaps_major_minor(yticks, yticksm, ybbox, ybboxm, ir, ic)
                ybboxm = hide_overlaps(yticksm, ybboxm, ir, ic)

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
                if xxs < 0:
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

    def _get_title_position(self):
        """Calculate the title position.

        self.title.position --> [left, right, top, bottom]
        """
        col_label = (self.label_col.size[1]
                     + self.ws_label_col * self.label_col.on) \
            + (self.label_wrap.size[1] + self.ws_label_col * self.label_wrap.on)
        self.title.position[0] = self.ncol / 2
        if self.label_wrap.size[1] > 0:
            title_height = 0
        else:
            # this is an offset from somewhere I don't know
            title_height = self.title.size[1] / 2 - 2
        self.title.position[3] = 1 + (self.ws_title_ax
                                      + col_label + title_height
                                      + self.label_wrap.size[1]) / self.axes.size[1]
        self.title.position[2] = \
            self.title.position[3] + (self.ws_title_ax + self.title.size[1]) / self.axes.size[1]

    def make_figure(self, data: 'Data', **kwargs):  # noqa: F821
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        # Create the subplots
        #   Note we don't have the actual element sizes until rendereing
        #   so we use an approximate size based upon what is known
        self._get_figure_size(data, **kwargs)
        self.fig.obj, self.axes.obj = \
            mplp.subplots(data.nrow, data.ncol,
                          figsize=[self.fig.size_inches[0], self.fig.size_inches[1]],
                          sharex=self.axes.share_x,
                          sharey=self.axes.share_y,
                          dpi=self.fig.dpi,
                          facecolor=self.fig.fill_color[0],
                          edgecolor=self.fig.edge_color[0],
                          linewidth=self.fig.edge_width,
                          )

        # Set default axes visibility
        self.axes.visible = np.array([[True] * self.ncol] * self.nrow)

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
                idx = [f + inst[i] * kwargs['height']
                       for i, f in enumerate(idx)]
                init_off = (total - 1) / 2 * kwargs['height']
                idx = list((idx - init_off).values)
        else:
            bar = ax.bar
            axx = 'x'
            if self.bar.stacked:
                kwargs['width'] = self.bar.width
                if iline > 0:
                    if isinstance(stacked, pd.Series):
                        stacked = stacked.loc[xvals[idx]].values
                    kwargs['bottom'] = stacked
            else:
                kwargs['width'] = self.bar.width / ngroups
                idx = [f + inst[i] * kwargs['width'] for i, f in enumerate(idx)]
                init_off = (total - 1) / 2 * kwargs['width']
                idx = list((idx - init_off).values)

        if self.bar.color_by_bar:
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
        except TypeError:
            # Show all labels
            getattr(ax, 'set_%sticks' % axx)(np.arange(0, len(xvals), 1))
        allowed_ticks = getattr(ax, 'get_%sticks' % axx)()  # mpl selected ticks
        allowed_ticks = list(set([int(f) for f in allowed_ticks if f >= 0 and f < len(xvals)]))
        getattr(ax, 'set_%sticks' % axx)(np.array(ixvals)[allowed_ticks])
        getattr(ax, 'set_%sticklabels' % axx)(xvals[allowed_ticks])

        if iline == 0:
            # Update ranges
            new_ticks = getattr(ax, 'get_%sticks' % axx)()
            tick_off = [f for f in new_ticks if f >= 0][0]
            if self.bar.horizontal:
                axx = 'y'
                data.ranges[ir, ic]['xmin'] = data.ranges[ir, ic]['ymin']
                data.ranges[ir, ic]['xmax'] = data.ranges[ir, ic]['ymax']
                data.ranges[ir, ic]['ymin'] = None
                data.ranges[ir, ic]['ymax'] = None
            else:
                axx = 'x'
            xoff = 3 * self.bar.width / 4
            if data.ranges[ir, ic]['%smin' % axx] is None:
                data.ranges[ir, ic]['%smin' % axx] = -xoff + tick_off
            else:
                data.ranges[ir, ic]['%smin' % axx] += tick_off
            if data.ranges[ir, ic]['%smax' % axx] is None:
                data.ranges[ir, ic]['%smax' % axx] = len(xvals) - 1 + xoff + tick_off
            else:
                data.ranges[ir, ic]['%smax' % axx] += tick_off

        # Legend
        if leg_name is not None:
            handle = [patches.Rectangle((0, 0), 1, 1,
                      color=self.bar.fill_color[(iline, leg_name)])]
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

    def plot_contour(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, z: str,
                     data: 'Data') -> ['MPL_contour_object', 'MPL_colorbar_object']:  # noqa: F821
        """Plot a contour plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            x: x-axis column name
            y: y-axis column name
            z: z-column name
            data: Data object

        Returns:
            reference to the contour plot object
            reference to the colorbar object
        """
        ax = self.axes.obj[ir, ic]
        ranges = data.ranges[ir, ic]

        # Convert data type
        xx = np.array(df[x])
        yy = np.array(df[y])
        zz = np.array(df[z])

        # Make the grid
        xi = np.linspace(min(xx), max(xx))
        yi = np.linspace(min(yy), max(yy))
        if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
            zi = mlab.griddata(xx, yy, zz, xi, yi, interp=self.contour.interp)
        else:
            zi = scipy.interpolate.griddata((xx, yy), zz,
                                            (xi[None, :], yi[:, None]),
                                            method=self.contour.interp)

        # Deal with out of range values
        zi = np.clip(zi, ranges['zmin'], ranges['zmax'])

        # Set the contours
        levels = np.linspace(ranges['zmin'] * 0.999, ranges['zmax'] * 1.001,
                             self.contour.levels)

        # Plot
        if self.contour.filled:
            cc = ax.contourf(xi, yi, zi, levels, cmap=self.cmap, zorder=2)
        else:
            cc = ax.contour(xi, yi, zi, levels,
                            linewidths=self.contour.width.values,
                            cmap=self.cmap, zorder=2)

        # Add a colorbar
        if self.cbar.on:
            self.cbar.obj[ir, ic] = self.add_cbar(ax, cc)
            new_ticks = cbar_ticks(self.cbar.obj[ir, ic], ranges['zmin'], ranges['zmax'])
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

        return cc, self.cbar.obj

    def plot_gantt(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str, y: str,
                   leg_name: str, yvals: list, ngroups: int):
        """Plot gantt graph.

        Args:
            ir: subplot row index
            ic: subplot column index
            iline: data subset index (from Data.get_plot_data)
            df: input data
            x: x-axis column name
            y: y-axis column name
            leg_name: legend value name if legend enabled
            yvals: list of tuples of groupling column values
            ngroups: total number of groups in the full data set based on
                data.get_plot_data

        """
        ax = self.axes.obj[ir, ic]
        bar = ax.broken_barh

        # Set the color values
        if self.gantt.color_by_bar:
            edgecolor = [self.gantt.edge_color[i]
                         for i, f in enumerate(df.index)]
            fillcolor = [self.gantt.fill_color[i]
                         for i, f in enumerate(df.index)]
        else:
            edgecolor = [self.gantt.edge_color[(iline, leg_name)]
                         for i, f in enumerate(df.index)]
            fillcolor = [self.gantt.fill_color[(iline, leg_name)]
                         for i, f in enumerate(df.index)]

        # Plot the bars
        for ii, (irow, row) in enumerate(df.iterrows()):
            if leg_name is not None:
                yi = yvals.index((row[y], leg_name))
            else:
                yi = yvals.index((row[y],))
            bar([(row[x[0]], row[x[1]] - row[x[0]])],
                (yi - self.gantt.height / 2, self.gantt.height),
                facecolor=fillcolor[ii],
                edgecolor=edgecolor[ii],
                linewidth=self.gantt.edge_width)

        # Adjust the yticklabels
        if iline + 1 == ngroups:
            yvals = [f[0] for f in yvals]
            ax.set_yticks(range(-1, len(yvals)))
            ax.set_yticklabels([''] + list(yvals))

        # Legend
        if leg_name is not None:
            handle = [patches.Rectangle((0, 0), 1, 1,
                      color=self.gantt.fill_color[(iline, leg_name)])]
            self.legend.add_value(leg_name, handle, 'lines')

    def plot_heatmap(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str,
                     z: str, data: 'Data') -> 'MPL_imshow_object':  # noqa: F821
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
        ranges = data.ranges[ir, ic]

        # Make the heatmap
        im = ax.imshow(df, self.cmap, vmin=ranges['zmin'],
                       vmax=ranges['zmax'],
                       interpolation=self.heatmap.interp)
        im.set_clim(ranges['zmin'], ranges['zmax'])

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
        if ranges['xmin'] is not None and ranges['xmin'] > 0:
            xticks = ax.get_xticks()
            ax.set_xticklabels([int(f + ranges['xmin']) for f in xticks])

        # Add the cbar
        if self.cbar.on:
            self.cbar.obj[ir, ic] = self.add_cbar(ax, im)

        # Loop over data dimensions and create text annotations
        if self.heatmap.text:
            for iy, yy in enumerate(df.index):
                for ix, xx in enumerate(df.columns):
                    if type(df.loc[yy, xx]) in [float, np.float32, np.float64] and np.isnan(df.loc[yy, xx]):
                        continue
                    text = ax.text(ix, iy, df.loc[yy, xx],  # noqa
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
        if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
            hist = self.axes.obj[ir, ic].hist(df[x], bins=self.hist.bins,
                                              color=self.hist.fill_color[iline],
                                              ec=self.hist.edge_color[iline],
                                              lw=self.hist.edge_width,
                                              zorder=3, align=self.hist.align,
                                              cumulative=self.hist.cumulative,
                                              normed=self.hist.normalize,
                                              rwidth=self.hist.rwidth,
                                              stacked=self.hist.stacked,
                                              orientation='vertical' if not self.hist.horizontal else 'horizontal',
                                              )
        else:
            hist = self.axes.obj[ir, ic].hist(df[x], bins=self.hist.bins,
                                              color=self.hist.fill_color[iline],
                                              ec=self.hist.edge_color[iline],
                                              lw=self.hist.edge_width,
                                              zorder=3, align=self.hist.align,
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
                x0 = np.linspace(data.ranges[ir, ic]['xmin'], data.ranges[ir, ic]['xmax'], 1000)
                y0 = kde(x0)
            else:
                y0 = np.linspace(data.ranges[ir, ic]['ymin'], data.ranges[ir, ic]['ymax'], 1000)
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
        ranges = data.ranges[ir, ic]

        # Make the heatmap (maybe pull these kwargs out to an imshow obj?)
        im = ax.imshow(df.dropna(axis=1, how='all'), self.cmap,
                       vmin=ranges['zmin'], vmax=ranges['zmax'],
                       interpolation=self.imshow.interp)
        im.set_clim(ranges['zmin'], ranges['zmax'])

        # Add a cmap
        if self.cbar.on:  # and (self.separate_ticks or ic == self.ncol - 1):
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
                                    alpha=self.markers.edge_alpha,
                                    zorder=40
                                    )
            else:
                # what is the use case here?
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
            if leg_name is not None \
                    and leg_name not in list(self.legend.values['Key']):
                self.legend.add_value(leg_name, points if points is not None else lines, line_type_name)

    def save(self, filename: str, idx: int = 0):
        """Save a plot window.

        Args:
            filename: name of the file
            idx (optional): figure index in order to set the edge and face color of the
                figure correctly when saving. Defaults to 0.
        """
        kwargs = {'edgecolor': self.fig.edge_color[idx],
                  'facecolor': self.fig.fill_color[idx]}
        if LooseVersion(mpl.__version__) < LooseVersion('3.3'):
            kwargs['linewidth'] = self.fig.edge_width
        self.fig.obj.savefig(filename, **kwargs)

    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        axes = self._get_axes()

        # for ax in axes:
        try:
            axes[0].obj[ir, ic].set_facecolor(axes[0].fill_color[utl.plot_num(ir, ic, self.ncol)])
        except:  # noqa
            axes[0].obj[ir, ic].set_axis_bgcolor(axes[0].fill_color[utl.plot_num(ir, ic, self.ncol)])
        for f in ['bottom', 'top', 'right', 'left']:
            if len(axes) > 1:
                axes[0].obj[ir, ic].spines[f].set_visible(False)
            if getattr(self.axes, 'spine_%s' % f):
                axes[-1].obj[ir, ic].spines[f].set_color(axes[0].edge_color[utl.plot_num(ir, ic, self.ncol)])
            else:
                axes[-1].obj[ir,
                             ic].spines[f].set_color(self.fig.fill_color[0])
            if self.axes.edge_width != 1:
                axes[-1].obj[ir,
                             ic].spines[f].set_linewidth(self.axes.edge_width)

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

    def set_axes_labels(self, ir: int, ic: int):
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        if self.name in ['pie']:
            return

        axis = ['x', 'x2', 'y', 'y2', 'z']
        for ax in axis:
            label = getattr(self, 'label_%s' % ax)
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
                if ax == 'x' and ir != self.nrow - 1 and self.nwrap == 0 and self.axes.visible[ir + 1, ic]:
                    continue
                if ax == 'x2' and ir != 0:
                    continue
                if ax == 'y' and ic != 0 and self.axes.visible[ir, ic - 1]:
                    continue
                if ax == 'y2' and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue
                if ax == 'z' and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue

            # Add the label
            label.obj[ir, ic], label.obj_bg[ir, ic] = \
                self.add_label(ir, ic, labeltext, **self.make_kw_dict(label))

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        if self.name in ['heatmap', 'pie']:  # skip these plot types
            return

        if ranges[ir, ic]['xmin'] is not None:
            self.axes.obj[ir, ic].set_xlim(left=ranges[ir, ic]['xmin'])
        if ranges[ir, ic]['x2min'] is not None:
            self.axes2.obj[ir, ic].set_xlim(left=ranges[ir, ic]['x2min'])
        if ranges[ir, ic]['xmax'] is not None:
            self.axes.obj[ir, ic].set_xlim(right=ranges[ir, ic]['xmax'])
        if ranges[ir, ic]['x2max'] is not None:
            self.axes2.obj[ir, ic].set_xlim(right=ranges[ir, ic]['x2max'])
        if ranges[ir, ic]['ymin'] is not None:
            self.axes.obj[ir, ic].set_ylim(bottom=ranges[ir, ic]['ymin'])
        if ranges[ir, ic]['y2min'] is not None:
            self.axes2.obj[ir, ic].set_ylim(bottom=ranges[ir, ic]['y2min'])
        if ranges[ir, ic]['ymax'] is not None:
            self.axes.obj[ir, ic].set_ylim(top=ranges[ir, ic]['ymax'])
        if ranges[ir, ic]['y2max'] is not None:
            self.axes2.obj[ir, ic].set_ylim(top=ranges[ir, ic]['y2max'])

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Wrap title
        if ir == 0 and ic == 0 and self.title_wrap.on:
            kwargs = self.make_kw_dict(self.title_wrap)
            if self.axes.edge_width == 0:
                kwargs['size'][0] -= 1
            if self.name == 'imshow' and not self.cbar.on and self.nrow == 1:
                kwargs['size'][0] -= 1  # don't understand this one
            if self.cbar.on:
                kwargs['size'][0] -= (self.ws_ax_cbar + self.cbar.size[0])
            self.title_wrap.obj, self.title_wrap.obj_bg = self.add_label(ir, ic, self.title_wrap.text, **kwargs)

        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_row.text_size[ir, ic]
            else:
                text_size = None

            if self.label_row.names:
                lab = f'{self.label_row.text}={self.label_row.values[ir]}'
            else:
                lab = self.label_row.values[ir]
            self.label_row.obj[ir, ic], self.label_row.obj_bg[ir, ic] = \
                self.add_label(ir, ic, lab, offset=True, **self.make_kw_dict(self.label_row))

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_col.text_size[ir, ic]
            else:
                text_size = None  # noqa
            if self.label_wrap.on:
                kwargs = self.make_kw_dict(self.label_wrap)
                if self.axes.edge_width == 0:
                    kwargs['size'][0] -= 1
                if self.name == 'imshow' and not self.cbar.on and self.nrow == 1:
                    kwargs['size'][0] -= 1
                text = ' | '.join([str(f) for f in utl.validate_list(
                    self.label_wrap.values[ir * self.ncol + ic])])
                self.label_wrap.obj[ir, ic], self.label_wrap.obj_bg[ir, ic] = \
                    self.add_label(ir, ic, text, **kwargs)
            else:
                if self.label_col.names:
                    lab = f'{self.label_col.text}={self.label_col.values[ic]}'
                else:
                    lab = self.label_col.values[ic]
                self.label_col.obj[ir, ic], self.label_col.obj_bg[ir, ic] = \
                    self.add_label(ir, ic, lab, **self.make_kw_dict(self.label_col))

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
            if hasattr(getattr(self, f'axes{lab}'), 'share_x%s' % lab) \
                    and getattr(getattr(self, f'axes{lab}'), 'share_x%s' % lab) is True \
                    and (ir != 0 or ic != 0):
                skipx = False
            if hasattr(getattr(self, f'axes{lab}'), 'share_y%s' % lab) \
                    and getattr(getattr(self, f'axes{lab}'), 'share_y%s' % lab) \
                    and (ir != 0 or ic != 0):
                skipy = False

            # Turn off scientific
            if ia == 0:
                if not skipx:
                    self.set_scientific(axes[ia], tp)
            elif self.axes.twin_y or self.axes.twin_x:
                if not skipy:
                    self.set_scientific(axes[ia], tp, 2)

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
                axes[0].tick_params(axis='x',
                                    which='major',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_major_x.color[0],
                                    labelcolor=self.tick_labels_major_x.font_color,
                                    labelsize=self.tick_labels_major_x.font_size,
                                    top=False,
                                    bottom=self.ticks_major_x.on,
                                    right=False,
                                    left=False,
                                    length=self.ticks_major_x._size[0],
                                    width=self.ticks_major_x._size[1],
                                    direction=self.ticks_major.direction,
                                    zorder=100,
                                    )
                axes[0].tick_params(axis='y',
                                    which='major',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_major_y.color[0],
                                    labelcolor=self.tick_labels_major_y.font_color,
                                    labelsize=self.tick_labels_major_y.font_size,
                                    top=False,
                                    bottom=False,
                                    right=False if self.axes.twin_x
                                    else self.ticks_major_y.on,
                                    left=self.ticks_major_y.on,
                                    length=self.ticks_major_y._size[0],
                                    width=self.ticks_major_y._size[1],
                                    direction=self.ticks_major_y.direction,
                                    zorder=100,
                                    )
                axes[0].tick_params(axis='x',
                                    which='minor',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_minor_x.color[0],
                                    labelcolor=self.tick_labels_minor_x.font_color,
                                    labelsize=self.tick_labels_minor_x.font_size,
                                    top=False,
                                    bottom=self.ticks_minor_x.on,
                                    right=False,
                                    left=False,
                                    length=self.ticks_minor_x._size[0],
                                    width=self.ticks_minor_x._size[1],
                                    direction=self.ticks_minor_x.direction,
                                    )
                axes[0].tick_params(axis='y',
                                    which='minor',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_minor_y.color[0],
                                    labelcolor=self.tick_labels_minor_y.font_color,
                                    labelsize=self.tick_labels_minor_y.font_size,
                                    top=False,
                                    bottom=False,
                                    right=False if self.axes.twin_x
                                    else self.ticks_minor_y.on,
                                    left=self.ticks_minor_y.on,
                                    length=self.ticks_minor_y._size[0],
                                    width=self.ticks_minor_y._size[1],
                                    direction=self.ticks_minor_y.direction,
                                    )
                if self.axes.twin_x:
                    if self.ticks_minor_y2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=self.ws_ticks_ax,
                                        colors=self.ticks_major_y2.color[0],
                                        labelcolor=self.tick_labels_major_y2.font_color,
                                        labelsize=self.tick_labels_major_y2.font_size,
                                        right=self.ticks_major_y2.on,
                                        length=self.ticks_major_y2.size[0],
                                        width=self.ticks_major_y2.size[1],
                                        direction=self.ticks_major_y2.direction,
                                        zorder=0,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=self.ws_ticks_ax,
                                        colors=self.ticks_minor_y2.color[0],
                                        labelcolor=self.tick_labels_minor_y2.font_color,
                                        labelsize=self.tick_labels_minor_y2.font_size,
                                        right=self.ticks_minor_y2.on,
                                        length=self.ticks_minor_y2._size[0],
                                        width=self.ticks_minor_y2._size[1],
                                        direction=self.ticks_minor_y2.direction,
                                        zorder=0,
                                        )
                elif self.axes.twin_y:
                    if self.ticks_minor_x2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=self.ws_ticks_ax,
                                        colors=self.ticks_major_x2.color[0],
                                        labelcolor=self.tick_labels_major_x2.font_color,
                                        labelsize=self.tick_labels_major_x2.font_size,
                                        top=self.ticks_major_x2.on,
                                        length=self.ticks_major_x2.size[0],
                                        width=self.ticks_major_x2.size[1],
                                        direction=self.ticks_major_x2.direction,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=self.ws_ticks_ax * 2,
                                        colors=self.ticks_minor.color[0],
                                        labelcolor=self.tick_labels_minor.font_color,
                                        labelsize=self.tick_labels_minor.font_size,
                                        top=self.ticks_minor_x2.on,
                                        length=self.ticks_minor._size[0],
                                        width=self.ticks_minor._size[1],
                                        direction=self.ticks_minor.direction,
                                        )

            # Set custom tick increment
            redo = True
            xinc = getattr(self, 'ticks_major_x%s' % lab).increment
            if not skipx and xinc is not None and self.name not in ['bar']:
                xstart = 0 if tp['x']['min'] == 0 else tp['x']['min'] + \
                    xinc - tp['x']['min'] % xinc
                axes[ia].set_xticks(np.arange(xstart, tp['x']['max'], xinc))
                redo = True
            yinc = getattr(self, 'ticks_major_y%s' % lab).increment
            if not skipy and yinc is not None:
                axes[ia].set_yticks(
                    np.arange(tp['y']['min'] + yinc - tp['y']['min'] % yinc,
                              tp['y']['max'], yinc))
                redo = True

            if redo:
                tp = mpl_get_ticks(axes[ia], True, True, minor_on)

            # Force ticks
            if self.separate_ticks or getattr(self, f'axes{lab}').share_x is False:
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    mplp.setp(axes[ia].get_xticklabels(), visible=True)
                else:
                    if self.axes.twin_x and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)
                    elif self.axes.twin_y and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labeltop=True)
                    else:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)

            if self.separate_ticks or getattr(self, f'axes{lab}').share_y is False:
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
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
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    mplp.setp(axes[ia].get_xticklabels()[1:], visible=True)
                elif self.axes.twin_y and ia == 1:
                    axes[ia].yaxis.set_tick_params(which='both', labeltop=True)
                else:
                    axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)

            if not self.separate_ticks and not self.axes.visible[ir, ic - 1]:
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
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

            if not self.separate_ticks and ir != 0 and self.axes.twin_y and ia == 1 and self.axes2.share_x:
                mplp.setp(axes[ia].get_xticklabels(), visible=False)

            # Major rotation
            axx = ['x', 'y']
            majmin = ['major', 'minor']
            for ax in axx:
                for mm in majmin:
                    if getattr(self, f'tick_labels_{mm}_x{lab}').on:
                        ticks_font = font_manager.FontProperties(
                            family=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font,
                            size=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_size,
                            style=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_style,
                            weight=getattr(self, f'tick_labels_{mm}_{ax}{lab}').font_weight)
                        style = dict(edgecolor=getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_color[0],
                                     facecolor=getattr(self, f'tick_labels_{mm}_{ax}{lab}').fill_color[0],
                                     linewidth=getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_width,
                                     alpha=max(getattr(self, f'tick_labels_{mm}_{ax}{lab}').edge_alpha,
                                               getattr(self, f'tick_labels_{mm}_{ax}{lab}').fill_alpha))
                        rotation = getattr(self, f'tick_labels_{mm}_{ax}{lab}').rotation
                        for text in getattr(axes[ia], f'get_{ax}ticklabels')(which=mm):
                            if rotation != 0:
                                text.set_rotation(rotation)
                            text.set_fontproperties(ticks_font)
                            text.set_bbox(style)

            # Tick label shorthand
            tlmajx = getattr(self, 'tick_labels_major_x%s' % lab)
            tlmajy = getattr(self, 'tick_labels_major_y%s' % lab)

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
            if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
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
                axl = '%s%s' % (axx, lab)
                tlmin = getattr(self, 'ticks_minor_%s' % axl)

                if ia == 1 and axx == 'x' and self.axes.twin_x:
                    continue

                if ia == 1 and axx == 'y' and self.axes.twin_y:
                    continue

                if getattr(self, 'ticks_minor_%s' % axl).number is not None:
                    num_minor = getattr(self, 'ticks_minor_%s' % axl).number
                    if getattr(self, 'axes%s' % lab).scale not in (LOG_ALLX if axx == 'x' else LOG_ALLY):
                        loc = None
                        loc = AutoMinorLocator(num_minor + 1)
                        getattr(axes[ia], '%saxis' %
                                axx).set_minor_locator(loc)
                        tp = mpl_get_ticks(axes[ia],
                                           getattr(self, 'ticks_major_x%s' % lab).on,
                                           getattr(self, 'ticks_major_y%s' % lab).on,
                                           True)

                if not self.separate_ticks and axl == 'x' and ir != self.nrow - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'y2' and ic != self.ncol - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'x2' and ir != 0 or \
                        not self.separate_ticks and axl == 'y' and ic != 0 or \
                        not self.separate_ticks and axl == 'y2' \
                        and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    axes[ia].tick_params(which='minor', **sides[axl])

                elif tlmin.on:
                    if not getattr(self, 'tick_labels_minor_%s' % axl).on:
                        continue
                    elif 'x' in axx and skipx:
                        continue
                    elif 'y' in axx and skipy:
                        continue
                    else:
                        tlminon = True  # noqa

                    # Determine the minimum number of decimals needed to display the minor ticks
                    m0 = len(tp[axx]['ticks'])
                    lim = getattr(axes[ia], 'get_%slim' % axx)()
                    vals = [f for f in tp[axx]['ticks'] if f > lim[0]]
                    label_vals = [f for f in tp[axx]
                                  ['label_vals'] if f > lim[0]]
                    inc = label_vals[1] - label_vals[0]
                    minor_ticks = [f[1] for f in tp[axx]['labels']][m0:]

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
                    decimals = utl.get_decimals(inc / number)

                    # Check for minor ticks below the first major tick for log axes
                    if getattr(self, 'axes%s' % lab).scale in (LOGX if axx == 'x' else LOGY):
                        extra_minor = [f for f in minor_ticks
                                       if f < label_vals[0] and f > lim[0]]
                        if len(extra_minor) > 0:
                            decimals += 1

                    # Set the tick decimal format
                    getattr(axes[ia], '%saxis' % axx).set_minor_formatter(
                        ticker.FormatStrFormatter('%%.%sf' % (decimals)))

    def set_colormap(self, data: 'Data', **kwargs):  # noqa: F821
        """Replace the color list with discrete values from a colormap.

        Args:
            data: Data object
            kwargs: keyword args
        """
        if not self.cmap or self.name in \
                ['contour', 'heatmap', 'imshow']:
            return

        try:
            # Conver the color map into discrete colors
            cmap = mplp.get_cmap(self.cmap)
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
            if self.legend.column is None:
                # NO IDEA HOW THIS CASE COULD BE, CONSIDER REMOVING
                if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
                    self.label_y.font_color = color_list[0]
                if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
                    self.label_y2.font_color = color_list[1]
                if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                    self.label_x.font_color = color_list[0]
                if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                    self.label_x2.font_color = color_list[1]

            self.lines.color.values = copy.copy(color_list)
            self.lines.color_alpha('color', 'alpha')
            self.markers.edge_color.values = copy.copy(color_list)
            self.markers.color_alpha('edge_color', 'edge_alpha')
            self.markers.fill_color.values = copy.copy(color_list)
            self.markers.color_alpha('fill_color', 'fill_alpha')

        except:  # noqa
            print('Could not find a colormap called "%s". Using default colors...' % self.cmap)

    def set_figure_final_layout(self, data: 'Data', **kwargs):  # noqa: F821
        """Final adjustment of the figure size and plot spacing.

        Args:
            data: Data object
            kwargs: keyword args
        """
        # Subplots within self.fig.obj are not currently in the right place and do not have the right size.  Before
        #   checking for overlaps with other elements we temporarily move the subplots to (0, 0) and size properly
        self._subplots_adjust_x0y0()

        # Render dummy figure to get the element sizes
        self._get_element_sizes(data)

        # Determine if extra whitespace is needed at the plot edge for the last tick
        self._get_tick_xs()

        # Resize the figure
        self._get_figure_size(data, **kwargs)
        self.fig.obj.set_size_inches((self.fig.size_inches[0], self.fig.size_inches[1]))

        # Adjust subplot spacing
        self._get_subplots_adjust()
        self.fig.obj.subplots_adjust(left=self.axes.position[0],
                                     right=self.axes.position[1],
                                     top=self.axes.position[2],
                                     bottom=self.axes.position[3],
                                     hspace=1.0 * self.ws_row / self.axes.size[1],
                                     wspace=1.0 * self.ws_col / self.axes.size[0],
                                     )

        # Tick overlap cleanup
        self._get_tick_label_sizes()  # update after axes reshape
        self._get_tick_overlaps()
        self._get_tick_overlaps('2')

        # Update the axes label positions
        self._get_axes_label_position()
        labels = ['x', 'x2', 'y', 'y2', 'z', 'row', 'col']
        for label in labels:
            # for ir, ic in np.ndindex(ax.obj.shape):
            lab = getattr(self, f'label_{label}')
            if not lab.on:
                continue
            x, y = getattr(self, f'label_{label}').position_xy
            for ir, ic in np.ndindex(lab.obj.shape):
                if lab.obj[ir, ic]:
                    lab.obj[ir, ic].set_position((x, y))
                    if label in ['x', 'x2', 'y', 'y2', 'z']:
                        offsetx = lab.size[0] / 2 + lab.bg_padding
                        offsety = lab.size[1] / 2 + lab.bg_padding
                        if lab.obj[ir, ic].get_rotation() == 90:
                            offsetx += 2  # this may not hold for all cases
                            offsety += 1
                        lab.obj_bg[ir, ic].set_x(x - offsetx / self.axes.size[0])
                        lab.obj_bg[ir, ic].set_y(y - offsety / self.axes.size[1])

        # Update the rc label positions
        self._get_rc_label_position()
        # row
        for ir, ic in np.ndindex(self.label_row.obj.shape):
            if self.label_row.obj[ir, ic]:
                if self.label_row.rotation == 270 and not self.label_col.on:
                    offset = -2
                else:
                    offset = 0
                x_rect = self.label_row.position[0]
                if not self.axes.visible[ir, self.ncol - 1]:
                    # weird cbar issue if the last column plot is not visible (still see a width offset that is ??)
                    x_rect -= ((self.ws_label_row + self.ws_ax_cbar if self.cbar.on else 0) +
                               self.cbar.size[0]) / self.axes.size[0]
                x_text = x_rect + (self.label_row.size[0] / 2 + offset) / self.axes.size[0]
                self.label_row.obj[ir, ic].set_position((x_text, 0.5))
                self.label_row.obj_bg[ir, ic].set_x(x_rect)
        # col
        for ir, ic in np.ndindex(self.label_col.obj.shape):
            if self.label_col.obj[ir, ic]:
                if self.label_col.rotation == 0 and not self.label_row.on:
                    offset = -2
                else:
                    offset = 0
                y_rect = self.label_col.position[3]
                y_text = y_rect + \
                    (self.label_col.size[1] / 2 + offset) / \
                    self.axes.size[1]
                self.label_col.obj[ir, ic].set_position((0.5, y_text))
                self.label_col.obj_bg[ir, ic].set_y(y_rect)
                self.label_col.obj_bg[ir, ic].set_width(1)
        # wrap label
        if self.name in ['imshow'] and self.ncol > 1:
            hack = 0  # some weirdness on cbar or imshow plots; here is a stupid hack
        elif self.name in ['imshow']:
            hack = -1
        else:
            hack = 1
        for ir, ic in np.ndindex(self.label_wrap.obj.shape):
            if self.label_wrap.obj[ir, ic]:
                offset = 0
                y_rect = self.label_wrap.position[3]
                y_text = y_rect + \
                    (self.label_wrap.size[1] / 2 + offset) / \
                    self.axes.size[1]
                self.label_wrap.obj[ir, ic].set_position((0.5, y_text))
                self.label_wrap.obj_bg[ir, ic].set_y(y_rect)
                self.label_wrap.obj_bg[ir, ic].set_width(1 + hack / self.axes.size[0])
        # wrap title
        if self.title_wrap.on:
            offset = 0
            y_rect = self.title_wrap.position[3]
            y_text = y_rect + (self.title_wrap.size[1] / 2 + offset) / self.axes.size[1]
            width = self.ncol + ((self.ncol - 1) * self.ws_col + hack) / self.axes.size[0]
            self.title_wrap.obj.set_position((width / 2, y_text))
            self.title_wrap.obj_bg.set_y(y_rect)
            cbar = (self._labtick_z - self.ws_label_tick + 3) / self.axes.size[0]  # hacky!
            self.title_wrap.obj_bg.set_width(width + cbar * self.cbar.on)

        # Update title position
        if self.title.on:
            self._get_title_position()
            self.title.obj.set_position(self.title.position_xy)
            self.title.obj_bg.set_width(1)
            self.title.obj_bg.set_height((self.title.size[1] + 5) / self.axes.size[1])
            self.title.obj_bg.set_x(0)

        # Update the legend position
        if self.legend.on and self.legend.location in [0, 11]:
            self._get_legend_position()
            self.legend.obj.set_bbox_to_anchor((self.legend.position[1],
                                                self.legend.position[2]))

        # Update the box labels
        if self.box_group_label.on:
            lab = self.box_group_label
            labt = self.box_group_title
            hh = self._box_label_heights / self.axes.size[1]

            # Iterate through labels
            offset = 1  # to make labels line up better at default font sizes
            for ir, ic in np.ndindex(lab.obj.shape):
                data.df_rc = data._subset(ir, ic)
                data.get_box_index_changes()
                if lab.obj[ir, ic] is None:
                    continue
                for ii, jj in np.ndindex(lab.obj[ir, ic].shape):
                    if lab.obj[ir, ic][ii, jj] is None:
                        continue

                    # group label background rectangle
                    lab.obj_bg[ir, ic][ii, jj].set_height(hh[ii])
                    lab.obj_bg[ir, ic][ii, jj].set_y(-hh[0:ii + 1].sum())

                    # group label text strings
                    changes = data.changes[data.changes[len(data.changes.columns) - ii - 1] == 1].index.values
                    changes = np.diff(np.append(changes, len(data.changes)))
                    divider = self.axes.size[0] / len(data.changes)
                    xtext = (divider * (changes[jj] / 2 + changes[0:jj].sum())) / self.axes.size[0]
                    ytext = -hh[ii] / 2 - hh[0:ii].sum() - offset / self.axes.size[1]

                    # apply an offset to better align the text
                    if lab.obj[ir, ic][ii, jj].get_rotation() == 90:
                        xtext += offset / self.axes.size[0]
                    else:
                        ytext -= offset / self.axes.size[1]

                    # set the position
                    lab.obj[ir, ic][ii, jj].set_position((xtext, ytext))

                # group title
                for ii, jj in np.ndindex(labt.obj[ir, ic].shape):
                    # find and set the label position
                    if labt.obj[ir, ic][ii, 0] is None:
                        continue
                    wtitle = labt.size_all.loc[(labt.size_all.ir == ir)
                                               & (labt.size_all.ic == ic) & (labt.size_all.ii == ii), 'width']
                    xtitle = 1 + (self.ws_ax_box_title + wtitle / 2) / self.axes.size[0]
                    ytitle = -hh[ii] / 2 - hh[0:ii].sum() - (ii + 2) * offset / self.axes.size[1]
                    labt.obj[ir, ic][ii, 0].set_position((xtitle, ytitle))

        # Text label adjustments
        if self.text.on:
            self.set_text_position()

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.on:
            self._get_title_position()
            self.title.obj, self.title.obj_bg = self.add_label(0, 0, self.title.text, offset=True,
                                                               **self.make_kw_dict(self.title))

    def set_scientific(self, ax: mplp.Axes, tp: dict, idx: int = 0) -> mplp.Axes:
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
        if self.name in ['hist'] and self.hist.horizontal is True and \
                self.hist.kde is False:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        elif not tick_labels_major_x_sci \
                and self.name not in ['box', 'heatmap'] \
                and not logx:
            try:
                ax.get_xaxis().get_major_formatter().set_scientific(False)
            except:  # noqa
                pass

        elif not tick_labels_major_x_sci \
                and self.name not in ['box', 'heatmap']:
            try:
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                # need to recompute the ticks after changing formatter
                tp = mpl_get_ticks(ax, minor=self.ticks_minor.on)
                for itick, tick in enumerate(tp['x']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['x']['label_text'][itick] = \
                        '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_xticklabels(tp['x']['label_text'])
            except:  # noqa
                pass
        elif (bestx and not logx
                or not bestx and tick_labels_major_x_sci and logx) \
                and self.name not in ['box', 'heatmap']:
            xlim = ax.get_xlim()
            dec = get_sci(tp['x']['ticks'], xlim)
            ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter(dec))

        logy = getattr(self, f'axes{lab}').scale in LOGY + SYMLOGY + LOGITY
        if self.name in ['hist'] and self.hist.horizontal is False and \
                self.hist.kde is False:
            ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        elif not tick_labels_major_y_sci \
                and self.name not in ['heatmap'] \
                and not logy:
            try:
                ax.get_yaxis().get_major_formatter().set_scientific(False)
            except:  # noqa
                pass

        elif not tick_labels_major_y_sci \
                and self.name not in ['heatmap']:
            try:
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
                # need to recompute the ticks after changing formatter
                tp = mpl_get_ticks(ax, minor=self.ticks_minor.on)
                for itick, tick in enumerate(tp['y']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['y']['label_text'][itick] = \
                        '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_yticklabels(tp['y']['label_text'])
            except:  # noqa
                pass
        elif (besty and not logy
                or not besty and tick_labels_major_y_sci and logy) \
                and self.name not in ['heatmap']:
            ylim = ax.get_ylim()
            dec = get_sci(tp['y']['ticks'], ylim)
            ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter(dec))

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

    def set_text_position(self):
        """Move text labels to the correct location."""
        obj = self.text

        for ir, ic in np.ndindex(self.axes.obj.shape):
            ax = self.axes.obj[ir, ic]
            offsetx = ir * self.axes.size[0]
            offsety = ic * self.axes.size[1]

            for itext, txt in enumerate(obj.text.values):
                # Set the coordinate so text is anchored to figure, axes, or the current
                #    data range
                coord = None if not hasattr(obj, 'coordinate') \
                    else self.text.coordinate.lower()
                if coord == 'figure':
                    transform = self.fig.obj.transFigure
                elif coord == 'data':
                    transform = ax.transData
                else:
                    transform = ax.transAxes  # noqa
                units = 'pixel' if not hasattr(obj, 'units') else getattr(obj, 'units')

                # Get position
                if 'position' in self.kwargs.keys():
                    position = copy.copy(self.kwargs['position'])
                elif hasattr(obj, 'position') and \
                        isinstance(getattr(obj, 'position'), RepeatedList):
                    position = copy.copy(getattr(obj, 'position')[itext])
                elif hasattr(obj, 'position'):
                    position = copy.copy(getattr(obj, 'position'))

                # Convert position to correct units
                if units == 'pixel' and coord == 'figure':
                    position[0] /= self.fig.size[0]
                    offsetx /= self.fig.size[0]
                    position[1] /= self.fig.size[1]
                    offsety /= self.fig.size[1]
                elif units == 'pixel' and coord != 'data':
                    position[0] /= self.axes.size[0]
                    offsetx /= self.axes.size[0]
                    position[1] /= self.axes.size[1]
                    offsety /= self.axes.size[1]

                # Something goes weird with x = 0 so we need to adjust slightly
                if position[0] == 0:
                    position[0] = 0.01

                # position
                obj.obj[ir, ic][itext].set_x(position[0])
                obj.obj[ir, ic][itext].set_y(position[1])

    def show(self, *args):
        """Display the plot window."""
        mplp.show(block=False)

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
                                     hspace=1.0 * self.ws_row / self.axes.size[1],
                                     wspace=1.0 * self.ws_col / self.axes.size[0],
                                     )
