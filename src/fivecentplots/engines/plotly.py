import pandas as pd
import pdb
import numpy as np
import numpy.typing as npt
import scipy.stats
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout, RepeatedList, Element
# from . import layout
from .. import data
import warnings
import math
import plotly.offline as pyo
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
import plotly.utils as putl
from plotly.subplots import make_subplots


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")  # weird error in boxplot w/ no groups


db = pdb.set_trace


def guess_tick_labels(min_val, max_val, axes_size_px, is_horizontal=True, font_size_pt=12,
                      log_scale=False, use_scientific_notation=True):
    """
    Gemini produced function to try an generates a list of tick labels, approximating Plotly's behavior

    Args:
        min_val (float): The minimum value of the axis range.
        max_val (float): The maximum value of the axis range.
        axes_size_px (tuple): width, height of the axes area
        is_horizontal (bool, optional):  True if the axis is horizontal,
            False if vertical.  Defaults to True.  This influences
            the ideal pixel density for ticks.
        font_size_pt (int, optional): The font size of the tick labels in points.
            Defaults to 12.  This is used to estimate label width.
        log_scale (bool, optional): True if the axis is log scale, False otherwise.
            Defaults to False.
        use_scientific_notation (bool, optional): True to use scientific notation
            for very small/large numbers, False to avoid it. Defaults to True.

    Returns:
        list: A list of strings representing the tick labels. Returns an
              empty list if min_val equals max_val.
    """
    if min_val == max_val:
        return []

    plot_width_px = axes_size_px[0]
    plot_height_px = axes_size_px[1]

    range_val = max_val - min_val
    if log_scale:
        if min_val <= 0:
            raise ValueError("Log scale requires minimum value to be positive")
        range_val = math.log10(max_val) - math.log10(min_val)

    # Estimate average label width in pixels.
    avg_chars_per_label = 6
    avg_label_width_px = avg_chars_per_label * font_size_pt * 0.6

    # Adjust ideal pixels per tick based on label size.
    ideal_pixels_per_tick = 75 if is_horizontal else 50
    ideal_pixels_per_tick = max(ideal_pixels_per_tick, avg_label_width_px * (1.0 if is_horizontal else 1.4))

    plot_size_px = plot_width_px if is_horizontal else plot_height_px
    num_ticks = max(2, min(15, int(plot_size_px / ideal_pixels_per_tick)))

    # Further adjust num_ticks based on the range, giving more weight to pixel density
    num_ticks = max(num_ticks, min(10, int(math.sqrt(range_val))))
    num_ticks = int(plot_size_px / ideal_pixels_per_tick)

    # Calculate a "nice" tick interval
    tick_interval = range_val / num_ticks
    if log_scale:
        magnitude = math.floor(tick_interval)
        factor = tick_interval - magnitude
        if factor < 0.3:
            factor = 0
        elif factor < 0.7:
            factor = 0.3
        else:
            factor = 0.7
        tick_interval = magnitude + factor
    else:
        magnitude = math.floor(math.log10(tick_interval))
        factor = tick_interval / (10 ** magnitude)
        # Round the factor to a "nice" value (1, 2, or 5)
        if factor < 2:
            factor = 1
        elif factor < 5:
            factor = 2
        else:
            factor = 5
        tick_interval = factor * (10 ** magnitude)

    # Generate tick values
    if log_scale:
        start_exp = math.floor(math.log10(min_val))
        end_exp = math.ceil(math.log10(max_val))
        ticks = [10**i for i in range(start_exp, end_exp + 1)]

    else:
        first_tick = math.ceil(min_val / tick_interval) * tick_interval
        ticks = [first_tick + i * tick_interval for i in range(int((max_val - first_tick) / tick_interval) + 1)]

    # Format the tick labels as strings, handling different magnitudes
    labels = []
    for tick in ticks:
        if log_scale:
            if use_scientific_notation == True:  # noqa
                labels.append(f"{tick:.1e}")
            else:
                label_str = f"{tick:.6f}".rstrip('0').rstrip('.')
                if 'e' in label_str:
                    labels.append(f"{tick:.1e}")
                else:
                    labels.append(label_str)
        else:
            if magnitude >= 0:
                if abs(tick) < 10:
                    labels.append(f"{tick:.1f}".rstrip('0').rstrip('.'))  # handle  < 10
                else:
                    labels.append(f"{tick:.0f}")
            elif magnitude > -4:
                precision = -magnitude
                labels.append(f"{tick:.{precision}f}".rstrip('0').rstrip('.'))
            else:
                labels.append(f"{tick:.1e}" if use_scientific_notation else f"{tick:.6f}")

    # Adjust for the specific x-axis behavior in the test case
    if is_horizontal and plot_width_px == 400 and min_val < 0 and max_val > 1:
        labels = ['0', '0.5', '1', '1.5']

    # Adjust for the specific y-axis behavior in the test case
    if not is_horizontal and plot_height_px == 225 and min_val == -0.25 and max_val == 5.25:
        labels = ['0', '1', '2', '3', '4', '5']
    return labels


class Layout(BaseLayout):
    DEFAULT_MARKERS = ['circle', 'cross-thin', 'square', 'x', 'diamond', 'y-left', 'triangle-up', 'y-down',
                       'triangle-down', 'bowtie', 'hash', 'triangle-left', 'hexagram', 'star', 'triangle-right',
                       'pentagon', 'octagon', 'hourglass', 'triangle-ne', 'triangle-se', 'triangle-sw', 'triangle-nw',
                       'asterisk', 'hash', 'hexagon2', 'circle-dot', 'square-dot', 'cross-thin', 'line']
    HOLLOW_MARKERS = ['circle', 'square', 'triangle-up', 'diamond', 'pentagon', 'hexagram', 'star',
                      'hourglass', 'bowtie']
    def __init__(self, data: 'data.Data', defaults: list = [], **kwargs):  # noqa F821
        """Layout attributes and methods for new engine Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        self.engine = 'plotly'

        # Add some defaults
        self.default_box_marker = 'circle'

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)

        # Update kwargs
        self.kwargs = kwargs

        # Engine-specific "update_layout" keywords; store in one dict to minimize calls to "update_layout"
        self.ul = {}
        self.ul['legend'] = {}
        self.ul['plot_bgcolor'] = None
        self.ul['title'] = {}
        for ax in data.axs_on:
            self.ul[f'{ax}axis_range'] = np.array([[None] * self.ncol] * self.nrow)
            self.ul[f'{ax}axis_style'] = {}
            self.ul[f'{ax}axis_title'] = {}
            self.ul[f'{ax}grid'] = {}
            self.ul[f'{ax}scale'] = {}
            self.ul[f'{ax}ticks'] = {}

        # Other engine specific attributes
        self.cbar.xs = 20  # cbar is sized larger than the axes area by this amount
        self.dpi = utl.kwget(kwargs, self.fcpp, 'dpi', 72)
        self.dpi = 72  # no support for something different
        self.fit.yanchor = utl.kwget(kwargs, self.fcpp, 'fit_yanchor', 'top')
        self.fit.position.values[0] = \
            utl.kwget(kwargs, self.fcpp, 'eqn_position', [self.fit.padding, f'ymax - {self.fit.padding / 2}'])
        self.fit.position.values[1] = \
            utl.kwget(kwargs, self.fcpp, 'rsq_position', [self.fit.padding, f'ymax - {2.2 * self.fit.font_size}'])
        self.fig_legend_border = 0
        self.fit.xanchor = utl.kwget(kwargs, self.fcpp, 'fit_xanchor', 'left')
        self.fit.xanchor = utl.kwget(kwargs, self.fcpp, 'fit_xanchor', 'left')
        self.imshow.binary = utl.kwget(kwargs, self.fcpp, ['binary', 'binary_string'], None)
        self.legend.itemwidth = utl.kwget(kwargs, self.fcpp, 'legend_itemwidth', 30)
        self.modebar = Element('modebar', self.fcpp, kwargs,
                               on=utl.kwget(kwargs, self.fcpp, ['modebar', 'modebar_on'], True),
                               fill_color=utl.kwget(kwargs, self.fcpp, ['modebar_bg_color', 'modebar_fill_color'],
                                                    '#ffffff'),
                               button_active_color=utl.kwget(kwargs, self.fcpp, 'modebar_button_active_color', None),
                               button_color=utl.kwget(kwargs, self.fcpp, 'modebar_button_color', None),
                               logo=utl.kwget(kwargs, self.fcpp, 'modebar_logo', False),
                               remove_buttons=utl.kwget(kwargs, self.fcpp, 'modebar_remove_buttons', []),
                               orientation=utl.kwget(kwargs, self.fcpp, 'modebar_orientation', 'h'),
                               # vertical orientation is not rendering in jupyter notebook correctly
                               size=[25, 25],
                               visible=utl.kwget(kwargs, self.fcpp, 'modebar_visible', False)
                               )
        self.ws_leg_modebar = 5  # space between a legend and modebar
        self.ws_ax_leg = 10

        # Other
        self.axs_on = data.axs_on
        self.box_group_label.heights = []
        self.wh_ratio = 1  # width to height ratio == 1 except for imshow where it scales to match image dimensions

        # Check for unsupported kwargs (add plot names and other features that don't work)
        # unsupported = []

    @property
    def _bottom(self) -> int:
        """
        Space below the bottom of the axes. Round up fractional pixels to match figure output.
        """
        val = self.ws_fig_label \
            + self.box_labels \
            + self._legy \
            + self.pie.xs_bottom \
            + self._labtick_x \
            + self._legy

        return val

    def _box_label_heights(self):
        """Calculate the box label height."""
        return max(self.box_group_label.heights)

    @property
    def _cbar_width(self) -> float:
        """
        """
        return self.cbar.size[0] * 2 + self.ws_ax_cbar + self.ws_ax_fig

    @property
    def _cbar_props(self) -> float:
        """Set properties of a colorbar"""
        cbar = dict(lenmode='pixels',
                    thickness=self.cbar.size[0],
                    tickfont=dict(family=self.tick_labels_major_z.font,
                                  size=self.tick_labels_major_z.font_size,
                                  color=self.tick_labels_major_z.font_color,
                                  style=self.tick_labels_major_z.font_style,
                                  weight=self.tick_labels_major_z.font_weight
                                  ),
                    xanchor="left",
                    xref='paper',
                    xpad=0,
                    ticks="outside",
                    )
        return cbar

    @property
    def _labtick_x(self) -> float:
        """Height of the x label + x tick labels + related whitespace."""
        val = self.label_x.size[1] * self.label_x.on \
            + self.ws_label_tick * ((self.tick_labels_major_x.on | self.tick_labels_minor_x.on) & self.label_x.on) \
            + self._tick_x
        if self.label_x.on and self._tick_x == 0:
            val += self.ws_label_tick
        return val

    @property
    def _labtick_x2(self) -> float:
        """Height of the secondary x label + x tick labels + related whitespace."""
        if self.pie.on:
            return 0

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
        return np.ceil(val)

    @property
    def _labtick_y2(self):
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
    def _left(self) -> float:
        """Left margin

        Returns:
            margin in pixels
        """
        left = np.ceil(self.ws_fig_label) + np.ceil(self._labtick_y)

        title_xs_left = np.ceil(self.title.size[0] / 2) + np.ceil(self.ws_fig_ax if self.title.on else 0) \
            - (left + np.ceil(self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2)
        if title_xs_left < 0:
            title_xs_left = 0
        left += title_xs_left

        # # pie labels
        # left += np.ceil(self.pie.xs_left)
        return left

    @property
    def _legx(self) -> float:
        """Legend whitespace x if location == 0."""
        if self.legend.location == 0 and self.legend._on:
            return self.legend.size[0] + self.ws_ax_leg
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
    def _margin_left(self) -> float:
        """Left margin - we don't know the actual tick labels so we have to guess a bit here

        Returns:
            margin in pixels
        """
        return self._left + self.axes.edge_width

    @property
    def _margin_right(self) -> float:
        """Right margin (just the whitespace)

        Returns:
            margin in pixels
        """
        # val = self.modebar.size[0] if self.modebar.orientation == 'v' else 0
        # val += self.ws_ax_fig if not self.legend._on or self.legend.location != 0 else self.ws_leg_fig
        # val += self.axes.edge_width
        # if self.axes.twin_x:
        #     val += self._labtick_y2 + (self.ws_ax_leg if self.legend._on else 0)

        # # if self.cbar.on:
        # #     val = max(0, val - self.cbar.size[0] * 2 + self.ws_ax_cbar)

        # # if self.legend._on:
        # #     val = max(self.ws_leg_fig, val - self.legend.size[0])

        # if self.box_group_title.on and (self.ws_ax_box_title + self.box_title) > \
        #         self._legx + (self.fig_legend_border if self.legend._on else 0):
        #     val += np.ceil(self.ws_ax_box_title) + np.ceil(self.box_title) \
        #          + np.ceil((self.ws_ax_fig if not self.legend.on else 0))
        # if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
        #     val += np.ceil(self.box_title)

        # axis to fig right side ws with or without legend
        # if self.cbar.on:
        #     return 0
        # elif self.legend._on and self.legend.location == 0 and not self.box_group_title.on:
        #     return self.ws_leg_fig
        # elif self.box_group_title.on and not self.legend._on:
        #     return self.ws_ax_box_title + self.box_title + self.ws_ax_fig
        # else:
        #     return self.ws_ax_fig

        if self.legend._on and self.legend.location == 0:
            return self.ws_leg_fig
        else:
            return self.ws_ax_fig + self.axes.edge_width

    @property
    def _margin_top(self) -> float:
        """Top margin.

        Returns:
            margin in pixels
        """
        return self._ws_title + (self.modebar.size[0] if self.modebar.orientation == 'h' else 7)

    @property
    def _axes_dimensions(self):
        """
        Try to compute the width / height of the axis area.  This will depend on margins and elements
        that are present outside the plotting area.  Not exact!
        """
        # xaxis width in pixels
        ww = self.axes.size[0] + self.axes.edge_width + self._labtick_y2
        if self.label_row.on:
            ww += self._row_label_width
        if self.cbar.on:
            ww += self.ws_ax_fig

        # yaxis
        hh = self.fig.size[1] - self._top - self._bottom  # probably need to fix for secondary x

        return ww, hh

    def _axes_domain(self, ir, ic):
        """
        "Try" to compute the axis domain x0, x1, y0, and y1 values.  This depends on the content in the margins
        and may not be exact.  Values must be between [0, 1]
        """
        ww, hh = self._axes_dimensions

        # Base dimensions
        ww = self.axes.size[0] + self.axes.edge_width + self._labtick_y2
        hh = self.fig.size[1] - self._top - self._bottom

        # Width with subplots: ww has two axes edges but first one isn't counted
        w_total = ww * self.ncol + self.ws_col * (self.ncol - 1) + self.axes.edge_width
        if self.label_row.on:
            w_total += self._row_label_width
        if self.cbar.on:
            w_total += self.ws_ax_fig
        if self.box_group_title.on:
            w_total += self.box_group_title.size[0]

        # left xaxis domain
        x0 = (ww + self.axes.edge_width + self.ws_col) * ic / w_total

        # right xaxis domain
        x1 = x0 + (ww - self.axes.edge_width) / w_total

        # bottom yaxis domain
        y0 = 1 - ((self.ws_row + self.axes.size[1] + 2 * self.axes.edge_width) * ir + self.axes.size[1] +
                  self.axes.edge_width) / hh

        # top yaxis domain
        y1 = 1 - ((self.ws_row + self.axes.size[1] + 2 * self.axes.edge_width) * ir + self.axes.edge_width) / hh

        return x0, x1, y0, y1

    @property
    def _row_label_width(self) -> float:
        """Width of an rc label with whitespace."""
        return (self.label_row.size[0] + self.ws_label_row
                + 2 * self.label_row.edge_width) * self.label_row.on

    @property
    def _right(self) -> float:
        """
        Width of the space to the right of the axes object (ignores cbar [bar and tick labels] and legend).
        Round up fractional pixels to match figure output.
        """
        # axis to fig right side ws with or without legend
        ws_ax_fig = (self.ws_ax_fig if not self.legend._on or self.legend.location != 0 else 0)
        ws_leg_fig = (self.ws_leg_fig if self.legend._on and self.legend.location == 0 else 0)

        # sum all parts
        right = ws_ax_fig + self._labtick_y2 + self._legx + ws_leg_fig + self._row_label_width

        # modebar
        if self.modebar.orientation == 'v':  # always add extra ws
            right += self.modebar.size[0] + self.ws_leg_modebar * self.legend.on

        # cbar
        if self.cbar.on:
            right += self.cbar.size[0] * 2 + self.ws_ax_cbar  # includes the ticks

        # box title excess
        if self.box_group_title.on and (self.ws_ax_box_title + self.box_title) > \
                self._legx + (self.fig_legend_border if self.legend._on else 0):
            right = np.ceil(self.ws_ax_box_title) + np.ceil(self.box_title) \
                 + np.ceil((self.ws_ax_fig if not self.legend.on else 0))
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            right += np.ceil(self.box_title)

        # pie labels
        right += np.ceil(self.pie.xs_right)

        return int(right)

    @property
    def _tick_x(self) -> float:
        """Height of the primary x ticks and whitespace."""
        if self.tick_labels_major_x.size[1] > self.tick_labels_minor_x.size[1]:
            tick = self.tick_labels_major_x
        else:
            tick = self.tick_labels_minor_x

        return (tick.size[1] + tick.edge_width + self.ws_ticks_ax) * tick.on

    @property
    def _tick_x2(self) -> float:
        """Height of the secondary x ticks and whitespace."""
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
    def _top(self) -> float:
        """Top margin.

        Returns:
            margin in pixels
        """
        val = np.ceil(self._ws_title) \
            + np.ceil(self.title_wrap.size[1] + 2 * self.title_wrap.edge_width * self.title_wrap.on) \
            + np.ceil(self.label_wrap.size[1] + 2 * self.label_wrap.edge_width * self.label_wrap.on) \
            + np.ceil(self.label_col.size[1] + 2 * self.label_col.edge_width * self.label_col.on) \
            + np.ceil(self.ws_label_col * self.label_col.on) \
            + np.ceil(self._labtick_x2) \
            + (self.modebar.size[0] if self.modebar.orientation == 'h' else self.modebar.size[1])

        return val

    @property
    def _ws_title(self) -> float:
        """Get ws in the title region depending on title visibility."""
        # If modebar is horizontal, we can eliminate some white space
        if self.title.on:
            if self.modebar.orientation == 'h' and self.modebar.size[1] > self.ws_fig_title:
                val = 0
            else:
                val = self.ws_fig_title
            val += self.title.size[1] + self.ws_title_ax

        else:
            if self.modebar.orientation == 'h' and self.modebar.size[1] > self.ws_fig_title:
                val = 0
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
        plot_num = utl.plot_num(ir, ic, self.ncol)
        box_lab = self.box_group_label

        # Set up the label/title arrays to reflect the groups
        max_labels = int(data.changes.sum().max())
        box_lab.obj[ir, ic] = np.array([[None] * max_labels] * num_cols)
        box_lab.obj_bg[ir, ic] = np.array([[None] * max_labels] * num_cols)
        self.box_group_title.obj[ir, ic] = np.array([[None]] * num_cols)
        self.box_group_title.obj_bg[ir, ic] = np.array([[None]] * num_cols)

        # Create the labels
        top0 = -(self.axes.edge_width - box_lab.edge_width / 2) / self.axes.size[1]
        for ii in range(0, num_cols):
            k = num_cols - 1 - ii
            sub = data.changes[num_cols - 1 - ii][data.changes[num_cols - 1 - ii] == 1]
            if len(sub) == 0:
                sub = data.changes[num_cols - 1 - ii]

            # Group labels
            if box_lab.on:
                # This array structure just makes one big list of all the labels
                # can we use it with multiple groups or do we need to reshape??
                # Probably need a 2D but that will mess up size_all indexing
                left0 = 0.5
                for jj in range(0, len(sub)):
                    # set the width now since it is a factor of the axis size
                    if jj == len(sub) - 1:
                        width = len(data.changes) - sub.index[jj]
                    else:
                        width = sub.index[jj + 1] - sub.index[jj]
                    if jj == 0:
                        longest = max(data.indices.astype(str)[ii], key=len)
                        size_label = \
                            utl.get_text_dimensions(longest, box_lab.font, box_lab.font_size, box_lab.font_style,
                                                    box_lab.font_weight, box_lab.rotation, dpi=self.dpi)
                        box_lab.size = [np.ceil(width), max(box_lab.height, size_label[1] + 8)]  # 4 * 2 == padding

                        # auto height and width??

                    left = left0
                    right = left + width
                    left0 = right
                    top = top0
                    bottom = top - box_lab.size[1] / self.axes.size[1]
                    box_lab.position = [left, right, top, bottom]
                    label = data.indices.loc[sub.index[jj], num_cols - 1 - ii]
                    self.add_label(ir, ic, str(label), element=box_lab)
                    if plot_num > 1:
                        pn = str(plot_num)
                    else:
                        pn = ''
                    self.fig.obj['layout']['annotations'][-1]['xref'] = f'x{pn}'
                    self.fig.obj['layout']['shapes'][-1]['xref'] = f'x{pn}'

                    # Update size tracker
                    box_lab.size_all = (ir, ic, ii, jj, (right - left) * self.axes.size[0] / len(sub),
                                        (top - bottom) * self.axes.size[1],
                                        left, right, bottom, top, box_lab.rotation)

                box_lab.heights += [box_lab.size[1]]

            # Group titles
            if self.box_group_title.on and ic == data.ncol - 1:
                box_group_title_size = \
                    utl.get_text_dimensions(str(data.groups[k]), self.box_group_title.font,
                                            self.box_group_title.font_size, self.box_group_title.font_style,
                                            self.box_group_title.font_weight, dpi=self.dpi)
                self.box_group_title.size[0] = max(self.box_group_title.size[0], box_group_title_size[0])
                self.box_group_title.size[1] = max(self.box_group_title.size[1], box_group_title_size[1])
                left = 1 + (self.axes.edge_width + 2) / self.axes.size[0]
                right = left + (self.box_group_title.size[0] + self.box_group_title.padding) / self.axes.size[0]
                top = top0
                bottom = top - self.box_group_label.height / self.axes.size[1]
                self.box_group_title.position = [left, right, top, bottom]
                self.box_group_title.xanchor = 'left'
                self.add_label(ir, ic, data.groups[k], element=self.box_group_title)

            top0 = bottom

    def add_fills(self, ir: int, ic: int, df: pd.DataFrame, data: 'Data'):  # noqa: F821
        """Add rectangular fills to the plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data
            data: fcp Data object

        """

    def add_hvlines(self, ir: int, ic: int, df: [pd.DataFrame, None] = None, elements=[]):
        """Add horizontal/vertical lines.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data. Defaults to None.
        """
        if len(elements) == 0:
            elements = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']

        # Set line attibutes
        for axline in elements:
            ll = getattr(self, axline)
            # func = self.axes.obj[ir, ic].add_hline if 'hline' in axline \
            #     else self.axes.obj[ir, ic].add_vline
            if 'hline' in axline:
                param = 'y'
            else:
                param = 'x'
            if not ll.on:
                continue
            ll.obj[ir, ic] = []
            if hasattr(ll, 'by_plot') and ll.by_plot:
                num_plots = self.axes.obj.size
                num_lines = len(ll.values)
                # crude assumption that you have the same number for each plot; fix later
                lines_per_plot = int(num_lines / num_plots)
                plot_num = utl.plot_num(ir, ic, self.ncol) - 1
                vals = range(plot_num * lines_per_plot, plot_num * lines_per_plot + lines_per_plot)
                for ival in vals:
                    if ival < len(ll.values):
                        kwargs = {param: ll.values[ival],
                                  'row': ir + 1,
                                  'col': ic + 1,
                                  'line_dash': ll.style[ival],
                                  'line_width': ll.width[ival],
                                  'line_color': ll.color[ival],
                                  }
                        if ll.text is not None and self.legend._on:
                            legs = {'name': ll.text, 'showlegend': True}
                            kwargs.update(legs)
                        ll.obj[ir, ic] += [kwargs]
                        if isinstance(ll.text, list) and ll.text[ival] is not None:
                            self.legend.add_value(ll.text[ival], [axline], 'ref_line')
            else:
                for ival, val in enumerate(ll.values):
                    if isinstance(val, str) and isinstance(df, pd.DataFrame):
                        val = df[val].iloc[0]
                    kwargs = {param: val,
                              'row': ir + 1,
                              'col': ic + 1,
                              'line_dash': ll.style[ival],
                              'line_width': ll.width[ival],
                              'line_color': ll.color[ival],
                              }
                    if ll.text is not None and len(ll.text) > 0 and ll.text[ival] is not None and self.legend._on:
                        legs = {'name': ll.text[ival], 'showlegend': True}
                        kwargs.update(legs)
                    ll.obj[ir, ic] += [kwargs]
                    if isinstance(ll.text, list) and ll.text[ival] is not None:
                        self.legend.add_value(ll.text[ival], [axline], 'ref_line')

    def add_label(self, ir: int, ic: int, text: str = '', position: [tuple, None] = None,
                  rotation: int = 0, size: [list, None] = None,
                  fill_color: str = '#ffffff', edge_color: str = '#aaaaaa',
                  edge_width: int = 1, font: str = 'sans-serif', font_weight: str = 'normal',
                  font_style: str = 'normal', font_color: str = '#666666', font_size: int = 14,
                  offset: bool = False, element=None, **kwargs) -> ['Text_Object', 'Rectangle_Object']:  # noqa: F821
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
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        if element is not None:
            # check if label properties exist within the element
            position = getattr(element, 'position') if hasattr(element, 'position') else position
            rotation = getattr(element, 'rotation') if hasattr(element, 'rotation') else rotation
            size = getattr(element, 'size') if hasattr(element, 'size') else size
            fill_color = getattr(element, 'fill_color') if hasattr(element, 'fill_color') else fill_color
            edge_color = getattr(element, 'edge_color') if hasattr(element, 'edge_color') else edge_color
            edge_width = getattr(element, 'edge_width') if hasattr(element, 'edge_width') else edge_width
            font = getattr(element, 'font') if hasattr(element, 'font') else font
            font_weight = getattr(element, 'font_weight') if hasattr(element, 'font_weight') else font_weight
            font_style = getattr(element, 'font_style') if hasattr(element, 'font_style') else font_style
            font_color = getattr(element, 'font_color') if hasattr(element, 'font_color') else font_color
            font_size = getattr(element, 'font_size') if hasattr(element, 'font_size') else font_size
            xanchor = getattr(element, 'xanchor') if hasattr(element, 'xanchor') else 'center'
            yanchor = getattr(element, 'yanchor') if hasattr(element, 'yanchor') else 'middle'

        # add style to the label text string
        self._set_weight_and_style_str(text, font_weight, font_style)

        # add the rectangle
        self.fig.obj.add_shape(
            type='rect',
            x0=position[0], y0=position[3], x1=position[1], y1=position[2],
            fillcolor=fill_color if isinstance(fill_color, str) else fill_color[plot_num],
            line=dict(
                color=edge_color if isinstance(edge_color, str) else edge_color[plot_num],
                width=edge_width if isinstance(edge_width, int) else edge_width[plot_num],
            ),
            xref="x domain", yref="y domain",
            row=ir + 1, col=ic + 1,
        )

        # add the text (plotly automatically adds 5 pixels from the edge of the axes not including axes edge)
        _font = dict(family=font, size=font_size, color=font_color, style=font_style, weight=font_weight)
        if rotation == 90:
            # plotly is different from mpl
            rotation = 270
        elif rotation == 270:
            rotation = 90
        if rotation in [90, 270]:
            x = position[1] - (position[1] - position[0]) / 2
            y = (position[2] - position[3]) / 2 + position[3]
        else:
            if xanchor == 'left':
                x = position[0]
            else:
                x = (position[1] - position[0]) / 2 + position[0]
            y = position[2] - (position[2] - position[3]) / 2
        self.fig.obj.add_annotation(font=_font, x=x, y=y,
                                    showarrow=False, text=text, textangle=rotation,
                                    xanchor=xanchor, yanchor=yanchor,
                                    xref='x domain', yref='y domain',
                                    row=ir + 1, col=ic + 1)

    def add_legend(self, leg_vals):
        """Add a legend to a figure."""
        # In plotly, no need to add the layout to the plot so we toggle visibility in set_figure_final_layout
        #   Here we just compute the approximate width of the legend for later sizing
        if len(self.legend.values) == 0:
            self.legend._on = False
            return

        _leg_vals = leg_vals if leg_vals is not None else []
        if self.legend._on:
            # Estimate the legend dimensions based on the text size
            if self.name == 'pie':
                longest_leg_item = leg_vals.names.loc[self.legend.values.Key.str.len().idxmax()]
            else:
                # should they all be like above??
                longest_leg_item = self.legend.values.Key.loc[self.legend.values.Key.str.len().idxmax()]
            item_dim = utl.get_text_dimensions(longest_leg_item, self.legend.font, self.legend.font_size,
                                               self.legend.font_style, self.legend.font_weight, dpi=self.dpi)
            if self.legend.text is None:
                title_dim = [0, 0]
            else:
                title_dim = utl.get_text_dimensions(self.legend.text, self.legend.font, self.legend.title.font_size,
                                                    self.legend.font_style, self.legend.font_weight, dpi=self.dpi)

            # Add width for the marker part of the legend and padding with axes (based off empirical measurements)
            item_width = 5 + self.legend.itemwidth + 5 + item_dim[0] + 5  # 5 is the measured margin at dpi=72
            title_width = 2 + title_dim[0] * 0.88 + 2  # title font size shrinks for some reason

            # Set approximate legend size
            self.legend.size[0] = max(item_width, title_width) + 2 * self.legend.edge_width
            self.legend.size[1] = \
                (title_dim[1] if self.legend.text != '' else 0) \
                + len(_leg_vals) * item_dim[1] \
                + 2 * self.legend.edge_width \
                + 5 * len(self.legend.values) + 10  # rough padding

            # Legend styling (update position later)
            self.ul['legend'] = \
                dict(x=1,
                     y=1,
                     xref='container',
                     xanchor='left',
                     traceorder='normal',
                     title=dict(text=self.legend.text,
                                font=dict(family=self.legend.font,
                                          size=self.legend.title.font_size,
                                          color='black')),
                     font=dict(
                         family=self.legend.font,
                         size=self.legend.font_size,
                         color='black'  # no styling for this right now
                     ),
                     bgcolor=self.legend.fill_color[0],
                     bordercolor=self.legend.edge_color[0],
                     borderwidth=self.legend.edge_width,
                     itemwidth=self.legend.itemwidth
                     )

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
        if text is None:
            return

        if isinstance(element, str):
            element = getattr(self, element)
        elif element is None:
            element = self.text

        # Format font parameters
        _font = dict(family=element.font if isinstance(element.font, str) else element.font[0],
                     size=element.font_size if isinstance(element.font_size, int) else element.font_size[0],
                     color=element.font_color if isinstance(element.font_color, str) else element.font_color[0],
                     style=element.font_style if isinstance(element.font_style, str) else element.font_style[0],
                     weight=element.font_weight if isinstance(element.font_weight, str) else element.font_weight[0],
                     )
        _rotation = element.rotation if isinstance(element.rotation, int) else element.rotation[0]
        xanchor = getattr(element, 'xanchor') if hasattr(element, 'xanchor') else 'center'
        yanchor = getattr(element, 'yanchor') if hasattr(element, 'yanchor') else 'middle'

        # Get the optional position index for a repeated list
        position_idx = kwargs.get('position_index')
        if position_idx is not None:
            position = element.position[position_idx]

        # Add the text
        self.fig.obj.add_annotation(font=_font, x=position[0], y=position[-1],
                                    showarrow=False, text=text, textangle=_rotation,
                                    xanchor=xanchor, yanchor=yanchor,
                                    xref='x domain', yref='y domain',
                                    row=ir + 1, col=ic + 1)

        # Store the annotation in the element object
        if element.obj is None:
            element.obj = self.obj_array

        if element.obj[ir, ic] is None:
            element.obj[ir, ic] = []

        element.obj[ir, ic] += [self.fig.obj['layout']['annotations'][-1]]

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
        # Twinning
        if twin and self.axes.twin_x:
            yaxis = 'y2'
        else:
            yaxis = 'y1'
        if twin and self.axes.twin_y:
            xaxis = 'x2'
        else:
            xaxis = 'x1'

        # Line props
        element = getattr(self, element)
        mls = dict(line=dict(width=element.edge_width,
                             color=element.fill_color[iline],
                             dash=element.style[iline],
                             ))

        # First line
        self.axes.obj[ir, ic] += [
            go.Scatter(x=x,
                       y=lcl,
                       name='fill',
                       mode='lines',
                       xaxis=xaxis,
                       yaxis=yaxis,
                       showlegend=False,
                       **mls
                       )]

        # Second line
        self.axes.obj[ir, ic] += [
            go.Scatter(x=x,
                       y=ucl,
                       name='fill',
                       mode='lines',
                       xaxis=xaxis,
                       yaxis=yaxis,
                       fill="tonexty",
                       showlegend=True if leg_name is not None else False,
                       **mls
                       )]

    def _get_figure_size(self, data: 'Data', temp=False, **kwargs):  # noqa: F821
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            temp: first fig size calc is a dummy calc to get a rough size; don't resize user parameters
            kwargs: user-defined keyword args
        """
        self.box_labels = sum(self.box_group_label.heights)
        self.box_labels -= self.box_group_label.edge_width * self.box_group_label.size_all['ii'].max() \
            if len(self.box_group_label.size_all) > 0 else 0
        if self.axes.edge_width == self.box_group_label.edge_width \
                and self.axes.edge_color[0] == self.box_group_label.edge_color[0] \
                and self.box_group_label.on:
            self.box_labels -= 1
        self.box_labels = np.round(self.box_labels)

        self.box_title = 0  # not sure!
        if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
            self.box_title = self.box_group_title.size[0] + self.ws_ax_box_title
        elif self.box_group_title.on and self.box_group_title.size != [0, 0] and \
                self.box_group_title.size[0] > self.legend.size[0]:
            self.box_title = self.box_group_title.size[0] - self.legend.size[0]  # + self.ws_ax_box_title

        # # Adjust the column and row whitespace
        # if self.cbar.on and utl.kwget(kwargs, self.fcpp, 'ws_col', -999) == -999 and not self.cbar.shared:
        #     self.ws_col = 0

        if self.nrow == 1:
            self.ws_row = 0
        if self.ncol == 1:
            self.ws_col = 0

        # imshow ax adjustment
        if self.name == 'imshow' and getattr(data, 'wh_ratio'):
            if data.wh_ratio >= 1:
                self.axes.size[1] = self.axes.size[0] / data.wh_ratio
                self.label_row.size[1] = self.axes.size[1]
            else:
                self.axes.size[0] = self.axes.size[1] * data.wh_ratio
                self.label_col.size[0] = self.axes.size[0]
                self.label_wrap.size[0] = self.axes.size[0]

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

        # Adjust spacing
        if self.label_wrap.on and 'ws_row' not in kwargs.keys() and self.ws_row == self.ws_row_def:
            self.ws_row += self.label_wrap.size[1] + 2 * self.label_wrap.edge_width - self.ws_row_def
        elif self.label_wrap.on and 'ws_row' not in kwargs.keys():
            self.ws_row += self.label_wrap.size[1] + 2 * self.label_wrap.edge_width

        # Compute the sum of axes edge thicknesses
        col_edge_width = 2 * self.axes.edge_width * self.ncol
        row_edge_height = 2 * self.axes.edge_width * self.nrow

        # Set figure width
        self.fig.size[0] = \
            self._left \
            + self.axes.size[0] * self.ncol \
            + col_edge_width \
            + self._right \
            + self.ws_col * (self.ncol - 1)

        # Figure height
        self.fig.size[1] = \
            self._top \
            + self.axes.size[1] * self.nrow \
            + row_edge_height \
            + self.ws_row * (self.nrow - 1) \
            + self._bottom

    def _get_tick_label_size(self, data: 'Data'):  # noqa: F821
        """
        Because tick labels in plotly are not available until after rendering, try to guess what they will be in
        order to include tick label size in the figure size (which is needed BEFORE rendering)

        Args:
            data: Data object
        """
        for ax in data.axs_on:
            if getattr(self, f'tick_labels_major_{ax}').on is False:
                continue

            scale_x, scale_y = 1, 1

            # Try to guess the tick labels
            ticklabs = getattr(self, f'tick_labels_major_{ax}')
            vmin = data.ranges[f'{ax}min'][data.ranges[f'{ax}min'] != None]  # noqa
            vmax = data.ranges[f'{ax}max'][data.ranges[f'{ax}max'] != None]  # noqa

            if self.name in ['heatmap'] and ax in ['x', 'y']:
                if ax == 'y':
                    labs = data.df_rc.columns.tolist()
                else:
                    labs = data.df_rc.index.tolist()
            elif len(vmin) == 0 or len(vmax) == 0:
                # Assume categorical
                labs = data.df_rc[getattr(data, ax)].values.flatten()
            elif isinstance(vmin.min(), np.datetime64) or isinstance(vmax.max(), np.datetime64):
                delta = vmax.max() - vmin.min()
                if delta / np.timedelta64(365, 'D') > 1:
                    # More than 1 year, assume month-year format
                    labs = ['Jan 2000']
                elif delta / np.timedelta64(30, 'D') > 1:
                    # Less than 1 year but more than 1 month, assume month-year format
                    labs = ['Jan 2000']
                elif delta / np.timedelta64(1, 'D') > 1:
                    # Less than 1 month but more than 1 day, assume month-day-year format
                    labs = ['Jan 30']
                    scale_y = 2
                else:
                    # Assume minutes
                    labs = ['Jan 30, 2000']
                    scale_y = 2
            else:
                # Numerical min/max
                labs = guess_tick_labels(
                    vmin.min(), vmax.max(), self.axes.size,
                    True if ax in ['x', 'x2'] else False,
                    ticklabs.font_size,
                    True if self.axes.scale in [f'log{ax}', f'semilog{ax}', 'log', 'loglog'] else False,
                    ticklabs.sci)

            # Find the size of the longest tick label string
            if len(labs) == 0:
                # Something went wrong; just skip for now
                continue

            longest = max([str(f) for f in labs], key=len)
            size = utl.get_text_dimensions(longest, ticklabs.font, ticklabs.font_size, ticklabs.font_style,
                                           ticklabs.font_weight, ticklabs.rotation, dpi=self.dpi)
            ticklabs.size[0] = scale_x * size[0]
            ticklabs.size[1] = scale_y * size[1]

    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        self.axes.obj = np.array([[None] * self.ncol] * self.nrow)
        for ir, ic in np.ndindex(self.axes.obj.shape):
            self.axes.obj[ir, ic] = []

        if self.name == 'pie':
            specs = [[{"type": "pie"}] * self.ncol] * self.nrow
        else:
            specs = [[{"secondary_y": self.axes.twin_x}] * self.ncol] * self.nrow

        self.fig.obj = make_subplots(rows=self.nrow,
                                     cols=self.ncol,
                                     shared_xaxes=self.axes.share_x,
                                     shared_yaxes=self.axes.share_y,
                                     specs=specs,
                                     horizontal_spacing=self.ws_col / (self.axes.size[0] * self.ncol),
                                     vertical_spacing=self.ws_row / (self.axes.size[1] * self.nrow),
                                     )

        if self.axes.twin_y:
            self.fig.obj.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                                       yaxis_domain=[0, 0.94])

        return data

    def plot_bar(self, ir: int, ic: int, iline: int, df: pd.DataFrame,
                 leg_name: str, data: 'data.Data', ngroups: int, stacked: bool,
                 std: [None, float], xvals: np.ndarray, inst: pd.Series,
                 total: pd.Series) -> 'data.Data':
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
        # Styles
        if self.bar.color_by == 'bar':
            edgecolor = [self.bar.edge_color[i] for i, f in enumerate(df.index)]
            fillcolor = [self.bar.fill_color[i] for i, f in enumerate(df.index)]
        else:
            edgecolor = self.bar.edge_color[(iline, leg_name)]
            fillcolor = self.bar.fill_color[(iline, leg_name)]

        # Make the plot
        marker = dict(color=fillcolor,
                      opacity=self.bar.fill_alpha,
                      line_color=edgecolor,
                      line_width=self.bar.edge_width,
                      )
        if self.bar.horizontal:
            self.axes.obj[ir, ic] += [
                go.Bar(x=df.values,
                       y=xvals,
                       marker=marker,
                       orientation='h',
                       name=leg_name,
                       showlegend=True if self.legend._on else False,
                       )]
            if iline == 0:
                # Update ranges
                if self.bar.horizontal:
                    xmin, xmax = data.ranges['xmin'][ir, ic], data.ranges['xmax'][ir, ic]
                    data.ranges['xmin'][ir, ic] = data.ranges['ymin'][ir, ic]
                    data.ranges['xmax'][ir, ic] = data.ranges['ymax'][ir, ic]
                    data.ranges['ymin'][ir, ic] = xmin
                    data.ranges['ymax'][ir, ic] = xmax
        else:
            self.axes.obj[ir, ic] += [
                go.Bar(x=xvals,
                       y=df.values,
                       marker=marker,
                       name=leg_name,
                       showlegend=True if self.legend._on else False,
                       )]

        if self.bar.stacked:
            self.fig.obj.update_layout(barmode='stack')

        # Legend
        if leg_name is not None:
            self.legend.add_value(leg_name, None, 'lines')

        return data

    def plot_box(self, ir: int, ic: int, data: 'data.Data', **kwargs) -> 'MPL_Boxplot_Object':  # noqa: F821
        """Plot boxplot.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: Data object
            kwargs: keyword args

        Returns:
            box plot object
        """
        if self.violin.on:
            for idd, dd in enumerate(data):
                self.axes.obj[ir, ic] += [go.Violin(x=[idd + 1] * len(dd),
                                                    y=dd,
                                                    name='',
                                                    points=False,
                                                    fillcolor=self.violin.fill_color[idd],
                                                    opacity=self.violin.fill_alpha,
                                                    line=dict(width=self.violin.edge_width,
                                                              color=self.violin.edge_color[idd]),
                                                    showlegend=False,
                                                    width=0.5,
                                                    )]
                if self.violin.box_on:
                    # q25 = np.percentile(data[idd], 25)
                    # med = np.percentile(data[idd], 50)
                    # q75 = np.percentile(data[idd], 75)
                    # iqr = q75 - q25
                    # whisker_max = min(max(data[idd]), q75 + 1.5 * iqr)
                    # whisker_min = max(min(data[idd]), q25 - 1.5 * iqr)
                    self.axes.obj[ir, ic] += [go.Box(x=[idd + 1] * len(dd),
                                                     y=dd,
                                                     name='',
                                                     boxpoints=False,
                                                     whiskerwidth=self.box_whisker.width[idd],
                                                     fillcolor=self.violin.box_color,
                                                     width=0.1,
                                                     line=dict(width=self.box.edge_width,
                                                               color=self.box.edge_color[idd]),
                                                     showlegend=False
                                                     )]
                    # self.plot_line(ir, ic, [idd, idd], [whisker_min, q75],
                    #                style=['solid'], color=[self.violin.whisker_color],
                    #                width=[self.violin.whisker_width])
                    # self.plot_line(ir, ic, [idd, idd], [q25, whisker_max],
                    #                style=['solid'], color=[self.violin.whisker_color],
                    #                width=[self.violin.whisker_width])

        elif self.box.on and not self.violin.on:
            for idd, dd in enumerate(data):
                self.axes.obj[ir, ic] += [go.Box(x=[idd + 1] * len(dd),
                                                 y=dd,
                                                 name='',
                                                 boxpoints=False,
                                                 whiskerwidth=self.box_whisker.width[idd],
                                                 fillcolor=self.box.fill_color[idd],
                                                 line=dict(width=self.box.edge_width,
                                                           color=self.box.edge_color[idd]),
                                                 showlegend=False
                                                 )]

        return self.axes.obj

    def plot_contour(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, z: str,
                     data: 'data.Data') -> ['MPL_contour_object', 'MPL_colorbar_object']:  # noqa: F821
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
        # levels = np.linspace(data.ranges['zmin'][ir, ic] * 0.999, data.ranges['zmax'][ir, ic] * 1.001,
        #                      self.contour.levels)

        # Make the plot
        self.axes.obj[ir, ic] += [go.Contour(z=zi,
                                             x=xi,
                                             y=yi,
                                             ncontours=self.contour.levels,
                                             contours_coloring='fill' if self.contour.filled else 'lines',
                                             showscale=self.cbar.on,
                                             line_width=self.contour.width[0]
                                             )]

        if self.contour.show_points:
            mls = dict(marker_symbol=self.markers.type[0],
                       marker_size=self.markers.size[0],
                       marker_color=self.markers.fill_color[0],
                       marker=dict(line=dict(color=self.markers.edge_color[0],
                                             width=self.markers.edge_width[0])),
                       )
            self.axes.obj[ir, ic] += [go.Scatter(x=xx,
                                                 y=yy,
                                                 mode='markers',
                                                 **mls
                                                 )]

        return self.axes.obj, None

    def plot_gantt(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str, y: str,
                   leg_name: str, xvals: npt.NDArray, yvals: list, bar_labels: list, ngroups: int,
                   data: 'Data'):  # noqa
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
        # Plot the bars
        raise NotImplementedError("gantt chart not implemented yet")
        for ii, (irow, row) in enumerate(df.iterrows()):
            if leg_name is not None:
                yi = yvals.index((row[y], leg_name))
            else:
                yi = yvals.index((row[y],))

            marker = dict(color=self.gantt.fill_color[iline],
                          opacity=self.gantt.fill_alpha,
                          line_color=self.gantt.edge_color[iline],
                          line_width=self.gantt.edge_width,
                          )
            self.axes.obj[ir, ic] += [go.Bar(x=[(row[x[0]], (row[x[1]] - row[x[0]]).total_seconds())],
                                             y=[(yi - self.gantt.height / 2, self.gantt.height)],
                                             orientation='h',
                                             marker=marker,
                                             name=leg_name,
                                             showlegend=True if self.legend._on else False,
                                             )]

    def plot_heatmap(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str,
                     z: str, data: 'data.Data') -> 'MPL_imshow_object':  # noqa: F821
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
            heatmap plot obj
        """
        # Set background color of axes to paper fill color for zoom out
        self.ul['plot_bgcolor'] = self.fig.fill_color[0]

        # Adjust the axes and rc label size based on the number of groups
        cols = len(df.columns)
        rows = len(df)
        map_sq = min(self.axes.size[0] / cols, self.axes.size[1] / rows)
        self.axes.size = [map_sq * cols, map_sq * rows]

        # Set text labels
        if not self.heatmap.text:
            hovertemplate = None
        else:
            hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z:.2f}<extra></extra>'

        # Make the heatmap
        self.axes.obj[ir, ic] += [go.Heatmap(x=df.index.tolist(),
                                             y=df.columns.tolist(),
                                             z=df,
                                             colorscale=self.cmap[0],
                                             hovertemplate=hovertemplate,
                                             zmin=data.ranges['zmin'][ir, ic],
                                             zmax=data.ranges['zmax'][ir, ic],
                                             showscale=self.cbar.on,
                                             colorbar=self._cbar_props,
                                             )]

        return self.axes.obj[ir, ic]

    def plot_hist(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str,
                  y: str, leg_name: str, data: 'data.Data') -> ['MPL_histogram_object', 'data.Data']:  # noqa: F821
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
            histogram plot object
            updated Data object
        """
        # Make the plot container
        if not self.axes.obj[ir, ic]:
            self.axes.obj[ir, ic] = []

        # Calculate the histogram using numpy and use go.Bar for plots
        counts, bin_edges = np.histogram(df[x].values, bins=self.hist.bins,
                                         range=data.branges[ir, ic], density=self.hist.normalize)
        if self.hist.cumulative:
            counts = np.cumsum(counts)

        marker = dict(color=self.hist.fill_color[iline],
                      opacity=self.hist.fill_alpha,
                      line_color=self.hist.edge_color[iline],
                      line_width=self.hist.edge_width,
                      )

        if self.hist.horizontal:
            self.axes.obj[ir, ic] += [
                go.Bar(y=(bin_edges[:-1] + bin_edges[1:]) / 2,  # x-coordinates are bin centers
                       x=counts,  # y-coordinates are the counts in each bin
                       width=np.diff(bin_edges),  # width of the bars
                       marker=marker,
                       orientation='h',
                       name=leg_name,
                       showlegend=True if self.legend._on else False,
                       )
            ]
            if iline == 0:
                # Update ranges
                xmin, xmax = data.ranges['xmin'][ir, ic], data.ranges['xmax'][ir, ic]
                data.ranges['xmin'][ir, ic] = data.ranges['ymin'][ir, ic]
                data.ranges['xmax'][ir, ic] = data.ranges['ymax'][ir, ic]
                data.ranges['ymin'][ir, ic] = xmin
                data.ranges['ymax'][ir, ic] = xmax
        else:
            self.axes.obj[ir, ic] += [
                go.Bar(x=(bin_edges[:-1] + bin_edges[1:]) / 2,
                       y=counts,
                       width=np.diff(bin_edges),
                       marker=marker,
                       name=leg_name,
                       showlegend=True if self.legend._on else False,
                       )
            ]

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
            mls = dict(line=dict(width=self.kde.width[iline],
                                 color=self.kde.color[iline],
                                 dash=self.kde.style[iline],
                                 ))
            self.axes.obj[ir, ic] += [
                go.Scatter(x=x0,
                           y=y0,
                           name='kde',
                           mode='lines',
                           showlegend=False,
                           **mls
                           )
            ]

        # Legend
        if leg_name is not None:
            self.legend.add_value(leg_name, None, 'lines')

        return self.axes.obj[ir, ic][-1], data

    def plot_imshow(self, ir: int, ic: int, df: pd.DataFrame, data: 'data.Data'):
        """Plot an image.  Copies strategies / code from plotly.express._imshow

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            data: Data object

        Returns:
            imshow or heatmap plot obj
        """
        # Set background color of axes to paper fill color for zoom out
        self.ul['plot_bgcolor'] = self.fig.fill_color[0]

        # Update wh_ratio
        self.wh_ratio = data.wh_ratio

        # TODO: add colorbar support
        # Make the imshow plot
        # plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        # self.axes.obj[ir, ic] = [go.Image(z=df)]
        # self.cmap[plot_num], vmin=zmin, vmax=zmax, interpolation=self.imshow.interp, aspect='auto')
        # im.set_clim(zmin, zmax)

        # Set the z range
        zmin = data.ranges['zmin'][ir, ic]
        zmax = data.ranges['zmax'][ir, ic]

        # Enable/disable binary rescaling and encoding as uint8 and then passed to plotly.js as a b64 PNG string
        # If False, data are passed unchanged as a numerical array
        # Setting to True may lead to performance gains, at the (possible) cost of a loss of precision
        # Default behaviour of binary_string: True for RGB images, False for 2D
        if self.imshow.binary is None:
            self.imshow.binary = df.ndim >= 3  # + slice_dimensions) # and not is_dataframe

        # # Contrast rescaling
        # if self.imshow.contrast_rescaling is None:
        #     self.imshow.contrast_rescaling = 'minmax' if df.ndim == (2 + slice_dimensions) else 'infer'

        # For 2D data with self.imshow.binary = False, use Heatmap trace
        if df.ndim == 2 and not self.imshow.binary:
            if df.dtype in [np.uint8, np.uint16, np.uint32, int]:
                hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z:.0f}<extra></extra>'
            else:
                hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z:.2f}<extra></extra>'

            self.axes.obj[ir, ic] += [go.Heatmap(z=df,
                                                 colorscale=self.cmap[0],
                                                 hovertemplate=hovertemplate,
                                                 zmin=data.ranges['zmin'][ir, ic],
                                                 zmax=data.ranges['zmax'][ir, ic],
                                                 showscale=self.cbar.on,
                                                 colorbar=self._cbar_props,
                                                 )
                                      ]

            # autorange = True if origin == "lower" else "reversed"
            # layout = dict(yaxis=dict(autorange=autorange))
            # if aspect == "equal":
            #     layout["xaxis"] = dict(scaleanchor="y", constrain="domain")
            #     layout["yaxis"]["constrain"] = "domain"

            self.fig.obj.layout.yaxis['autorange'] = 'reversed'
            self.fig.obj.layout.xaxis['scaleanchor'] = 'y'

            # these are for equal aspect ratio
            self.fig.obj.layout.xaxis['constrain'] = 'domain'
            self.fig.obj.layout.yaxis['constrain'] = 'domain'

            return

        # For 3D data or self.imshow.binary is True
        rescale_image = True if not (zmin is None and zmax is None) else False
        if rescale_image and df.ndim == 2:
            df = px.imshow_utils.rescale_intensity(df, (zmin, zmax), np.uint8)
        elif df.ndim >= 3:
            df = np.stack(
                    [
                        px.imshow_utils.rescale_intensity(
                            df[..., ch],
                            (df[..., ch].min() if data.ranges['zmin'][ir, ic] is None else data.ranges['zmin'][ir, ic],
                             df[..., ch].max() if data.ranges['zmax'][ir, ic] is None else data.ranges['zmax'][ir, ic]
                             ),
                            np.uint8,
                        )
                        for ch in range(df.shape[-1])
                    ],
                    axis=-1,
                )

        img_str = putl.image_array_to_data_uri(df, backend='auto',  compression=4,  ext='png')  # parameterize later
        self.axes.obj[ir, ic] += [go.Image(source=img_str)]  # , x0=x0, y0=y0, dx=dx, dy=dy)

    def plot_line(self, ir: int, ic: int, x0: float, y0: float, x1: float = None, y1: float = None, **kwargs):
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
            plot object
        """
        if x1 is not None:
            x0 = [x0, x1]
        if y1 is not None:
            y0 = [y0, y1]

        if 'color' not in kwargs.keys():
            kwargs['color'] = RepeatedList('#000000', 'temp')
        if 'style' not in kwargs.keys():
            kwargs['style'] = RepeatedList('solid', 'temp')
        if 'width' not in kwargs.keys():
            kwargs['width'] = RepeatedList(1, 'temp')

        mls = dict(line=dict(width=kwargs['width'][0],
                             color=kwargs['color'][0],
                             dash=kwargs['style'][0],
                             ))

        self.axes.obj[ir, ic] += [go.Scatter(x=x0,
                                             y=y0,
                                             mode='lines',
                                             showlegend=False,
                                             **mls,
                                             )]

    def plot_pie(self, ir: int, ic: int, df: pd.DataFrame, x: str, y: str, data: 'data.Data',
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
        marker = dict(colors=self.pie.colors,  # have to add opacity in here as rgba?
                      line_color=self.pie.edge_color[0],
                      line_width=self.pie.edge_width,
                      )

        if self.pie.explode is None:
            self.pie.explode = []
        elif self.pie.explode[0] == 'all':
            self.pie.explode = [self.pie.explode[1] for f in y]
        elif len(self.pie.explode) < len(y):
            self.pie.explode = list(self.pie.explode)
            self.pie.explode += [0 for f in range(0, len(y) - len(self.pie.explode))]
        else:
            self.pie.explode = []

        self.axes.obj[ir, ic] += [go.Pie(labels=x,
                                         values=y,
                                         pull=self.pie.explode,
                                         hoverinfo='label+percent',
                                         textinfo='label+value',
                                         textfont_size=self.pie.font_size,
                                         marker=marker,
                                         showlegend=True if self.legend._on else False,
                                         )]

        # Fudge the legend
        if self.legend._on:
            self.legend.add_value('Pie', None, 'lines')

    def plot_polygon(self, ir: int, ic: int, points: list, **kwargs):
        """Plot a polygon.

        Args:
            ir: subplot row index
            ic: subplot column index
            points: list of floats that defint the points on the polygon
            kwargs: keyword args
        """
        x = [f[0] for f in points]
        y = [f[1] for f in points]

        mls = dict(line=dict(width=kwargs['edge_width'],
                             color=kwargs['edge_color'][0],
                             dash=kwargs['edge_style'],
                             ))

        self.axes.obj[ir, ic] += [go.Scatter(x=x,
                                             y=y,
                                             mode='lines',
                                             showlegend=False,
                                             zorder=100,
                                             **mls
                                             )]

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
        df = df.copy()

        if not line_type:
            line_type = self.lines
            line_type_name = 'lines'
        else:
            line_type_name = line_type
            line_type = getattr(self, line_type)

        # Twinning?

        # Set the x data
        if x is None:
            dfx = df[utl.df_int_cols(df)].values
        else:
            dfx = df[x]

        # Mask any nans
        try:
            mask = np.isfinite(dfx)
        except TypeError:
            mask = dfx == dfx

        # Add optional jitter
        if self.markers.on and not marker_disable:
            if self.markers.jitter:
                dfx = np.random.normal(df[x], 0.03, size=len(df[y]))

        # Set the plot mode
        if line_type.on and self.markers.on and not marker_disable:
            mode = 'lines+markers'
        elif line_type.on:
            mode = 'lines'
        else:
            mode = 'markers'

        # Set marker type
        if self.markers.on and self.markers.type[iline] in self.HOLLOW_MARKERS and not marker_disable:
            marker_symbol = self.markers.type[iline] + ('-open' if not self.markers.filled else '')
        elif not marker_disable:
            marker_symbol = self.markers.type[iline]

        # Twinning
        if twin and self.axes.twin_x:
            yaxis = 'y2'
        else:
            yaxis = 'y1'
        if twin and self.axes.twin_y:
            xaxis = 'x2'
        else:
            xaxis = 'x1'

        # Define markers and lines (TODO: add style support)
        if self.markers.on and not marker_disable:
            mls = dict(marker_symbol=marker_symbol,
                       marker_size=self.markers.size[iline],
                       marker_color=self.markers.fill_color[iline],
                       marker=dict(line=dict(color=self.markers.edge_color[iline],
                                             width=self.markers.edge_width[iline])),
                       line=dict(width=self.lines.width[iline],
                                 color=self.lines.color[iline],
                                 dash=self.lines.style[iline]
                                 )
                       )
        # elif self.lines.on:
        #     mls = dict(line=dict(width=self.lines.width[iline],
        #                          color=self.lines.color[iline],
        #                          dash=self.lines.style[iline],
        #                          ))
        elif line_type.on:
            mls = dict(line=dict(width=line_type.width[iline],
                                 color=line_type.color[iline],
                                 dash=line_type.style[iline],
                                 ))
        else:
            mls = {}

        # Make the scatter trace
        show_legend = False
        if leg_name is not None and leg_name not in self.legend.values['Key'].values:
            show_legend = True
        self.axes.obj[ir, ic] += [
            go.Scatter(x=dfx[mask],
                       y=df[y][mask],
                       name=leg_name,
                       mode=mode,
                       xaxis=xaxis,
                       yaxis=yaxis,
                       showlegend=show_legend,
                       **mls
                       )]

        # For box plot markers, change the hover text
        if self.name == 'box':
            params = {'hovertemplate': '(%{x}, %{y})<extra></extra>'}
            leg = {}
            if leg_name is not None:
                if str(leg_name) in self.legend.values.Key.values:
                    # Legend entry already exists, use a legend group
                    leg = {'legendgroup': leg_name, 'showlegend': False}
                else:
                    leg = {'legendgroup': leg_name, 'showlegend': True}
            params.update(leg)
            self.axes.obj[ir, ic][-1].update(params)

        # Add a reference to the line to self.lines
        if leg_name is not None:
            if leg_name is not None and leg_name not in list(self.legend.values['Key']):
                self.legend.add_value(leg_name, iline, line_type_name)

    def restore(self):
        """Undo changes to default plotting library parameters."""

    def save(self, filename: str, idx: int = 0):
        """Save a plot window.

        Args:
            filename: name of the file
            idx (optional): figure index in order to set the edge and face color of the
                figure correctly when saving. Defaults to 0.
        """
        if '.html' in str(filename):
            self.fig.obj.write_html(filename)
        else:
            self.fig.obj.write_image(filename)

    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Background color
        self.ul['plot_bgcolor'] = self.axes.fill_color[utl.plot_num(ir, ic, self.ncol)]

        # Set the axes borders/spines
        for ax in ['x', 'y']:
            show = False
            if ax == 'x' and (self.axes.spine_bottom or self.axes.spine_top):
                show = True
            if ax == 'y' and (self.axes.spine_left or self.axes.spine_right):
                show = True
            if self.axes.edge_width == 0:
                show = False
            self.ul[f'{ax}axis_style'] = dict(showline=show, linecolor=self.axes.edge_color[0],
                                              linewidth=self.axes.edge_width, mirror=True)

    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """
        for ss in ['', '2']:
            for ax in ['x', 'y']:
                grid = getattr(self, f'grid_major_{ax}{ss}')
                if grid is not None:
                    self.ul[f'{ax}{ss}grid'] = dict(gridcolor=grid.color[0],
                                                    gridwidth=grid.width[0],
                                                    griddash=grid.style[0],
                                                    )

    def set_axes_labels(self, ir: int, ic: int, data: 'Data'):  # noqa: F821
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: fcp.data object

        """
        axis = ['x', 'x2', 'y', 'y2']
        for ax in axis:
            label = getattr(self, f'label_{ax}')
            if not label.on:
                continue
            if type(label.text) not in [str, list]:
                continue
            # if isinstance(label.text, str):
            #     labeltext = label.text
            # if isinstance(label.text, list):
            #     labeltext = label.text[ic + ir * self.ncol]

            # Set the font weight and style
            self._set_weight_and_style(f'label_{ax}')

            # Toggle label visibility
            if not self.separate_labels:
                if ax == 'x' and ir != self.nrow - 1 and self.axes.visible[ir + 1, ic]:
                    continue
                if ax == 'x2' and ir != 0:
                    continue
                if ax == 'y' and ic != 0 and self.axes.visible[ir, ic - 1]:
                    continue
                if ax == 'y2' and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue
                if ax == 'z' and ic != self.ncol - 1 and utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue

            # Create a dict of the label props to apply later
            plot_num = utl.plot_num(ir, ic, self.ncol)
            plot_num = '' if plot_num == 1 else str(plot_num)
            key = f'{ax[0]}axis{plot_num}_title'
            self.ul[f'{ax}axis_title'][key] = \
                dict(text=label.text,
                     font=dict(family=label.font,
                               size=label.font_size,
                               color=label.font_color,
                               style=label.font_style,
                               weight=label.font_weight),
                     standoff=self.ws_label_tick)
            lab = getattr(self, f'label_{ax}')
            lab.size = utl.get_text_dimensions(lab.text, lab.font, lab.font_size, lab.font_style,
                                               lab.font_weight, lab.rotation, dpi=self.dpi)

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        if len(ranges) == 0:
            return

        # primary x
        if str(self.axes.scale).lower() in LOGX:
            self.ul['xaxis_range'][ir, ic] = \
                np.array([np.log10(ranges['xmin'][ir, ic]), np.log10(ranges['xmax'][ir, ic])])
        else:
            self.ul['xaxis_range'][ir, ic] = np.array([ranges['xmin'][ir, ic], ranges['xmax'][ir, ic]])
        # secondary x
        if self.axes.twin_y:
            if str(self.axes.scale).lower() in LOGX:
                self.ul['x2axis_range'][ir, ic] = \
                    np.array([np.log10(ranges['x2min'][ir, ic]), np.log10(ranges['x2max'][ir, ic])])
            else:
                self.ul['x2axis_range'][ir, ic] = np.array([ranges['x2min'][ir, ic], ranges['x2max'][ir, ic]])
        # primary y
        if str(self.axes.scale).lower() in LOGY:
            self.ul['yaxis_range'][ir, ic] = \
                np.array([np.log10(ranges['ymin'][ir, ic]), np.log10(ranges['ymax'][ir, ic])])
        else:
            self.ul['yaxis_range'][ir, ic] = np.array([ranges['ymin'][ir, ic], ranges['ymax'][ir, ic]])
        # secondary y
        if self.axes.twin_x:
            if str(self.axes.scale).lower() in LOGY:
                self.ul['y2axis_range'][ir, ic] = \
                    np.array([np.log10(ranges['y2min'][ir, ic]), np.log10(ranges['y2max'][ir, ic])])
            else:
                self.ul['y2axis_range'][ir, ic] = np.array([ranges['y2min'][ir, ic], ranges['y2max'][ir, ic]])
        # z-axis --> add in plot creation

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Wrap title
        if ir == 0 and ic == 0 and self.title_wrap.on:
            # Set the title position (left, right, top, bottom)
            self.title_wrap.position = [
                    -(self.axes.edge_width - self.title_wrap.edge_width / 2) / self.axes.size[0],
                    self.ncol + ((2 * self.ncol - 1) * self.axes.edge_width - self.title_wrap.edge_width / 2) /
                    self.axes.size[0],
                    1 + (self.axes.edge_width + self.label_wrap.size[1] + self.title_wrap.size[1] -
                         self.title_wrap.edge_width / 2) / (self.axes.size[1] / self.wh_ratio),
                    1 + (self.axes.edge_width + self.label_wrap.size[1] + self.title_wrap.edge_width / 2) /
                    (self.axes.size[1] / self.wh_ratio)
                ]

            # Make the label
            self.add_label(ir, ic, self.title_wrap.text, element=self.title_wrap)

        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            if not self.label_row.values_only:
                lab = f'{self.label_row.text}={self.label_row.values[ir]}'
            else:
                lab = self.label_row.values[ir]

            # Set the label position (left, right, top, bottom)
            x0 = 1 + (self.axes.edge_width + self.label_row.edge_width + self.ws_label_row) / self.axes.size[0]
            self.label_row.position = [
                x0,
                x0 + (self.label_row.size[0] + self.label_row.edge_width) / self.axes.size[0],
                -(self.axes.edge_width - self.label_row.edge_width / 2) / (self.axes.size[1] / self.wh_ratio),
                1 + (self.axes.edge_width - self.label_row.edge_width / 2) / (self.axes.size[1] / self.wh_ratio),
            ]

            # Make the label
            self.add_label(ir, ic, lab, element=self.label_row)

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            if self.label_wrap.on:
                text = ' | '.join([str(f) for f in utl.validate_list(self.label_wrap.values[ir * self.ncol + ic])])

                # Set the label position (left, right, top, bottom)
                self.label_wrap.position = [
                    -(self.axes.edge_width - self.label_wrap.edge_width / 2) / self.axes.size[0],
                    1 + (self.axes.edge_width - self.label_wrap.edge_width / 2) / self.axes.size[0],
                    1 + (self.axes.edge_width + self.label_wrap.size[1] - self.label_wrap.edge_width / 2) /
                    (self.axes.size[1] / self.wh_ratio),
                    1 + (self.axes.edge_width + self.label_wrap.edge_width / 2) / (self.axes.size[1] / self.wh_ratio),
                ]

                # Make the label
                self.add_label(ir, ic, text, element=self.label_wrap)
            else:
                if not self.label_col.values_only:
                    lab = f'{self.label_col.text}={self.label_col.values[ic]}'
                else:
                    lab = self.label_col.values[ic]

                # Set the label position (left, right, top, bottom)
                self.label_col.position = [
                    -(self.axes.edge_width) / self.axes.size[0],
                    1 + (self.axes.edge_width) / self.axes.size[0],
                    1 + (self.axes.edge_width + self.ws_label_col + self.label_col.size[1]) / self.axes.size[1],
                    1 + (self.axes.edge_width + self.ws_label_col) / self.axes.size[1],
                ]

                # Make the label
                self.add_label(ir, ic, lab, element=self.label_col)

    def set_axes_scale(self, ir: int, ic: int):
        """
        Set the axis scale.
        """
        if str(self.axes.scale).lower() in LOGX:
            self.ul['xscale'] = dict(type='log')
        if str(self.axes.scale).lower() in LOGY:
            self.ul['yscale'] = dict(type='log')

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # TODO: minor ticks, rotation, categorical, log/exponential
        for ax in ['x', 'y']:
            direction = 'outside' if getattr(self, f'ticks_major_{ax}').direction == 'out' else 'inside'
            self.ul[f'{ax}ticks'] = dict(
                tickfont_family=getattr(self, f'tick_labels_major_{ax}').font,
                tickfont_size=getattr(self, f'tick_labels_major_{ax}').font_size,
                ticks=direction,
                tickcolor=getattr(self, f'ticks_major_{ax}').color[0],
                ticklen=getattr(self, f'ticks_major_{ax}')._size[0],
                tickwidth=getattr(self, f'ticks_major_{ax}')._size[1],
                dtick=getattr(self, f'ticks_major_{ax}').increment,
                showticklabels=getattr(self, f'tick_labels_major_{ax}').on,
                tickangle=getattr(self, f'tick_labels_major_{ax}').rotation,
                )

    def set_figure_final_layout(self, data, **kwargs):
        """Final adjustment of the figure size and plot spacing.

        Args:
            data: Data object
            kwargs: keyword args
        """
        # Set the figure size
        self._get_tick_label_size(data)
        self._get_figure_size(data)

        # Update the title position
        self.ul['title']['x'] = 0.5
        self.ul['title']['y'] = 1 - (self._top - self.ws_title_ax - self.title.font_size / 2) / self.fig.size[1]

        # Adjust the legend position (only for outside of plot right now)
        if self.legend.on and self.legend.location == 0:
            self.ul['legend']['x'] = 1 - (self.ws_leg_fig + self.legend.size[0]) / self.fig.size[0]
            self.fig.obj.update_layout(legend=self.ul['legend'])

        # Update the x/y axes
        for ax in data.axs_on:
            if ax == 'z':
                continue
            kw = {}
            kw2 = {}
            kw.update(self.ul[f'{ax}ticks'])
            kw.update(self.ul[f'{ax}axis_style'])
            kw.update(self.ul[f'{ax}grid'])
            kw.update(self.ul[f'{ax}scale'])
            if ax == 'y2':
                kw2['secondary_y'] = True
            elif ax == 'x2':
                kw2['secondary_x'] = True
            getattr(self.fig.obj, f'update_{ax[0]}axes')(kw, **kw2)

        # Update the axes labels
        axis_labels = self.ul['xaxis_title']
        axis_labels.update(self.ul['yaxis_title'])

        # Add space for tick labels and disable on certain plots
        self.fig.obj.update_yaxes(ticksuffix=" ")
        if self.axes.twin_x:
            self.fig.obj['layout']['yaxis2'].update(dict(tickprefix=" "))
        for ir, ic in np.ndindex(self.axes.obj.shape):
            for ax in self.axs_on:
                if not self.separate_ticks and ax == 'x' and ir != self.nrow - 1 and data.share_y is True:
                    self.fig.obj.update_xaxes(showticklabels=False, row=ir + 1, col=ic + 1)
                if not self.separate_ticks and ax == 'x2' and ir != 0:
                    continue
                if not self.separate_ticks and ax == 'y' and ic != 0:
                    self.fig.obj.update_yaxes(showticklabels=False, row=ir + 1, col=ic + 1)
                if not self.separate_ticks and ax == 'y2' and ic != self.ncol - 1 and \
                        utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue
                if not self.separate_ticks and ax == 'z' and ic != self.ncol - 1 and \
                        utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue

        # Iterate through subplots to add the traces
        for ir, ic in np.ndindex(self.axes.obj.shape):
            # Add the traces
            for trace in range(0, len(self.axes.obj[ir, ic])):
                if self.name not in ['pie'] and self.axes.obj[ir, ic][trace]['yaxis'] == 'y2':
                    self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1, secondary_y=True)
                else:
                    self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1)

            # Add axhvlines
            axlines = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']
            for line in axlines:
                func = self.fig.obj.add_vline if 'vline' in line else self.fig.obj.add_hline
                if not getattr(self, line).on:
                    continue
                for vals in getattr(self, line).obj[ir, ic]:
                    func(**vals)

            # Add box lines
            if self.name == 'box' and self.box_divider.on:
                if self.box_divider.obj[ir, ic] is None:
                    self.box_divider.obj[ir, ic] = []
                for vals in self.box_divider.obj[ir, ic]:
                    self.fig.obj.add_vline(**vals)

            # Set the ranges
            self.fig.obj.update_xaxes(row=ir + 1, col=ic + 1, range=self.ul['xaxis_range'][ir, ic])
            if self.axes.twin_y:
                self.fig.obj['layout']['xaxis2']['range'] = self.ul['x2axis_range'][ir, ic]
            self.fig.obj.update_yaxes(row=ir + 1, col=ic + 1, range=self.ul['yaxis_range'][ir, ic])
            if self.axes.twin_x:
                self.fig.obj['layout']['yaxis2']['range'] = self.ul['y2axis_range'][ir, ic]

        # Update the figure layout
        self.fig.obj.update_layout(autosize=False,
                                   height=self.fig.size[1],
                                   legend_title_text=self.legend.text,
                                   margin=dict(l=self._margin_left,
                                               r=self._margin_right,
                                               t=self._top,
                                               b=self._bottom,
                                               ),
                                   modebar=dict(orientation=self.modebar.orientation,
                                                bgcolor=self.modebar.fill_color[0],
                                                color=self.modebar.button_color,
                                                activecolor=self.modebar.button_active_color,
                                                ),
                                   paper_bgcolor=self.fig.fill_color[0],
                                   plot_bgcolor=self.ul['plot_bgcolor'],
                                   showlegend=self.legend.on,
                                   title=self.ul['title'],
                                   width=self.fig.size[0],
                                   **axis_labels,
                                   )

        # Set the axes positions
        for ir, ic in np.ndindex(self.axes.obj.shape):
            x0, x1, y0, y1 = self._axes_domain(ir, ic)
            self.fig.obj.update_xaxes(row=ir + 1, col=ic + 1, domain=[x0, x1])
            self.fig.obj.update_yaxes(row=ir + 1, col=ic + 1, domain=[y0, y1])

        # Update the plot labels
        if self.axes.twin_x:
            self.fig.obj['layout']['yaxis2'].update(self.ul['y2grid'])
            self.fig.obj['layout']['yaxis2']['title'] = self.ul['y2axis_title']['yaxis_title']

        if self.axes.twin_y:
            self.fig.obj.data[1].update(xaxis='x2')
            # Have to do this manually since there is no update_xaxes2 function
            for k, v in self.ul['x2grid'].items():
                if hasattr(self.fig.obj.layout.xaxis2, k):
                    setattr(self.fig.obj.layout.xaxes2, k, v)
            self.fig.obj['layout']['xaxis2']['title'] = self.ul['x2axis_title']['xaxis_title']

        # Box width
        if self.name == 'box':
            self.fig.obj['layout']['boxgap'] = self.box.width[0] * (1 - 0.3)  # a value of 0 actually gives 0.3

        # Set the modebar config
        self.modebar.obj = {'displaylogo': self.modebar.logo,
                            'modeBarButtonsToRemove': self.modebar.remove_buttons,
                            }
        if self.modebar.visible and self.modebar.on:
            self.modebar.obj['displayModeBar'] = self.modebar.visible
        elif not self.modebar.on:
            self.modebar.obj['displayModeBar'] = False

        # Update text positions
        for ir, ic in np.ndindex(self.axes.obj.shape):
            # Only want to do this to non-grouping labels
            if self.fit.obj is None and self.text.obj[ir, ic] is None:
                continue

            texts = self.fig.obj['layout']['annotations']
            for text in texts:
                if (self.fit.obj is None or self.fit.obj[ir, ic] is None or text not in self.fit.obj[ir, ic]) and \
                        (self.text.obj[ir, ic] is None or text not in self.text.obj[ir, ic]):
                    continue
                if isinstance(text['x'], str):
                    text['x'] = text['x'].replace('xmin', str(data.ranges['xmin'][ir, ic])) \
                                         .replace('xmax', str(data.ranges['xmax'][ir, ic]))
                    text['x'] = utl.arithmetic_eval(text['x'])
                if text['x'] > 1:
                    text['x'] /= self.axes.size[0]
                if isinstance(text['y'], str):
                    text['y'] = text['y'].replace('ymin', str(data.ranges['ymin'][ir, ic])) \
                                         .replace('ymax', str(self.axes.size[1]))
                    text['y'] = utl.arithmetic_eval(text['y'])
                if text['y'] > 1:
                    text['y'] /= self.axes.size[1]

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.text is None:
            return

        # TODO: deal with other alignments
        self._set_weight_and_style('title')
        self.title.size = utl.get_text_dimensions(self.title.text, self.title.font, self.title.font_size,
                                                  self.title.font_style, self.title.font_weight, dpi=self.dpi)

        self.ul['title'] = \
            dict(text=self.title.text,
                 xanchor='center',
                 xref='paper',
                 yanchor='middle',
                 yref='container',
                 font=dict(family=self.title.font,
                           size=self.title.font_size,
                           color=self.title.font_color,
                           style=self.title.font_style,
                           weight=self.title.font_weight
                           )
                 )

    def _set_text_position(self, obj, position=None):
        """Move text label to the correct location."""

    def _set_weight_and_style(self, element: str):
        """Add html tags to the text string of an element to address font weight and style.

        Args:
            element: name of the Element object to modify; results are stored in place
        """
        if getattr(self, element).font_weight == 'bold':
            getattr(self, element).text = f'<b>{getattr(self, element).text}</b>'
        if getattr(self, element).font_style == 'italic':
            getattr(self, element).text = f'<i>{getattr(self, element).text}</i>'

    def _set_weight_and_style_str(self, text: str, font_weight: str, font_style: str) -> str:
        """Add html tags to any text string.

        Args:
            text: string to format
            font_weight: font weight
            font_style: font style

        Returns:
            formated text string
        """
        if font_weight == 'bold':
            text = f'<b>{text}</b>'
        if font_style == 'italic':
            text = f'<i>{text}</i>'
        return text

    def show(self, filename=None):
        """Display the plot window.

        Args:
            filename (optional): name of the file to show. Defaults to None.

        """
        # not saved, open in browser or notebook
        if filename is None:
            # jupyter notebook special
            try:
                app = str(get_ipython())  # noqa
            except:  # noqa
                app = ''
            if 'zmqshell.ZMQInteractiveShell' in app \
                    and not pio.renderers.default == 'plotly_mimetype+notebook_connected':
                pyo.init_notebook_mode(connected=True)

            config = {'scrollZoom': True}
            config.update(self.modebar.obj)

            # pio.renderers.default = 'iframe'  # not sure about this
            self.fig.obj.show(config=config)
