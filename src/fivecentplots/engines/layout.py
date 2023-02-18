import pandas as pd
import pdb
import numpy as np
import copy
from .. colors import DEFAULT_COLORS
from .. utilities import RepeatedList
from .. import utilities as utl
from distutils.version import LooseVersion
from collections import defaultdict
import warnings
import abc
from .. import data
import matplotlib as mpl
db = pdb.set_trace


def custom_formatwarning(msg, *args, **kwargs):
    """Ignore everything except the message."""
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")
warnings.simplefilter(action='ignore', category=FutureWarning)


DEFAULT_MARKERS = ['o', '+', 's', 'x', 'd', 'Z', '^', 'Y', 'v', r'\infty',
                   r'\#', r'<', u'\u2B21', u'\u263A', '>', u'\u29C6', r'\$',
                   u'\u2B14', u'\u2B1A', u'\u25A6', u'\u229E', u'\u22A0',
                   u'\u22A1', u'\u20DF', r'\gamma', r'\sigma', r'\star', ]

LEGEND_LOCATION = defaultdict(int,
                              {'outside': 0,  # always "best" outside of the damn plot
                               'upper right': 1, 1: 1,
                               'upper left': 2, 2: 2,
                               'lower left': 3, 3: 3,
                               'lower right': 4, 4: 4,
                               'right': 5, 5: 5,
                               'center left': 6, 6: 6,
                               'center right': 7, 7: 7,
                               'lower center': 8, 8: 8,
                               'upper center': 9, 9: 9,
                               'center': 10, 10: 10,
                               'below': 11, 11: 11},
                              )

LOGX = ['logx', 'semilogx', 'log', 'loglog']
LOGY = ['logy', 'semilogy', 'log', 'loglog']
LOGZ = []
SYMLOGX = ['symlogx', 'symlog']
SYMLOGY = ['symlogy', 'symlog']
LOGITX = ['logitx', 'logit']
LOGITY = ['logity', 'logit']
LOG_ALLX = LOGX + SYMLOGX + LOGITX
LOG_ALLY = LOGY + SYMLOGY + LOGITY
ENGINE = ''


class BaseLayout:
    def __init__(self, data: 'Data', defaults: list = [], **kwargs):  # noqa F821
        """Generic layout properties class

        Args:
            data: Data class object for the plot
            defaults: items from the theme file
            kwargs: user-defined keyword args

        """
        # Retain the original kwargs as the kwargs variable will be modified
        # during the various _init functions
        kwargs_orig = kwargs.copy()

        # Set the plot type name
        self.name = data.name

        # Reload default file
        if len(defaults) > 0:
            self.fcpp = defaults[0].copy()
            self.color_list = defaults[1].copy()
            marker_list = defaults[2].copy()
        else:
            self.fcpp, self.color_list, marker_list, rcParams = utl.reload_defaults(kwargs.get('theme', None))

        # Class attribute definition (alphabetical)
        self.auto_tick_threshold = None  # Threshold value for placement of auto-generated ticks
        self.ax = ['x', 'y', 'x2', 'y2']  # list of valid axes
        self.ax_hlines = None  # Element object for horizontal lines
        self.ax_vlines = None  # Element object for vertical lines
        self.ax2_hlines = None  # Element object for horizontal lines on secondary axis
        self.ax2_vlines = None  # Element object for vertical lines on secondary axis
        self.axes = None  # Element object for the axis
        self.axes2 = None  # Element object for the secondary axis
        self.bar = None  # Element object for barchart plot
        self.box = None  # Element object for box plot
        self.box_divider = None  # Element object for divider lines between box groups
        self.box_grand_mean = None  # Element object for grand mean line plot on box plot
        self.box_grand_median = None  # Element object for grand median line plot on box plot
        self.box_group_means = None  # Element object for group mean line plots on box plot
        self.box_group_label = None  # Element object for box plot group label text
        self.box_group_title = None  # Element object for box plot group title text
        self.box_mean_diamonds = None  # Element object for mean diamond overlays on box plot
        self.box_range_lines = None  # Element object for box plot range line styling
        self.box_scale = False  # auto-scale axes width for boxplot
        self.box_stat_line = None  # Element object for arbitrary stat line plot on box plot
        self.box_whisker = None  # Element object for bow wisker line styling
        self.cbar = None  # Element object for colorbar
        self.color_list_unique = None  # color list sans duplicates
        self.contour = None  # Element object for contour plot
        self.cmap = None  # color map to use in plot
        self.fig = None  # Element object for the figure
        self.fit = None  # Element object for fit line
        self.gantt = None  # Element object for gannt chart
        self.grid_major = None  # Element object with default values for major grids not explicitly defined
        self.grid_major_x = None  # Element object for x major grid (defaults to self.grid_major)
        self.grid_major_x2 = None  # Element object for x2 major grid (defaults to self.grid_major)
        self.grid_major_y = None  # Element object for y major grid (defaults to self.grid_major)
        self.grid_major_y2 = None  # Element object for y2 major grid (defaults to self.grid_major)
        self.grid_minor = None  # same as above but for minor grid
        self.grid_minor_x = None  # same as above but for minor grid
        self.grid_minor_x2 = None  # same as above but for minor grid
        self.grid_minor_y = None  # same as above but for minor grid
        self.grid_minor_y2 = None  # same as above but for minor grid
        self.heatmap = None  # Element object for heatmap plot
        self.hist = None  # Element object for histogram plot
        self.imshow = None  # Element object for imshow plot
        self.interval = None  # Element object for conf, percentile, nq interval ranges
        self.label_col = None  # Element object for col label text
        self.label_row = None  # Element object for row label text
        self.label_wrap = None  # Element object for wrap label text
        self.label_x = None  # Element object for the x-label
        self.label_x2 = None  # Element object for the secondary x-label
        self.label_y = None  # Element object for the y-label
        self.label_y2 = None  # Element object for the secondary y-label
        self.label_z = None  # Element object for the z-label
        self.lcl = None  # Element object for lower control limit shading
        self.legend = None  # Legend_Element for figure legend
        self.lines = None  # Element object for plot lines
        self.markers = None  # Element object for markers
        self.ncol = None  # number of subplot columns
        self.nrow = None  # number of subplot rows
        self.obj_array = None  # row x column array for Element objects
        self.pie = None  # Element object for pie chart
        self.ref_line = None  # Element object for reference line
        self.rolling_mean = None  # Element object for the rolling mean XY plot on bar chart
        self.text = None  # Element object for arbitrary text
        self.tick_labels_major = None  # Element object with default values for tick labels not explicitly defined
        self.tick_labels_major_x = None  # Element object for x major tick labels (defaults to self.tick_labels_major)
        self.tick_labels_major_x2 = None  # Element object for x2 major tick labels (defaults to self.tick_labels_major)
        self.tick_labels_major_y = None  # Element object for y major tick labels (defaults to self.tick_labels_major)
        self.tick_labels_major_y2 = None  # Element object for y2 major tick labels (defaults to self.tick_labels_major)
        self.tick_labels_major_z = None  # Element object for z major tick labels (defaults to self.tick_labels_major)
        self.tick_labels_minor = None  # same as above but for minor tick labels
        self.tick_labels_minor_x = None  # same as above but for minor tick labels
        self.tick_labels_minor_x2 = None  # same as above but for minor tick labels
        self.tick_labels_minor_y = None  # same as above but for minor tick labels
        self.tick_labels_minor_y2 = None  # same as above but for minor tick labels
        self.ticks_major = None  # Element object with default values for ticks not explicitly defined
        self.ticks_major_x = None  # Element object for x major ticks (defaults to self.ticks_major)
        self.ticks_major_x2 = None  # Element object for x2 major ticks (defaults to self.ticks_major)
        self.ticks_major_y = None  # Element object for y major ticks (defaults to self.ticks_major)
        self.ticks_major_y2 = None  # Element object for y2 major ticks (defaults to self.ticks_major)
        self.ticks_minor = None  # same as above but for minor ticks
        self.ticks_minor_x = None  # same as above but for minor ticks
        self.ticks_minor_x2 = None  # same as above but for minor ticks
        self.ticks_minor_y = None  # same as above but for minor ticks
        self.ticks_minor_y2 = None  # same as above but for minor ticks
        self.ticks_minor_x_number = None  # number of x-axis minor ticks
        self.ticks_minor_x2_number = None  # number of x2-axis minor ticks
        self.ticks_minor_y_number = None  # number of y-axis minor ticks
        self.ticks_minor_y2_number = None  # number of y2-axis minor ticks
        self.title = None  # Element object for the plot title
        self.title_wrap = None  # Element object for title text in wrap plot
        self.violin = None  # Element object for box plot violins
        self.ucl = None  # Element object for upper control limit shading
        self.ws_ax_box_title = 0  # white space between axis right edge and left edge of box label text
        self.ws_ax_cbar = 0  # white space between axis right edge and cbar
        self.ws_ax_fig = 0  # white space between axis right edge and figure right edge (w/ no legend)
        self.ws_ax_label_xs = 0  # excess white space needed for long labels
        self.ws_ax_leg = 0  # white space between right axis edge and legend left edge
        self.ws_col = 0  # white space between columns (gets adjusted to fit unless explicitly defined)
        self.ws_col_def = 0  # default white space between columns
        self.ws_fig_ax = 0  # white space between figure left edge and axis left edge
        self.ws_fig_label = 0  # white space between left figure edge and y-axis label
        self.ws_fig_title = 0  # white space between figure top edge and title top edge
        self.ws_label_col = 0  # white space between x label and col label
        self.ws_label_fig = 0  # white space between x-axis label bottom edge and figure bottom edge
        self.ws_label_row = 0  # white space between y label and row label
        self.ws_label_tick = 0  # white space between axis label and tick labels
        self.ws_leg_fig = 0  # white space between legend right edge and figure right edge
        self.ws_row = 0  # white space between rows (gets adjusted to fit unless explicitly defined)
        self.ws_row_def = 0  # default white space between
        self.ws_tick_minimum = 0  # minimum white space between ticks
        self.ws_ticks_ax = 0  # white space between tick labels and axis edge
        self.ws_title_ax = 0  # white space between plot title and top axis edge

        # Init the elements and their parameters
        self._init_layout_rc(data)
        kwargs = self._init_figure(kwargs)
        kwargs = self._init_colors(kwargs)
        kwargs = self._init_axes(data, kwargs)
        kwargs = self._init_axes_labels(kwargs)
        kwargs = self._init_title(kwargs)
        kwargs = self._init_ticks(kwargs)
        kwargs = self._init_markers(kwargs, data)
        kwargs = self._init_lines(kwargs)
        kwargs = self._init_fit(kwargs)
        kwargs = self._init_control_limit(kwargs)
        kwargs = self._init_ref(kwargs)
        kwargs = self._init_legend(kwargs, data)
        kwargs = self._init_cbar(kwargs)
        kwargs = self._init_contour(kwargs)
        kwargs = self._init_imshow(kwargs, kwargs_orig)
        kwargs = self._init_heatmap(kwargs, kwargs_orig, data)
        kwargs = self._init_bar(kwargs)
        kwargs = self._init_gantt(kwargs)
        kwargs = self._init_pie(kwargs)
        kwargs = self._init_hist(kwargs)
        kwargs = self._init_box(kwargs)
        kwargs = self._init_axhvlines(kwargs)
        kwargs = self._init_grid(kwargs)
        kwargs = self._init_rc_labels(kwargs)
        kwargs = self._init_intervals(kwargs)
        kwargs = self._init_text_box(kwargs)
        self._init_white_space(kwargs)

        # Some extra kwargs
        self.inline = utl.kwget(kwargs, self.fcpp, 'inline', None)
        self.separate_labels = utl.kwget(kwargs, self.fcpp,
                                         'separate_labels', False)
        self.separate_ticks = utl.kwget(kwargs, self.fcpp,
                                        'separate_ticks', self.separate_labels)
        if self.separate_labels:
            self.separate_ticks = True
        self.tick_cleanup = utl.kwget(kwargs, self.fcpp, 'tick_cleanup', 'shrink')
        if isinstance(self.tick_cleanup, str):
            self.tick_cleanup = self.tick_cleanup.lower()

        # Overrides for specific plot types
        if 'bar' in self.name:
            if self.bar.horizontal:
                self.grid_major_y.on = False
                self.grid_minor_y.on = False
                self.ticks_major_y.on = False
                self.ticks_minor_y.on = False
            else:
                self.grid_major_x.on = False
                self.grid_minor_x.on = False
                self.ticks_major_x.on = False  # may want to not do this but hard to see the ticks anyway
                self.ticks_minor_x.on = False
        if 'box' in self.name:
            self.grid_major_x.on = False
            self.grid_minor_x.on = False
            self.ticks_major_x.on = False
            self.ticks_minor_x.on = False
            self.tick_labels_major_x.on = False
            self.tick_labels_minor_x.on = False
            self.label_x.on = False
        if 'heatmap' in self.name or 'imshow' in self.name:
            self.grid_major_x.on = False
            self.grid_major_y.on = False
            self.grid_minor_x.on = False
            self.grid_minor_y.on = False
            self.ticks_major_x.on = False
            self.ticks_major_y.on = False
            self.ticks_minor_x.on = False
            self.ticks_minor_y.on = False
            self.tick_labels_minor_x.on = False
            self.tick_labels_minor_y.on = False
        if 'pie' in self.name:
            self.grid_major_x.on = False
            self.grid_major_y.on = False
            self.grid_minor_x.on = False
            self.grid_minor_y.on = False
            self.ticks_major_x.on = False
            self.ticks_major_y.on = False
            self.ticks_minor_x.on = False
            self.ticks_minor_y.on = False
            self.tick_labels_minor_x.on = False
            self.tick_labels_minor_y.on = False
            self.label_x.on = False
            self.label_y.on = False
        self.kwargs_mod = kwargs

        # kwargs changes made in the data object
        self._update_from_data(data)
        self._update_wrap(data, kwargs)

        # Update the label text
        self._set_label_text(data)

    def _init_axes(self, data, kwargs: dict) -> dict:
        """Create the axes object

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Axis
        spines = utl.kwget(kwargs, self.fcpp, 'spines', True)
        self.axes = Element('ax', self.fcpp, kwargs,
                            obj=self.obj_array,
                            size=utl.kwget(kwargs, self.fcpp, 'ax_size', [400, 400]),
                            edge_color=utl.kwget(kwargs, self.fcpp, 'ax_edge_color', '#aaaaaa'),
                            fill_color=utl.kwget(kwargs, self.fcpp, 'ax_fill_color', '#eaeaea'),
                            primary=True,
                            scale=utl.kwget(kwargs, self.fcpp, 'ax_scale', kwargs.get('ax_scale', None)),
                            share_x=utl.kwget(kwargs, self.fcpp, 'share_x', kwargs.get('share_x', None)),
                            share_y=utl.kwget(kwargs, self.fcpp, 'share_y', kwargs.get('share_y', None)),
                            share_z=utl.kwget(kwargs, self.fcpp, 'share_z', kwargs.get('share_z', None)),
                            share_x2=utl.kwget(kwargs, self.fcpp, 'share_x2', kwargs.get('share_x2', None)),
                            share_y2=utl.kwget(kwargs, self.fcpp, 'share_y2', kwargs.get('share_y2', None)),
                            share_col=utl.kwget(kwargs, self.fcpp, 'share_col', kwargs.get('share_col', None)),
                            share_row=utl.kwget(kwargs, self.fcpp, 'share_row', kwargs.get('share_row', None)),
                            spine_bottom=utl.kwget(kwargs, self.fcpp, ['spine_bottom', 'ax_edge_bottom'], spines),
                            spine_left=utl.kwget(kwargs, self.fcpp, ['spine_left', 'ax_edge_left'], spines),
                            spine_right=utl.kwget(kwargs, self.fcpp, ['spine_right', 'ax_edge_right'], spines),
                            spine_top=utl.kwget(kwargs, self.fcpp, ['spine_top', 'ax_edge_top'], spines),
                            twin_x=kwargs.get('twin_x', False),
                            twin_y=kwargs.get('twin_y', False),
                            )
        # auto-boxplot size option
        if self.axes.size == 'auto':
            self.box_scale = 'auto'
            self.axes.size = [400, 400]

        # twinned axes
        twinned = kwargs.get('twin_x', False) or kwargs.get('twin_y', False)
        if not twinned:
            self.axes2 = Element('ax', self.fcpp, kwargs, on=False,
                                 scale=kwargs.get('ax2_scale', None))
            return kwargs

        self.axes2 = Element('ax', self.fcpp, kwargs,
                             on=True if twinned else False,
                             edge_color=self.axes.edge_color,
                             fill_color=self.axes.fill_color,
                             primary=False,
                             scale=kwargs.get('ax2_scale', None),
                             xmin=kwargs.get('x2min', None),
                             xmax=kwargs.get('x2max', None),
                             ymin=kwargs.get('y2min', None),
                             ymax=kwargs.get('y2max', None),
                             )

        return kwargs

    def _init_axes_labels(self, kwargs: dict) -> dict:
        """Set the axes label elements parameters except for text related
        parameters which are set later in self._set_label_text (to make sure
        any updates after init of data obj are included)

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Axes labels
        label = Element('label', self.fcpp, kwargs,
                        obj=self.obj_array,
                        font_style='italic',
                        font_weight='bold',
                        bg_padding=utl.kwget(kwargs, self.fcpp, 'label_bg_padding', 2),
                        )
        labels = ['x', 'x2', 'y', 'y2', 'z']
        rotations = [0, 0, 90, 270, 270]
        for ilab, lab in enumerate(labels):
            # Copy base label object and set default rotation
            setattr(self, 'label_%s' % lab, copy.deepcopy(label))
            getattr(self, 'label_%s' % lab).rotation = rotations[ilab]

            # Override params
            keys = [f for f in kwargs.keys() if 'label_%s' % lab in f]
            for k in keys:
                v = kwargs[k]
                if k == 'label_%s' % lab or '_text' in k:
                    continue  # k = 'label_%s_text' % lab
                setattr(getattr(self, 'label_%s' % lab), k.replace('label_%s_' % lab, ''), v)

            # Update alphas
            getattr(self, 'label_%s' % lab).color_alpha('fill_color', 'fill_alpha')
            getattr(self, 'label_%s' % lab).color_alpha('edge_color', 'edge_alpha')

        # Turn off secondary labels
        if not self.axes.twin_y:
            self.label_x2.on = False
        if not self.axes.twin_x:
            self.label_y2.on = False

        # Twinned label colors
        if 'legend' not in kwargs.keys():
            self.color_list_unique = pd.Series(self.color_list).unique()
            if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
                self.label_y.font_color = self.color_list_unique[0]
            if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
                self.label_y2.font_color = self.color_list_unique[1]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x.font_color = self.color_list_unique[0]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x2.font_color = self.color_list_unique[1]

        return kwargs

    def _init_axhvlines(self, kwargs: dict) -> dict:
        """Set the vertical/horizontal line element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Axhlines/axvlines
        axlines = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']
        # Todo: list
        for axline in axlines:
            val = kwargs.get(axline, False)
            if not isinstance(val, tuple):
                vals = utl.validate_list(val)
            else:
                vals = [val]
            values = []
            colors = []
            styles = []
            widths = []
            alphas = []
            labels = []
            for ival, val in enumerate(vals):
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 1:
                    values += [val[0]]
                else:
                    values += [val]
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 1:
                    colors += [val[1]]
                else:
                    colors += [utl.kwget(kwargs, self.fcpp, '%s_color' % axline, '#000000')]
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 2:
                    styles += [val[2]]
                else:
                    styles += [utl.kwget(kwargs, self.fcpp, '%s_style' % axline, '-')]
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 3:
                    widths += [val[3]]
                else:
                    widths += [utl.kwget(kwargs, self.fcpp, '%s_width' % axline, 1)]
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 4:
                    alphas += [val[4]]
                else:
                    alphas += [utl.kwget(kwargs, self.fcpp, '%s_alpha' % axline, 1)]
                if (isinstance(val, list) or isinstance(val, tuple)) and len(val) > 5:
                    labels += [val[5]]
                elif (isinstance(val, list) or isinstance(val, tuple)) and isinstance(val[0], str):
                    labels += [val[0]]
                else:
                    labels += [utl.kwget(kwargs, self.fcpp, '%s_label' % axline, None)]
            by_plot = utl.kwget(kwargs, self.fcpp, '%s_by_plot' % axline, False)
            if by_plot:
                labels = labels[0]

            setattr(self, axline,
                    Element(axline, self.fcpp, kwargs,
                            on=True if axline in kwargs.keys() else False,
                            values=values, color=colors, style=styles,
                            width=widths, alpha=alphas, text=labels,
                            by_plot=utl.kwget(kwargs, self.fcpp, '%s_by_plot' % axline, False),
                            zorder=utl.kwget(kwargs, self.fcpp, '%s_zorder' % axline, 1),
                            ))

        return kwargs

    def _init_bar(self, kwargs: dict) -> dict:
        """Set the bar plot element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'bar':
            self.bar = Element('bar', self.fcpp, kwargs, on=False,
                               horizontal=False)
            return kwargs

        self.bar = Element('bar', self.fcpp, kwargs,
                           on=True,
                           width=utl.kwget(kwargs, self.fcpp, ['bar_width', 'width'],
                                           kwargs.get('width', 0.8)),
                           align=utl.kwget(kwargs, self.fcpp, 'bar_align',
                                           kwargs.get('align', 'center')),
                           edge_color=utl.kwget(kwargs, self.fcpp, 'bar_edge_color',
                                                copy.copy(self.color_list)),
                           edge_width=utl.kwget(kwargs, self.fcpp, 'bar_edge_width', 0),
                           fill_alpha=utl.kwget(kwargs, self.fcpp, 'bar_fill_alpha', 0.75),
                           fill_color=utl.kwget(kwargs, self.fcpp, 'bar_fill_color',
                                                copy.copy(self.color_list)),
                           horizontal=utl.kwget(kwargs, self.fcpp, ['bar_horizontal', 'horizontal'],
                                                kwargs.get('horizontal', False)),
                           stacked=utl.kwget(kwargs, self.fcpp, ['bar_stacked', 'stacked'],
                                             kwargs.get('stacked', False)),
                           error_bars=utl.kwget(kwargs, self.fcpp, ['bar_error_bars', 'error_bars'],
                                                kwargs.get('error_bars', False)),
                           error_color=utl.kwget(kwargs, self.fcpp, ['bar_error_color', 'error_color'],
                                                 kwargs.get('error_color', '#555555')),
                           color_by_bar=utl.kwget(kwargs, self.fcpp, ['bar_color_by_bar', 'color_by_bar'],
                                                  kwargs.get('color_by_bar', False)),
                           )
        if 'colors' in kwargs.keys():
            self.bar.color_by_bar = True

        # rolling mean options
        rolling = utl.kwget(kwargs, self.fcpp, ['bar_rolling_mean', 'rolling_mean',
                                                'bar_rolling', 'rolling'], False)
        if isinstance(rolling, int):
            self.rolling_mean = Element('rolling_mean', self.fcpp, kwargs,
                                        on=utl.kwget(kwargs, self.fcpp, 'rolling_mean', False),
                                        color=utl.kwget(kwargs, self.fcpp,
                                                        ['rolling_mean_line_color', 'rolling_mean_color'],
                                                        DEFAULT_COLORS[1]),
                                        width=utl.kwget(kwargs, self.fcpp, 'rolling_mean_line_width', 2),
                                        window=rolling,
                                        )
            if not kwargs.get('markers', False):
                self.markers.on = False
            else:
                self.markers.edge_color = self.rolling_mean.color

        # need to fix all the kwargs only defaults!!
        return kwargs

    def _init_box(self, kwargs: dict) -> dict:
        """Set the box plot element parameters.  Creates the following box-plot
        required elements:

        * box: main element
        * box_grand_mean: element to control style of overall mean value of plot
        * box_grand_median: element to control style of overall median value of plot
        * box_mean_diamonds: element to control style of discrete box mean diamonds
        * box_whisker: element to control style of box whisker lines
        * box_divider: element to control style of divider lines between groups in box plot
        * box_range_lines: element to control style of range lines from a given box
        * box_stat_line: element to control style of stat line connecting various boxes in a plot
        * box_group_title: element to control style of the title text of a grouping row
        * box_group_label: element to control style of the value label for a given group
            (displayed under the box)
        * violin: element to control style of violin plots if used instead of boxes

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'box':
            self.box = Element('box', self.fcpp, kwargs, on=False)
            self.box_grand_mean = Element('box_grand_mean', self.fcpp, kwargs, on=False)
            self.box_grand_median = Element('box_grand_median', self.fcpp, kwargs, on=False)
            self.box_group_means = Element('box_group_means', self.fcpp, kwargs, on=False)
            self.box_mean_diamonds = Element('box_mean_diamonds', self.fcpp, kwargs, on=False)
            self.box_whisker = Element('box_whisker', self.fcpp, kwargs, on=False)
            self.box_divider = Element('box_divider', self.fcpp, kwargs, on=False)
            self.box_range_lines = Element('box_range_lines', self.fcpp, kwargs, on=False)
            self.box_stat_line = Element('box_stat_line', self.fcpp, kwargs, on=False)
            self.box_group_title = Element('box_group_title', self.fcpp, kwargs, on=False)
            self.box_group_label = Element('box_group_label', self.fcpp, kwargs, on=False)
            self.violin = Element('violin', self.fcpp, kwargs, on=False)
            return kwargs

        self.box_group_title = Element('box_group_title', self.fcpp, kwargs,
                                       on=True if kwargs.get('box_labels_on', True) else False,
                                       obj=self.obj_array,
                                       font_color=utl.kwget(kwargs, self.fcpp, 'box_group_title_font_color', '#666666'),
                                       font_size=utl.kwget(kwargs, self.fcpp, 'box_group_title_font_size', 13),
                                       padding=utl.kwget(kwargs, self.fcpp, 'box_group_title_padding', 10),  # percent
                                       )
        self.box_group_label = Element('box_group_label', self.fcpp, kwargs,
                                       align={},
                                       on=True if 'box' in self.name and kwargs.get('box_labels_on', True) else False,
                                       obj=self.obj_array,
                                       edge_color=utl.kwget(kwargs, self.fcpp, 'box_group_label_edge_color', '#aaaaaa'),
                                       font_color=utl.kwget(kwargs, self.fcpp, 'box_group_label_font_color', '#666666'),
                                       font_size=utl.kwget(kwargs, self.fcpp, 'box_group_label_font_size', 12),
                                       padding=utl.kwget(kwargs, self.fcpp, 'box_group_label_padding', 10),  # percent
                                       rotation=utl.kwget(kwargs, self.fcpp, 'box_group_label_rotation', 0),
                                       )

        # Must have groups to have group labels
        if 'groups' not in kwargs.keys():
            self.box_group_title.on = False
            self.box_group_label.on = False

        # Other boxplot elements
        self.violin = Element('violin', self.fcpp, kwargs,
                              on=utl.kwget(kwargs, self.fcpp, ['box_violin', 'violin'], kwargs.get('violin', False)),
                              box_color=utl.kwget(kwargs, self.fcpp, 'violin_box_color', '#555555'),
                              box_on=utl.kwget(kwargs, self.fcpp, 'violin_box_on', True),
                              edge_color=utl.kwget(kwargs, self.fcpp, 'violin_edge_color', '#aaaaaa'),
                              fill_alpha=utl.kwget(kwargs, self.fcpp, 'violin_fill_alpha', 0.5),
                              fill_color=kwargs.get('color', utl.kwget(kwargs, self.fcpp,
                                                    'violin_fill_color', DEFAULT_COLORS[0])),
                              markers=kwargs.get('markers', utl.kwget(kwargs, self.fcpp,
                                                                      'violin_markers', False)),
                              median_color=utl.kwget(kwargs, self.fcpp,
                                                     'violin_median_color', '#ffffff'),
                              median_marker=utl.kwget(kwargs, self.fcpp,
                                                      'violin_median_marker', 'o'),
                              median_size=utl.kwget(kwargs, self.fcpp,
                                                    'violin_median_size', 2),
                              whisker_color=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_color', '#555555'),
                              whisker_style=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_style', '-'),
                              whisker_width=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_width', 1.5),
                              )

        self.box = Element('box', self.fcpp, kwargs,
                           on=True if 'box' in self.name and kwargs.get('box_on', True) else False,
                           edge_color=utl.kwget(kwargs, self.fcpp, 'box_edge_color', '#aaaaaa'),
                           edge_width=utl.kwget(kwargs, self.fcpp, 'box_edge_width', 0.5),
                           fill_color=utl.kwget(kwargs, self.fcpp, 'box_fill_color', '#ffffff'),
                           median_color=utl.kwget(kwargs, self.fcpp, 'box_median_color', '#ff7f0e'),
                           notch=utl.kwget(kwargs, self.fcpp, ['box_notch', 'notch'],
                                           kwargs.get('notch', False)),
                           width=utl.kwget(kwargs, self.fcpp, ['box_width', 'width'],
                                           kwargs.get('width', 0.5 if not self.violin.on else 0.15)),
                           )

        self.box_grand_mean = Element('box_grand_mean', self.fcpp, kwargs,
                                      on=utl.kwget(kwargs, self.fcpp,
                                                   ['box_grand_mean', 'grand_mean'],
                                                   kwargs.get('grand_mean', False)),
                                      color=utl.kwget(kwargs, self.fcpp,
                                                      ['box_grand_mean_color', 'grand_mean_color'],
                                                      kwargs.get('grand_mean_color', '#555555')),
                                      style=utl.kwget(kwargs, self.fcpp,
                                                      ['box_grand_mean_style', 'grand_mean_style'],
                                                      kwargs.get('grand_mean_style', '--')),
                                      width=utl.kwget(kwargs, self.fcpp,
                                                      ['box_grand_mean_width', 'grand_mean_width'],
                                                      kwargs.get('grand_mean_width', 1)),
                                      zorder=30)

        self.box_grand_median = Element('box_grand_median', self.fcpp, kwargs,
                                        on=utl.kwget(kwargs, self.fcpp,
                                                     ['box_grand_median', 'grand_median'],
                                                     kwargs.get('grand_median', False)),
                                        color=utl.kwget(kwargs, self.fcpp,
                                                        ['box_grand_median_color', 'grand_median_color'],
                                                        kwargs.get('grand_median_color', '#0000ff')),
                                        style=utl.kwget(kwargs, self.fcpp,
                                                        ['box_grand_median_style', 'grand_median_style'],
                                                        kwargs.get('grand_median_style', '--')),
                                        width=utl.kwget(kwargs, self.fcpp,
                                                        ['box_grand_median_width', 'grand_median_width'],
                                                        kwargs.get('grand_median_width', 1)),
                                        zorder=30)

        self.box_group_means = Element('box_group_means', self.fcpp, kwargs,
                                       on=utl.kwget(kwargs, self.fcpp,
                                                    ['box_group_means', 'group_means'],
                                                    kwargs.get('group_means', False)),
                                       color=utl.kwget(kwargs, self.fcpp,
                                                       ['box_group_means_color', 'group_means_color'],
                                                       kwargs.get('group_means_color', '#FF00FF')),
                                       style=utl.kwget(kwargs, self.fcpp,
                                                       ['box_group_means_style', 'group_means_style'],
                                                       kwargs.get('group_means_style', '--')),
                                       width=utl.kwget(kwargs, self.fcpp,
                                                       ['box_group_means_width', 'group_means_width'],
                                                       kwargs.get('group_means_width', 1)),
                                       zorder=30)

        self.box_mean_diamonds = Element('box_mean_diamonds', self.fcpp, kwargs,
                                         on=utl.kwget(kwargs, self.fcpp, ['box_mean_diamonds', 'mean_diamonds'],
                                                      kwargs.get('mean_diamonds', False)),
                                         alpha=utl.kwget(kwargs, self.fcpp,
                                                         ['box_mean_diamonds_alpha', 'mean_diamonds_alpha'],
                                                         kwargs.get('mean_diamonds_alpha', 1)),
                                         conf_coeff=utl.kwget(kwargs, self.fcpp, 'conf_coeff', 0.95),
                                         edge_color=utl.kwget(kwargs, self.fcpp,
                                                              ['box_mean_diamonds_edge_color',
                                                               'mean_diamonds_edge_color'],
                                                              kwargs.get('mean_diamonds_edge_color', '#00FF00')),
                                         edge_style=utl.kwget(kwargs, self.fcpp,
                                                              ['box_mean_diamonds_edge_style',
                                                               'mean_diamonds_edge_style'],
                                                              kwargs.get('mean_diamonds_edge_style', '-')),
                                         edge_width=utl.kwget(kwargs, self.fcpp,
                                                              ['box_mean_diamonds_edge_width',
                                                               'mean_diamonds_edge_width'],
                                                              kwargs.get('mean_diamonds_edge_width', 0.7)),
                                         fill_color=utl.kwget(kwargs, self.fcpp,
                                                              ['box_mean_diamonds_fill_color',
                                                               'mean_diamonds_fill_color'],
                                                              kwargs.get('mean_diamonds_fill_color', None)),
                                         width=utl.kwget(kwargs, self.fcpp,
                                                         ['box_mean_diamonds_width',
                                                          'mean_diamonds_width'],
                                                         kwargs.get('mean_diamonds_width', 0.8)),
                                         zorder=30)

        self.box_whisker = Element('box_whisker', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp, ['box_whisker', 'whisker'], self.box.on),
                                   color=utl.kwget(kwargs, self.fcpp, ['box_whisker_color'], self.box.edge_color),
                                   style=utl.kwget(kwargs, self.fcpp, ['box_whisker_style'], self.box.style),
                                   width=utl.kwget(kwargs, self.fcpp, ['box_whisker_width'], self.box.edge_width),
                                   )
        if not self.box_whisker.on:
            self.box_whisker.width.values = [0]

        self.box_stat_line = \
            Element('box_stat_line', self.fcpp, kwargs,
                    on=True if 'box' in self.name and kwargs.get('box_stat_line_on', True) else False,
                    color=utl.kwget(kwargs, self.fcpp, 'box_stat_line_color', '#666666'),
                    stat=kwargs.get('box_stat_line', 'mean'),
                    zorder=utl.kwget(kwargs, self.fcpp, 'box_stat_line_zorder', 7),
                    )

        self.box_divider = Element('box_divider', self.fcpp, kwargs,
                                   on=kwargs.get('box_divider', kwargs.get('box', True)),
                                   color=utl.kwget(kwargs, self.fcpp,
                                                   ['box_divider_color', 'box_divider_line_color'],
                                                   '#bbbbbb'),
                                   text=None,
                                   zorder=2,
                                   )

        self.box_range_lines = Element('box_range_lines', self.fcpp, kwargs,
                                       on=kwargs.get('box_range_lines', not kwargs.get('violin', False)),
                                       color=utl.kwget(kwargs, self.fcpp, 'box_range_lines_color', '#cccccc'),
                                       style=utl.kwget(kwargs, self.fcpp, 'box_range_lines_style', '-'),
                                       style2=RepeatedList('--', 'style2'),
                                       zorder=utl.kwget(kwargs, self.fcpp, 'box_range_lines', 3),
                                       )
        if 'box' in self.name:
            self.lines.on = False
        if self.violin.on:
            self.markers.on = self.violin.markers

        if 'box' in self.name:
            # edge color
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_edge_color') \
                    and not self.legend._on:
                self.markers.edge_color = self.color_list[1]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            elif not kwargs.get('colors') and not kwargs.get('marker_edge_color'):
                self.markers.edge_color = self.color_list[1:] + [self.color_list[0]]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_fill_color') \
                    and not self.legend._on:
                self.markers.fill_color = DEFAULT_COLORS[1]
                self.markers.color_alpha('fill_color', 'fill_alpha')
            elif not kwargs.get('colors'):
                self.markers.fill_color = self.color_list[1:] + [self.color_list[0]]
                self.markers.color_alpha('fill_color', 'fill_alpha')

            # For some marker attributes, prioritize kwargs which can have keywords with/without prefix box, then
            # check self.fcpp only for those keywords that are prefixed by box
            self.markers.edge_width = \
                utl.kwget(kwargs, {}, ['box_marker_edge_width', 'marker_edge_width'],
                          utl.kwget(self.fcpp, {}, 'box_marker_edge_width', self.markers.edge_width))
            self.markers.edge_alpha = \
                utl.kwget(kwargs, {}, ['box_marker_edge_alpha', 'marker_edge_alpha'],
                          utl.kwget(self.fcpp, {}, 'box_marker_edge_alpha', self.markers.edge_alpha))
            self.markers.edge_color = \
                utl.kwget(kwargs, {}, ['box_marker_edge_color', 'marker_edge_color'],
                          utl.kwget(self.fcpp, {}, 'box_marker_edge_color', self.markers.edge_color))
            self.markers.color_alpha('edge_color', 'edge_alpha')

            self.markers.fill_alpha = \
                utl.kwget(kwargs, {}, ['box_marker_fill_alpha', 'marker_fill_alpha'],
                          utl.kwget(self.fcpp, {}, 'box_marker_fill_alpha', self.markers.fill_alpha))
            self.markers.fill_color = \
                utl.kwget(kwargs, {}, ['box_marker_fill_color', 'marker_fill_color'],
                          utl.kwget(self.fcpp, {}, 'box_marker_fill_color', self.markers.fill_color))
            self.markers.color_alpha('fill_color', 'fill_alpha')

            self.markers.filled = \
                utl.kwget(kwargs, {}, ['box_marker_fill', 'marker_fill'],
                          utl.kwget(self.fcpp, {}, 'box_marker_fill', self.markers.filled))
            self.markers.size = \
                utl.kwget(kwargs, {}, ['box_marker_size', 'marker_size'],
                          utl.kwget(self.fcpp, {}, 'box_marker_size', 4))
            self.markers.zorder = \
                utl.kwget(kwargs, {}, ['box_marker_zorder', 'marker_zorder'],
                          utl.kwget(self.fcpp, {}, 'box_marker_zorder', self.markers.zorder))
            self.markers.jitter = utl.kwget(kwargs, self.fcpp, 'jitter', True)

            # treat the marker type
            if 'box_marker_type' in kwargs.keys():
                self.markers.type = RepeatedList(kwargs['box_marker_type'], 'box_marker_type')
            elif 'marker_type' in kwargs.keys():
                self.markers.type = RepeatedList(kwargs['marker_type'], 'marker_type')
            elif 'box_marker_type' in self.fcpp.keys():
                self.markers.type = RepeatedList(self.fcpp['box_marker_type'], 'marker_type')
            elif not self.legend._on:
                self.markers.type = RepeatedList('o', 'marker_type')

            # convert some attributes to RepeatedList
            vals = ['size', 'edge_width', 'edge_color', 'fill_color']
            for val in vals:
                if not isinstance(getattr(self.markers, val), RepeatedList):
                    setattr(self.markers, val, RepeatedList(getattr(self.markers, val), val))

        return kwargs

    def _init_cbar(self, kwargs: dict) -> dict:
        """Set the color bar element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        cbar_size = utl.kwget(kwargs, self.fcpp, 'cbar_size', 30)
        self.cbar = Element('cbar', self.fcpp, kwargs,
                            on=kwargs.get('cbar', False),
                            obj=self.obj_array,
                            size=[cbar_size if not isinstance(cbar_size, list) else cbar_size[0], self.axes.size[1]],
                            )
        if not self.cbar.on:
            self.label_z.on = False
            self.tick_labels_major_z.on = False

        return kwargs

    def _init_colors(self, kwargs: dict) -> dict:
        """Set the color elements (color_list, cmap).

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Set the color list which determines line color order
        if kwargs.get('colors'):
            colors = utl.validate_list(kwargs.get('colors'))
            for icolor, color in enumerate(colors):
                if isinstance(color, int):
                    colors[icolor] = DEFAULT_COLORS[color]
            self.color_list = colors
        elif not self.color_list:
            self.color_list = copy.copy(DEFAULT_COLORS)

        # Set the cmap
        self.cmap = kwargs.get('cmap', None)
        if self.name in ['contour', 'heatmap', 'imshow']:
            self.cmap = utl.kwget(kwargs, self.fcpp, 'cmap', 'inferno')

        # Program any legend value specific color overrides
        vals = ['fill_alpha', 'fill_color', 'edge_alpha', 'edge_color', 'color']
        for val in vals:
            if '%s_override' % val in kwargs.keys():
                kwargs['%s_override' % val] = utl.kwget(kwargs, self.fcpp, '%s_override' % val, {})
            else:
                kwargs['%s_override' % val] = utl.kwget(kwargs, self.fcpp, 'color_override', {})

        return kwargs

    def _init_control_limit(self, kwargs: dict) -> dict:
        """_summary_

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        keys = ['lcl', 'ucl']
        for key in keys:
            setattr(self, key, Element(key, self.fcpp, kwargs,
                                       on=True if kwargs.get(key, False) else False,
                                       edge_color=utl.kwget(kwargs, self.fcpp, f'{key}_edge_color',
                                                            copy.copy(self.color_list)),
                                       edge_alpha=utl.kwget(kwargs, self.fcpp, f'{key}_edge_alpha', 0.25),
                                       edge_style=utl.kwget(kwargs, self.fcpp, f'{key}_edge_style', '-'),
                                       edge_width=utl.kwget(kwargs, self.fcpp, f'{key}_edge_width', 1),
                                       fill_color=utl.kwget(kwargs, self.fcpp, f'{key}_fill_color',
                                                            copy.copy(self.color_list)),
                                       fill_alpha=utl.kwget(kwargs, self.fcpp, f'{key}_fill_alpha', 0.2),
                                       value=utl.validate_list(kwargs.get(key, None)),
                                       key=key,
                                       ))
        self.control_limit_side = utl.kwget(kwargs, self.fcpp, 'control_limit_side', 'outside')

        return kwargs

    def _init_contour(self, kwargs: dict) -> dict:
        """Set the contour plot element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'contour':
            self.contour = Element('contour', self.fcpp, kwargs, on=False)
            return kwargs

        self.contour = Element('contour', self.fcpp, kwargs,
                               on=True,
                               filled=utl.kwget(kwargs, self.fcpp, ['contour_filled', 'filled'],
                                                kwargs.get('filled', True)),
                               levels=utl.kwget(kwargs, self.fcpp, ['contour_levels', 'levels'],
                                                kwargs.get('levels', 20)),
                               interp=utl.kwget(kwargs, self.fcpp, ['contour_interp', 'interp'],
                                                kwargs.get('interp', 'cubic')),
                               show_points=utl.kwget(kwargs, self.fcpp, ['contour_show_points', 'show_points'],
                                                     kwargs.get('show_points', False)),
                               )

        return kwargs

    def _init_figure(self, kwargs: dict) -> dict:
        """Set the figure element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        self.fig = Element('fig', self.fcpp, kwargs, edge_width=3)

        return kwargs

    def _init_fit(self, kwargs: dict) -> dict:
        """Set the line fit element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        self.fit = Element('fit', self.fcpp, kwargs,
                           on=True if kwargs.get('fit', False) else False,
                           color=utl.kwget(kwargs, self.fcpp, ['fit_color', 'fit_line_color'], '#000000'),
                           conf_band=utl.kwget(kwargs, self.fcpp, ['fit_conf_band', 'conf_band'], False),
                           edge_color='none',
                           eqn=utl.kwget(kwargs, self.fcpp, 'fit_eqn', False),
                           fill_color='none',
                           font_size=utl.kwget(kwargs, self.fcpp, 'fit_font_size', 12),
                           padding=utl.kwget(kwargs, self.fcpp, 'fit_padding', 10),
                           rsq=utl.kwget(kwargs, self.fcpp, 'fit_rsq', False),
                           size=[0, 0],
                           )

        self.fit.legend_text = utl.kwget(kwargs, self.fcpp, 'fit_legend_text', None)
        self.fit.position[0] = self.fit.padding
        self.fit.position[1] = self.axes.size[1] - \
            (self.fit.padding + self.fit.font_size)

        return kwargs

    def _init_gantt(self, kwargs: dict) -> dict:
        """Set the gantt chart element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'gantt':
            self.gantt = Element('gantt', self.fcpp, kwargs, on=False)
            return kwargs

        self.gantt = Element('gantt', self.fcpp, kwargs,
                             on=True,
                             height=utl.kwget(kwargs, self.fcpp, 'gantt_height', kwargs.get('height', 0.9)),
                             sort=utl.kwget(kwargs, self.fcpp, 'sort', 'descending'),
                             edge_color=utl.kwget(kwargs, self.fcpp, 'gantt_edge_color', copy.copy(self.color_list)),
                             edge_width=utl.kwget(kwargs, self.fcpp, 'gantt_edge_width', 0),
                             fill_alpha=utl.kwget(kwargs, self.fcpp, 'gantt_fill_alpha', 0.75),
                             fill_color=utl.kwget(kwargs, self.fcpp, 'gantt_fill_color', copy.copy(self.color_list)),
                             color_by_bar=utl.kwget(kwargs, self.fcpp, ['gantt_color_by_bar', 'color_by_bar'],
                                                    kwargs.get('color_by_bar', False)),
                             order_by_legend=utl.kwget(kwargs, self.fcpp, ['gantt_order_by_legend', 'order_by_legend'],
                                                       kwargs.get('order_by_legend', False)),
                             label_x=utl.kwget(kwargs, self.fcpp, 'gantt_label_x', kwargs.get('gantt_label_x', '')),
                             tick_labels_x_rotation=utl.kwget(kwargs, self.fcpp, 'gantt_tick_labels_x_rotation',
                                                              kwargs.get('gantt_tick_labels_x_rotation', 90)),
                             )
        if self.gantt.on and \
                ('tick_labels_major_rotation' not in kwargs.keys()
                 or 'tick_labels_major_x_rotation' not in kwargs.keys()
                 or 'tick_labels_x_rotation' not in kwargs.keys()):
            self.tick_labels_major_x.rotation = self.gantt.tick_labels_x_rotation
        if self.gantt.on and \
                ('grid_major' not in kwargs.keys()
                 or 'grid_major_y' not in kwargs.keys()):
            kwargs['grid_major_y'] = False
        if self.gantt.on and \
                'grid_minor_y' not in kwargs.keys():
            kwargs['grid_minor_y'] = False
        if self.gantt.on and \
                ('ticks_major' not in kwargs.keys()
                 or 'ticks_major_y' not in kwargs.keys()):
            self.ticks_major_y.on = False
        if self.gantt.on and 'label_x' not in kwargs.keys():
            self.label_x.text = self.gantt.label_x
            kwargs['label_x'] = self.gantt.label_x
            self.label_x.on = False  # disable unless explicitly added

        return kwargs

    def _init_grid(self, kwargs: dict) -> dict:
        """Set the gridline element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Major gridlines
        self.grid_major = Element('grid_major', self.fcpp, kwargs,
                                  on=kwargs.get('grid_major', True),
                                  color=utl.kwget(kwargs, self.fcpp, 'grid_major_color', '#ffffff'),
                                  style=utl.kwget(kwargs, self.fcpp, 'grid_major_style', '-'),
                                  width=utl.kwget(kwargs, self.fcpp, 'grid_major_width', 1.3),
                                  )
        secondary = ['y2'] if kwargs.get('grid_major_y2') is True else [] + \
                    ['x2'] if kwargs.get('grid_major_x2') is True else []
        for ax in ['x', 'y'] + secondary:
            # secondary axes cannot get the grid
            setattr(self, 'grid_major_%s' % ax,
                    Element('grid_major_%s' % ax, self.fcpp, kwargs,
                            on=kwargs.get('grid_major_%s' % ax, self.grid_major.on),
                            color=utl.kwget(kwargs, self.fcpp, f'grid_major_{ax}_color', self.grid_major.color),
                            style=utl.kwget(kwargs, self.fcpp, f'grid_major_{ax}_style', self.grid_major.style),
                            width=utl.kwget(kwargs, self.fcpp, f'grid_major_{ax}_width', self.grid_major.width),
                            zorder=utl.kwget(kwargs, self.fcpp, f'grid_major_{ax}_zorder', self.grid_major.zorder),
                            ))
            if getattr(getattr(self, 'grid_major_%s' % ax), 'on') \
                    and ('ticks' not in kwargs.keys() or kwargs['ticks'] is not False) \
                    and ('ticks_%s' % ax not in kwargs.keys() or kwargs['ticks_%s' % ax] is not False) \
                    and ('ticks_major' not in kwargs.keys() or kwargs['ticks_major'] is not False) \
                    and ('ticks_major_%s' % ax not in kwargs.keys() or kwargs['ticks_major_%s' % ax] is not False):
                setattr(getattr(self, 'ticks_major_%s' % ax), 'on', True)

        # Minor gridlines
        self.grid_minor = Element('grid_minor', self.fcpp, kwargs,
                                  on=kwargs.get('grid_minor', False),
                                  color=utl.kwget(kwargs, self.fcpp, 'grid_minor_color', '#ffffff'),
                                  width=utl.kwget(kwargs, self.fcpp, 'grid_minor_width', 0.5),
                                  )
        secondary = ['y2'] if kwargs.get('grid_major_y2') is True else [] + \
                    ['x2'] if kwargs.get('grid_major_x2') is True else []
        for ax in ['x', 'y'] + secondary:
            # secondary axes cannot get the grid
            setattr(self, 'grid_minor_%s' % ax,
                    Element('grid_minor_%s' % ax, self.fcpp, kwargs,
                            on=kwargs.get('grid_minor_%s' % ax, self.grid_minor.on),
                            color=utl.kwget(kwargs, self.fcpp, 'grid_minor_color_%s' % ax, self.grid_minor.color),
                            style=utl.kwget(kwargs, self.fcpp, f'grid_minor_{ax}_style', self.grid_minor.style),
                            width=utl.kwget(kwargs, self.fcpp, f'grid_minor_{ax}_width', self.grid_minor.width),
                            zorder=utl.kwget(kwargs, self.fcpp, f'grid_minor_{ax}_zorder', self.grid_minor.zorder),
                            ))
            if getattr(self, 'grid_minor_%s' % ax).on and \
                    ('ticks' not in kwargs.keys() or kwargs['ticks'] is not False) and \
                    ('ticks_minor' not in kwargs.keys() or kwargs['ticks_minor'] is not False) and \
                    ('ticks_minor_%s' % ax not in kwargs.keys() or kwargs['ticks_minor_%s' % ax] is not False):
                getattr(self, 'ticks_minor_%s' % ax).on = True

        return kwargs

    def _init_heatmap(self, kwargs: dict, kwargs_orig: dict, data: 'data.Data') -> dict:
        """Set the heatmap element parameters.

        Args:
            kwargs: user-defined keyword args (modified by other _init functions)
            kwargs_orig: original, unmodified user-defined keyword args
            data: Data class object for the plot

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'heatmap':
            self.heatmap = Element('heatmap', self.fcpp, kwargs, on=False)
            return kwargs

        if 'cell_size' in kwargs.keys():
            kwargs['heatmap_cell_size'] = kwargs['cell_size']
        self.heatmap = Element('heatmap', self.fcpp, kwargs,
                               on=True,
                               cell_size=utl.kwget(kwargs, self.fcpp, 'heatmap_cell_size',
                                                   60 if 'ax_size' not in kwargs else None),
                               edge_width=0,
                               font_color=utl.kwget(kwargs, self.fcpp, 'heatmap_font_color', '#ffffff'),
                               font_size=12,
                               interp=utl.kwget(kwargs, self.fcpp, ['heatmap_interp', 'interp'],
                                                kwargs.get('interp', 'none')),
                               text=utl.kwget(kwargs, self.fcpp, 'data_labels', False),
                               )
        if self.heatmap.on and data.x != ['Column']:
            self.tick_labels_major_x.rotation = utl.kwget(kwargs, self.fcpp, 'tick_labels_major_x', 90)

        # Enable cbars by default
        if self.heatmap.on and kwargs.get('cbar', True):
            self.cbar.on = True
            self.label_z.on = True
            self.tick_labels_major_z.on = True

        # Special gridline/tick/axes defaults for heatmap
        grids = [f for f in kwargs.keys() if f in
                 ['grid_major', 'grid_major_x', 'grid_major_y',
                  'grid_minor', 'grid_minor_x', 'grid_minor_y']]
        if len(grids) == 0:
            kwargs['grid_major'] = False
            kwargs['grid_minor'] = False
            kwargs['ticks_major'] = True
        if 'ax_edge_width' not in kwargs.keys():
            self.axes.edge_width = 0
        if 'x' in kwargs.keys():
            kwargs['tick_cleanup'] = False
        self.ticks_major_x.on = \
            utl.kwget(kwargs_orig, self.fcpp,
                      ['ticks_major_x', 'ticks_major'], False)
        self.ticks_major_y.on = \
            utl.kwget(kwargs_orig, self.fcpp,
                      ['ticks_major_y', 'ticks_major'], False)

        return kwargs

    def _init_hist(self, kwargs: dict) -> dict:
        """Set the hist and kde plot element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'hist':
            self.hist = Element('hist', self.fcpp, kwargs, on=False,
                                cdf=utl.kwget(kwargs, self.fcpp, ['cdf'],
                                              kwargs.get('cdf', False)),
                                horizontal=False)
            return kwargs

        self.hist = Element('hist', self.fcpp, kwargs,
                            on=True if 'hist' in self.name and kwargs.get('hist_on', True) else False,
                            align=utl.kwget(kwargs, self.fcpp, 'hist_align', 'mid'),
                            bins=utl.kwget(kwargs, self.fcpp, ['hist_bins', 'bins'], kwargs.get('bins', 20)),
                            edge_color=copy.copy(self.color_list),
                            edge_width=utl.kwget(kwargs, self.fcpp, ['hist_edge_width'], 0),
                            fill_alpha=utl.kwget(kwargs, self.fcpp, ['hist_fill_alpha'], 0.5),
                            fill_color=copy.copy(self.color_list),
                            cumulative=utl.kwget(kwargs, self.fcpp, ['hist_cumulative', 'cumulative'],
                                                 kwargs.get('cumulative', False)),
                            kde=utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'], kwargs.get('kde', False)),
                            normalize=utl.kwget(kwargs, self.fcpp, ['hist_normalize', 'normalize'],
                                                kwargs.get('normalize', False)),
                            rwidth=utl.kwget(kwargs, self.fcpp, 'hist_rwidth', None),
                            horizontal=utl.kwget(kwargs, self.fcpp, ['hist_horizontal', 'horizontal'],
                                                 kwargs.get('horizontal', False)),
                            )

        # kde element defined separately from self.hist to store unique parameters
        self.kde = Element('kde', self.fcpp, kwargs,
                           on=utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'], kwargs.get('kde', False)),
                           color=copy.copy(self.color_list),
                           width=utl.kwget(kwargs, self.fcpp, ['hist_kde_width', 'kde_width'], 2),
                           zorder=5,
                           )
        if self.kde.on:
            self.hist.normalize = True

        return kwargs

    def _init_imshow(self, kwargs, kwargs_orig) -> dict:
        """Set the imshow plot element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'imshow':
            self.imshow = Element('imshow', self.fcpp, kwargs, on=False)
            return kwargs

        self.imshow = Element('imshow', self.fcpp, kwargs,
                              on=True,
                              interp=utl.kwget(kwargs, self.fcpp,
                                               ['imshow_interp', 'interp'],
                                               kwargs.get('interp', 'none')),
                              )

        # unless the color map is explicity defined in kwargs or self.fcpp,
        # override the inferno default for imshow
        if self.imshow.on and not utl.kwget(kwargs, self.fcpp, 'cmap', None):
            self.cmap = 'gray'

        # Special gridline/tick/axes defaults for heatmap
        grids = [f for f in kwargs.keys() if f in
                 ['grid_major', 'grid_major_x', 'grid_major_y',
                  'grid_minor', 'grid_minor_x', 'grid_minor_y']]
        if len(grids) == 0:
            kwargs['grid_major'] = False
            kwargs['grid_minor'] = False
            kwargs['ticks_major'] = True
        if 'ax_edge_width' not in kwargs.keys():
            self.axes.edge_width = 0
        if 'x' in kwargs.keys():
            kwargs['tick_cleanup'] = False
        self.ticks_major_x.on = utl.kwget(kwargs_orig, self.fcpp, ['ticks_major_x', 'ticks_major'], False)
        self.ticks_major_y.on = utl.kwget(kwargs_orig, self.fcpp, ['ticks_major_y', 'ticks_major'], False)
        self.label_x.on = utl.kwget(kwargs_orig, self.fcpp, ['label_x'], False)
        self.label_y.on = utl.kwget(kwargs_orig, self.fcpp, ['label_y'], False)
        self.tick_labels_major_x.rotation = utl.kwget(kwargs_orig, self.fcpp, 'tick_labels_major_x', 0)
        self.tick_labels_major_x.on = \
            utl.kwget(kwargs_orig, self.fcpp, ['tick_labels_major', 'tick_labels_major_x'], False)
        self.tick_labels_major_x.font_size = \
            utl.kwget(kwargs_orig, self.fcpp, ['tick_labels_major_font_size', 'tick_labels_major_x_font_size'], 10)
        self.tick_labels_major_y.on = \
            utl.kwget(kwargs_orig, self.fcpp, ['tick_labels_major', 'tick_labels_major_y'], False)
        self.tick_labels_major_y.font_size = \
            utl.kwget(kwargs_orig, self.fcpp, ['tick_labels_major_font_size', 'tick_labels_major_y_font_size'], 10)
        self.tick_labels_major_z.font_size = \
            utl.kwget(kwargs_orig, self.fcpp, ['tick_labels_major_font_size', 'tick_labels_major_z_font_size'], 10)

        return kwargs

    def _init_intervals(self, kwargs: dict) -> dict:
        """Set the interval element parameters which determines point-by-point
        confidence/percentile/nq intervals around a data point

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        if kwargs.get('perc_int'):
            itype = 'percentile'
            on = True
            key = 'perc_int'
        elif kwargs.get('nq_int'):
            itype = 'nq'
            on = True
            key = 'nq_int'
        elif kwargs.get('conf_int'):
            itype = 'confidence'
            on = True
            key = 'conf_int'
        else:
            itype = None
            on = False
            key = None
        self.interval = Element('interval', self.fcpp, kwargs,
                                on=on,
                                type=itype,
                                edge_color=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_edge_color',
                                                     copy.copy(self.color_list)),
                                edge_alpha=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_edge_alpha',
                                                     0.25),
                                edge_style=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_edge_style', '-'),
                                edge_width=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_edge_width', 1),
                                fill_color=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_fill_color',
                                                     copy.copy(self.color_list)),
                                fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                     f'{key}_fill_alpha',
                                                     0.2),
                                value=utl.validate_list(kwargs.get(key, None)),
                                key=key,
                                )

        # Case of percentages instead of decimals between 0 and 1
        if self.interval.type and self.interval.type != 'nq':
            for ival, val in enumerate(self.interval.value):
                if float(val) > 1:
                    self.interval.value[ival] = val / 100

        # error check
        if itype in ['percentile', 'nq'] and \
                len(utl.validate_list(self.interval.value)) != 2:
            raise ValueError(f'{itype} interval must have only two values (lower and upper limits)')

        return kwargs

    def _init_layout_rc(self, data: 'data.Data') -> dict:
        """Initialize layout attributes for rows and columns

        Args:
            data: Data class object for the plot

        """
        self.ncol = data.ncol
        self.nrow = data.nrow
        self.obj_array = np.array([[None] * self.ncol] * self.nrow)

    def _init_legend(self, kwargs, data: 'data.Data') -> dict:
        """Set the legend element parameters

        Args:
            kwargs: user-defined keyword args
            data: Data class object for the plot

        Returns:
            updated kwargs
        """
        kwargs['legend'] = kwargs.get('legend', None)
        if isinstance(kwargs['legend'], list):
            kwargs['legend'] = ' | '.join(utl.validate_list(kwargs['legend']))
        self.legend = Legend_Element('legend', self.fcpp, kwargs,
                                     on=True if (kwargs.get('legend')
                                                 and kwargs.get('legend_on', True)) else False,
                                     column=kwargs['legend'],
                                     edge_color=utl.kwget(kwargs, self.fcpp,
                                                          'legend_edge_color',
                                                          '#ffffff'),
                                     edge_width=utl.kwget(kwargs, self.fcpp,
                                                          'legend_edge_width',
                                                          1),
                                     font_size=12,
                                     location=LEGEND_LOCATION[utl.kwget(kwargs,
                                                                        self.fcpp, 'legend_location', 0)],
                                     marker_alpha=utl.kwget(kwargs, self.fcpp,
                                                            'legend_marker_alpha',
                                                            1),
                                     marker_size=utl.kwget(kwargs, self.fcpp,
                                                           'legend_marker_size',
                                                           7),
                                     nleg=utl.kwget(kwargs, self.fcpp, 'nleg', -1),
                                     points=utl.kwget(kwargs, self.fcpp,
                                                      'legend_points', 1),
                                     ordered_curves=[],
                                     ordered_fits=[],
                                     ordered_ref_lines=[],
                                     overflow=0,
                                     text=kwargs.get('legend_title',
                                                     kwargs.get('legend') if kwargs.get('legend') is not True else ''),
                                     )

        # For pie plot user must force legend enabled
        if self.legend._on and self.name == 'pie':
            self.legend.on = True

        # Special case: reference line added to legend without legend grouping column
        if not self.legend._on and self.ref_line.on:
            for ref_line_legend_text in self.ref_line.legend_text.values:
                self.legend.values[ref_line_legend_text] = []
            self.legend.on = True
            self.legend.text = ''

        # Special case: fit line
        if not self.legend._on and self.fit.on \
                and not (('legend' in kwargs.keys() and kwargs['legend'] is False)
                         or ('legend_on' in kwargs.keys() and kwargs['legend_on'] is False)):
            self.legend.on = True
            self.legend.text = ''

        # Set legend.values for some specific cases
        y = utl.validate_list(kwargs.get('y'))
        if not self.axes.twin_x and y is not None and len(y) > 1 and \
                self.name != 'box' and \
                (kwargs.get('wrap') != 'y'
                 and kwargs.get('row') != 'y'
                 and kwargs.get('col') != 'y') \
                and kwargs.get('legend') is not False:
            self.legend.values = self.legend.get_default_values_df()
            self.legend.on = True

        return kwargs

    def _init_lines(self, kwargs: dict) -> dict:
        """Set the line element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        for k in list(kwargs.keys()):
            # Update any kwargs with `line_` to `lines_` to match the actual element name
            if 'line_' in k and '%ss_%s' % (k.split('_')[0], k.split('_')[1]) \
                    not in kwargs.keys():
                kwargs['%ss_%s' % (k.split('_')[0], k.split('_')[1])] = kwargs[k]
        self.lines = Element('lines', self.fcpp, kwargs,
                             on=kwargs.get('lines', True),
                             color=RepeatedList(copy.copy(self.color_list), 'colors'),
                             values=[],
                             )

        return kwargs

    def _init_markers(self, kwargs: dict, data: 'data.Data') -> dict:
        """Set the markers element parameters

        Args:
            kwargs: user-defined keyword args
            data: Data class object for the plot

        Returns:
            updated kwargs
        """
        if 'marker_type' in kwargs.keys():
            marker_list = kwargs['marker_type']
        elif kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        markers = RepeatedList(marker_list, 'markers')
        marker_edge_color = utl.kwget(kwargs, self.fcpp, 'marker_edge_color', self.color_list)
        marker_fill_color = utl.kwget(kwargs, self.fcpp, 'marker_fill_color', self.color_list)
        if kwargs.get('marker_fill_color'):
            kwargs['marker_fill'] = True
        self.markers = Element('marker', self.fcpp, kwargs,
                               on=utl.kwget(kwargs, self.fcpp, 'markers', True),
                               filled=utl.kwget(kwargs, self.fcpp, 'marker_fill', False),
                               edge_color=copy.copy(marker_edge_color),
                               edge_width=utl.kwget(kwargs, self.fcpp, 'marker_edge_width', 1.5),
                               fill_color=copy.copy(marker_fill_color),
                               jitter=utl.kwget(kwargs, self.fcpp, ['marker_jitter', 'jitter'],
                                                kwargs.get('jitter', False)),
                               size=utl.kwget(kwargs, self.fcpp, 'marker_size', 6),
                               type=markers,
                               zorder=utl.kwget(kwargs, self.fcpp, 'zorder', 2),
                               )
        if isinstance(self.markers.size, str) and self.markers.size in data.df_all.columns:
            pass
        elif not isinstance(self.markers.size, RepeatedList):
            self.markers.size = RepeatedList(self.markers.size, 'marker_size')
        if not isinstance(self.markers.edge_width, RepeatedList):
            self.markers.edge_width = RepeatedList(self.markers.edge_width, 'marker_edge_width')

        return kwargs

    def _init_pie(self, kwargs: dict) -> dict:
        """Set the pie chart element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # If this plot type is disabled, create minimal set of element parameters
        if self.name != 'pie':
            self.pie = Element('pie', self.fcpp, kwargs, on=False,
                               label_sizes=[(0, 0), (0, 0)])
            self.pie.xs_left = 0
            self.pie.xs_right = 0
            self.pie.xs_top = 0
            self.pie.xs_bottom = 0
            return kwargs

        self.pie = Element('pie', self.fcpp, kwargs,
                           on=True,
                           alpha=utl.kwget(kwargs, self.fcpp, ['pie_fill_alpha', 'fill_alpha'],
                                           kwargs.get('fill_alpha', 0.85)),
                           colors=utl.kwget(kwargs, self.fcpp, 'pie_colors', copy.copy(self.color_list)),
                           counter_clock=utl.kwget(kwargs, self.fcpp, ['pie_counter_clock', 'counter_clock'],
                                                   kwargs.get('counter_clock', False)),
                           edge_color=utl.kwget(kwargs, self.fcpp, 'pie_edge_color',
                                                kwargs.get('pie_edge_color', '#ffffff')),
                           edge_style=utl.kwget(kwargs, self.fcpp, 'pie_edge_style',
                                                kwargs.get('pie_edge_style', '-')),
                           edge_width=1,
                           explode=utl.kwget(kwargs, self.fcpp, ['pie_explode', 'explode'],
                                             kwargs.get('explode', None)),
                           inner_radius=utl.kwget(kwargs, self.fcpp, ['pie_inner_radius', 'inner_radius'],
                                                  kwargs.get('inner_radius', None)),
                           label_distance=utl.kwget(kwargs, self.fcpp, ['pie_label_distance', 'label_distance'],
                                                    kwargs.get('label_distance', 1.1)),
                           label_sizes=[(0, 0), (0, 0)],
                           percents=utl.kwget(kwargs, self.fcpp, ['pie_percents', 'percents'],
                                              kwargs.get('percents', None)),
                           percents_distance=utl.kwget(kwargs, self.fcpp,
                                                       ['pie_percents_distance', 'percents_distance'],
                                                       kwargs.get('percents_distance', 0.6)),
                           percents_font_color=utl.kwget(kwargs, self.fcpp,
                                                         ['pie_percents_font_color', 'percents_font_color'],
                                                         kwargs.get('percents_font_color', '#444444')),
                           percents_font_size=utl.kwget(kwargs, self.fcpp,
                                                        ['pie_percents_font_size', 'percents_font_size'],
                                                        kwargs.get('percents_font_size', 11)),
                           percents_font_weight=utl.kwget(kwargs, self.fcpp,
                                                          ['pie_percents_font_weight', 'percents_font_weight'],
                                                          kwargs.get('percents_font_weight', 'normal')),
                           radius=utl.kwget(kwargs, self.fcpp, ['pie_radius', 'radius'],
                                            kwargs.get('radius', 1)),
                           rotate_labels=utl.kwget(kwargs, self.fcpp, ['pie_rotate_labels', 'rotate_labels'],
                                                   kwargs.get('rotate_labels', False)),
                           shadow=utl.kwget(kwargs, self.fcpp, ['pie_shadow', 'shadow'],
                                            kwargs.get('shadow', False)),
                           start_angle=utl.kwget(kwargs, self.fcpp, ['pie_start_angle', 'start_angle'],
                                                 kwargs.get('start_angle', 90)),
                           )
        if self.pie.inner_radius is None:
            self.pie.inner_radius = self.pie.radius
        self.pie.xs_left = 0
        self.pie.xs_right = 0
        self.pie.xs_top = 0
        self.pie.xs_bottom = 0

        if self.pie.percents is True:
            self.pie.percents = '%1.1f%%'
        elif self.pie.percents is False:
            self.pie.percents = None

        return kwargs

    def _init_ref(self, kwargs: dict) -> dict:
        """Set the reference line plot element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        ref_line = kwargs.get('ref_line', False)
        if isinstance(ref_line, bool):
            self.ref_line = Element('ref_line', self.fcpp, kwargs, on=False)
            return kwargs

        if isinstance(ref_line, pd.Series):
            # dtype == pandas Series
            ref_col = 'Ref Line'
        else:
            # dtype == list
            ref_line = utl.validate_list(ref_line)
            ref_col = [f for f in ref_line if f in kwargs['df'].columns]
            missing = [f for f in ref_line if f not in ref_col]
            if len(missing) > 0:
                missing = ', '.join(missing)
                print(f'Could not find DataFrame columns for the folling ref lines: {missing}')
            if not kwargs.get('ref_line_legend_text'):
                kwargs['ref_line_legend_text'] = ref_col

        self.ref_line = Element('ref_line', self.fcpp, kwargs,
                                on=False if not ref_col else True,
                                column=RepeatedList(ref_col, 'ref_col') if ref_col else None,
                                color=utl.kwget(kwargs, self.fcpp, 'ref_line_color', '#000000'),
                                legend_text=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                                   'ref_line_legend_text', 'Ref Line'),
                                                         'ref_line_legend_text'),
                                )

        return kwargs

    def _init_rc_labels(self, kwargs: dict) -> dict:
        """Set the row/column/wrap label element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Row column label
        label_rc = DF_Element('label_rc', self.fcpp, kwargs,
                              on=True,
                              obj=self.obj_array,
                              size=utl.kwget(kwargs, self.fcpp, 'label_rc_size', 30),
                              edge_color=utl.kwget(kwargs, self.fcpp, 'label_rc_edge_color', '#8c8c8c'),
                              edge_width=utl.kwget(kwargs, self.fcpp, 'label_rc_edge_width', 0),
                              fill_color=utl.kwget(kwargs, self.fcpp, 'label_rc_fill_color', '#8c8c8c'),
                              font_color=utl.kwget(kwargs, self.fcpp, 'label_rc_font_color', '#ffffff'),
                              font_size=utl.kwget(kwargs, self.fcpp, 'label_rc_font_size', 16),
                              font_style=utl.kwget(kwargs, self.fcpp, 'label_rc_font_style', 'normal'),
                              font_weight=utl.kwget(kwargs, self.fcpp, 'label_rc_font_weight', 'bold'),
                              names=utl.kwget(kwargs, self.fcpp, 'label_rc_names', True),
                              align=utl.kwget(kwargs, self.fcpp, 'label_rc_align', 'center'),
                              )
        self.label_row = copy.deepcopy(label_rc)
        self.label_row.on = utl.kwget(kwargs, self.fcpp, 'label_row_on', True) \
            if kwargs.get('row') not in [None, 'y'] else False
        self.label_row.column = kwargs.get('row')
        self.label_row.edge_color = utl.kwget(kwargs, self.fcpp, 'label_row_edge_color', label_rc.edge_color)
        self.label_row.edge_alpha = utl.kwget(kwargs, self.fcpp, 'label_row_edge_alpha', label_rc.edge_alpha)
        self.label_row.edge_width = utl.kwget(kwargs, self.fcpp, 'label_row_edge_width', label_rc.edge_width)
        self.label_row.fill_color = utl.kwget(kwargs, self.fcpp, 'label_row_fill_color', label_rc.fill_color)
        self.label_row.font_color = utl.kwget(kwargs, self.fcpp, 'label_row_font_color', label_rc.font_color)
        self.label_row.names = utl.kwget(kwargs, self.fcpp, 'label_row_names', label_rc.names)
        self.label_row.size = [utl.kwget(kwargs, self.fcpp, 'label_row_size', label_rc._size), self.axes.size[1]]
        self.label_row.text_size = None
        self.label_row.rotation = 270

        self.label_col = copy.deepcopy(label_rc)
        self.label_col.on = utl.kwget(kwargs, self.fcpp, 'label_col_on', True) \
            if kwargs.get('col') not in [None, 'x'] else False
        self.label_row.column = kwargs.get('col')
        self.label_col.edge_color = utl.kwget(kwargs, self.fcpp, 'label_col_edge_color', label_rc.edge_color)
        self.label_col.edge_width = utl.kwget(kwargs, self.fcpp, 'label_col_edge_width', label_rc.edge_width)
        self.label_col.edge_alpha = utl.kwget(kwargs, self.fcpp, 'label_col_edge_alpha', label_rc.edge_alpha)
        self.label_col.fill_color = utl.kwget(kwargs, self.fcpp, 'label_col_fill_color', label_rc.fill_color)
        self.label_col.font_color = utl.kwget(kwargs, self.fcpp, 'label_col_font_color', label_rc.font_color)
        self.label_col.names = utl.kwget(kwargs, self.fcpp, 'label_col_names', label_rc.names)
        self.label_col.size = [self.axes.size[0], utl.kwget(kwargs, self.fcpp, 'label_col_size', label_rc._size)]
        self.label_col.text_size = None

        # Wrap label
        self.label_wrap = DF_Element('label_wrap', self.fcpp, kwargs,
                                     on=utl.kwget(kwargs, self.fcpp, 'label_wrap_on', True)
                                     if kwargs.get('wrap') else False,
                                     obj=self.obj_array,
                                     column=kwargs.get('wrap'),
                                     size=[self.axes.size[0], utl.kwget(kwargs, self.fcpp, 'label_wrap_size', 30)],
                                     edge_color=utl.kwget(kwargs, self.fcpp, 'label_wrap_edge_color',
                                                          label_rc.edge_color),
                                     edge_width=utl.kwget(kwargs, self.fcpp, 'label_wrap_edge_width',
                                                          label_rc.edge_width),
                                     edge_alpha=utl.kwget(kwargs, self.fcpp, 'label_wrap_edge_alpha',
                                                          label_rc.edge_alpha),
                                     fill_color=utl.kwget(kwargs, self.fcpp, 'label_wrap_fill_color',
                                                          label_rc.fill_color),
                                     fill_alpha=utl.kwget(kwargs, self.fcpp, 'label_wrap_fill_alpha',
                                                          label_rc.fill_alpha),
                                     font=utl.kwget(kwargs, self.fcpp, 'label_wrap_font', label_rc.font),
                                     font_color=utl.kwget(kwargs, self.fcpp, 'label_wrap_font_color',
                                                          label_rc.font_color),
                                     font_size=utl.kwget(kwargs, self.fcpp, 'label_wrap_font_size',
                                                         label_rc.font_size),
                                     font_style=utl.kwget(kwargs, self.fcpp, 'label_wrap_font_style',
                                                          label_rc.font_style),
                                     font_weight=utl.kwget(kwargs, self.fcpp, 'label_wrap_font_weight',
                                                           label_rc.font_weight),
                                     text_size=None,
                                     )

        self.title_wrap = Element('title_wrap', self.fcpp, kwargs,
                                  on=utl.kwget(kwargs, self.fcpp, 'title_wrap_on', True)
                                  if kwargs.get('wrap') else False,
                                  size=utl.kwget(kwargs, self.fcpp, 'title_wrap_size', label_rc.size),
                                  edge_color=utl.kwget(kwargs, self.fcpp, 'title_wrap_edge_color', '#5f5f5f'),
                                  edge_width=utl.kwget(kwargs, self.fcpp, 'title_wrap_edge_width', label_rc.edge_width),
                                  edge_alpha=utl.kwget(kwargs, self.fcpp, 'title_wrap_edge_alpha', label_rc.edge_alpha),
                                  fill_color=utl.kwget(kwargs, self.fcpp, 'title_wrap_fill_color', '#5f5f5f'),
                                  fill_alpha=utl.kwget(kwargs, self.fcpp, 'title_wrap_fill_alpha', label_rc.fill_alpha),
                                  font=utl.kwget(kwargs, self.fcpp, 'title_wrap_font', label_rc.font),
                                  font_color=utl.kwget(kwargs, self.fcpp, 'title_wrap_font_color', label_rc.font_color),
                                  font_size=utl.kwget(kwargs, self.fcpp, 'title_wrap_font_size', 16),
                                  font_style=utl.kwget(kwargs, self.fcpp, 'title_wrap_font_style', label_rc.font_style),
                                  font_weight=utl.kwget(kwargs, self.fcpp, 'title_wrap_font_weight',
                                                        label_rc.font_weight),
                                  text=kwargs.get('title_wrap', None),
                                  )

        if not isinstance(self.title_wrap.size, list):
            self.title_wrap.size = [self.axes.size[0], self.title_wrap.size]

        return kwargs

    def _init_text_box(self, kwargs: dict) -> dict:
        """Set the text box element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        position = utl.kwget(kwargs, self.fcpp, 'text_position', [0, 0])
        if not isinstance(position[0], list):
            position = [position]
        self.text = Element('text', self.fcpp, {},
                            on=True if utl.kwget(kwargs, self.fcpp, 'text', None)
                            is not None else False,
                            obj=self.obj_array,
                            edge_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_edge_color', 'none'),
                                                    'text_edge_color'),
                            fill_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_fill_color', 'none'),
                                                    'text_fill_color'),
                            font=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                              'text_font', 'sans-serif'), 'text_font'),
                            font_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_font_color', '#000000'),
                                                    'text_font_color'),
                            font_size=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                   'text_font_size', 14),
                                                   'text_font_size'),
                            font_style=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_font_style', 'normal'),
                                                    'text_font_style'),
                            font_weight=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                               'text_font_weight', 'normal'),
                                                     'text_font_weight'),
                            position=RepeatedList(position, 'text_position'),
                            coordinate=utl.kwget(kwargs, self.fcpp, 'text_coordinate', 'axis'),
                            rotation=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                  'text_rotation', 0), 'text_rotation'),
                            units=utl.kwget(kwargs, self.fcpp,
                                            'text_units', 'pixel'),
                            text=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                        'text', ''), 'text'),
                            )

        for ir, ic in np.ndindex(self.axes.obj.shape):
            self.text.obj[ir, ic] = np.zeros(len(self.text.text.values), dtype=object)

        return kwargs

    def _init_ticks(self, kwargs: dict) -> dict:
        """Set the tick marks and tick labels element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # Major ticks
        if 'ticks' in kwargs.keys() and 'ticks_major' not in kwargs.keys():
            kwargs['ticks_major'] = kwargs['ticks']
        ticks_length = utl.kwget(kwargs, self.fcpp, 'ticks_length', 6.2)
        ticks_width = utl.kwget(kwargs, self.fcpp, 'ticks_width', 2.2)
        self.ticks_major = Element('ticks_major', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp, 'ticks_major', True),
                                   color=utl.kwget(kwargs, self.fcpp, 'ticks_major_color', '#ffffff'),
                                   direction=utl.kwget(kwargs, self.fcpp, 'ticks_major_direction', 'in'),
                                   increment=utl.kwget(kwargs, self.fcpp, 'ticks_major_increment', None),
                                   size=[utl.kwget(kwargs, self.fcpp, 'ticks_major_length', ticks_length),
                                         utl.kwget(kwargs, self.fcpp, 'ticks_major_width', ticks_width)],
                                   )
        kwargs = self._from_list(self.ticks_major, ['color', 'increment'], 'ticks_major', kwargs)
        for ia, ax in enumerate(self.ax):
            setattr(self, 'ticks_major_%s' % ax,
                    Element('ticks_major_%s' % ax, self.fcpp, kwargs,
                            on=utl.kwget(kwargs, self.fcpp,
                                         'ticks_major_%s' % ax, self.ticks_major.on),
                            color=copy.copy(self.ticks_major.color),
                            direction=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major_%s_direction' % ax,
                                                self.ticks_major.direction),
                            increment=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major_%s_increment' % ax,
                                                self.ticks_major.increment),
                            size=self.ticks_major.size,
                            ))
        if 'tick_labels' in kwargs.keys() \
                and 'tick_labels_major' not in kwargs.keys():
            kwargs['tick_labels_major'] = kwargs['tick_labels']
        for k, v in kwargs.copy().items():
            if 'tick_labels' in k and 'major' not in k and 'minor' not in k:
                kwargs['tick_labels_major%s' % k.split('tick_labels')[1]] = v
                if k != 'tick_labels':
                    kwargs['tick_labels_minor%s' % k.split('tick_labels')[1]] = v
        self.tick_labels_major = \
            Element('tick_labels_major', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_major',
                                 kwargs.get('tick_labels', True)),
                    edge_alpha=0 if not kwargs.get('tick_labels_edge_alpha', None)
                    and not kwargs.get('tick_labels_major_edge_alpha', None)
                    and not kwargs.get('tick_labels_major_edge_color', None)
                    else 1,
                    fill_alpha=0 if not kwargs.get('tick_labels_fill_alpha', None)
                    and not kwargs.get('tick_labels_major_fill_alpha', None)
                    and not kwargs.get('tick_labels_major_fill_color', None)
                    else 1,
                    font_size=13,
                    offset=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_major_offset', False),
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_major_padding', 4),
                    scale_factor=1.5,
                    )
        kwargs = self._from_list(self.tick_labels_major,
                                 ['font', 'font_color', 'font_size',
                                  'font_style', 'font_weight', 'padding',
                                  'rotation'], 'tick_labels_major', kwargs)
        for ax in self.ax + ['z']:
            fill_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_fill_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_fill_alpha',
                                             None))
            fill_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_fill_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_fill_color',
                                             None))
            if not fill_alpha and fill_color:
                fill_alpha = 1
            elif not fill_alpha and not fill_color:
                fill_alpha = 0
            if not fill_color:
                fill_color = copy.copy(self.tick_labels_major.fill_color)
            edge_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_edge_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_edge_alpha',
                                             None))
            edge_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_edge_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_edge_color',
                                             None))
            if not edge_alpha and edge_color:
                edge_alpha = 1
            elif not edge_alpha and not edge_color:
                edge_alpha = 0
            if not edge_color:
                edge_color = copy.copy(self.tick_labels_major.edge_color)

            if '2' in ax:
                axl = '2'
            else:
                axl = ''
            if getattr(self, 'axes%s' % axl).scale in globals()['LOG%s' % ax[0].upper()] and \
                    not utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, False) and \
                    'sci_%s' % ax not in kwargs.keys():
                kwargs['sci_%s' % ax] = 'best'

            setattr(self, 'tick_labels_major_%s' % ax,
                    Element('tick_labels_major_%s' % ax, self.fcpp, kwargs,
                            on=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_major_%s' % ax,
                                         self.tick_labels_major.on),
                            obj=self.obj_array,
                            edge_color=edge_color,
                            edge_alpha=edge_alpha,
                            edge_width=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_major_{ax}_edge_width',
                                                  'tick_labels_major_edge_width'],
                                                 self.tick_labels_major.edge_width),
                            fill_color=fill_color,
                            fill_alpha=fill_alpha,
                            font=utl.kwget(kwargs, self.fcpp,
                                           [f'tick_labels_major_{ax}_font', 'tick_labels_major_font'],
                                           self.tick_labels_major.font),
                            font_color=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_major_{ax}_font_color', 'tick_labels_major_font_color'],
                                                 self.tick_labels_major.font_color),
                            font_size=utl.kwget(kwargs, self.fcpp,
                                                [f'tick_labels_major_{ax}_font_size', 'tick_labels_major_font_size'],
                                                self.tick_labels_major.font_size),
                            font_style=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_major_{ax}_font_style', 'tick_labels_major_font_style'],
                                                 self.tick_labels_major.font_style),
                            font_weight=utl.kwget(kwargs, self.fcpp,
                                                  [f'tick_labels_major_{ax}_font_weight',
                                                   'tick_labels_major_font_weight'],
                                                  self.tick_labels_major.font_weight),
                            offset=utl.kwget(kwargs, self.fcpp,
                                             [f'tick_labels_major_{ax}_offset', 'tick_labels_major_offset'],
                                             self.tick_labels_major.offset),
                            rotation=utl.kwget(kwargs, self.fcpp,
                                               [f'tick_labels_major_{ax}_rotation', 'tick_labels_major_rotation'],
                                               self.tick_labels_major.rotation),
                            size=[0, 0],
                            scale_factor=self.tick_labels_major.scale_factor,
                            sci=utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, 'best'),
                            ))
        self.auto_tick_threshold = utl.kwget(kwargs, self.fcpp, 'auto_tick_threshold', [1e-6, 1e6])

        # Minor ticks
        self.ticks_minor = Element('ticks_minor', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp,
                                                'ticks_minor', False),
                                   color='#ffffff',
                                   direction=utl.kwget(kwargs, self.fcpp, 'ticks_minor_direction', 'in'),
                                   number=utl.kwget(kwargs, self.fcpp, 'ticks_minor_number', 3),
                                   size=[utl.kwget(kwargs, self.fcpp, 'ticks_minor_length', ticks_length * 0.67),
                                         utl.kwget(kwargs, self.fcpp, 'ticks_minor_width', ticks_width * 0.6)],
                                   )
        kwargs = self._from_list(self.ticks_minor, ['color', 'number'], 'ticks_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'ticks_minor_%s' % ax,
                    Element('ticks_minor_%s' % ax, self.fcpp, kwargs,
                            on=utl.kwget(kwargs, self.fcpp,
                                         'ticks_minor_%s' % ax, self.ticks_minor.on),
                            color=copy.copy(self.ticks_minor.color),
                            direction=utl.kwget(kwargs, self.fcpp,
                                                'ticks_minor_%s_direction' % ax,
                                                self.ticks_minor.direction),
                            number=utl.kwget(kwargs, self.fcpp,
                                             'ticks_minor_%s_number' % ax,
                                             self.ticks_minor.number),
                            size=self.ticks_minor._size,
                            ))
            if 'ticks_minor_%s_number' % ax in kwargs.keys():
                getattr(self, 'ticks_minor_%s' % ax).on = True

        self.tick_labels_minor = \
            Element('tick_labels_minor', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_minor',
                                 False),
                    edge_alpha=0 if not kwargs.get('tick_labels_edge_alpha', None)
                    and not kwargs.get('tick_labels_minor_edge_alpha', None)
                    and not kwargs.get('tick_labels_minor_edge_color', None)
                    else 1,
                    fill_alpha=0 if not kwargs.get('tick_labels_fill_alpha', None)
                    and not kwargs.get('tick_labels_minor_fill_alpha', None)
                    and not kwargs.get('tick_labels_minor_fill_color', None)
                    else 1,
                    font_size=10,
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_minor_padding', 3),
                    )
        kwargs = self._from_list(self.tick_labels_minor,
                                 ['font', 'font_color', 'font_size',
                                  'font_style', 'font_weight', 'padding',
                                  'rotation'], 'tick_labels_minor', kwargs)
        for ax in self.ax:
            fill_alpha = utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_%s_fill_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_fill_alpha', None))
            fill_color = utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_%s_fill_color' % ax,
                                   utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_fill_color', None))
            if not fill_alpha and fill_color:
                fill_alpha = 1
            elif not fill_alpha and not fill_color:
                fill_alpha = 0
            if not fill_color:
                fill_color = copy.copy(self.tick_labels_minor.fill_color)

            edge_alpha = utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_%s_edge_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_edge_alpha', None))
            edge_color = utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_%s_edge_color' % ax,
                                   utl.kwget(kwargs, self.fcpp, 'tick_labels_minor_edge_color', None))
            if not edge_alpha and edge_color:
                edge_alpha = 1
            elif not edge_alpha and not edge_color:
                edge_alpha = 0
            if not edge_color:
                edge_color = copy.copy(self.tick_labels_minor.edge_color)

            setattr(self, 'tick_labels_minor_%s' % ax,
                    Element('tick_labels_minor_%s' % ax, self.fcpp, kwargs,
                            on=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_minor_%s' % ax,
                                         self.tick_labels_minor.on),
                            obj=self.obj_array,
                            edge_color=edge_color,
                            edge_alpha=edge_alpha,
                            edge_width=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_minor_{ax}_edge_width', 'tick_labels_minor_edge_width'],
                                                 self.tick_labels_minor.edge_width),
                            fill_color=fill_color,
                            fill_alpha=fill_alpha,
                            font=utl.kwget(kwargs, self.fcpp,
                                           [f'tick_labels_minor_{ax}_font', 'tick_labels_minor_font'],
                                           self.tick_labels_minor.font),
                            font_color=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_minor_{ax}_font_color', 'tick_labels_minor_font_color'],
                                                 self.tick_labels_minor.font_color),
                            font_size=utl.kwget(kwargs, self.fcpp,
                                                [f'tick_labels_minor_{ax}_font_size', 'tick_labels_minor_font_size'],
                                                self.tick_labels_minor.font_size),
                            font_style=utl.kwget(kwargs, self.fcpp,
                                                 [f'tick_labels_minor_{ax}_font_style', 'tick_labels_minor_font_style'],
                                                 self.tick_labels_minor.font_style),
                            font_weight=utl.kwget(kwargs, self.fcpp,
                                                  [f'tick_labels_minor_{ax}_font_weight',
                                                   'tick_labels_minor_font_weight'],
                                                  self.tick_labels_minor.font_weight),
                            rotation=utl.kwget(kwargs, self.fcpp,
                                               [f'tick_labels_minor_{ax}_rotation', 'tick_labels_minor_rotation'],
                                               self.tick_labels_minor.rotation),
                            size=[0, 0],
                            sci=utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, False),
                            ))

            if getattr(self, 'tick_labels_minor_%s' % ax).on:
                getattr(self, 'ticks_minor_%s' % ax).on = True

        self.tick_y_top_xs = 0

        return kwargs

    def _init_title(self, kwargs: dict) -> dict:
        """Set the plot title element parameters

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        title = utl.kwget(kwargs, self.fcpp, 'title', None)
        self.title = Element('title', self.fcpp, kwargs,
                             on=True if title is not None else False,
                             obj=self.obj_array,
                             text=str(title) if title is not None else None,
                             font_color='#333333',
                             font_size=18,
                             font_weight='bold',
                             align='center',
                             )
        return kwargs

    def _init_white_space(self, kwargs: dict) -> dict:
        """Set the whitespace for various elements defined previously

        Args:
            kwargs: user-defined keyword args

        Returns:
            updated kwargs
        """
        # cbar
        if self.cbar.on:
            self.ws_ax_cbar = utl.kwget(kwargs, self.fcpp, 'ws_ax_cbar', 10)

        # rc labels
        ws_label_rc = utl.kwget(kwargs, self.fcpp, 'ws_label_rc', 10)
        self.ws_label_col = utl.kwget(kwargs, self.fcpp,
                                      'ws_label_col', ws_label_rc)
        self.ws_label_row = utl.kwget(kwargs, self.fcpp,
                                      'ws_label_row', ws_label_rc)
        self.ws_col = utl.kwget(kwargs, self.fcpp, 'ws_col', 30)
        self.ws_col_def = int(self.ws_col)
        self.ws_row = utl.kwget(kwargs, self.fcpp, 'ws_row', 30)
        self.ws_row_def = int(self.ws_row)

        # figure
        self.ws_fig_label = utl.kwget(kwargs, self.fcpp, 'ws_fig_label', 10)
        self.ws_leg_fig = utl.kwget(kwargs, self.fcpp, 'ws_leg_fig', 10)
        self.ws_fig_ax = utl.kwget(kwargs, self.fcpp, 'ws_fig_ax', 10)
        self.ws_fig_title = utl.kwget(kwargs, self.fcpp, 'ws_fig_title', 10)
        self.ws_label_fig = utl.kwget(kwargs, self.fcpp, 'ws_label_fig',
                                      self.ws_fig_label)

        # axes
        self.ws_label_tick = utl.kwget(kwargs, self.fcpp, 'ws_label_tick', 10)
        self.ws_ax_leg = utl.kwget(kwargs, self.fcpp, 'ws_ax_leg', 5)
        self.ws_ticks_ax = utl.kwget(kwargs, self.fcpp, 'ws_ticks_ax', 5)
        self.ws_title_ax = utl.kwget(kwargs, self.fcpp, 'ws_title_ax', 10)
        self.ws_ax_fig = utl.kwget(kwargs, self.fcpp, 'ws_ax_fig', 10)
        self.ws_ax_label_xs = utl.kwget(kwargs, self.fcpp, 'ws_ax_label_xs', 5)

        # ticks
        self.ws_tick_minimum = utl.kwget(kwargs, self.fcpp,
                                         'ws_tick_minimum', 10)

        # box
        self.ws_ax_box_title = utl.kwget(kwargs, self.fcpp, 'ws_ax_box_title', 10)

    def _from_list(self, base, attrs, name, kwargs) -> dict:
        """Supports definition of object attributes for multiple axes using a list

        Index 0 is always 'x'
        Index 1 is always 'y'
        Index 2 is for twinned axes

        Args:
            base (Element): object class
            attrs (list): attributes to check from the base class
            name (str): name of the base class
            kwargs (dict): keyword dict

        Returns:
            updated kwargs
        """
        for attr in attrs:
            if isinstance(getattr(base, attr), list):
                setattr(base, attr, getattr(base, attr) + [None] * (len(getattr(base, attr)) - 3))
                kwargs['%s_x_%s' % (name, attr)] = getattr(base, attr)[0]
                kwargs['%s_y_%s' % (name, attr)] = getattr(base, attr)[1]
                if 'twin_x' in kwargs.keys() and kwargs['twin_x']:
                    kwargs['%s_y2_%s' % (name, attr)] = getattr(base, attr)[2]
                if 'twin_y' in kwargs.keys() and kwargs['twin_y']:
                    kwargs['%s_x2_%s' % (name, attr)] = getattr(base, attr)[2]

        return kwargs

    def _get_axes(self):
        """Return list of active axes."""
        return [f for f in [self.axes, self.axes2] if f.on]

    @abc.abstractmethod
    def _get_element_sizes(self, data: 'data.Data'):
        """Calculate the actual rendered size of select elements by pre-plotting
        them.  This is needed to correctly adjust the figure dimensions.

        Args:
            data: fcp Data object

        Returns:
            updated version of `data`
        """

    @abc.abstractmethod
    def _get_figure_size(self, data: 'data.Data', **kwargs):
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            kwargs: user-defined keyword args
        """

    def _set_label_text(self, data: 'data.Data'):
        """Set the default label text for x, y, z axes and col, row, wrap
        grouping labels

        Args:
            data: Data class object for the plot
        """
        kwargs = self.kwargs_mod.copy()  # alias

        # Set the label text
        labels = ['x', 'y', 'z', 'col', 'row', 'wrap']
        for ilab, lab in enumerate(labels):
            dd = getattr(data, lab)
            if not dd:
                continue

            # Get label override name if in kwargs
            if 'label_%s' % lab in kwargs.keys():
                lab_text = str(kwargs.get('label_%s' % lab))
                lab_text2 = str(kwargs.get('label_%s2' % lab))
            elif 'label_%s_text' % lab in kwargs.keys():
                lab_text = str(kwargs.get('label_%s_text' % lab))
                lab_text2 = str(kwargs.get('label_%s2_text' % lab))
            else:
                lab_text = None
                lab_text2 = None

            if lab == 'x' and self.axes.twin_y:
                getattr(self, 'label_x').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_x2').text = \
                    lab_text2 if lab_text2 is not None else getattr(data, '%s2' % lab)[
                    0]
            elif lab == 'y' and self.axes.twin_x:
                getattr(self, 'label_y').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_y2').text = \
                    lab_text2 if lab_text2 is not None else getattr(data, '%s2' % lab)[
                    0]
            else:
                if lab == 'wrap':
                    # special case
                    val = 'title_wrap'
                else:
                    val = 'label_%s' % lab
                if isinstance(dd, list):
                    if data.wrap == 'y' and lab == 'y' \
                            or data.wrap == 'x' and lab == 'x':
                        getattr(self, val).text = data.wrap_vals
                    elif lab == 'x' and data.col == 'x':
                        getattr(self, val).text = data.x_vals * self.nrow
                    elif lab == 'y' and data.row == 'y':
                        yvals = []
                        for yval in data.y_vals:
                            yvals += [yval] * self.ncol
                        getattr(self, val).text = yvals
                    else:
                        getattr(self, val).text = \
                            lab_text if lab_text is not None \
                            else ' & '.join([str(f) for f in dd])
                else:
                    getattr(self, val).text = dd
                if lab != 'z' and hasattr(self, 'label_%s2' % lab):
                    getattr(self, 'label_%s2' % lab).text = \
                        lab_text2 if lab_text2 is not None \
                        else ' & '.join([str(f) for f in dd])

            if hasattr(data, '%s_vals' % lab):
                getattr(self, 'label_%s' % lab).values = \
                    getattr(data, '%s_vals' % lab)

        if 'hist' in self.name:
            if self.hist.normalize:
                self.label_y.text = kwargs.get('label_y_text', 'Normalized Counts')
            elif self.hist.horizontal:
                tmp = self.label_y.text
                self.label_y.text = self.label_x.text
                self.label_x.text = tmp

        if 'bar' in self.name and self.bar.horizontal:
            temp = self.label_x.text
            self.label_x.text = self.label_y.text
            self.label_y.text = temp

    def _update_from_data(self, data: 'data.Data'):
        """Update certain attributes of the layout based on values calculated
        in the data object

        Args:
            data: Data class object for the plot
        """
        self.groups = data.groups
        self.ngroups = data.ngroups
        self.nwrap = data.nwrap
        self.axes.share_x = data.share_x
        self.axes2.share_x = data.share_x2
        self.axes.share_y = data.share_y
        self.axes2.share_y = data.share_y2
        self.axes.share_z = data.share_z
        self.axes.share_col = data.share_col
        self.axes.share_row = data.share_row
        self.axes.scale = data.ax_scale
        self.axes2.scale = data.ax2_scale

    def _update_wrap(self, data, kwargs):
        """Update certain figure properties based on wrap plot parameters

        Args:
            data: Data class object for the plot
            kwargs: user-defined keyword args
        """
        if data.wrap == 'y' or data.wrap == 'x':
            self.title_wrap.on = False
            self.label_wrap.on = False
            self.separate_labels = kwargs.get('separate_labels', True)
            self.separate_ticks = kwargs.get('separate_ticks', True) if not self.separate_labels else True
        elif data.wrap:
            self.separate_labels = kwargs.get('separate_labels', False)
            self.separate_ticks = kwargs.get('separate_ticks', False) if not self.separate_labels else True
            self.ws_row = kwargs.get('ws_row', self.label_wrap._size[1])
            self.ws_row_def = int(self.ws_row)
            self.ws_col = kwargs.get('ws_col', 0)
            self.ws_col_def = 0

    @abc.abstractmethod
    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
            data: fcp Data object
        """

    @abc.abstractmethod
    def add_hvlines(self, ir: int, ic: int, df: [pd.DataFrame, None] = None):
        """Add horizontal/vertical lines.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data. Defaults to None.
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def add_legend(self, leg_vals: pd.DataFrame):
        """Add a legend to a figure.

        Args:
            data.legend_vals, used to ensure proper sorting
        """

    @abc.abstractmethod
    def close(self):
        """Close an inline plot window."""

    @abc.abstractmethod
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

    @abc.abstractmethod
    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """

    def make_kw_dict(self, element: 'Element', pop: list = []) -> dict:
        """Extract certain parameters from a given element as a dictionary
        that can be passed to another function as **kwargs

        Args:
            element: an element object [ex. self.label_x, self.box_group_label]
            pop: optional list of parameters that can be dropped from the output
                dictionary. Defaults to [].

        Returns:
            dict
        """
        kwargs = {}
        kwargs['position'] = copy.copy(element.position)
        kwargs['size'] = copy.copy(element.size)
        kwargs['rotation'] = copy.copy(element.rotation)
        kwargs['fill_color'] = copy.copy(element.fill_color)
        kwargs['edge_color'] = copy.copy(element.edge_color)
        kwargs['edge_width'] = copy.copy(element.edge_width)
        kwargs['font'] = copy.copy(element.font)
        kwargs['font_weight'] = copy.copy(element.font_weight)
        kwargs['font_style'] = copy.copy(element.font_style)
        kwargs['font_color'] = copy.copy(element.font_color)
        kwargs['font_size'] = copy.copy(element.font_size)
        kwargs['color'] = copy.copy(element.color)
        kwargs['width'] = copy.copy(element.width)
        kwargs['style'] = copy.copy(element.style)
        kwargs['zorder'] = copy.copy(element.zorder)
        for pp in pop:
            if pp in kwargs.keys():
                kwargs.pop(pp)

        return kwargs

    # Note: plot functions follow the following ordering scheme for input args
    # -> ir, ic, iline, df, x, y, z, leg_name, data, ngroups, twin, others at will...
    # simply skip any that are not relevant for a give plot function
    @abc.abstractmethod
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

    @abc.abstractmethod
    def plot_box(self, ir: int, ic: int, data: 'data.Data', **kwargs) -> 'MPL_Boxplot_Object':  # noqa: F821
        """Plot boxplot.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: Data object
            kwargs: keyword args

        Returns:
            box plot MPL object
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @abc.abstractmethod
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
            imshow plot obj
        """

    @abc.abstractmethod
    def plot_hist(self, ir: int, ic: int, iline: int, df: pd.DataFrame, x: str,
                  y: str, leg_name: str, data: 'data.Data') -> ['MPL_histogram_object', 'Data']:  # noqa: F821
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

    @abc.abstractmethod
    def plot_imshow(self, ir: int, ic: int, df: pd.DataFrame, data: 'data.Data'):
        """Plot an image.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            data: Data object

        Returns:
            imshow plot obj
        """

    @abc.abstractmethod
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
            plot object
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def plot_polygon(self, ir: int, ic: int, points: list, **kwargs):
        """Plot a polygon.

        Args:
            ir: subplot row index
            ic: subplot column index
            points: list of floats that defint the points on the polygon
            kwargs: keyword args
        """

    @abc.abstractmethod
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

    @abc.abstractmethod
    def save(self, filename: str, idx: int = 0):
        """Save a plot window.

        Args:
            filename: name of the file
            idx (optional): figure index in order to set the edge and face color of the
                figure correctly when saving. Defaults to 0.
        """

    @abc.abstractmethod
    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index
        """

    @abc.abstractmethod
    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """

    @abc.abstractmethod
    def set_axes_labels(self, ir: int, ic: int):
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    @abc.abstractmethod
    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """

    @abc.abstractmethod
    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    @abc.abstractmethod
    def set_axes_scale(self, ir: int, ic: int):
        """Set the scale type of the axes.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    @abc.abstractmethod
    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    @abc.abstractmethod
    def set_figure_final_layout(self, data: 'data.Data', **kwargs):
        """Final adjustment of the figure size and plot spacing."""

    @abc.abstractmethod
    def set_figure_title(self):
        """Set a figure title."""

    @abc.abstractmethod
    def show(self, filename: str = None):
        """Display the plot window.

        Args:
            filename (optional): name of the file to show. Defaults to None.

        """


class Element:
    def __init__(self, name: str = 'None', fcpp: dict = {}, others: dict = {},
                 obj: [None, 'ObjectArray'] = None, **kwargs):
        """Element object is a container for storing/accessing the attributes
        that are applied to a given plot element.  Examples of plot elements
        include axes labels, ticks, fit lines, or a type of plot (box, hist, etc.).
        The attributes of the elements may differ based on the use of an object
        but can include things like font (size, weight, type), color, padding, or
        specific options relevant to a plot type (for example, a hist plot contains
        attributes to set the number of bins or enable/disable a kde overlay.

        Args:
            name (optional): Name of the element. Defaults to 'None'.
            fcpp (optional): Default kwargs loaded from a theme file.
                Defaults to {}.
            others (optional): Other kwargs which override those in fcpp.
                Typically, these are the user-defined kwargs set in the
                plotting function call.  Defaults to {}.
            obj (optional): if this element is unique on each subplot,
                pass an ObjectArray class to define one for each row x column
                index; else defaults to None.
        """
        # Update kwargs
        for k, v in others.items():
            if k not in kwargs.keys():
                kwargs[k] = v

        # Defaults
        self._on = kwargs.get('on', True)  # visbile or not
        self.name = name
        self.dpi = utl.kwget(kwargs, fcpp, 'dpi', 100)
        if obj is None:
            self.obj = None
            self.obj_bg = None
            self.limits = []
        else:
            self.obj = obj.copy()  # plot object reference
            self.obj_bg = obj.copy()  # background rect
            self.limits = obj.copy()
        self.position = kwargs.get('position', [0, 0, 0, 0])  # left, right, top, bottom
        self._size = kwargs.get('size', [0, 0])  # width, height
        self._size_orig = kwargs.get('size', [0, 0])
        self._text = kwargs.get('text', True)  # text label
        self._text_orig = kwargs.get('text')
        self.rotation = utl.kwget(kwargs, fcpp, '%s_rotation' % name,
                                  kwargs.get('rotation', 0))
        self.zorder = utl.kwget(kwargs, fcpp, '%s_zorder' % name,
                                kwargs.get('zorder', 0))

        # For some elements that are unique by axes, track sizes as DataFrame
        self._size_all = pd.DataFrame()
        self._size_all_bg = pd.DataFrame()
        self.size_cols = ['ir', 'ic', 'ii', 'jj', 'width', 'height', 'x0', 'x1', 'y0', 'y1', 'rotation']

        # fill and edge colors
        if 'fill_alpha' not in kwargs:
            self.fill_alpha = utl.kwget(kwargs, fcpp, '%s_fill_alpha' % name, kwargs.get('fill_alpha', 1))
        else:
            self.fill_alpha = kwargs['fill_alpha']
        if 'fill_color' not in kwargs:
            self.fill_color = utl.kwget(kwargs, fcpp, '%s_fill_color' % name, kwargs.get('fill_color', '#ffffff'))
        else:
            self.fill_color = kwargs['fill_color']
        if not isinstance(self.fill_color, RepeatedList) \
                and self.fill_color is not None \
                or self.fill_alpha != 1:
            self.color_alpha('fill_color', 'fill_alpha')
        if 'edge_width' not in kwargs:
            self.edge_width = utl.kwget(kwargs, fcpp, '%s_edge_width' % name, kwargs.get('edge_width', 1))
        else:
            self.edge_width = kwargs['edge_width']
        if 'edge_alpha' not in kwargs:
            self.edge_alpha = utl.kwget(kwargs, fcpp, '%s_edge_alpha' % name, kwargs.get('edge_alpha', 1))
        else:
            self.edge_alpha = kwargs['edge_alpha']
        if 'edge_color' not in kwargs:
            self.edge_color = utl.kwget(kwargs, fcpp, '%s_edge_color' % name, kwargs.get('edge_color', '#ffffff'))
        else:
            self.edge_color = kwargs['edge_color']
        if not isinstance(self.edge_color, RepeatedList) or self.edge_alpha != 1:
            self.color_alpha('edge_color', 'edge_alpha')

        # fonts
        if 'font' not in kwargs:
            self.font = utl.kwget(kwargs, fcpp, '%s_font' % name, kwargs.get('font', 'sans-serif'))
        else:
            self.font = kwargs['font']
        if 'font_color' not in kwargs:
            self.font_color = utl.kwget(kwargs, fcpp, '%s_font_color' % name, kwargs.get('font_color', '#000000'))
        else:
            self.font_color = kwargs['font_color']
        if 'font_size' not in kwargs:
            self.font_size = utl.kwget(kwargs, fcpp, '%s_font_size' % name, kwargs.get('font_size', 14))
        else:
            self.font_size = kwargs['font_size']
        if 'font_style' not in kwargs:
            self.font_style = utl.kwget(kwargs, fcpp, '%s_font_style' % name, kwargs.get('font_style', 'normal'))
        else:
            self.font_style = kwargs['font_style']
        if 'font_weight' not in kwargs:
            self.font_weight = utl.kwget(kwargs, fcpp, '%s_font_weight' % name, kwargs.get('font_weight', 'normal'))
        else:
            self.font_weight = kwargs['font_weight']

        # lines
        if 'alpha' not in kwargs:
            self.alpha = utl.kwget(kwargs, fcpp, '%s_alpha' % name, kwargs.get('alpha', 1))
        else:
            self.alpha = kwargs['alpha']
        if 'color' not in kwargs:
            self.color = utl.kwget(kwargs, fcpp, ['%s_color' % name, 'color'], kwargs.get('color', '#000000'))
        else:
            self.color = kwargs['color']
        if not isinstance(self.color, RepeatedList) or self.alpha != 1:
            self.color_alpha('color', 'alpha')
        if self.name != 'bar':
            if 'width' not in kwargs:
                self.width = utl.kwget(kwargs, fcpp, '%s_width' % name, kwargs.get('width', 1))
            else:
                self.width = kwargs['width']
            if not isinstance(self.width, RepeatedList):
                self.width = RepeatedList(self.width, 'width')
        if 'style' not in kwargs:
            self.style = utl.kwget(kwargs, fcpp, '%s_style' % name, kwargs.get('style', '-'))
        else:
            self.style = kwargs['style']
        if not isinstance(self.style, RepeatedList):
            self.style = RepeatedList(self.style, 'style')

        # overrides
        attrs = ['color', 'fill_color', 'edge_color']
        for attr in attrs:
            if getattr(self, attr) is None:
                continue
            getattr(self, attr).override = others.get('%s_override' % attr, {})

        # kwargs to ignore
        skip_keys = ['df', 'x', 'y', 'z']
        for k, v in kwargs.items():
            try:
                if not hasattr(self, k) and k not in skip_keys:
                    setattr(self, k, v)
            except AttributeError:
                pass

    @property
    def kwargs(self):
        """kwargs without df."""
        temp = self.__dict__
        if 'df' in temp.keys():
            temp.pop('df')
        return temp

    @property
    def on(self):
        """Check if element is visible."""
        return self._on

    @on.setter
    def on(self, state: bool):
        """Set element visibility.

        Args:
            state: True = visible | False = not visible
        """
        self._on = state

        if not self.on:
            self._size = [0, 0]  # no size if not visible
            self._text = None

        else:
            self._size = self._size_orig
            self._text = self._text_orig

    @property
    def position_xy(self):
        """Return the x, y coordinates of the element."""
        x, y = map(self.position.__getitem__, [0, 3])
        return x, y

    @property
    def size(self):
        """Return the element size, if enabled."""
        if self.on:
            return self._size
        else:
            return [0, 0]

    @size.setter
    def size(self, value: list):
        """Set the element size.

        Args:
            value: width, height
        """
        if self._size_orig is None and value is not None:
            self._size_orig = value

        self._size = value

    @property
    def size_all(self):
        """Return the DataFrame of all the element sizes by subplot index."""
        return self._size_all

    @size_all.setter
    def size_all(self, vals: tuple):
        """Add a row to the table tracking element size by subplot index.

        Args:
            vals: 'ir', 'ic', 'ii', 'width', 'height', 'x0', 'x1', 'y0', 'y1', 'rotation'
                   each value can be a single item or a list
        """
        data = {}
        if len(vals) != len(self.size_cols):
            raise ValueError('incorrect size_all table values')

        for icol, col in enumerate(self.size_cols):
            data[col] = utl.validate_list(vals[icol])

        # temp = pd.DataFrame(data)

        if len(self._size_all) == 0:
            self._size_all = pd.DataFrame(data)

        else:
            self._size_all = pd.concat([self._size_all, pd.DataFrame(data)], axis=0).reset_index(drop=True)

    @property
    def size_all_bg(self):
        """Some elements have a background with a different size (like labels)."""
        return self._size_all_bg

    @size_all_bg.setter
    def size_all_bg(self, vals: tuple):
        """Add a row to the table tracking element background size by subplot index.

        Args:
            vals: 'ir', 'ic', 'ii', 'jj', 'width', 'height', 'x0', 'x1', 'y0', 'y1'
                each value can be a single item or a list
        """
        data = {}
        if len(vals) != len(self.size_cols):
            raise ValueError('incorrect size_all table values')

        for icol, col in enumerate(self.size_cols):
            data[col] = utl.validate_list(vals[icol])

        temp = pd.DataFrame(data)

        if len(self._size_all_bg) == 0:
            self._size_all_bg = temp.copy()

        else:
            self._size_all_bg = pd.concat([self._size_all_bg, temp]).reset_index(drop=True)

    @property
    def size_inches(self):
        """Return the element size in inches, not pixels."""
        if self.on:
            return [self._size[0] / self.dpi, self._size[1] / self.dpi]
        else:
            return [0, 0]

    @property
    def text(self):
        """Return the element text."""
        return self._text

    @text.setter
    def text(self, value: str):
        """Set the element text.

        Args:
            value: element text value
        """
        if self._text_orig is None and value is not None:
            self._text_orig = value

        self._text = value

    def color_alpha(self, attr: str, alpha: str):
        """Add alpha to each color in the color list and make it a RepeatedList.

        Args:
            attr: kwarg key of the "color" attribute to set of the element
                Ex: 'fill_color'
            alpha:  kwarg key of the "alpha" attribute to set of the element
                Ex: 'fill_alpha'
        """
        # MPL < v2 does not support alpha in hex color code
        skip_alpha = False
        if ENGINE == 'mpl' and LooseVersion(mpl.__version__) < LooseVersion('2'):
            skip_alpha = True

        alpha = RepeatedList(getattr(self, alpha), 'temp')

        if not isinstance(getattr(self, attr), RepeatedList):
            self.color_list = utl.validate_list(getattr(self, attr))

            for ic, color in enumerate(self.color_list):
                if isinstance(color, int):
                    color = DEFAULT_COLORS[color]
                if color[0] != '#' and color != 'none':
                    color = '#' + color
                if skip_alpha or color == 'none':
                    astr = ''
                else:
                    astr = str(hex(int(alpha[ic] * 255))
                               )[-2:].replace('x', '0')
                self.color_list[ic] = color[0:7].lower() + astr

            setattr(self, attr, RepeatedList(self.color_list, attr))

        else:
            # Update existing RepeatedList alphas
            setattr(self, attr, copy.copy(getattr(self, attr)))
            new_vals = []
            for ival, val in enumerate(getattr(self, attr).values):
                if skip_alpha:
                    astr = ''
                else:
                    astr = str(hex(int(alpha[ival] * 255))
                               )[-2:].replace('x', '0')
                if len(val) > 7:
                    new_vals += [val[0:-2] + astr]
                else:
                    new_vals += [val + astr]

            getattr(self, attr).values = new_vals

    def size_all_reset(self):
        """Reset the size_all arrays."""

        self._size_all = pd.DataFrame()
        self._size_all_bg = pd.DataFrame()


class DF_Element(Element):
    def __init__(self, name: str = 'None', fcpp: dict = {}, others: dict = {}, **kwargs):
        """Wrapper for Element that is only visible if the `values` attribute
        exists and contains items.  Used for rc labels and legends

        Args:
            name (optional): Name of the element. Defaults to 'None'.
            fcpp (optional): Default kwargs loaded from a theme file.
                Defaults to {}.
            others (optional): Other kwargs which override those in fcpp.
                Typically, these are the user-defined kwargs set in the
                plotting function call.  Defaults to {}.
            kwargs
        """
        super().__init__(name=name, fcpp=fcpp, others=others, **kwargs)

        if not hasattr(self, 'column'):
            self.column = None
        if not hasattr(self, 'values'):
            self.values = []

    @property
    def on(self):
        """Return visibility state; True only if self.values is not None."""
        return True if self._on and self.values is not None \
            and len(self.values) > 0 else False

    @on.setter
    def on(self, state):
        """Set element visibility.

        Args:
            state: True = visible | False = not visible
        """
        self._on = state

        if not self.on:
            self._size = [0, 0]
            self._text = None

        else:
            self._size = self._size_orig
            self._text = self._text_orig


class Legend_Element(DF_Element):
    def __init__(self, name='None', fcpp={}, others={}, **kwargs):
        """Wrapper for DF_Elements for legends

        Args:
            name (optional): Name of the element. Defaults to 'None'.
            fcpp (optional): Default kwargs loaded from a theme file.
                Defaults to {}.
            others (optional): Other kwargs which override those in fcpp.
                Typically, these are the user-defined kwargs set in the
                plotting function call.  Defaults to {}.
            kwargs
        """
        self.cols = ['Key', 'Curve', 'LineType']
        self.default = pd.DataFrame(columns=self.cols, data=[
                                    ['NaN', None, None]], index=[0])

        if not kwargs.get('legend'):
            self._values = pd.DataFrame(columns=self.cols)
        else:
            self._values = self.get_default_values_df()
        if kwargs.get('sort') is True:
            self.sort = True
        else:
            self.sort = False

        super().__init__(name=name, fcpp=fcpp, others=others, **kwargs)

    @property
    def values(self):
        """Get the legend values properly ordered."""
        if len(self._values) <= 1:
            return self._values

        # Re-order single fit lines
        if 'Fit' in self._values.Key.values \
                or 'ref_line' in self._values.LineType.values \
                or 'fill' in self._values.LineType.values:
            df = self._values[self._values.LineType == 'lines']
            fit = self._values[self._values.LineType == 'fit']
            ref = self._values[self._values.LineType == 'ref_line']
            fill = self._values[self._values.LineType == 'fill']
            return pd.concat([df, fit, ref, fill]).reset_index(drop=True)
        else:
            return self._values.sort_index()

    @values.setter
    def values(self, value: pd.DataFrame):
        """Set the legend values.

        Args:
            value: a new legend value formulated as a legend-style DataFrame
                with columns 'Key', 'Curve', and 'LineType'
        """
        self._values = value

    def add_value(self, key: str, curve: 'PlotObj', line_type_name: str):  # noqa: F821
        """Add a new curve to the values dataframe.

        Args:
            key: string name for legend label
            curve: reference to curve obj (plotting engine specific)
            line_type_name: line type description
        """
        temp = pd.DataFrame({'Key': key, 'Curve': curve, 'LineType': line_type_name},
                            index=[len(self._values)])

        # don't add duplicates - this could miss a case where the curve type is actually different
        if len(self.values.loc[(self.values.Key == key)
               & (self.values.LineType == line_type_name)]) == 0:
            self._values = pd.concat([self.values, temp], sort=True)

    def del_value(self, key: str):
        """Delete a value from the values DataFrame.

        Args:
            key: key name to delete
        """
        df = self.values.copy()
        self._values = df[df.Key != key].copy()

    def get_default_values_df(self):
        """Return the default values DataFrame."""
        return self.default.copy()


class ObjectArray:
    def __init__(self):
        """Automatically appending np.array."""
        self._obj = np.array([])

    def __len__(self):
        """Return the array length."""
        return len(self.obj)

    def __getitem__(self, idx: int):
        """Get an array by index.

        Args:
            idx: index of the item to get

        """
        return self.obj[idx]

    @property
    def obj(self):
        """Return the objects."""
        return self._obj

    @obj.setter
    def obj(self, new_obj):
        """Append a new object to the array.

        Args:
            new_obj (multiple): the new object to add to the array
        """
        self._obj = np.append(self._obj, new_obj)

    def reshape(self, r: int, c: int):
        """Reshape the object array.

        Args:
            r: new object array row size
            c: new object array column size
        """
        self._obj = self._obj.reshape(r, c)
