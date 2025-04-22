import pandas as pd
import pdb
import numpy as np
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout, RepeatedList, Element
# from . import layout
from .. import data
import warnings
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


# More works is needed on markers as this set is rather limiting
DEFAULT_MARKERS = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'pentagon', 'hexagram', 'star',
                   'hourglass', 'bowtie', 'asterisk', 'hash', 'y', 'line']
HOLLOW_MARKERS = ['circle', 'square', 'triangle-up', 'diamond', 'pentagon', 'hexagram', 'star', 'hourglass', 'bowtie']


class Layout(BaseLayout):
    def __init__(self, data: 'data.Data', defaults: list = [], **kwargs):  # noqa F821
        """Layout attributes and methods for new engine Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        self.engine = 'plotly'

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)
        self.engine = 'plotly'

        # Update kwargs
        self.kwargs = kwargs
        self.update_markers()

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
        self.imshow.binary = utl.kwget(kwargs, self.fcpp, ['binary', 'binary_string'], None)
        # Add other imshow params later

        self.modebar = Element('modebar', self.fcpp, kwargs,
                               on=utl.kwget(kwargs, self.fcpp, ['modebar', 'modebar_on'], True),
                               bg_color=utl.kwget(kwargs, self.fcpp, 'modebar_bg_color', None),
                               button_active_color=utl.kwget(kwargs, self.fcpp, 'modebar_button_active_color', None),
                               button_color=utl.kwget(kwargs, self.fcpp, 'modebar_button_color', None),
                               logo=utl.kwget(kwargs, self.fcpp, 'modebar_logo', False),
                               remove_buttons=utl.kwget(kwargs, self.fcpp, 'modebar_remove_buttons', []),
                               orientation=utl.kwget(kwargs, self.fcpp, 'modebar_orientaiton', 'v'),
                               visible=utl.kwget(kwargs, self.fcpp, 'modebar_visible', False)
                               )
        self.modebar.size[0] = 25  # the plotly toolbar
        self.ws_leg_modebar = 5  # space between a legend and modebar

        # Check for unsupported kwargs
        # unsupported = []

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
            + np.ceil(self._legy)

        return int(val)

    @property
    def _cbar(self) -> float:
        """Width of all the cbars and cbar ticks/labels and z-labels."""
        val = 0
        # if not self.cbar.on:
        #     return 0

        # val = \
        #     (self.ws_ax_cbar + self.cbar.size[0] + self.tick_labels_major_z.size[0]) \
        #     * (self.ncol if not self.cbar.shared else 1) \
        #     + (self.label_z.size[0] * (self.ncol if self.separate_labels else 1)
        #        + self.ws_ticks_ax * (self.ncol - 1 if not self.cbar.shared else 0)  # btwn z-ticks and the next axes
        #        + self.ws_ticks_ax * self.label_z.on)  # this is between the z-ticks and label_z
        return val

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
        """Left margin.

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
        return int(left)

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
    def _right(self) -> float:
        """
        Width of the space to the right of the axes object (ignores cbar [bar and tick labels] and legend).
        Round up fractional pixels to match figure output.
        """
        # axis to fig right side ws with or without legend
        ws_ax_fig = (self.ws_ax_fig if not self.legend._on or self.legend.location != 0 else 0)

        # sum all parts
        right = np.ceil(ws_ax_fig) + np.ceil(self._labtick_y2) + self._legx

        # modebar
        if self.modebar.orientation == 'v':  # always add extra ws
            right += self.modebar.size[0] + self.ws_leg_modebar * self.legend.on

        # # box title excess
        # if self.box_group_title.on and (self.ws_ax_box_title + self.box_title) > \
        #         self._legx + (self.fig_legend_border if self.legend._on else 0):
        #     right = np.ceil(self.ws_ax_box_title) + np.ceil(self.box_title) \
        #          + np.ceil((self.ws_ax_fig if not self.legend.on else 0))
        # if self.box_group_title.on and self.legend.size[1] > self.axes.size[1]:
        #     right += np.ceil(self.box_title)

        # # Main figure title excess size
        # title_xs_right = np.ceil(self.title.size[0] / 2) \
        #     - np.ceil((right + (self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1)) / 2)) \
        #     - np.ceil(self.legend.size[0])
        # if title_xs_right < 0:
        #     title_xs_right = 0

        # right += title_xs_right

        # # pie labels
        # right += np.ceil(self.pie.xs_right)

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
        toolbar = self.modebar.size[0] if self.modebar.orientation == 'h' else 0
        padding = self.ws_fig_title if self.title.on else self.ws_fig_ax
        return padding + self.title.size[1] + self.ws_title_ax * self.title.on + toolbar

    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
            data: fcp Data object
        """

    def add_fills(self, ir: int, ic: int, df: pd.DataFrame, data: 'Data'):  # noqa: F821
        """Add rectangular fills to the plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data
            data: fcp Data object

        """

    def add_hvlines(self, ir: int, ic: int, df: [pd.DataFrame, None] = None):
        """Add horizontal/vertical lines.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: current data. Defaults to None.
        """

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

    def add_legend(self, leg_vals):
        """Add a legend to a figure."""
        # In plotly, no need to add the layout to the plot so we toggle visibility in set_figure_final_layout
        #   Here we just compute the approximate width of the legend for later sizing
        if self.legend.on:
            # Guesstimate the legend dimensions based on the text size
            longest_key = self.legend.values.Key.loc[self.legend.values.Key.str.len().idxmax()]
            key_dim = utl.get_text_dimensions(longest_key, self.legend.font, self.legend.font_size,
                                              self.legend.font_style, self.legend.font_weight)
            title_dim = utl.get_text_dimensions(self.legend.text, self.legend.font, self.legend.title_font_size,
                                                self.legend.font_style, self.legend.font_weight, scale_x=1, scale_y=1)

            # Add width for the marker part of the legend and padding with axes (based off empirical measurements)
            marker_leg_edge_to_text = 40
            text_to_leg_edge = 5
            legend_key_width = marker_leg_edge_to_text + key_dim[0] + text_to_leg_edge

            # Set approximate legend size
            self.legend.size[0] = max(title_dim[0], legend_key_width) + 2 * self.legend.edge_width
            self.legend.size[1] = \
                (title_dim[1] if self.legend.text != '' else 0) \
                + len(leg_vals) * key_dim[1] \
                + 2 * self.legend.edge_width \
                + 5 * len(self.legend.values) + 10  # padding
            print(self.legend.size)
            print(max(title_dim[0], legend_key_width))

            # Legend styling
            self.ul['legend'] = \
                dict(x=1,
                     y=1,
                     xanchor='left',
                     traceorder='normal',
                     title=dict(text=self.legend.text,
                                font=dict(family=self.legend.font, size=self.legend.title_font_size)),
                     font=dict(
                         family=self.legend.font,
                         size=self.legend.font_size,
                         color='black'  # no styling for this right now
                     ),
                     bgcolor=self.legend.fill_color[0],
                     bordercolor=self.legend.edge_color[0],
                     borderwidth=self.legend.edge_width
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

    def _get_figure_size(self, data: 'Data', temp=False, **kwargs):  # noqa: F821
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            temp: first fig size calc is a dummy calc to get a rough size; don't resize user parameters
            kwargs: user-defined keyword args
        """
        # SKIPPING A LOT FROM MPL, CONSIDER REUSING
        self.box_labels = 0

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
            + self.ws_col * (self.ncol - 1) \
            + self._cbar

        # Figure height
        self.fig.size[1] = \
            self._top \
            + self.axes.size[1] * self.nrow \
            + row_edge_height \
            + self.ws_row * (self.nrow - 1) \
            + self._bottom

    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        self.axes.obj = np.array([[None] * self.ncol] * self.nrow)
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

        # # Need more spacing to account for labels and stuff, same problem as mpl
        # #   Note: this is not the full final size; margins are added in set_figure_final_layout
        # self.fig.size = [self.axes.size[0] * self.ncol, self.axes.size[1] * self.nrow]

        # if self.title.on:
        #     self.ws_title = self.ws_fig_title + self.title.size[1] + self.ws_title_ax
        # else:
        #     self.ws_title = self.ws_fig_ax
        # self.box_labels = 0
        # self._legy = 0

        # self.fig.size[1] = int(
        #     self.ws_title
        #     + (self.label_col.size[1] + self.ws_label_col) * self.label_col.on
        #     + self.title_wrap.size[1] + self.label_wrap.size[1]
        #     # + self._labtick_x2
        #     + self.axes.size[1] * self.nrow
        #     # + self._labtick_x
        #     + self.ws_fig_label
        #     + self.ws_row * (self.nrow - 1)
        #     + self.box_labels) \
        #     + self._legy \
        #     + self.pie.xs_top \
        #     + self.pie.xs_bottom \
        #     + self.tick_y_top_xs

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

    def plot_imshow(self, ir: int, ic: int, df: pd.DataFrame, data: 'data.Data'):
        """Plot an image.  Copies strategies / code from plotly.express._imshow

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            data: Data object

        Returns:
            imshow plot obj
        """
        # TODO: add colorbar support
        # Make the imshow plot
        # plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        # self.axes.obj[ir, ic] = [go.Image(z=df)]
        # self.cmap[plot_num], vmin=zmin, vmax=zmax, interpolation=self.imshow.interp, aspect='auto')
        # im.set_clim(zmin, zmax)

        # # Add a cmap
        # if self.cbar.on and (not self.cbar.shared or ic == self.ncol - 1):
        #     self.cbar.obj[ir, ic] = self.add_cbar(ax, im)

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

        # For 2D data, use Heatmap trace, unless self.imshow.binary is True
        if df.ndim == 2 and not self.imshow.binary:
            if df.dtype in [np.uint8, np.uint16, np.uint32, int]:
                hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z:.0f}<extra></extra>'
            else:
                hovertemplate = 'x: %{x}<br>y: %{y}<br>z: %{z:.2f}<extra></extra>'
            self.axes.obj[ir, ic] = [go.Heatmap(z=df,
                                                colorscale='gray',
                                                hovertemplate=hovertemplate
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
                            df[..., ch], (df[..., ch].min(), df[..., ch].max()), np.uint8,
                        )
                        for ch in range(df.shape[-1])
                    ],
                    axis=-1,
                )

        img_str = putl.image_array_to_data_uri(df, backend='auto',  compression=4,  ext='png')  # parameterize later
        self.axes.obj[ir, ic] = [go.Image(source=img_str)]  # , x0=x0, y0=y0, dx=dx, dy=dy)

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

    def plot_polygon(self, ir: int, ic: int, points: list, **kwargs):
        """Plot a polygon.

        Args:
            ir: subplot row index
            ic: subplot column index
            points: list of floats that defint the points on the polygon
            kwargs: keyword args
        """

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
        if line_type.on and self.markers.on:
            mode = 'lines+markers'
        elif line_type.on:
            mode = 'lines'
        else:
            mode = 'markers'

        # Make the plot
        if not self.axes.obj[ir, ic]:
            self.axes.obj[ir, ic] = []

        # Set marker type
        if self.markers.on and self.markers.type[iline] in HOLLOW_MARKERS:
            marker_symbol = self.markers.type[iline] + ('-open' if not self.markers.filled else '')
        else:
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
        if self.markers.on:
            mls = dict(marker_symbol=marker_symbol,
                       marker_size=self.markers.size[iline],
                       marker_color=self.markers.fill_color[iline],
                       marker=dict(line=dict(color=self.markers.edge_color[iline],
                                             width=self.markers.edge_width[iline])),
                       line=dict(width=self.lines.width[iline])
                       )
        elif self.lines.on:
            mls = dict(line=dict(width=self.lines.width[iline],
                                 color=self.lines.color[iline]))
        else:
            mls = {}

        # Make the scatter trace
        show_legend = False
        if leg_name not in self.legend.values['Key'].values:
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
            self.ul[f'{ax}axis_style'] = dict(showline=show, linecolor=self.axes.edge_color[0],
                                              linewidth=self.axes.edge_width, mirror=True)

    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """
        dash_lookup = {'-': 'solid', '--': 'dash', '.': 'dot'}
        for ss in ['', '2']:
            for ax in ['x', 'y']:
                grid = getattr(self, f'grid_major_{ax}{ss}')
                if grid is not None:
                    dash = dash_lookup.get(grid.style[0], grid.style[0])
                    self.ul[f'{ax}{ss}grid'] = dict(gridcolor=grid.color[0],
                                                    gridwidth=grid.width[0],
                                                    griddash=dash,
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

            # Create a dict of the label props to apply later
            plot_num = utl.plot_num(ir, ic, self.ncol)
            plot_num = '' if plot_num == 1 else str(plot_num)
            key = f'{ax[0]}axis{plot_num}_title'
            self.ul[f'{ax}axis_title'][key] = \
                dict(text=label.text,
                     font=dict(family=label.font,
                               size=label.font_size,
                               color=label.font_color),
                     standoff=self.ws_label_tick)
            lab = getattr(self, f'label_{ax}')
            lab.size = utl.get_text_dimensions(lab.text, lab.font, lab.font_size, lab.font_style,
                                               lab.font_weight, lab.rotation)

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        # TODO address secondary axes and what happens if None
        if str(self.axes.scale).lower() in LOGX:
            self.ul['xaxis_range'][ir, ic] = \
                np.array([np.log10(ranges['xmin'][ir, ic]), np.log10(ranges['xmax'][ir, ic])])
        else:
            self.ul['xaxis_range'][ir, ic] = np.array([ranges['xmin'][ir, ic], ranges['xmax'][ir, ic]])
        if str(self.axes.scale).lower() in LOGY:
            self.ul['yaxis_range'][ir, ic] = \
                np.array([np.log10(ranges['ymin'][ir, ic]), np.log10(ranges['ymax'][ir, ic])])
        else:
            self.ul['yaxis_range'][ir, ic] = np.array([ranges['ymin'][ir, ic], ranges['ymax'][ir, ic]])

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Wrap title

        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            if self.label_row.names:
                lab = f'{self.label_row.text}={self.label_row.values[ir]}'
            else:
                lab = self.label_row.values[ir]
            lab = self._set_weight_and_style_str(lab, self.label_row)

            # add margin to the figure for the labels (one time only) and set style
            if ir == 0 and ic == 0:
                self.fig.obj.update_layout(margin=dict(r=self.ws_label_row + self.label_row.size[1]))

            # add the rectangle
            self.fig.obj.add_shape(type='rect', x0=1.05, y0=1, x1=1.2, y1=0,
                                   fillcolor=self.label_row.fill_color[ir, ic],
                                   line=dict(
                                       color=self.label_row.edge_color[ir, ic],
                                       width=self.label_row.edge_width,
                                   ),
                                   xref="x domain", yref="y domain", row=ir + 1, col=ic + 1,
                                   )

            # add the label
            font = dict(family=self.label_row.font, size=self.label_row.font_size, color=self.label_row.font_color)
            self.fig.obj.add_annotation(font=font, x=1.18, y=0.5,
                                        showarrow=False, text=lab, textangle=90,
                                        xanchor='right', yanchor='middle',
                                        xref='x domain', yref='y domain', row=ir + 1, col=ic + 1)

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
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
                lab = self._set_weight_and_style_str(lab, self.label_col)

                # add margin to the figure for the labels (one time only) and set style
                if ir == 0 and ic == 0:
                    self.fig.obj.update_layout(margin=dict(t=self.ws_label_col + self.label_col.size[1]))

                # add the rectangle
                self.fig.obj.add_shape(type='rect', x0=0, y0=1.05, x1=1, y1=1.2,
                                       fillcolor=self.label_col.fill_color[ir, ic],
                                       line=dict(color=self.label_col.edge_color[ir, ic],
                                                 width=self.label_col.edge_width),
                                       xref="x domain", yref="y domain", row=ir + 1, col=ic + 1,
                                       )

                # add the label
                font = dict(family=self.label_col.font, size=self.label_col.font_size, color=self.label_col.font_color)
                self.fig.obj.add_annotation(font=font, y=1.16, x=0.5,
                                            showarrow=False, text=lab, textangle=0,
                                            xanchor='center', yanchor='top',
                                            xref='x domain', yref='y domain', row=ir + 1, col=ic + 1)

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
                )

    def set_figure_final_layout(self, data, **kwargs):
        """Final adjustment of the figure size and plot spacing.

        Args:
            data: Data object
            kwargs: keyword args
        """
        # Set the figure size
        self._get_figure_size(data)

        # Update the title position
        self.ul['title']['x'] = \
            0.5 + (self.label_y.font_size + self.tick_labels_major_y.font_size) / self.fig.size[0]
        self.ul['title']['y'] = \
            1 - ((self.modebar.size[0] if (self.modebar.visible and self.modebar.orientation == 'h') else 0) +
                 self.title.font_size / 2) / self.fig.size[1]
        # ws_leg_modebar??

        # Update the x/y axes
        for ax in data.axs_on:
            kw = {}
            kw.update(self.ul[f'{ax}ticks'])
            kw.update(self.ul[f'{ax}axis_style'])
            kw.update(self.ul[f'{ax}grid'])
            kw.update(self.ul[f'{ax}scale'])
            try:
                getattr(self.fig.obj, f'update_{ax}axes')(kw)
            except:  # noqa
                pass

        # Update the axes labels
        axis_labels = self.ul['xaxis_title']
        axis_labels.update(self.ul['yaxis_title'])

        # Add space for tick labels
        self.fig.obj.update_yaxes(ticksuffix=" ")
        if self.axes.twin_x:
            self.fig.obj['layout']['yaxis2'].update(dict(tickprefix=" "))

        # Iterate through subplots to add the traces
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                # Add the traces
                for trace in range(0, len(self.axes.obj[ir, ic])):
                    if self.axes.obj[ir, ic][trace]['yaxis'] == 'y2':
                        self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1, secondary_y=True)
                    else:
                        self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1)

                # Set the ranges
                self.fig.obj.update_layout(xaxis_range=self.ul['xaxis_range'][ir, ic])
                self.fig.obj.update_layout(yaxis_range=self.ul['yaxis_range'][ir, ic])

        # Get the tick sizes
        for ax in data.axs_on:
            self._set_tick_sizes(data)

        # Update the figure layout
        self.fig.obj.update_layout(autosize=False,  # do we need this?
                                   height=self.fig.size[1],
                                   legend_title_text=self.legend.text,
                                   margin=dict(l=self._left,
                                               r=self._right,
                                               t=self._top,
                                               b=self._bottom),
                                   modebar=dict(orientation=self.modebar.orientation,
                                                bgcolor=self.modebar.bg_color,
                                                color=self.modebar.button_color,
                                                activecolor=self.modebar.button_active_color),
                                   paper_bgcolor=self.fig.fill_color[0],
                                   plot_bgcolor=self.ul['plot_bgcolor'],
                                   showlegend=self.legend.on,
                                   title=self.ul['title'],
                                   width=self.fig.size[0],
                                   **axis_labels)

        # Update the plot labels
        self.fig.obj.update_layout()

        if self.axes.twin_x:
            self.fig.obj['layout']['yaxis2'].update(self.ul['y2grid'])
            self.fig.obj['layout']['yaxis2']['title'] = self.ul['y2axis_title']['yaxis_title']

        if self.axes.twin_y:
            self.fig.obj.data[1].update(xaxis='x2')
            # Have to do this manually since there is no update_xaxes2 function
            for k, v in self.ul['x2grid'].items():
                if hasattr(self.fig.obj.layout.xaxis2, k):
                    setattr(self.fig.obj.layout.xaxes2, k, v)
            for k, v in self.ul['x2axis_title'].items():
                setattr(self.fig.obj.layout.xaxis2.title, k, v)

        # Adjust the legend position (only for outside of plot right now)
        if self.legend.on:
            leg_x = (self._labtick_y2 + self.ws_ax_leg + self.axes.edge_width) / self.axes.size[0]
            self.ul['legend']['x'] += leg_x
            self.ul['legend']['y'] += self.axes.edge_width / self.axes.size[1]
            self.fig.obj.update_layout(legend=self.ul['legend'])

        # Set the modebar config
        self.modebar.obj = {'displaylogo': self.modebar.logo,
                            'modeBarButtonsToRemove': self.modebar.remove_buttons}
        if self.modebar.visible:
            self.modebar.obj['displayModeBar'] = self.modebar.visible

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.text is None:
            return

        # TODO: deal with other alignments
        self._set_weight_and_style('title')
        self.title.size[1] = self.title.font_size
        self.ul['title'] = \
            dict(text=self.title.text,
                 xanchor='center',
                 font=dict(family=self.title.font,
                           size=self.title.font_size,
                           color=self.title.font_color,
                           )
                 )

    def _set_tick_sizes(self, data):
        """
        Try to guess the tick rounding in order to approximate the tick size.  Auto tick labels are calculated by js
        and cannot be queried by python

        See: https://github.com/plotly/plotly.js/blob/4ed586a6402073cc5c50a40cad5f652d7472fcce/src/plots/cartesian/axes.js#L652

        Args:
            data: data object
        """  # noqa: E501
        return
        # def getBase(v, roughDTick):
        #     return v ** np.floor(np.log(roughDTick) / np.log(10))

        # def roundUp(val, arrayIn, reverse):
        #     """
        #     Modified from plotly.js

        #     Return the smallest element from (sorted) array arrayIn that's bigger than val,
        #     or (reverse) the largest element smaller than val
        #     used to find the best tick given the minimum (non-rounded) tick
        #     particularly useful for date/time where things are not powers of 10
        #     binary search is probably overkill here...
        #     """
        #     low = 0
        #     high = len(arrayIn) - 1
        #     c = 0
        #     dlow = 0 if reverse else 1
        #     dhigh = 1 if reverse else 0
        #     rounded = np.ceil if reverse else np.floor
        #     # c is just to avoid infinite loops if there's an error
        #     while low < high and c < 100:
        #         mid = rounded((low + high) / 2)
        #         if arrayIn[mid] <= val:
        #             low = mid + dlow
        #         else:
        #             high = mid - dhigh
        #         c += 1
        #     return arrayIn[low]

        # def roundDTick(roughDTick, base, roundingSet):
        #     return base * roundUp(roughDTick / base, roundingSet)

        # roundBase10 = [2, 5, 10]
        # roundBase24 = [1, 2, 3, 6, 12]
        # roundBase60 = [1, 2, 5, 10, 15, 30]
        # # 2&3 day ticks are weird, but need something btwn 1&7
        # roundDays = [1, 2, 3, 7, 14]
        # # these don't have to be exact, just close enough to round to the right value
        # # approx. tick positions for log axes, showing all (1) and just 1, 2, 5 (2)
        # roundLog1 = [-0.046, 0, 0.301, 0.477, 0.602, 0.699, 0.778, 0.845, 0.903, 0.954, 1]
        # roundLog2 = [-0.301, 0, 0.301, 0.699, 1]
        # roundAngles = [15, 30, 45, 90, 180]

        # for ax in data.axs_on:
        #     ww, hh = 0, 0
        #     for ir, ic in np.ndindex(self.axes.obj.shape):
        #         vmin, vmax = data.ranges[f'{ax}min'][ir, ic], data.ranges[f'{ax}max'][ir, ic]
        #         if self.axes.scale in getattr(layout, f'LOG{ax.upper()}'):
        #             db()
        #         else:
        #             # ax.tick0 = 0;
        #             # base = getBase(10);
        #             # ax.dtick = roundDTick(roughDTick, base, roundBase10);

        #             db()

        # for ax in data.axs_on:
        #     tick = getattr(self, f'tick_labels_major_{ax}')
        #     longest = ''
        #     for trace in self.fig.obj['data']:
        #         longest = max([str(f) for f in trace[ax]] + [longest], key=len)
        #     tick.size = utl.get_text_dimensions(longest, tick.font, tick.font_size, tick.font_style,
        #                                         tick.font_weight, tick.rotation)
        pass

    def _set_weight_and_style(self, element: str):
        """Add html tags to the text string of an element to address font weight and style.

        Args:
            element: name of the Element object to modify; results are stored in place
        """
        if getattr(self, element).font_weight == 'bold':
            getattr(self, element).text = f'<b>{getattr(self, element).text}</b>'
        if getattr(self, element).font_style == 'italic':
            getattr(self, element).text = f'<i>{getattr(self, element).text}</i>'

    def _set_weight_and_style_str(self, text: str, element: Element):
        """Add html tags to any text string.

        Args:
            text: string to format
            element: Element object to get the style parameters

        Returns:
            formated text string
        """
        if element.font_weight == 'bold':
            text = f'<b>{text}</b>'
        if element.font_style == 'italic':
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

            self.fig.obj.show(config=self.modebar.obj)

    def update_markers(self):
        """Update the marker list to valid option for new engine."""

        if 'marker_type' in self.kwargs.keys():
            marker_list = self.kwargs['marker_type']
        elif self.kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(self.kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        self.markers.type = RepeatedList(marker_list, 'markers')
