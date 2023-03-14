import pandas as pd
import pdb
import numpy as np
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout, RepeatedList
from .. import data
import warnings
import plotly.offline as pyo
import plotly.graph_objects as go
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

        # Engine-specific kwargs; store update parameters in kwargs to avoid calling "update_layout" multiple times
        self.kwargs['ul_plot_bgcolor'] = None
        self.kwargs['ul_title'] = {}
        self.kwargs['ul_xaxis_range'] = np.array([[None] * self.ncol] * self.nrow)
        self.kwargs['ul_xaxis_style'] = {}
        self.kwargs['ul_xgrid'] = {}
        self.kwargs['ul_x2grid'] = {}
        self.kwargs['ul_xscale'] = {}
        self.kwargs['ul_xticks'] = {}
        self.kwargs['ul_xaxis_title'] = {}
        self.kwargs['ul_x2axis_title'] = {}
        self.kwargs['ul_yaxis_range'] = np.array([[None] * self.ncol] * self.nrow)
        self.kwargs['ul_yaxis_style'] = {}
        self.kwargs['ul_ygrid'] = {}
        self.kwargs['ul_y2grid'] = {}
        self.kwargs['ul_yscale'] = {}
        self.kwargs['ul_yticks'] = {}
        self.kwargs['ul_yaxis_title'] = {}
        self.kwargs['ul_y2axis_title'] = {}

        # Other engine specific attributes
        self.ws_toolbar = 20

        # Check for unsupported kwargs
        # unsupported = []

    @property
    def _bottom(self) -> float:
        """Bottom margin.

        Returns:
            margin in pixels
        """
        return 0  # self.ws_label_tick + self.label_y.size[1]

    @property
    def _left(self) -> float:
        """Left margin.

        Returns:
            margin in pixels
        """

        return 0  # self.ws_label_tick + self.label_y.size[1]

    @property
    def _right(self) -> float:
        """Right margin.

        Returns:
            margin in pixels
        """
        return self.legend.size[0] * self.legend.on + (self.label_y2.size[1] + self.ws_label_tick) * self.axes.twin_x

    @property
    def _top(self) -> float:
        """Top margin.

        Returns:
            margin in pixels
        """
        return self.ws_fig_title + self.title.size[1] + self.ws_title_ax * self.title.on + self.ws_toolbar

    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
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
            longest = self.legend.values.Key.loc[self.legend.values.Key.str.len().idxmax()]
            self.legend.size[0] = utl.get_text_dimensions(longest, self.legend.font,
                                                          self.legend.font_size, self.legend.font_style,
                                                          self.legend.font_weight)[0]
            # add width for the marker part of the legend and padding with axes
            self.legend.size[0] += 3 * self.markers.size.max() + 10

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

    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        self.axes.obj = np.array([[None] * self.ncol] * self.nrow)
        specs = [[{"secondary_y": self.axes.twin_x}] * self.ncol] * self.nrow

        # self.label_col.obj = np.array([None] * self.ncol)
        # self.label_wrap.obj = np.array([[None] * self.ncol] * self.nrow)
        # self.label_row.obj = np.array([None] * self.nrow)
        # for ir in range(0, self.nrow):
        #     for ic in range(0, self.ncol):
        #         self.axes.obj[ir, ic] = go.Figure(
        #             layout={'width': self.axes.size[0], 'height': self.axes.size[1]},
        #         )
        self.fig.obj = make_subplots(rows=self.nrow,
                                     cols=self.ncol,
                                     shared_xaxes=self.axes.share_x,
                                     shared_yaxes=self.axes.share_y,
                                     specs=specs,
                                     )

        if self.axes.twin_y:
            self.fig.obj.update_layout(xaxis2={'anchor': 'y', 'overlaying': 'x', 'side': 'top'},
                                       yaxis_domain=[0, 0.94])

        # Need more spacing to account for labels and stuff, same problem as mpl
        #   Note: this is not the full final size; margins are added in set_figure_final_layout
        self.fig.size = self.axes.size[0] * self.ncol, self.axes.size[1] * self.nrow

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
        """Plot an image.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data to plot
            data: Data object

        Returns:
            imshow plot obj
        """

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

        # Make the scatter trace
        self.axes.obj[ir, ic] += [
            go.Scatter(x=dfx[mask],
                       y=df[y][mask],
                       name=leg_name,
                       mode=mode,
                       line=None,
                       marker_symbol=marker_symbol,
                       marker_size=self.markers.size[iline],
                       marker_color=self.markers.fill_color[iline],
                       marker=dict(line=dict(color=self.markers.edge_color[iline],
                                             width=self.markers.edge_width[iline])),
                       xaxis=xaxis,
                       yaxis=yaxis,
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

    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Background color
        self.kwargs['ul_plot_bgcolor'] = self.axes.fill_color[utl.plot_num(ir, ic, self.ncol)]

        # Set the axes borders/spines
        for ax in ['x', 'y']:
            show = False
            if ax == 'x' and (self.axes.spine_bottom or self.axes.spine_top):
                show = True
            if ax == 'y' and (self.axes.spine_left or self.axes.spine_right):
                show = True
            self.kwargs[f'ul_{ax}axis_style'] = dict(showline=show, linecolor=self.axes.edge_color[0],
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
                    self.kwargs[f'ul_{ax}{ss}grid'] = dict(gridcolor=grid.color[0],
                                                           gridwidth=grid.width[0],
                                                           griddash=dash,
                                                           )

    def set_axes_labels(self, ir: int, ic: int):
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Set the font weight and style
        self._set_weight_and_style('label_x')
        self._set_weight_and_style('label_x2')  # not implemented
        self._set_weight_and_style('label_y')
        self._set_weight_and_style('label_y2')  # not implemented
        self._set_weight_and_style('label_z')  # not implemented

        # Set axes label properties
        for ss in ['', '2']:
            for ax in ['x', 'y']:
                self.kwargs[f'ul_{ax}{ss}axis_title'] = \
                    dict(text=getattr(self, f'label_{ax}{ss}').text,
                         font=dict(family=getattr(self, f'label_{ax}{ss}').font,
                                   size=getattr(self, f'label_{ax}{ss}').font_size,
                                   color=getattr(self, f'label_{ax}{ss}').font_color,
                                   ),
                         standoff=self.ws_label_tick,
                         )
                lab = getattr(self, f'label_{ax}{ss}')
                lab.size = utl.get_text_dimensions(lab.text, lab.font, lab.font_size, lab.font_style, lab.font_weight)

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        # TODO address secondary axes and what happens if None
        if str(self.axes.scale).lower() in LOGX:
            self.kwargs['ul_xaxis_range'][ir, ic] = [np.log10(ranges[ir, ic]['xmin']), np.log10(ranges[ir, ic]['xmax'])]
        else:
            self.kwargs['ul_xaxis_range'][ir, ic] = [ranges[ir, ic]['xmin'], ranges[ir, ic]['xmax']]
        if str(self.axes.scale).lower() in LOGY:
            self.kwargs['ul_yaxis_range'][ir, ic] = [np.log10(ranges[ir, ic]['ymin']), np.log10(ranges[ir, ic]['ymax'])]
        else:
            self.kwargs['ul_yaxis_range'][ir, ic] = [ranges[ir, ic]['ymin'], ranges[ir, ic]['ymax']]

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    def set_axes_scale(self, ir: int, ic: int):
        """
        Set the axis scale.
        """
        if str(self.axes.scale).lower() in LOGX:
            self.kwargs['ul_xscale'] = dict(type='log')
        if str(self.axes.scale).lower() in LOGY:
            self.kwargs['ul_yscale'] = dict(type='log')

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # TODO: minor ticks, rotation, categorical, log/exponential
        for ax in ['x', 'y']:
            direction = 'outside' if getattr(self, f'ticks_major_{ax}').direction == 'out' else 'inside'
            self.kwargs[f'ul_{ax}ticks'] = dict(
                tickfont_size=getattr(self, f'tick_labels_major_{ax}').font_size,
                ticks=direction,
                tickcolor=getattr(self, f'ticks_major_{ax}').color[0],
                ticklen=getattr(self, f'ticks_major_{ax}')._size[0],
                tickwidth=getattr(self, f'ticks_major_{ax}')._size[1],
                )

    def set_figure_final_layout(self, data, **kwargs):
        """Final adjustment of the figure size and plot spacing.

        Args:
            data: Data object
            kwargs: keyword args
        """
        # Update the x/y axes
        for ax in ['x', 'y']:
            kw = {}
            kw.update(self.kwargs[f'ul_{ax}ticks'])
            kw.update(self.kwargs[f'ul_{ax}axis_style'])
            kw.update(self.kwargs[f'ul_{ax}grid'])
            kw.update(self.kwargs[f'ul_{ax}scale'])
            getattr(self.fig.obj, f'update_{ax}axes')(kw)

        # Iterate through subplots
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                # Add the traces
                for trace in range(0, len(self.axes.obj[ir, ic])):
                    if self.axes.obj[ir, ic][trace]['yaxis'] == 'y2':
                        self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1, secondary_y=True)
                    else:
                        self.fig.obj.add_trace(self.axes.obj[ir, ic][trace], row=ir + 1, col=ic + 1)

                # Set the ranges
                self.fig.obj.update_layout(xaxis_range=self.kwargs['ul_xaxis_range'][ir, ic])
                self.fig.obj.update_layout(yaxis_range=self.kwargs['ul_yaxis_range'][ir, ic])

        # Update the layout
        self.fig.obj.update_layout(autosize=True,  # do we need this?
                                   height=self.fig.size[1] + self._top + self._bottom,
                                   legend_title_text=self.legend.text,
                                   margin=dict(l=self._left,
                                               r=self._right,
                                               t=self._top,
                                               b=self._bottom),
                                   paper_bgcolor=self.fig.fill_color[0],
                                   plot_bgcolor=self.kwargs['ul_plot_bgcolor'],
                                   showlegend=self.legend.on,
                                   title=self.kwargs['ul_title'],
                                   width=self.fig.size[0] + self._left + self._right,
                                   xaxis_title=self.kwargs['ul_xaxis_title'],
                                   yaxis_title=self.kwargs['ul_yaxis_title'],
                                   )

        if self.axes.twin_x:
            leg = dict(yanchor='top', y=0.99, xanchor='right', x=1.55)  # this needs an algo
            self.fig.obj['layout']['yaxis2'].update(self.kwargs['ul_y2grid'])
            self.fig.obj.update_layout(yaxis2_title=self.kwargs['ul_y2axis_title'], legend=leg)

        if self.axes.twin_y:
            self.fig.obj.data[1].update(xaxis='x2')
            self.fig.obj.update_xaxes2(self.kwargs['ul_x2grid'])

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.text is None:
            return

        # TODO: deal with other alignments
        self._set_weight_and_style('title')
        self.title.size[1] = self.title.font_size
        self.kwargs['ul_title'] = \
            dict(text=self.title.text,
                 xanchor='center',
                 x=0.5 + (self.label_y.font_size + self.tick_labels_major_y.font_size) / self.fig.size[0],
                 y=1 - (self.ws_toolbar + self.title.font_size / 2) / self.fig.size[1],
                 font=dict(family=self.title.font,
                           size=self.title.font_size,
                           color=self.title.font_color,
                           )
                 )

    def _set_weight_and_style(self, element: str):
        """Add html tags to the text string of an element to address font weight and style.

        Args:
            element: name of the Element class to modify; results are stored in place
        """
        if getattr(self, element).font_weight == 'bold':
            getattr(self, element).text = f'<b>{getattr(self, element).text}</b>'
        if getattr(self, element).font_style == 'italic':
            getattr(self, element).text = f'<i>{getattr(self, element).text}</i>'

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
            if 'zmqshell.ZMQInteractiveShell' in app:
                pyo.init_notebook_mode()
            self.fig.obj.show()

    def update_markers(self):
        """Update the marker list to valid option for new engine."""

        if 'marker_type' in self.kwargs.keys():
            marker_list = self.kwargs['marker_type']
        elif self.kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(self.kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        self.markers.type = RepeatedList(marker_list, 'markers')
