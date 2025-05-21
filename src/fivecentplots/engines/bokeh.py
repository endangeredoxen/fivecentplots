import pandas as pd
import pdb
import numpy as np
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout, RepeatedList, Element
from .. import data
import warnings
import bokeh.plotting as bp
import bokeh.layouts as bl
import bokeh.models as bm
import bokeh.io.state as bs


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")  # weird error in boxplot w/ no groups


db = pdb.set_trace


DASHES = {'-': 'solid',
          'solid': 'solid',
          '--': 'dashed',
          'dashed': 'dashed',
          '.': 'dotted',
          'dotted': 'dotted',
          '.-': 'dotdash',
          'dotdash': 'dotdash',
          '-.': 'dashdot',
          'dashdot': 'dashdot'}


DEFAULT_MARKERS = ['circle', 'cross', 'square', 'x', 'diamond', 'asterisk', 'inverted_triangle', 'y',
                   'star', 'plus', 'dash', 'dot', 'square_pin', 'triangle_pin', 'circle_cross', 'square_cross',
                   'diamond_cross', 'circle_x', 'square_x', 'circle_y', 'circle_dot', 'square_dot', 'diamond_dot',
                   'triangle_dot', 'hex_dot', 'star_dot',
                   ]


def fill_alpha(hexstr):
    """Break an 8-digit hex string into a hex color and a fractional alpha value."""
    if len(hexstr) == 6:
        return hexstr, 1

    return hexstr[0:7], int(hexstr[-2:], 16) / 255


def format_marker(fig, marker):
    """Get the marker function."""
    return getattr(fig, marker)


class Layout(BaseLayout):
    def __init__(self, data: 'data.Data', defaults: list = [], **kwargs):  # noqa F821
        """Layout attributes and methods for bokeh Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        self.engine = 'bokeh'

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)

        # Update kwargs
        if not kwargs.get('save_ext'):
            kwargs['save_ext'] = '.html'
        self.kwargs = kwargs
        self.update_markers()

        # Engine-specific kwargs
        location = utl.kwget(kwargs, self.fcpp, 'toolbar_location', 'below')
        self.toolbar = Element('toolbar', self.fcpp, kwargs,
                               on=True if location else None,
                               location=location,
                               sticky=utl.kwget(kwargs, self.fcpp, ['toolbar_sticky', 'sticky'], True),
                               tools=utl.kwget(kwargs, self.fcpp, ['toolbar_tools', 'tools'],
                                               'pan,wheel_zoom,box_zoom,reset'),
                               # active_zoom=utl.kwget(kwargs, self.fcpp, 'toolbar_active_zoom', 'wheel_zoom,box_zoom')
                               )

        # Check for unsupported kwargs
        # unsupported = []

    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
            data: fcp Data object
        """
        pass

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
        pass

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
        pass

    def add_legend(self, leg_vals):
        """Add a legend to a figure."""
        if not self.legend.on or len(self.legend.values) == 0:
            return

        title = self.axes.obj[0, -1].circle(0, 0, size=1.00000001, color=None)
        tt = [(self.legend.text, [title])]
        for irow, row in self.legend.values.iterrows():
            if row['Key'] == 'NaN':
                continue
            tt += [(row['Key'], [row['Curve']])]
        legend = bm.Legend(items=tt, location='top_right')

        # Style the legend
        legend.border_line_width = self.legend.edge_width
        legend.border_line_color = self.legend.edge_color[0]
        legend.border_line_alpha = self.legend.edge_alpha
        legend.label_text_font_size = f'{self.legend.font_size}px'
        legend.title_text_font_size = f'{self.legend.font_size}px'  # could separate this parameter in future

        # Add the legend
        self.axes.obj[0, -1].add_layout(legend, 'right')

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
        pass

    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        self.axes.obj = np.array([[None] * self.ncol] * self.nrow)
        self.label_col.obj = np.array([None] * self.ncol)
        self.label_wrap.obj = np.array([[None] * self.ncol] * self.nrow)
        self.label_row.obj = np.array([None] * self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                x_type, y_type = self._set_axes_type(ir, ic, data)
                x_range, y_range = self._set_axes_custom_range(ir, ic, data)
                self.axes.obj[ir, ic] = bp.figure(x_axis_type=x_type,
                                                  y_axis_type=y_type,
                                                  frame_width=self.axes.size[0],  # sizing is so easy! thank you bokeh
                                                  frame_height=self.axes.size[1],
                                                  tools=self.toolbar.tools,
                                                  toolbar_sticky=self.toolbar.sticky,
                                                  )
                if self.label_row.on:
                    self.label_row.obj[ir] = bm.Title()
                if self.label_col.on:
                    self.label_col.obj[ic] = bm.Title()
                if self.label_wrap.on:
                    self.label_wrap.obj[ir, ic] = bm.Title()

                # Twinning
                if self.axes.twin_x:
                    self.axes.obj[ir, ic].extra_y_ranges = {'y2': bm.Range1d(start=0, end=1)}
                    self.axes.obj[ir, ic].add_layout(bm.LinearAxis(y_range_name='y2'), 'right')
                elif self.axes.twin_y:
                    self.axes.obj[ir, ic].extra_x_ranges = {'x2': bm.Range1d(start=0, end=1)}
                    self.axes.obj[ir, ic].add_layout(bm.LinearAxis(x_range_name='x2'), 'above')

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def plot_polygon(self, ir: int, ic: int, points: list, **kwargs):
        """Plot a polygon.

        Args:
            ir: subplot row index
            ic: subplot column index
            points: list of floats that defint the points on the polygon
            kwargs: keyword args
        """
        pass

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

        # TWINNING
        if self.axes.twin_x and twin:
            y_range_name = 'y2'
        else:
            y_range_name = 'default'
        if self.axes.twin_y and twin:
            x_range_name = 'x2'
        else:
            x_range_name = 'default'

        # Make the points
        points = None
        if self.markers.on and not marker_disable:
            if self.markers.jitter:
                df[x] = np.random.normal(df[x], 0.03, size=len(df[y]))
            marker = format_marker(self.axes.obj[ir, ic],
                                   self.markers.type[iline])
            ecolor, ealpha = fill_alpha(self.markers.edge_color[iline])
            fcolor, falpha = fill_alpha(self.markers.fill_color[iline])
            if marker != 'None':
                points = marker(df[x], df[y],
                                fill_color=fcolor if self.markers.filled else None,
                                fill_alpha=falpha,
                                line_color=ecolor,
                                line_alpha=ealpha,
                                size=self.markers.size[iline],
                                x_range_name=x_range_name,
                                y_range_name=y_range_name,
                                )
            else:
                points = marker(df[x], df[y],
                                color=line_type.color[iline],
                                linestyle=line_type.style[iline],
                                linewidth=line_type.width[iline],
                                x_range_name=x_range_name,
                                y_range_name=y_range_name,
                                )

        # Make the line
        lines = None
        if line_type.on:
            lines = self.axes.obj[ir, ic].line(df[x], df[y],
                                               color=line_type.color[iline][0:7],
                                               line_dash=DASHES[line_type.style[iline]],
                                               line_width=line_type.width[iline],
                                               x_range_name=x_range_name,
                                               y_range_name=y_range_name,
                                               )

        # Add a reference to the line to self.lines
        if self.legend.location == 0:
            if ir == 0 and ic == self.ncol - 1:
                self.legend.add_value(leg_name, points if points is not None else lines, line_type_name)
        elif leg_name is not None and leg_name not in list(self.legend.values['Key']):
            self.legend.add_value(leg_name, points if points is not None else lines, line_type_name)

    def restore(self):
        """Undo changes to default plotting library parameters."""
        pass

    def save(self, filename: str, idx: int = 0):
        """Save a plot window.

        Args:
            filename: name of the file
            idx (optional): figure index in order to set the edge and face color of the
                figure correctly when saving. Defaults to 0.
        """
        bp.output_file(filename)
        # by row
        if self.ncol == 1 and self.nrow > 1:
            self.saved = bl.column(self.axes.obj.flatten().tolist())
        # by column
        elif self.ncol > 1 and self.nrow == 1:
            self.saved = bl.row(self.axes.obj.flatten().tolist())
        # grid
        else:
            self.saved = bl.gridplot(self.axes.obj.tolist(),
                                     ncols=self.ncol if self.ncol > 1 else None)
        bp.save(self.saved)

    def set_axes_colors(self, ir: int, ic: int):
        """Set axes colors (fill, alpha, edge).

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Set the axes fill colors
        fill, alpha = fill_alpha(self.axes.fill_color[utl.plot_num(ir, ic, self.ncol)])
        self.axes.obj[ir, ic].background_fill_color = fill
        self.axes.obj[ir, ic].background_fill_alpha = alpha

        # Set the axes edge colors (not sure how to handle the top and right spines)
        self.axes.obj[ir, ic].outline_line_color = self.axes.edge_color[0]
        if self.axes.spine_bottom:
            self.axes.obj[ir, ic].xaxis.axis_line_color = self.axes.edge_color[0]
        if self.axes.spine_left:
            self.axes.obj[ir, ic].yaxis.axis_line_color = self.axes.edge_color[0]

    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """
        axis = ['x', 'y']

        for ax in axis:
            grid = getattr(self.axes.obj[ir, ic], '%sgrid' % ax)
            grid.grid_line_color = \
                self.grid_major.color[0][0:7] if self.grid_major.on else None
            grid.grid_line_width = self.grid_major.width[0]
            grid.grid_line_dash = DASHES[self.grid_major.style[0]]

            grid.minor_grid_line_color = \
                self.grid_minor.color[0][0:7] if self.grid_minor.on else None
            grid.minor_grid_line_width = self.grid_minor.width[0]
            grid.minor_grid_line_dash = DASHES[self.grid_minor.style[0]]

        # Twinning -- may not be possible

    def set_axes_labels(self, ir: int, ic: int, data: 'Data'):  # noqa: F821
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: fcp.data object

        """
        axis = ['x', 'y', 'x2', 'y2']  # , 'z']
        for ax in axis:
            # Twinning
            if ax == 'x2' and not self.axes.twin_y:
                continue
            if ax == 'y2' and not self.axes.twin_x:
                continue

            # Get the label
            label = getattr(self, 'label_%s' % ax)
            if not label.on:
                continue
            if type(label.text) not in [str, list]:
                continue
            if isinstance(label.text, str):
                labeltext = label.text
            if isinstance(label.text, list):
                labeltext = label.text[ic + ir * self.ncol]

            # Toggle label visibility
            if not self.separate_labels:
                if ax == 'x' and ir != self.nrow - 1 and \
                        self.nwrap == 0 and self.axes.visible[ir + 1, ic]:
                    continue
                if ax == 'x2' and ir != 0:
                    continue
                if ax == 'y' and ic != 0 and self.axes.visible[ir, ic - 1]:
                    continue
                if ax == 'y2' and ic != self.ncol - 1 and \
                        utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue

            lkwargs = self.make_kw_dict(label)
            if ax == 'y2' or ax == 'x2':
                laxis = self.axes.obj[ir, ic].axis[-1]
            else:
                laxis = getattr(self.axes.obj[ir, ic], '%saxis' % ax)
            laxis.axis_label = labeltext
            laxis.axis_label_text_font = lkwargs['font']
            laxis.axis_label_text_font_size = '%spt' % lkwargs['font_size']
            laxis.axis_label_text_color = lkwargs['font_color']
            laxis.axis_label_text_font_style = lkwargs['font_style']  # no way to do bold and italic?

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        if self.name in ['heatmap', 'pie']:  # skip these plot types
            return

        if 'xmin' in ranges and ranges['xmin'][ir, ic] is not None:
            self.axes.obj[ir, ic].x_range.start = ranges['xmin'][ir, ic]
        if 'x2min' in ranges and ranges['x2min'][ir, ic] is not None:
            self.axes.obj[ir, ic].extra_x_ranges['x2'].start = ranges['x2min'][ir, ic]
        if 'xmax' in ranges and ranges['xmax'][ir, ic] is not None:
            self.axes.obj[ir, ic].x_range.end = ranges['xmax'][ir, ic]
        if 'x2max' in ranges and ranges['x2max'][ir, ic] is not None:
            self.axes.obj[ir, ic].extra_x_ranges['x2'].end = ranges['x2max'][ir, ic]
        if 'ymin' in ranges and ranges['ymin'][ir, ic] is not None:
            self.axes.obj[ir, ic].y_range.start = ranges['ymin'][ir, ic]
        if 'y2min' in ranges and ranges['y2min'][ir, ic] is not None:
            self.axes.obj[ir, ic].extra_y_ranges['y2'].start = ranges['y2min'][ir, ic]
        if 'ymax' in ranges and ranges['ymax'][ir, ic] is not None:
            self.axes.obj[ir, ic].y_range.end = ranges['ymax'][ir, ic]
        if 'y2max' in ranges and ranges['y2max'][ir, ic] is not None:
            self.axes.obj[ir, ic].extra_y_ranges['y2'].end = ranges['y2max'][ir, ic]

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            title = self.label_row.obj[ir]
            title.text = f'{self.label_row.text}={self.label_row.values[ir]}'
            title.align = self.label_row.align
            title.text_color = self.label_row.font_color
            title.text_font_size = '%spt' % self.label_row.font_size
            title.background_fill_alpha = self.label_row.fill_alpha
            title.background_fill_color = self.label_row.fill_color[0][0:7]
            self.axes.obj[ir, ic].add_layout(title, 'right')

        # Col/wrap labels
        if (ir == 0 and self.label_col.on):
            title = self.label_col.obj[ic]
            title.text = f'{self.label_col.text}={self.label_col.values[ic]}'
            title.align = self.label_col.align
            title.text_color = self.label_col.font_color
            title.text_font_size = '%spt' % self.label_col.font_size
            title.background_fill_alpha = self.label_col.fill_alpha
            title.background_fill_color = self.label_col.fill_color[0][0:7]
            self.axes.obj[ir, ic].add_layout(title, 'above')

        # Wrap labels
        if self.label_wrap.on:
            title = self.label_wrap.obj[ir, ic]
            title.text = ' | '.join([str(f) for f in utl.validate_list(self.label_wrap.values[ir * self.ncol + ic])])
            title.align = self.label_wrap.align
            title.text_color = self.label_wrap.font_color
            title.text_font_size = '%spt' % self.label_wrap.font_size
            title.background_fill_alpha = self.label_wrap.fill_alpha
            title.background_fill_color = self.label_wrap.fill_color[0][0:7]
            self.axes.obj[ir, ic].add_layout(title, 'above')

    def set_axes_scale(self, ir: int, ic: int):
        """
        This needs to happen at instantiation of the figure element, see _set_axes_type
        """

    def _set_axes_custom_range(self, ir: int, ic: int, data):
        """
        Customize the range of the plot.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: Data object

        Returns:
            x-axis range
            y-axis range
        """
        # how do we know the subset ranges???
        x_range, y_range = None, None

        return x_range, y_range

    def _set_axes_type(self, ir: int, ic: int, data):
        """
        Determine the axes type ('linear', 'log', 'datetime').

        This needs to happen at instantiation of the figure

        Args:
            ir: subplot row index
            ic: subplot column index
            data: Data object

        Returns:
            x-axis type
            y-axis type
        """
        if str(self.axes.scale).lower() in LOGX:
            x_type = 'log'
        elif data.df_all[data.x[0]].dtype == 'datetime64[ns]':
            x_type = 'datetime'
        # elif str(ax.scale).lower() in SYMLOGX:
        #     ax.obj[ir, ic].set_xscale('symlog')
        # elif str(ax.scale).lower() in LOGITX:
        #     ax.obj[ir, ic].set_xscale('logit')
        else:
            x_type = 'linear'

        if str(self.axes.scale).lower() in LOGY:
            y_type = 'log'
        elif data.df_all[data.y[0]].dtype == 'datetime64[ns]':
            y_type = 'datetime'
        # elif str(ax.scale).lower() in SYMLOGY:
        #     ax.obj[ir, ic].set_yscale('symlog')
        # elif str(ax.scale).lower() in LOGITY:
        #     ax.obj[ir, ic].set_yscale('logit')
        else:
            y_type = 'linear'

        return x_type, y_type

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        for mm in ['major', 'minor']:
            for xy in ['x', 'y']:
                ax = getattr(self.axes.obj[ir, ic], f'{xy}axis')
                if mm == 'major':
                    # tick label font
                    setattr(ax, f'{mm}_label_text_font_size',
                            '%spt' % getattr(self, f'tick_labels_{mm}_{xy}').font_size)
                # tick line color
                setattr(ax, f'{mm}_tick_line_color', getattr(self, f'ticks_{mm}_{xy}').color[0])
                # tick line width
                setattr(ax, f'{mm}_tick_line_width', int(getattr(self, f'ticks_{mm}_{xy}').size[1]))
                # tick line style
                # tick direction
                if getattr(self, f'ticks_{mm}').direction == 'in':
                    setattr(ax, f'{mm}_tick_in', int(getattr(self, f'ticks_{mm}_{xy}').size[0]))
                    setattr(ax, f'{mm}_tick_out', 0)
                else:
                    setattr(ax, f'{mm}_tick_out', int(getattr(self, f'ticks_{mm}_{xy}').size[0]))
                    setattr(ax, f'{mm}_tick_in', 0)

    def set_figure_final_layout(self, data, **kwargs):
        key_len = max(self.legend.values['Key'].apply(lambda x: len(x) if x else 0))
        self.legend.size[0] = 30 + max(utl.validate_list(self.markers.sizes)) + 10 + key_len

    def set_figure_title(self):
        """Set a figure title."""
        title = self.axes.obj[0, 0].title

        if self.title.on:
            tt = self.title
            title.text = tt.text
        elif self.title_wrap.on:
            tt = self.title_wrap
            self.title.text = tt.text
        else:
            return

        title.align = tt.align
        title.text_color = tt.font_color
        title.text_font_size = '%spt' % tt.font_size
        title.background_fill_alpha = tt.fill_alpha
        title.background_fill_color = tt.fill_color[0][0:7]

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
            if 'zmqshell.ZMQInteractiveShell' in app and not bs.curstate().notebook:
                bp.output_notebook()

            plots = bl.gridplot(list(self.axes.obj.flatten()), ncols=self.ncol, toolbar_location=self.toolbar.location)
            # plots.toolbar.active_multi = self.toolbar.active_zoom

            if self.title_wrap.on:
                # Bokeh doesn't support suptitles so need to add a custom Div above the plot for the wrap title
                css = 'text-align: left;\n'
                css += f'color: {self.title_wrap.font_color};\n'
                css += f'font-size: {self.title_wrap.font_size}px;\n'
                css += f'font-weight: {self.title_wrap.font_weight};\n'
                css += f'font-style: {self.title_wrap.font_style};\n'
                # add other css props here
                text = '<style type="text/css"> .title_wrap {' + css + '}</style><div class="title_wrap">'
                text += f'{self.title_wrap.text}</div>'
                title = bm.Div(text=text)
                bp.show(bl.column(title, plots))
            else:
                bp.show(plots)

        # other
        else:
            utl.show_file(filename)

    def update_markers(self):
        """Update the marker list to valid option for bokeh."""

        if 'marker_type' in self.kwargs.keys():
            marker_list = self.kwargs['marker_type']
        elif self.kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(self.kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        self.markers.type = RepeatedList(marker_list, 'markers')
