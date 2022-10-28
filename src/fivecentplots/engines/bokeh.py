import pandas as pd
import pdb
import numpy as np
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout
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


def fill_alpha(hexstr):
    """Break an 8-digit hex string into a hex color and a fractional alpha value."""
    if len(hexstr) == 6:
        return hexstr, 1

    return hexstr[0:7], int(hexstr[-2:], 16) / 255


def format_marker(fig, marker):
    """Format the marker string to mathtext."""
    markers = {'o': fig.circle,
               'circle': fig.circle,
               '+': fig.cross,
               'cross': fig.cross,
               's': fig.square,
               'square': fig.square,
               'x': fig.x,
               'd': fig.diamond,
               'diamond': fig.diamond,
               't': fig.triangle,
               'triangle': fig.triangle}

    return markers[marker]


class Layout(BaseLayout):
    def __init__(self, data: 'data.Data', defaults: list = [], **kwargs):  # noqa F821
        """Layout attributes and methods for matplotlib Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        global ENGINE
        ENGINE = 'bokeh'

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)

        # Update kwargs
        if not kwargs.get('save_ext'):
            kwargs['save_ext'] = '.html'
        self.kwargs = kwargs

    def add_box_labels(self, ir: int, ic: int, data):
        """Add box group labels and titles (JMP style).

        Args:
            ir: current axes row index
            ic: current axes column index
            data: fcp Data object
        """
        pass

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

        x = 0
        y = self.ncol - 1
        tt = list(self.legend.values.items())
        tt = [f for f in tt if f[0] != 'NaN']
        title = self.axes.obj[x, y].circle(0, 0, size=0.00000001, color=None)
        tt = [(self.legend.text, [title])] + tt
        legend = bm.Legend(items=tt, location='top_right')
        self.axes.obj[x, y].add_layout(legend, 'right')

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

    def _get_element_sizes(self, data: 'data.Data'):
        """Calculate the actual rendered size of select elements by pre-plotting
        them.  This is needed to correctly adjust the figure dimensions.

        Args:
            data: fcp Data object

        Returns:
            updated version of `data`
        """
        # Add label size
        self.label_x.size = \
            utl.get_text_dimensions(self.label_x.text, **self.make_kw_dict(self.label_x))
        self.label_y.size += \
            utl.get_text_dimensions(self.label_y.text, **self.make_kw_dict(self.label_y))

        # Ticks (rough)
        xlab = '%s.0' % int(data.ranges[0, 0]['xmax'])
        ylab = '%s.0' % int(data.ranges[0, 0]['ymax'])
        self.tick_labels_major_x.size = \
            utl.get_text_dimensions(xlab, **self.make_kw_dict(self.tick_labels_major_x))
        self.tick_labels_major_y.size += \
            utl.get_text_dimensions(ylab, **self.make_kw_dict(self.tick_labels_major_y))

        # title
        if self.title.text is not None:
            self.title.size = \
                utl.get_text_dimensions(self.title.text, **self.make_kw_dict(self.title))

        # rc labels
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                if self.label_row.text is not None:
                    text = '%s=%s' % (self.label_row.text, self.label_row.values[ir])
                    self.label_row.size[1] = \
                        max(self.label_row.size[1],
                            utl.get_text_dimensions(text, **self.make_kw_dict(self.label_row))[1])
                if self.label_col.text is not None:
                    text = '%s=%s' % (self.label_col.text, self.label_col.values[ir])
                    self.label_col.size[1] = \
                        max(self.label_col.size[1],
                            utl.get_text_dimensions(text, **self.make_kw_dict(self.label_col))[1])

        # Legend
        if data.legend_vals is None:
            return data

        # Get approx legend size
        name_size = 0
        for name in data.legend_vals['names']:
            name_size = max(name_size, utl.get_text_dimensions(str(name), **self.make_kw_dict(self.legend))[0])

        if self.legend.on:
            self.legend.size = [10 + 20 + 5 + name_size + 10, 0]

        return data

    def _get_figure_size(self, data: 'data.Data', **kwargs):
        """Determine the size of the mpl figure canvas in pixels and inches.

        Args:
            data: Data object
            kwargs: user-defined keyword args
        """
        self.axes.size[0] += self.legend.size[0]

    def make_figure(self, data: 'data.Data', **kwargs):
        """Make the figure and axes objects.

        Args:
            data: fcp Data object
            **kwargs: input args from user
        """
        self._update_from_data(data)
        self._update_wrap(data, kwargs)
        self._set_label_text(data)
        data = self._get_element_sizes(data)

        self.axes.obj = np.array([[None] * self.ncol] * self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                x_scale, y_scale = self._set_axes_scale2(ir, ic)
                x_size, y_size = self.axes.size
                # legend
                if ir == 0 and ic == self.ncol - 1:
                    x_size += self.legend.size[0]
                # rc labels
                y_size += self.label_col.size[1]
                # axes labels
                if ir == 0 and ic == 0:
                    x_size += self.label_x.size[1] + 15
                if ir == self.nrow - 1:
                    y_size += self.label_y.size[1] + 15
                # ticks (separate out to label and ticks later)
                x_size += self.tick_labels_major_x.size[0] + 15
                y_size += self.tick_labels_major_y.size[0] + 15
                # title
                y_size += self.title.size[0]

                self.axes.obj[ir, ic] = bp.figure(plot_width=int(x_size),
                                                  plot_height=int(y_size),
                                                  x_axis_type=x_scale,
                                                  y_axis_type=y_scale)

        self.axes.visible = np.array([[True] * self.ncol] * self.nrow)

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
        else:
            line_type = getattr(self, line_type)

        # TWINNING
        # Insert here

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
                                )
            else:
                points = marker(df[x], df[y],
                                color=line_type.color[iline],
                                linestyle=line_type.style[iline],
                                linewidth=line_type.width[iline])

        # Make the line
        lines = None
        if line_type.on:
            lines = self.axes.obj[ir, ic].line(df[x], df[y],
                                               color=line_type.color[iline][0:7],
                                               line_dash=DASHES[line_type.style[iline]],
                                               line_width=line_type.width[iline],
                                               )

        # Add a reference to the line to self.lines
        if leg_name is not None:
            leg_vals = []
            if self.markers.on and not marker_disable:
                leg_vals += [points]
            if line_type.on:
                leg_vals += [lines]

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
        axes = self._get_axes()

        fill, alpha = fill_alpha(axes[0].fill_color[utl.plot_num(ir, ic, self.ncol)])
        self.axes.obj[ir, ic].background_fill_color = fill
        self.axes.obj[ir, ic].background_fill_alpha = alpha

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

    def set_axes_labels(self, ir: int, ic: int):
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        axis = ['x', 'y']  # x2', 'y', 'y2', 'z']
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

            # Twinning?

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
            laxis = getattr(self.axes.obj[ir, ic], '%saxis' % ax)
            laxis.axis_label = labeltext
            laxis.axis_label_text_font = lkwargs['font']
            laxis.axis_label_text_font_size = '%spt' % lkwargs['font_size']
            laxis.axis_label_text_color = lkwargs['font_color']
            laxis.axis_label_text_font_style = lkwargs['font_style']

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """
        pass

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        title = self.axes.obj[ir, ic].title

        # Row labels
        if ic == self.ncol - 1 and self.label_row.on and not self.label_wrap.on:
            title.text = ' %s=%s ' % (self.label_row.text, self.label_row.values[ir])
            title.align = self.label_row.align
            title.text_color = self.label_row.font_color
            title.text_font_size = '%spt' % self.label_row.font_size
            title.background_fill_alpha = self.label_row.fill_alpha
            title.background_fill_color = self.label_row.fill_color[0][0:7]

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            title.text = ' %s=%s ' % (self.label_col.text, self.label_col.values[ic])
            title.align = self.label_col.align
            title.text_color = self.label_col.font_color
            title.text_font_size = '%spt' % self.label_col.font_size
            title.background_fill_alpha = self.label_col.fill_alpha
            title.background_fill_color = self.label_col.fill_color[0][0:7]

    def set_axes_scale(self, ir, ic):
        pass

    def _set_axes_scale2(self, ir, ic):
        """
        This appears to need to happen at instantiation of the figure
        """

        if str(self.axes.scale).lower() in LOGX:
            x_scale = 'log'
        # elif str(ax.scale).lower() in SYMLOGX:
        #     ax.obj[ir, ic].set_xscale('symlog')
        # elif str(ax.scale).lower() in LOGITX:
        #     ax.obj[ir, ic].set_xscale('logit')
        else:
            x_scale = 'linear'
        if str(self.axes.scale).lower() in LOGY:
            y_scale = 'log'
        # elif str(ax.scale).lower() in SYMLOGY:
        #     ax.obj[ir, ic].set_yscale('symlog')
        # elif str(ax.scale).lower() in LOGITY:
        #     ax.obj[ir, ic].set_yscale('logit')
        else:
            y_scale = 'linear'

        return x_scale, y_scale

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """
        self.axes.obj[ir, ic].xaxis.major_label_text_font_size = \
            '%spt' % self.tick_labels_major.font_size
        self.axes.obj[ir, ic].yaxis.major_label_text_font_size = \
            '%spt' % self.tick_labels_major.font_size

    def set_figure_final_layout(self, data, **kwargs):
        pass

    def set_figure_title(self):
        """Set a figure title."""
        if self.title.on:
            title = self.axes.obj[0, 0].title
            title.text = self.title.text
            title.align = self.title.align
            title.text_color = self.title.font_color
            title.text_font_size = '%spt' % self.title.font_size
            title.background_fill_alpha = self.title.fill_alpha
            title.background_fill_color = self.title.fill_color[0][0:7]

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

            bp.show(bl.gridplot(self.axes.obj.flatten(), ncols=self.ncol))

        # other
        else:
            utl.show_file(filename)
