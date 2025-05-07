import pandas as pd
import pdb
import numpy as np
from .. import utilities as utl
from . layout import LOGX, LOGY, BaseLayout, RepeatedList, Element  # noqa
from .. import data
import warnings
# import plot lib specific stuff here


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'


warnings.formatwarning = custom_formatwarning
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")  # weird error in boxplot w/ no groups


db = pdb.set_trace


DASHES = {}


DEFAULT_MARKERS = []


class Layout(BaseLayout):
    def __init__(self, data: 'data.Data', defaults: list = [], **kwargs):  # noqa F821
        """Layout attributes and methods for new engine Figure.

        Args:
            data: fcp Data object
            defaults: items from the theme file
            kwargs: input args from user
        """
        # Set the layout engine
        self.engine = 'new engine name here'

        # Inherit the base layout properties
        super().__init__(data, defaults, **kwargs)

        # Update kwargs
        self.kwargs = kwargs
        self.update_markers()

        # Engine-specific kwargs

        # Check for unsupported kwargs
        # unsupported = []

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

    def set_axes_grid_lines(self, ir: int, ic: int):
        """Style the grid lines and toggle visibility.

        Args:
            ir (int): subplot row index
            ic (int): subplot column index

        """

    def set_axes_labels(self, ir: int, ic: int, data: 'Data'):  # noqa: F821
        """Set the axes labels.

        Args:
            ir: subplot row index
            ic: subplot column index
            data: fcp.data object

        """

    def set_axes_ranges(self, ir: int, ic: int, ranges: dict):
        """Set the axes ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            ranges: min/max axes limits for each axis

        """

    def set_axes_rc_labels(self, ir: int, ic: int):
        """Add the row/column label boxes and wrap titles.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    def set_axes_scale(self, ir: int, ic: int):
        """
        This needs to happen at instantiation of the figure element, see _set_axes_type
        """

    def set_axes_ticks(self, ir: int, ic: int):
        """Configure the axes tick marks.

        Args:
            ir: subplot row index
            ic: subplot column index

        """

    def set_figure_final_layout(self, data, **kwargs):
        """Final adjustment of the figure size and plot spacing.

        Args:
            data: Data object
            kwargs: keyword args
        """

    def set_figure_title(self):
        """Set a figure title."""

    def show(self, filename=None):
        """Display the plot window.

        Args:
            filename (optional): name of the file to show. Defaults to None.

        """

    def update_markers(self):
        """Update the marker list to valid option for new engine."""

        if 'marker_type' in self.kwargs.keys():
            marker_list = self.kwargs['marker_type']
        elif self.kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(self.kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        self.markers.type = RepeatedList(marker_list, 'markers')
