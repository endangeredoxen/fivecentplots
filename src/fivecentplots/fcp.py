############################################################################
# fcp.py
#   Custom library of plot functions based on matplotlib to generate more
#   attractive plots more easily.  Part of the fivecentplots project.
############################################################################

# maybe subclass fig and ax in a new class that contains all the internal
# functions needed for an mpl plot.  then another for bokeh

__author__ = 'Steve Nicholes'
__copyright__ = 'Copyright (C) 2016 Steve Nicholes'
__license__ = 'GPLv3'
__url__ = 'https://github.com/endangeredoxen/fivecentplots'

import os
import numpy as np
import pandas as pd
import pdb
import copy
import shutil
import sys
from pathlib import Path
from . import utilities
from . import data
from . colors import DEFAULT_COLORS, BAYER  # noqa
from . import engines
import fivecentplots as fcp
try:
    # optional import - only used for paste_kwargs to use windows clipboard
    # to directly copy kwargs from ini file
    import fivecentfileio as fileio
    import win32clipboard  # noqa
except (ModuleNotFoundError, ImportError, NameError):
    pass
with open(Path(__file__).parent / 'version.txt', 'r') as fid:
    __version__ = fid.readlines()[0].replace('\n', '')
utl = utilities
HIST = utl.HIST
db = pdb.set_trace
osjoin = os.path.join
cur_dir = Path(__file__).parent
user_dir = Path.home()

# transfer theme file or use built-in
if not (user_dir / '.fivecentplots').exists():
    os.makedirs(user_dir / '.fivecentplots')
if not (user_dir / '.fivecentplots' / 'defaults.py'):
    shutil.copy(cur_dir / 'themes' / 'gray.py', user_dir / '.fivecentplots' / 'defaults.py')
if (user_dir / '.fivecentplots' / 'defaults.py').exists():
    sys.path = [str(user_dir / '.fivecentplots')] + sys.path
    from defaults import *  # noqa, use local file
else:
    from . themes.gray import *  # noqa

# install requirements for other packages beyond what is in setup.py
global INSTALL
INSTALL = {}
INSTALL['bokeh'] = ['bokeh']

# Global kwargs to override anything
global KWARGS
KWARGS = {}  # type: ignore


class EngineError(Exception):
    def __init__(self, *args, **kwargs):
        """Invalid engine kwargs error."""
        Exception.__init__(self, *args, **kwargs)


def bar(df, **kwargs):
    """Bar chart.

    Args:
        df (pandas.DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (str): x-axis column name [REQUIRED]
        y (str): y-axis column name [REQUIRED]
        bar_align (str): If ‘center’ aligns center of bar to x-axis value; if ‘edge’ aligns the left edge of the bar to
          the x-axis value. Defaults to ‘center’ .
        bar_color_by_bar|color_by_bar (bool): Color each bar differently. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Color-by-bar
        bar_edge_color (str): Hex color string for the edge of the bar. Defaults to fcp.DEFAULT_COLORS.
        bar_edge_width (float): Width of the edge of the bar in pixels. Defaults to 0.
        bar_error_bars|error_bars (bool): Display error bars on each bar. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Error-bars
        bar_error_color|error_color (str): Hex color string of the error bar lines. Defaults to #555555.
        bar_fill_alpha (float): Transparency value for the bars between 0-1. Defaults to 0.75.
        bar_fill_color (str): Hex color string of the bar fill . Defaults to fcp.DEFAULT_COLORS.
        bar_horizontal|horizontal (bool): Display bars horizontally. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Horizontal-bars
        bar_rolling|bar_rolling_mean|rolling|rolling_mean (int): Rolling mean window size [enables this curve]. No
          default. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Rolling-mean
        bar_stacked|stacked (bool): Stack bars of a given group . Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Stacked
        bar_width (float): Set the fractional width of the bars between 0-1; for stacked barplots the width corresponds
          to the height of the bars. Defaults to 0.8.
        rolling_mean_line_color (str): Hex color string for the rolling mean line. Defaults to fcp.DEFAULT_COLORS.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Custom-line-style
        rolling_mean_line_width (int): Width for the rolling mean line in pixels. Defaults to 2. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/barplot.html#Custom-line-style

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
        >>> fcp.bar(df, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25',
                    tick_labels_major_x_rotation=90)

            .. figure:: ../_static/images/example_bar.png
    """
    return plotter(data.Bar, **utl.dfkwarg(df, kwargs))


def boxplot(df, **kwargs):
    """Box plot modeled after the "Variability Chart" in JMP which Dummy function to return convenient,
    multi-level group labels automatically along the x-axis.

    Args:
        df (pandas.DataFrame): DataFrame containing data to plot

    Keyword Args:
        y (str): y-axis column name contining the box plot data [REQUIRED]
        BASIC:
        box_divider (bool): Toggle box divider visibility. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Dividers
        box_divider_color (str): Hex color string for the vertical line between groups. Defaults to #bbbbbb.
        box_divider_style (str): Line style for the box divider lines {‘-’, ‘--’, ‘-.’, ‘:’}. Defaults to -.
        box_divider_width (float): Width of the divider lines in pixels. Defaults to 1.
        box_edge_color (str): Hex color string for the edge of the box. Defaults to #aaaaaa. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Boxplot-example
        box_edge_width (float): Width of the edge of the boxes in pixels. Defaults to 0.5.
        box_fill_color (str): Hex color string of the bar fill . Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Boxplot-example
        box_median_color (str): Hex color string of the median line inside each box. Defaults to #ff7f0e.
        box_on (bool): Toggle box visibility. Defaults to True.
        box_range_lines (bool): Toggle the horizontal lines showing the min/max of the data range. Defaults to True.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Range-lines
        box_range_lines_color (str): Hex color string for the box range lines. Defaults to #cccccc.
        box_range_lines_style (str): Line style for the box range lines {‘-’, ‘--’, ‘-.’, ‘:’}. Defaults to --.
        box_range_lines_width (float): Width of the range lines in pixels. Defaults to 1.
        box_whisker (bool): Toggle range lines that extend from the box Q1/Q3 edges to the data min/max. Defaults to
          True. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Whiskers
        box_whisker_color (str): Hex color string for the box whisker lines. Defaults to #cccccc.
        box_whisker_style (str): Line style for the box whisker lines {‘-’, ‘--’, ‘-.’, ‘:’}. Defaults to -.
        box_whisker_width (float): Width of the whisker lines in pixels. Defaults to 0.5.
        box_width (float): Set the fractional width of the boxes between 0-1. Defaults to 0.5 [if violin on, 0.15].
        groups (str|list): Grouping columns for the box plot. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Groups
        notch (bool): Use a notched-style box instead of a rectangular box. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Notch
        GROUPING_TEXT:
        box_group_label_edge_alpha (float): Transparency value for group label line edge between 0-1. Defaults to 1.
        box_group_label_edge_color (str): Hex color string for the group label rectangle edge. Defaults to #ffffff.
        box_group_label_edge_width (float): Width of the edge of the line around the group labels in pixels. Defaults to
          1.
        box_group_label_fill_alpha (float): Transparency value for group label fill between 0-1. Defaults to 1.
        box_group_label_fill_color (str): Hex color string for the group label background color. Defaults to #ffffff.
        box_group_label_font (str): Font name for box group label. Defaults to Sans-serif.
        box_group_label_font_color (str): Hex color string for group label font. Defaults to #000000.
        box_group_label_font_size (float): Font size for group label text in pixels. Defaults to 12.
        box_group_label_font_style (str): Font style for the group label text {'normal', 'italic', 'oblique'}. Defaults
          to 'normal’.
        box_group_label_font_weight (str): Font weight for the group label text {'light', 'normal', 'medium',
          'semibold', 'bold', 'heavy', 'black'}. Defaults to 'normal’.
        box_group_title_edge_alpha (float): Transparency value for group title line edge between 0-1. Defaults to 1.
        box_group_title_edge_color (str): Hex color string for the group title rectangle edge. Defaults to #ffffff.
        box_group_title_edge_width (float): Width of the edge of the line around the group titles in pixels. Defaults to
          1.
        box_group_title_fill_alpha (float): Transparency value for group title fill between 0-1. Defaults to 1.
        box_group_title_fill_color (str): Hex color string for the group title background color. Defaults to #ffffff.
        box_group_title_font (str): Font name for box group title. Defaults to Sans-serif.
        box_group_title_font_color (str): Hex color string for group title font. Defaults to #000000.
        box_group_title_font_size (float): Font size for group title text in pixels. Defaults to 13.
        box_group_title_font_style (str): Font style for the group title text {'normal', 'italic', 'oblique'}. Defaults
          to 'normal’.
        box_group_title_font_weight (str): Font weight for the group title text {'light', 'normal', 'medium',
          'semibold', 'bold', 'heavy', 'black'}. Defaults to 'normal’.
        STAT_LINES:
        box_grand_mean (bool): Toggle visibility of a line showing the mean of all data on the plot. Defaults to False.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Grand-Mean/Median
        box_grand_mean_color|grand_mean_color (str): Hex color string for the grand mean line. Defaults to #555555.
        box_grand_mean_style|grand_mean_style (str): Line style for the box grand mean lines {‘-’, ‘--’, ‘-.’, ‘:’}.
          Defaults to '-’.
        box_grand_mean_width|grand_mean_width (float): Width of the grand mean line in pixels. Defaults to 1.
        box_grand_median (bool): Toggle visibility of a line showing the median of all data on the plot. Defaults to
          False. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Grand-Mean/Median
        box_grand_median_color|grand_median_color (str): Hex color string for the grand median line. Defaults to
          #0000ff.
        box_grand_median_style|grand_median_style (str): Line style for the box grand median lines {‘-’, ‘--’, ‘-.’,
          ‘:’}. Defaults to '-’.
        box_grand_median_width|grand_median_width (float): Width of the grand median line in pixels. Defaults to 1.
        box_group_mean (bool): Toggle visibility of a line showing the mean of each data group on the plot. Defaults to
          False. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Group-Means
        box_group_mean_color|group_mean_color (str): Hex color string for the group mean line. Defaults to #555555.
        box_group_mean_style|group_mean_style (str): Line style for the box group mean lines {‘-’, ‘--’, ‘-.’, ‘:’}.
          Defaults to '-’.
        box_group_mean_width|group_mean_width (float): Width of the group mean line in pixels. Defaults to 1.
        box_stat_line (str): Set the statistic for the connecting line {‘mean’, ‘median’, ‘std’, ‘qXX’ [qunatile where
          XX is a number between 0-100]}. Defaults to mean. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Stat-line
        box_stat_line_color (str): Hex color string for the stat line. Defaults to #666666.
        box_stat_line_on (bool): Toggle visibility of the stat line between boxes. Defaults to True.
        box_stat_line_width (float): Width of the stat line in pixels. Defaults to 1.
        DIAMONDS:
        box_mean_diamonds_alpha|mean_diamonds_alpha (float): Transparency value for the diamonds between 0-1. Defaults
          to 1.
        box_mean_diamonds_edge_color|mean_diamonds_edge_color (str): Hex color string for the edges of the diamond.
          Defaults to #FF0000.
        box_mean_diamonds_edge_style|mean_diamonds_edge_style (str): Line style for the diamonds lines {‘-’, ‘--’,
          ‘-.’, ‘:’}. Defaults to '-’.
        box_mean_diamonds_edge_width|mean_diamonds_edge_width (float): Width of the diamond lines in pixels. Defaults
          to 0.7.
        box_mean_diamonds_fill_color|mean_diamonds_fill_color (str): Hex color string for the fill of the diamond.
          Defaults to None.
        box_mean_diamonds_width|mean_diamonds_width (float): Set the fractional width of the diamonds between 0-1.
          Defaults to 0.8.
        box_mean_diamonds|mean_diamonds (bool): Toggle visibility of a diamond overlay on the box showing the group mean
          and a confidence interval. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Mean-Diamonds
        conf_coeff (float): Confidence interval from 0 to 1. Defaults to 0.95.
        VIOLINS:
        box_violin|violin (bool): Toggle visibility of violin plot showing the distribution of box plot data. Defaults
          to False. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Violins
        violin_box_color (str): Hex color string for the fill of an optional box overlay on the violin. Defaults to
          #555555.
        violin_box_on (bool): Toggle visibility of a box over the violin plot. Defaults to True.
        violin_edge_color (str): Hex color string for the edge of the violins. Defaults to #aaaaaa.
        violin_fill_alpha (float): Transparency value for the violin plots between 0-1. Defaults to 0.5.
        violin_fill_color (str): Hex color string for the fill of the violins. Defaults to fcp.DEFAULT_COLORS.
        violin_markers (bool): Toggle visibility of data point markers on the violin plots. Defaults to False.
        violin_median_color (str): Hex color string for the median point in each violin. Defaults to #ffffff.
        violin_median_marker (str): Marker type for the median point in each violin. Defaults to 'o’.
        violin_median_size (int): Size of the median point marker in each violin. Defaults to 2.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
        >>> fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'])

            .. figure:: ../_static/images/example_boxplot.png
    """
    return plotter(data.Box, **utl.dfkwarg(df, kwargs))


def contour(df, **kwargs):
    """Contour plot module.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (str): x-axis column name [REQUIRED]
        y (str): y-axis column name [REQUIRED]
        z (str): z-axis column name [REQUIRED]
        BASIC:
        cmap (str): Name of a color map . Defaults to inferno. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Filled-contour
        contour_width (float): Width of the contour lines. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Contour-lines
        filled (bool): Color area between contour lines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Filled-contour
        interp (str): Scipy interpolate.griddata method to make Z points {‘linear’, ‘nearest’, ‘cubic’}. Defaults to
          'cubic’. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Data-interpolation
        levels (int): Number of contour lines/levels to draw. Defaults to 20. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Contour-lines
        show_points (bool): Show points on top of the contour plot. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Contour-points
        COLOR_BAR:
        cbar (bool): Toggle colorbar on/off for contour and heatmap plots. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Filled-contour
        size (int): cbar width [height will match the height of the axes]. Defaults to 30.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
        >>> fcp.contour(cc, x='X', y='Y', z='Value', cbar=True, cbar_size=40, xmin=-4, xmax=2, ymin=-4, ymax=2)

            .. figure:: ../_static/images/example_contour.png

    """
    return plotter(data.Contour, **utl.dfkwarg(df, kwargs))


def deprecated(kwargs):
    """Automatically fix deprecated keyword args."""

    # leg_groups
    if kwargs.get('leg_groups'):
        kwargs['legend'] = kwargs['leg_groups']
        kwargs.pop('leg_groups')
        print('"leg_groups" is deprecated. Please use "legend" instead')

    # labels
    labels = ['x', 'x2', 'y', 'y2', 'z']
    for ilab, lab in enumerate(labels):
        # Deprecated style
        keys = [f for f in kwargs.keys() if '%slabel' % lab in f]
        if len(keys) > 0:
            print('"%slabel" is deprecated. Please use "label_%s" instead' % (lab, lab))
            for k in keys:
                kwargs[k.replace('%slabel' % lab, 'label_%s' % lab)] = kwargs[k]
                kwargs.pop(k)

    # twin + share
    vals = ['sharex', 'sharey', 'twinx', 'twiny']
    for val in vals:
        if val in kwargs:
            print('"%s" is deprecated.  Please use "%s_%s" instead' % (val, val[0:-1], val[-1]))
            kwargs['%s_%s' % (val[0:-1], val[-1])] = kwargs[val]
            kwargs.pop(val)

    return kwargs


def docs():
    import webbrowser
    webbrowser.open(r'https://endangeredoxen.github.io/fivecentplots/index.html')


def gantt(df, **kwargs):
    """Gantt chart plotting function.  This plot is built off of a horizontal
       implementation of `fcp.bar`.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (list): two x-axis column names containing Datetime values [REQUIRED]
            - 1) the start time for each item in the Gantt chart
            - 2) the stop time for each item in the Gantt chart
        y (str): y-axis column name [REQUIRED]
        gantt_color_by_bar|color_by_bar (bool): Color each Gantt bar differently. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Styling
        gantt_edge_color (str): Hex color string for the edge of the Gantt bars. Defaults to fcp.DEFAULT_COLORS.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Styling
        gantt_edge_width (float): Width of the edge of the Gantt bars in pixels. Defaults to 0. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Styling
        gantt_fill_alpha (int): Transparency value for the Gantt bars between 0-1. Defaults to 0.75. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Styling
        gantt_fill_color (str): Hex color string of the Gantt bar fill . Defaults to fcp.DEFAULT_COLORS.
        gantt_height|height (float): Set the fractional height of the Gantt bars between 0-1. Defaults to 0.9. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Styling
        gantt_label_x (str): By default, x-axis labels are disabled for this plot type. Defaults to '’.
        gantt_order_by_legend|order by legend (bool): Order the y-axis values based on the sort order of the legend
          values [requires legend]. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Legends
        gantt_tick_labels_x_rotation|tick_labels_x_rotation (int): Gantt-specific version of the this kwarg to ensure
          rotations are not applied globably to all plots from a theme file. Defaults to 90. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Legends
        sort (str): Sort order for the Gantt bars {‘ascending’, ‘descending’}. Defaults to 'descending’. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/gantt.html#Sorting

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt.csv')
        >>> fcp.gantt(df, x=['Start', 'Stop'], y='Task', ax_size=[600, 400])

            .. figure:: ../_static/images/example_gantt.png
    """

    return plotter(data.Gantt, **utl.dfkwarg(df, kwargs))


def heatmap(df, **kwargs):
    """Heatmap plot.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (str): x-axis column name [REQUIRED]
        y (str): y-axis column name [REQUIRED]
        z (str): z-axis column name [REQUIRED]
        BASIC:
        cell_size (int): Width of a heatmap cell in pixels. Defaults to 60. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/heatmap.html#Cell-size
        cmap (bool): Name of a color map to apply to the plot. Defaults to inferno. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/heatmap.html#No-data-labels
        data_labels (bool): Toggle visibility of value text labels on the heatmap cells. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/heatmap.html#With-data-labels
        heatmap_edge_width (float): Width of the edges of the heat map cells. Defaults to 0.
        heatmap_font_color (str): Hex color string for the value label text. Defaults to #ffffff.
        heatmap_font_size (int): Font size of the value label text. Defaults to 12.
        heatmap_interp|interp (str): imshow interpolation scheme [see matplotlib docs for more details]. Defaults to
          'none’.
        COLOR_BAR:
        cbar (bool): Toggle colorbar on/off for contour and heatmap plots. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Filled-contour
        size (int): cbar width [height will match the height of the axes]. Defaults to 30.

    Examples
    --------
    Categorical heatmap:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
        >>> fcp.heatmap(df, x='Category', y='Player', z='Average')

            .. figure:: ../_static/images/example_heatmap1.png

    Non-uniform numerical data:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
        >>> fcp.heatmap(df, x='X', y='Y', z='Value', row='Batch', col='Experiment',
                        cbar=True, share_z=True, ax_size=[400, 400], data_labels=False,
                        label_rc_font_size=12, filter='Batch==103', cmap='viridis')

            .. figure:: ../_static/images/example_heatmap2.png

    """
    return plotter(data.Heatmap, **utl.dfkwarg(df, kwargs))


def hist(df, **kwargs):
    """Histogram plot.

    Args:
        df (DataFrame | numpy array): DataFrame or numpy array containing data to plot
            [when passing a numpy array it is automatically converted to a DataFrame]

    Keyword Args:
        x (str): x-axis column name (i.e., the "value" column from which "counts" are calculated) [REQUIRED]
        bars (bool): Toggle between bars or a line plot for the counts (True=bars enabled, False=use line).  Defaults
          to True unless 2D image then False
        cdf (bool): Convert the histogram into a cumulative distribution plot. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#cdf
        cfa (str): Color-filter array pattern that is used to split data from a Bayer image into separate color planes.
          Defaults to None. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/imshow.html#split-color-planes
        hist_align (str): If "mid" aligns center of histogram bar to x-axis value; if "left" aligns the left edge of the
          histogram bar to the x-axis value {"left"; "mid"; "right"}. Defaults to mid. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#alignment
        hist_bins|bins (int): Number of histogram bins to use; when plotting the histogram of a raw image file the
          number of bins is automatically adjusted to enable one bin per DN code. Defaults to 20. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#bin-counts
        hist_cumulative|cumulative (bool): From matplotlib: If True then a histogram is computed where each bin gives
          the counts in that bin plus all bins for smaller values; if -1 direction of accumulation is reversed. Defaults
          to False. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#cumulative
        hist_edge_color (str): Hex color string for the edge of the histogram bar. Defaults to fcp.DEFAULT_COLORS.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#colors
        hist_edge_width (float): Width of the edge of the histogram bar in pixels. Defaults to 0. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#colors
        hist_fill_alpha (int): Transparency value for the histogram bars between 0-1. Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#colors
        hist_fill_color (str): Hex color string of the histogram bar fill . Defaults to fcp.DEFAULT_COLORS. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#colors
        hist_horizontal|horizontal (bool): Enable a horizontal histogram plot [default is vertical]. Defaults to False.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#horizontal-bars
        hist_kde|kde (bool): Toggle visibility of a kernel-density estimator curve over the histogram bars. Defaults to
          False. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#kernel-density-estimator
        hist_normalize|normalize (bool): Sets the "density" parameter for matplotlib-based plots; from matplotlib: if
          True draw and return a probability density: each bin will display each bin"s raw count divided by the total
          number of counts and the bin width so that the area under the histogram integrates to 1; automatically enabled
          if kde=True. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#cumulative
        hist_rwidth (float|None): From matplotlib: the relative width of the bars as a fraction of the bin width; None
          means auto-calculation. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#width
        pdf (bool): Convert the histogram into a probability density function plot. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#pdf

    Examples
    --------
    Simple histogram:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
        >>> fcp.hist(df, x='Value')

            .. figure:: ../_static/images/example_hist1.png

    Bayer-image histogram:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Make a dummy blue patch through an RGB filter
        >>> img_rgb = np.zeros([300, 300]).astype(np.uint16)
        >>> img_rgb[::2, ::2] = 180  # green_red
        >>> img_rgb[1::2, 1::2] = 180  # green_blue
        >>> img_rgb[::2, 1::2] = 10
        >>> img_rgb[1::2, ::2] = 255
        >>> # Add gaussian shading
        >>> x, y = np.meshgrid(np.linspace(-1,1,300), np.linspace(-1,1,300))
        >>> dst = np.sqrt(x*x+y*y)
        >>> sigma = 1
        >>> muu = 0.001
        >>> gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
        >>> img_rgb = (gauss * img_rgb).astype(float)
        >>> # Add random noise
        >>> img_rgb[::2, ::2] += np.random.normal(-0.1*img_rgb[::2, ::2].mean(), 0.1*img_rgb[::2, ::2].mean(),
        >>>                                       img_rgb[::2, ::2].shape)
        >>> img_rgb[1::2, ::2] += np.random.normal(-0.1*img_rgb[1::2, ::2].mean(), 0.1*img_rgb[1::2, ::2].mean(),
        >>>                                        img_rgb[1::2, ::2].shape)
        >>> img_rgb[1::2, 1::2] += np.random.normal(-0.1*img_rgb[1::2, 1::2].mean(), 0.1*img_rgb[1::2, 1::2].mean(),
        >>>                                         img_rgb[1::2, 1::2].shape)
        >>> img_rgb[::2, 1::2] += np.random.normal(-0.1*img_rgb[::2, 1::2].mean(), 0.1*img_rgb[::2, 1::2].mean(),
        >>>                                        img_rgb[::2, 1::2].shape)
        >>> img_rgb = img_rgb.astype(np.uint16)
        >>> fcp.hist(img_rgb, ax_size=[600, 400], legend='Plane', cfa='grbg', colors=fcp.BAYER, **fcp.HIST)

            .. figure:: ../_static/images/example_hist2.png
    """

    return plotter(data.Histogram, **utl.dfkwarg(df, kwargs))


def imshow(df, **kwargs):
    """Image show plotting function.

    Args:
        df (DataFrame | numpy array): DataFrame or numpy array containing 2D row/column
            image data to plot [when passing a numpy array it is automatically converted
            to a DataFrame]

    Keyword Args:
        cfa (str): Color-filter array pattern that is used to split data from a Bayer image into separate color planes.
          Defaults to None. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/imshow.html#split-color-planes
        cmap (bool): Name of a color map to apply to the plot. Defaults to gray. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/imshow.html#color-maps
        imshow_interp|interp (str): imshow interpolation scheme [see matplotlib docs for more details]. Defaults to
          'none’.
        stretch (float|list): Calculate "stretch" times the standard deviation above and below the mean to set new
          z-limits. Can be a single value used as +/- limits or a two-value list for the lower/upper multiplier values.
          Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/imshow.html#Contrast-stretching

    Examples
    --------
    Basic:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> import imageio
        >>> # Read an image from the world-wide web
        >>> url = 'https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&rect=0%2C214%2C2000%2C1214&' \
        >>>       + 'poi=%5B920%2C546%5D&w=2000&h=1000&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads' \
        >>>       + '%2Fsites%2F47%2F2020%2F10%2F07%2Fcat-in-pirate-costume-380541532-2000.jpg'
        >>> imgr = imageio.imread(url)
        >>> # Convert to grayscale
        >>> img = fcp.utilities.img_grayscale(imgr)
        >>> fcp.imshow(img, ax_size=[600, 600])

            .. figure:: ../_static/images/example_imshow1.png

    With +/- 3 sigma contrast stretching:

        >>> uu = img.stack().mean()
        >>> ss = img.stack().std()
        >>> fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600], zmin=uu-3*ss, zmax=uu+3*ss)

            .. figure:: ../_static/images/example_imshow2.png
    """

    kwargs['tick_labels'] = kwargs.get('tick_labels', True)

    return plotter(data.ImShow, **utl.dfkwarg(df, kwargs))


def nq(df, **kwargs):
    """Plot the normal quantiles of a data set.

    Args:
        df (DataFrame | numpy array): DataFrame containing a column of
            values data or a DataFrame or numpy array containing a set of 2D values
            that can be used to calculate quantiles at various "sigma" intervals

    Keyword Args:
        BASIC:
        x (str): x-axis column name (if using a 1D dataset). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/nq.html#1D-dataset
        CALCULATION:
        sigma (float): Maximum sigma value to use for the calculation; range will be +/- this value. Defaults to Auto-
          calculated based on the dataset using "fcp.utilities.sigma". Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/nq.html#custom-sigma-range
        step_inner (float): Delta between sigma values outside of the tail (around sigma=0). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/nq.html#point-density
        step_tail (float): Delta between sigma values in the tails (all value >= and <= to keyword "tail"). Defaults to
          0.2. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/nq.html#point-density
        tail (float): Sigma value that represents the start of the tail of the distribution. Defaults to 3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/nq.html#custom-tail

    Examples
    --------
    "Normal" distribution:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Make a normal distribution from noise
        >>> img = np.ones([1000, 1000]) * 2**12 / 2
        >>> img += np.random.normal(-0.025*img.mean(), 0.025*img.mean(), img.shape)
        >>> fcp.nq(img, marker_size=4, line_width=2)

            .. figure:: ../_static/images/example_nq.png
    """

    kwargs['tick_labels'] = kwargs.get('tick_labels', True)

    return plotter(data.NQ, **utl.dfkwarg(df, kwargs))


def paste_kwargs(kwargs: dict) -> dict:
    """Get the kwargs from contents of the clipboard in ini file format.

    Args:
        kwargs: originally inputted kwargs

    Returns:
        copied kwargs
    """
    # Read the pasted data using the ConfigFile class and convert to dict
    try:
        fileio.ConfigFile
        config = fileio.ConfigFile(paste=True)
        new_kw = list(config.config_dict.values())[0]

        # Maintain any kwargs originally specified
        for k, v in kwargs.items():
            new_kw[k] = v

        return new_kw

    except:  # noqa
        print('This feature requires the fivecentfileio package '
              '(download @ https://github.com/endangeredoxen/fivecentfileio) '
              'and pywin32 for the win32clipboard module')


def pie(df, **kwargs):
    """Pie chart

    Args:
        df (DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (str): x-axis column name with categorical data [REQUIRED]
        y (str): y-axis column name with values [REQUIRED]
        pie_colors|colors (str|list): Wedge fill colors. Defaults to fcp.DEFAULT_COLORS. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#colors
        pie_counter_clock|counter_clock (bool): Places wedges in a counter-clockwise fashion. Defaults to False.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#counter-clockwise
        pie_edge_color|edge_color (str): Hex color string for the edge of the pie wedges. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#colors
        pie_edge_style|edge_style (str): Line style for the wedge edge lines {‘-’, ‘--’, ‘-.’, ‘:’}. Defaults to '-’.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#colors
        pie_edge_width|edge_width (float): Width of the wedge edge lines in pixels. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#colors
        pie_explode|explode (list of float): Emphasize one or more wedges by offsetting it from the center of the pie by
          some amount. Defaults to None. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#explode
        pie_fill_alpha|fill_alpha (float): Transparency value for the bars between 0-1. Defaults to 0.85.
        pie_font_color|font_color (str): Font color for the wedge labels. Defaults to #444444. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#wedge-labels
        pie_font_size|font_size (float): Font size for the wedge labels. Defaults to 11. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#wedge-labels
        pie_font_weight|font_weight (str): Font weight for the wedge labels {'light', 'normal', 'medium', 'semibold',
          'bold', 'heavy', 'black'}. Defaults to 'normal'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#wedge-labels
        pie_inner_radius|inner_radius (float): Distance from the center of the pie to the inner edge; used to make donut
          plots. Defaults to pie.html#donut. Example: nan
        pie_label_distance|label_distance (float): Distance from the center of the pie to the category labels. Defaults
          to 1.1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#wedge-labels
        pie_percents_distance|percents_distance (float): Distance from center [0] to edge [pie_radius] at which
          percentage labels are placed. Defaults to 0.6. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#labeled-percents
        pie_percents_font_color|percents_font_color (str): Font color for the percentage labels. Defaults to #444444.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#labeled-percents
        pie_percents_font_size|percents_font_size (float): Font size for the percentage labels. Defaults to 11. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#labeled-percents
        pie_percents_font_weight|percents_font_weight (str): Font weight for the percentage labels {'light', 'normal',
          'medium', 'semibold', 'bold', 'heavy', 'black'}. Defaults to 'normal'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#labeled-percents
        pie_percents|percents (bool): Label each pie wedge with the percentage for that category. Defaults to False.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#labeled-percents
        pie_radius|radius (float): Sets the radius of the pie chart. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#basic-plot
        pie_rotate_labels|rotate_labels (bool): Rotate the pie labels to align with the bisection line from center of
          the pie through the wedge. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#wedge-labels
        pie_shadow|shadow (bool): Add a shadow to give a 3D appearance to the pie chart. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#shadow
        pie_start_angle|start_angle (float): The angle at which the first wedge starts with [3 o'clock = 0; 12 o'clock
          =90; etc]. Defaults to 90. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/pie.html#start-angle

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
        >>> df.loc[df.pH < 0, 'pH'] = -df.pH
        >>> fcp.pie(df, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25')

            .. figure:: ../_static/images/example_pie.png
    """
    return plotter(data.Pie, **utl.dfkwarg(df, kwargs))


def plot(df, **kwargs):
    """XY plot.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Keyword Args:
        x (str | list): x-axis column name(s) [REQUIRED]
        y (str | list): y-axis column name(s) [REQUIRED]
        LINES:
        cmap (str): Color map name (overrides all other color parameters). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Colormap
        line_alpha (str|list): Transparency value for the line(s) between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_color (str|list): Hex color string or list of hex color strings for the plot lines. Defaults to
          fcp.DEFAULT_COLORS. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_style (str|list): Matplotlib string character for line style {'-'; '--'; '-.' ':'}. Defaults to '-'.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_width (int|list): Line width in pixels. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        lines (boolean): Enable/disable plotting of lines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Scatter
        MARKERS:
        marker_edge_color (str|list): Hex color string for the marker edges. Defaults to fcp.DEFAULT_COLORS. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_edge_width (float): Marker edge line width in pixels. Defaults to 1.
        marker_fill (boolean): Enable/disable color fill in markers. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_fill_color (str|list): Hex color string for the fill color of markers. Defaults to fcp.DEFAULT_COLORS.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_jitter|jitter (boolean): For boxplots add random noise on x-axis to show separation between markers.
          Defaults to True. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Jitter
        marker_size (float): Size in pixels of the data point markers. Defaults to 6.
        markers (boolean): Enable/disable data point markers. Defaults to True.
        AX_[H|V]LINES:
        ax_hlines|ax2_hlines (float|list of tuples and floats): Add horizontal lines to the plot; if only float value is
          provided add a solid black line with width=1 pixel at that value; if tuple add any one or more of the
          following in order: [1] float value or DataFrame column name [required]; [2] hex string for line color; [3]
          line style str; [4] line width in pixels; [5] line alpha transparency value from 0-1; [6] legend text  [added
          automatically if using a column name for value]. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Horizontal-&-vertical-lines
        ax_vlines|ax2_vlines (float|list of tuples and floats): Add vertical lines to the plot [same parameters as
          ax_hlines]. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Horizontal-&-vertical-lines
        ax_[h|v]lines (list of values): Add a line with a different value to each subplot when using row/col/wrap
          grouping. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Horizontal-&-vertical-lines
        CONTROL_LIMITS:
        control_limit_side (str): Determines if shaded region is <= lcl and >= ucl {"outside"} or between the lcl and
          ucl {"inside"}. Defaults to outside. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Control-limits
        lcl (float): Float value to start the lower control limit shading region. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Control-limits
        lcl|ucl_edge_alpha (float): Transparency value for the line starting the control limit shaded region between
          0-1. Defaults to 0.25.
        lcl|ucl_edge_color (str): Hex color string for the the line starting the control limit shaded region. Defaults
          to fcp.DEFAULT_COLORS.
        lcl|ucl_edge_style (str): Line style for the line starting the control limit shaded region {‘-’, ‘--’, ‘-.’,
          ‘:’}. Defaults to '-'.
        lcl|ucl_edge_width (float): Width of the line starting the control limit shaded region in pixels. Defaults to 1.
        lcl|ucl_fill_alpha (float): Transparency value for the control limit shaded region fill between 0-1. Defaults to
          0.20. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Control-limits
        lcl|ucl_fill_color (str): Hex color string for the control limit shaded region fill. Defaults to
          fcp.DEFAULT_COLORS. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Control-limits
        ucl (float): Float value to start the upper control limit shading region. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Control-limits
        CONFIDENCE_INTERVALS:
        conf_int (float): Interval with upper and lower bounds based on a single confidence value between 0-1
          (typical=0.95). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_edge_alpha (float): Transparency value for the lines bounding the interval shaded
          region between 0-1. Defaults to 0.25. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_edge_color (str): Hex color string for the the lines bounding the interval shaded
          region. Defaults to fcp.DEFAULT_COLORS. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_edge_style (str): Line style for the lines bounding the interval shaded region {‘-’,
          ‘--’, ‘-.’, ‘:’}. Defaults to '-'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_edge_width (float): Width of the lines bounding the interval shaded region in pixels.
          Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_fill_alpha (float): Transparency value for the interval shaded region fill
          between 0-1. Defaults to 0.20.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        conf_int_|perc_int_|nq_int_fill_color (str): Hex color string for the interval shaded region fill. Defaults to
          fcp.DEFAULT_COLORS. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-
          interval
        nq_int (list of float): Interval with upper and lower bounds based on values of sigma (where the mean of a
          distribution is sigma=0). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        perc_int (list of float): Interval with upper and lower bounds based on percentiles between 0-1. Defaults to
          None. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Confidence-interval
        FIT:
        fit (int): Polynomial degree for the fit. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Line-fit
        fit_color (str): Hex color string for the fit line. Defaults to #000000.
        fit_eqn (boolean): Display the fit equation on the plot. Defaults to False.
        fit_font_size (float): Font size of the fit eqn and rsq value. Defaults to 12.
        fit_padding (int): Padding in pixels from the top of the plot to the location of the fit eqn. Defaults to 10.
        fit_range_x (list): Compute the fit only over a given range of x-values. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Curve-fitting
        fit_range_y (list): Compute the fit only over a given range of y-values. Defaults to None.
        fit_rsq (boolean): Display the rsq of the fit on the plot. Defaults to False.
        REFERENCE_LINES:
        ref_line (list|pd.Series): The name of one or more columns in the DataFrame or a pandas Series with the same
          number of rows as the x column. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        ref_line_alpha (str|list): Transparency value for the reference line(s) between 0-1 (use list if more than one
          ref_line plotted). Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        ref_line_color (str|list): Hex color string or list of hex color strings for the reference line (use list if
          more than one ref_line plotted). Defaults to #000000. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        ref_line_legend_text (str|list): Custom string label(s) to add to a legend for the reference line data (use list
          if more than one ref_line plotted). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        ref_line_style (str|list): Matplotlib string character for reference line style {'-'; '--'; '-.' ':'} (use list
          if more than one ref_line plotted). Defaults to '-'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        ref_line_width (int|list): Reference line width in pixels (use list if more than one ref_line plotted). Defaults
          to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Reference-line
        STAT_LINES:
        stat (str): Calculate a statistic on a data set (any stat value supported by pandas.groupby is valid {'mean',
          'std', etc}. Defaults to None. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Stat-
          lines
        stat_line_xxx (various): Stat-line styling is controlled by the regular line_xxx values. Defaults to None.
        stat_val (str): Alternate column name used as a pseudo x-axis for the stat calculation for cases in which the
          plotted x-column values are not perfectly aligned. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Stat-lines

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2')

            .. figure:: ../_static/images/example_plot.png

    """
    return plotter(data.XY, **utl.dfkwarg(df, kwargs))


def plot_bar(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot data as boxplot

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    # would need to update to support multiple x
    if not kwargs.get('sort', True):
        xvals = df_rc[data.x[0]].unique()
    else:
        xvals = np.sort(df_rc[data.x[0]].unique())

    stacked = pd.DataFrame(index=xvals)
    ss = []

    if data.legend:
        df_rc = df_rc.reset_index(drop=True)
        for n, g in df_rc.groupby(data.x[0]):
            for ii, (nn, gg) in enumerate(g.groupby(data.legend)):
                df_rc.loc[df_rc.index.isin(gg.index), 'Instance'] = ii
                df_rc.loc[df_rc.index.isin(gg.index), 'Total'] = len(g.groupby(data.legend))
    else:
        df_rc['Instance'] = 0
        df_rc['Total'] = 1

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):

        df2 = df.groupby(x).sum()[y].loc[xvals]
        inst = df.groupby(x).mean()['Instance']
        total = df.groupby(x).mean()['Total']

        if layout.bar.error_bars:
            std = df.groupby(x).std()[y]
        else:
            std = None

        if layout.bar.stacked:
            stacked = pd.concat([stacked, df2])

        data = layout.plot_bar(ir, ic, iline, df2, leg_name, data, ngroups, ss,
                               std, xvals, inst, total)

        if layout.rolling_mean.on and not layout.bar.stacked:
            dfrm = df2.rolling(layout.rolling_mean.window).mean()
            dfrm = dfrm.reset_index().reset_index()
            if layout.legend._on:
                legend_name = f'Moving Average = {layout.rolling_mean.window}'
            else:
                legend_name = None
            layout.plot_xy(ir, ic, iline, dfrm, 'index', data.y[0], legend_name, False,
                           line_type='rolling_mean')

        if layout.bar.stacked:
            ss = stacked.groupby(stacked.index).sum()[0]

    return data


def plot_box(dd, layout, ir, ic, df_rc, kwargs):
    """
    Plot data as boxplot

    Args:
        dd (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    Keywords:

    """

    # Init arrays
    data = []
    labels = []
    dividers = []
    stats = []

    if dd.groups is not None:
        col = dd.changes.columns

        # Plot the groups
        for irow, row in dd.indices.iterrows():
            gg = df_rc.copy().sort_values(by=dd.groups)
            gg = gg.set_index(dd.groups)
            if len(gg) > 1:
                gg = gg.loc[tuple(row)]
            if isinstance(gg, pd.Series):
                gg = pd.DataFrame(gg).T
            else:
                gg = gg.reset_index()
            temp = gg[dd.y].dropna()
            temp['x'] = irow + 1
            data += [temp]
            ss = str(layout.box_stat_line.stat).lower()
            if ss == 'median':
                stats += [temp.median().iloc[0]]
            elif ss == 'std':
                stats += [temp.std().iloc[0]]
            elif 'q' in ss:
                if float(ss.strip('q')) < 1:
                    stats += [temp.quantile(float(ss.strip('q'))).iloc[0]]
                else:
                    stats += [temp.quantile(float(ss.strip('q')) / 100).iloc[0]]
            else:
                stats += [temp.mean().iloc[0]]
            # if not isinstance(row, tuple):  # don't believe this case is possible
            #     row = [row]
            # else:
            row = [str(f) for f in row]
            labels += ['']

            if len(dd.changes.columns) > 1 and \
                    dd.changes[col[0]].iloc[irow] == 1 \
                    and len(kwargs['groups']) > 1:
                dividers += [irow + 0.5]

            # Plot points
            if isinstance(dd.legend_vals, pd.DataFrame):
                for jj, jrow in dd.legend_vals.iterrows():
                    temp = gg.loc[gg[dd.legend] == jrow['names']][dd.y].dropna()
                    temp['x'] = irow + 1
                    if len(temp) > 0:
                        layout.plot_xy(ir, ic, jj, temp, 'x', dd.y[0], jrow['names'], False, zorder=10)
            else:
                if len(temp) > 0:
                    layout.plot_xy(ir, ic, irow, temp, 'x', dd.y[0], None, False, zorder=10)

    else:
        data = [df_rc[dd.y].dropna()]
        labels = ['']
        data[0]['x'] = 1
        if isinstance(dd.legend_vals, pd.DataFrame):
            for jj, jrow in dd.legend_vals.iterrows():
                temp = data[0].loc[df_rc[dd.legend] == jrow['names']].index
                layout.plot_xy(ir, ic, jj, data[0].loc[temp], 'x', dd.y[0], jrow['names'], False, zorder=10)
        else:
            layout.plot_xy(ir, ic, 0, data[0], 'x', dd.y[0], None, False, zorder=10)

    # Remove lowest divider
    dividers = [f for f in dividers if f > 0.5]

    # Remove temporary 'x' column
    for dat in data:
        del dat['x']

    # Range lines
    if layout.box_range_lines.on:
        for id, dat in enumerate(data):
            kwargs = layout.box_range_lines.kwargs.copy()
            layout.plot_line(ir, ic, id + 1 - 0.2, dat.max().iloc[0], x1=id + 1 + 0.2, y1=dat.max().iloc[0], **kwargs)
            layout.plot_line(ir, ic, id + 1 - 0.2, dat.min().iloc[0], x1=id + 1 + 0.2, y1=dat.min().iloc[0], **kwargs)
            kwargs['style'] = kwargs['style2']
            layout.plot_line(ir, ic, id + 1, dat.min().iloc[0], x1=id + 1, y1=dat.max().iloc[0], **kwargs)

    # Add boxes
    for ival, val in enumerate(data):
        data[ival] = val[dd.y[0]].values
    layout.plot_box(ir, ic, data, **kwargs)

    # Add divider lines
    if layout.box_divider.on and len(dividers) > 0:
        layout.ax_vlines = copy.deepcopy(layout.box_divider)
        layout.ax_vlines.values = dividers
        layout.ax_vlines.color = copy.copy(layout.box_divider.color)
        layout.ax_vlines.style = copy.copy(layout.box_divider.style)
        layout.ax_vlines.width = copy.copy(layout.box_divider.width)
        layout.add_hvlines(ir, ic)
        layout.ax_vlines.values = []

    # Add mean/median connecting lines
    if layout.box_stat_line.on and len(stats) > 0:
        x = np.linspace(1, dd.ngroups, dd.ngroups)
        layout.plot_line(ir, ic, x, stats, **layout.box_stat_line.kwargs)

    # add group means
    if layout.box_group_means.on is True:
        mgroups = df_rc.groupby(dd.groups[0:1])
        x = -0.5
        for ii, (nn, mm) in enumerate(mgroups):
            y = mm[dd.y[0]].mean()
            y = [y, y]
            if ii == 0:
                x2 = len(mm[dd.groups].drop_duplicates()) + x + 1
            else:
                x2 = len(mm[dd.groups].drop_duplicates()) + x
            layout.plot_line(ir, ic, [x, x2], y, **layout.box_group_means.kwargs)
            x = x2

    # add grand mean
    if layout.box_grand_mean.on is True:
        x = np.linspace(0.5, dd.ngroups + 0.5, dd.ngroups)
        mm = df_rc[dd.y[0]].mean()
        y = [mm for f in x]
        layout.plot_line(ir, ic, x, y, **layout.box_grand_mean.kwargs)

    # add grand mean
    if layout.box_grand_median.on:
        x = np.linspace(0.5, dd.ngroups + 0.5, dd.ngroups)
        mm = df_rc[dd.y[0]].median()
        y = [mm for f in x]
        layout.plot_line(ir, ic, x, y, **layout.box_grand_median.kwargs)

    # add mean confidence diamonds
    if layout.box_mean_diamonds.on:
        mgroups = df_rc.groupby(dd.groups)
        for ii, (nn, mm) in enumerate(mgroups):
            low, high = utl.ci(mm[dd.y[0]], layout.box_mean_diamonds.conf_coeff)
            mm = mm[dd.y[0]].mean()
            x1 = -layout.box_mean_diamonds.width[0] / 2
            x2 = layout.box_mean_diamonds.width[0] / 2
            points = [[ii + 1 + x1, mm],
                      [ii + 1, high],
                      [ii + 1 + x2, mm],
                      [ii + 1, low],
                      [ii + 1 + x1, mm],
                      [ii + 1 + x2, mm]]
            layout.plot_polygon(ir, ic, points, **layout.box_mean_diamonds.kwargs)

    return dd


def plot_contour(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot contour data

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):
        layout.plot_contour(ir, ic, df, x, y, z, data)

    return data


def plot_control_limit(ir: int, ic: int, iline: int, layout: 'engines.Layout', data: 'data.Data'):
    """Add control limit shading to a plot.

    Args:

    """

    x = [data.ranges[ir, ic]['xmin'], data.ranges[ir, ic]['xmax']]
    if layout.lcl.on:
        if layout.ucl.on and layout.control_limit_side == 'inside':
            lower = np.ones(2) * layout.lcl.value[0]
            upper = np.ones(2) * layout.ucl.value[0]
            leg_name = u'lcl \u2192 ucl'
        elif layout.lcl.on and layout.control_limit_side == 'inside':  # use the ucl for this
            lower = np.ones(2) * layout.lcl.value[0]
            upper = np.ones(2) * data.ranges[ir, ic]['ymax']
            leg_name = 'lcl'
        elif layout.lcl.on:
            upper = np.ones(2) * layout.lcl.value[0]
            lower = np.ones(2) * data.ranges[ir, ic]['ymin']
            leg_name = 'lcl'
        if not layout.legend._on:
            leg_name = None
        layout.fill_between_lines(ir, ic, iline, x, lower, upper, 'lcl', leg_name=leg_name, twin=False)

    if layout.ucl.on and layout.lcl.on and layout.control_limit_side == 'inside':
        return data
    if layout.ucl.on:
        if layout.control_limit_side == 'inside':
            upper = np.ones(2) * layout.ucl.value[0]
            lower = np.ones(2) * data.ranges[ir, ic]['ymin']
            leg_name = 'ucl'
        else:
            lower = np.ones(2) * layout.ucl.value[0]
            upper = np.ones(2) * data.ranges[ir, ic]['ymax']
            leg_name = 'ucl'
        if not layout.legend._on:
            leg_name = None
        layout.fill_between_lines(ir, ic, iline, x, lower, upper, 'ucl', leg_name=leg_name, twin=False)

    return data


def plot_fit(data, layout, ir, ic, iline, df, x, y, twin, leg_name, ngroups):
    """
    Plot a fit line

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        iline (int): iterator
        df (pd.DataFrame): input data
        x (str): x-axis column name
        y (str): y-axis column name
        twin (bool): denote twin axis
        leg_name (str): legend value
        ngroups (int): number of groups in this data set

    """
    if not layout.fit.on:
        return

    df, coeffs, rsq = data.get_fit_data(ir, ic, df, x, y)
    if layout.legend._on:
        if layout.fit.legend_text is not None:
            leg_name = layout.fit.legend_text
        elif (data.wrap_vals is not None and ngroups / data.nwrap > 1
                or ngroups / (data.nrow * data.ncol) > 1
                or len(np.unique(layout.fit.color.values)) > 1) \
                and data.legend_vals is not None \
                and layout.label_wrap.column is None:
            leg_name = '%s [Fit]' % leg_name
        else:
            leg_name = 'Fit'
    else:
        leg_name = None
    layout.plot_xy(ir, ic, iline, df, '%s Fit' % x, '%s Fit' % y,
                   leg_name, twin, line_type='fit',
                   marker_disable=True)

    if layout.fit.eqn:
        eqn = 'y='
        for ico, coeff in enumerate(coeffs[0:-1]):
            if coeff > 0 and ico > 0:
                eqn += '+'
            if len(coeffs) - 1 - ico > 1:
                power = '^%s' % str(len(coeffs) - 1 - ico)
            else:
                power = ''
            eqn += '%s*x%s' % (round(coeff, 3), power)
        if coeffs[-1] > 0:
            eqn += '+'
        eqn += '%s' % round(coeffs[-1], 3)
        layout.add_text(ir, ic, eqn, 'fit')

    if layout.fit.rsq:
        layout.add_text(ir, ic, 'R^2=%s' % round(rsq, 4), 'fit',
                        offsety=-2.2 * layout.fit.font_size)

    return data


def plot_gantt(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot data as gantt chart

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    # Sort the values
    ascending = False if layout.gantt.sort.lower() == 'descending' else True
    df_rc = df_rc.sort_values(data.x[0], ascending=ascending)
    if layout.gantt.order_by_legend:
        df_rc = df_rc.sort_values(data.legend, ascending=ascending)

    cols = data.y
    if data.legend is not None:
        cols += [f for f in utl.validate_list(data.legend)
                 if f is not None and f not in cols]

    yvals = [tuple(f) for f in df_rc[cols].values]

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):
        layout.plot_gantt(ir, ic, iline, df, data.x, y, leg_name, yvals, ngroups)

    return data


def plot_heatmap(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot heatmap data data

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):

        # Make the plot
        layout.plot_heatmap(ir, ic, df, x, y, z, data)

    return data


def plot_hist(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot data as histogram

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """
    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):
        # if kwargs.get('groups', False):  # no use case for this
        #     for nn, gg in df.groupby(utl.validate_list(kwargs['groups'])):
        #         hist, data = layout.plot_hist(ir, ic, iline, gg, x, y, leg_name, data)
        # else:
        hist, data = layout.plot_hist(ir, ic, iline, df, x, y, leg_name, data)

    return data


def plot_imshow(data, layout, ir, ic, df_rc, kwargs):
    """
    Show an image

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):
        layout.plot_imshow(ir, ic, df, data)

    return data


def plot_interval(ir, ic, iline, data, layout, df, x, y, twin):
    """
    Add a point-by-point interval based color band around a line plot
    """

    getattr(data, f'get_interval_{layout.interval.type}')(df, x, y)

    leg_name = None
    if layout.legend._on:
        if layout.interval.type == 'nq':
            leg_name = f'nq = [{layout.interval.value[0]}, {layout.interval.value[1]}]'
        elif layout.interval.type == 'percentile':
            leg_name = f'q = [{layout.interval.value[0]}, {layout.interval.value[1]}]'
        else:
            leg_name = f'ci = {layout.interval.value[0]}'

    if twin:
        leg_name = f'{leg_name} '
    layout.fill_between_lines(ir, ic, iline, data.stat_idx, data.lcl, data.ucl,
                              'interval', leg_name=leg_name, twin=twin)

    return data


def plot_nq(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot data as normal quantiles by sigma

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    plot_xy(data, layout, ir, ic, df_rc, kwargs)

    return data


def plot_pie(data, layout, ir, ic, df, kwargs):
    """
    Plot a pie chart

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """
    if data.sort:
        x = df.groupby(data.x[0]).sum().index.values
    else:
        x = df[data.x[0]].drop_duplicates().values
    y = df.groupby(data.x[0]).sum().loc[x][data.y[0]].values

    if any(y < 0):
        print('Pie plot had negative values.  Dropping bad values...')
        negs = np.where(y <= 0)[0]
        y = np.delete(y, negs)
        x = np.delete(x, negs)

    layout.plot_pie(ir, ic, df, x, y, data, layout.pie.__dict__)

    return data


def plot_ref(ir, ic, iline, data, layout, df, x, y):
    """
    Plot a reference line
    """

    if not layout.ref_line.on:
        return

    for iref in range(0, len(layout.ref_line.column.values)):
        layout.plot_xy(ir, ic, iref, df, x, layout.ref_line.column[iref],
                       layout.ref_line.legend_text[iref], False,
                       line_type='ref_line', marker_disable=True)
        layout.legend.ordered_curves = layout.legend.ordered_curves[0:-1]

    return data


def plot_stat(ir, ic, iline, data, layout, df, x, y, leg_name=None, twin=False):
    """
    Plot a line calculated by stats
    """

    df_stat = data.get_stat_data(df, x, y)

    if df_stat is None or len(df_stat) == 0 or layout.fit.on:
        return

    layout.lines.on = True
    layout.plot_xy(ir, ic, iline, df_stat, x, y, leg_name, twin, marker_disable=True)

    return data


def plot_xy(data, layout, ir, ic, df_rc, kwargs):
    """
    Plot xy data

    Args:
        data (obj): Data object
        layout (obj): layout object
        ir (int): current subplot row number
        ic (int): current subplot column number
        df_rc (pd.DataFrame): data subset
        kwargs (dict): keyword args

    """

    for iline, df, x, y, z, leg_name, twin, ngroups in data.get_plot_data(df_rc):
        if data.stat is not None:
            layout.lines.on = False
        if not layout.lines.on and not layout.markers.on:
            pass
        elif kwargs.get('groups', False):
            for nn, gg in df.groupby(utl.validate_list(kwargs['groups']), sort=data.sort):
                layout.plot_xy(ir, ic, iline, gg, x, y, leg_name, twin)
                plot_fit(data, layout, ir, ic, iline, gg,
                         x, y, twin, leg_name, ngroups)
        else:
            layout.plot_xy(ir, ic, iline, df, x, y, leg_name, twin)
            plot_fit(data, layout, ir, ic, iline, df,
                     x, y, twin, leg_name, ngroups)

        plot_ref(ir, ic, iline, data, layout, df, x, y)
        if not layout.lines.on and not layout.markers.on:
            plot_stat(ir, ic, iline, data, layout, df, x, y, leg_name, twin)
        else:
            plot_stat(ir, ic, iline, data, layout, df, x, y, twin=twin)

        # add intervals (prioritized so multiple cannot be added)
        if layout.interval.on:
            plot_interval(ir, ic, iline, data, layout, df, x, y, twin)

        # add control limits
        if layout.lcl.on or layout.ucl.on:
            plot_control_limit(ir, ic, iline, layout, data)

    return data


def plotter(dobj, **kwargs):
    """ Main plotting function

    UPDATE At minimum, it requires a pandas DataFrame with at
    least two columns and two column names for the x and y axis.  Plots can be
    customized and enhanced by passing keyword arguments as defined below.
    Default values that must be defined in order to generate the plot are
    pulled from the fcp_params default dictionary

    Args:
        dobj (Data object):  data class for the specific plot type

    Keyword Args:
        Defined by the specific plot type

    Returns:
        plots
    """

    # Check for deprecated kwargs
    kwargs = deprecated(kwargs)

    # Apply globals if they don't exist
    for k, v in fcp.KWARGS.items():
        if k not in kwargs.keys():
            kwargs[k] = v

    # Build the Timer
    kwargs['timer'] = utl.Timer(print=kwargs.get('timer', False), start=True, units='ms')

    # Set the plotting engine
    verbose = kwargs.get('verbose', False)
    defaults = utl.reload_defaults(kwargs.get('theme', None), verbose=verbose)
    engine = utl.kwget(kwargs, defaults[0], 'engine', 'mpl')
    if not hasattr(engines, engine):
        if engine in INSTALL.keys():
            installs = '\npip install '.join(INSTALL[engine])
            raise EngineError(f'Plotting engine "{engine}" is supported by not installed! '
                              f'Please run the following:\npip install {installs}')
        else:
            raise EngineError(f'Plotting engine "{engine}" is not supported')
        return
    else:
        engine = getattr(engines, engine)
    kwargs['timer'].get('Layout obj')

    # Build the data object and update kwargs
    dd = dobj(fcpp=defaults[0], **kwargs)
    for k, v in kwargs.items():
        if k in dd.__dict__.keys():
            kwargs[k] = getattr(dd, k)
    kwargs['timer'].get('Data obj')

    # Iterate over discrete figures
    for ifig, fig_item, fig_cols, dd in dd.get_df_figure():
        kwargs['timer'].get('dd.get_df_figure')
        # Create a layout object
        layout = engine.Layout(dd, defaults, **kwargs)
        kwargs = layout.kwargs
        kwargs['timer'].get('layout class')

        # Make the figure
        dd = layout.make_figure(dd, **kwargs)
        kwargs['timer'].get('ifig=%s | make_figure' % ifig)

        # Turn off empty subplots and populate layout.axes.visible)
        for ir, ic, df_rc in dd.get_rc_subset():
            if len(df_rc) == 0:  # could set this value in Data after first time to avoid recalc
                if dd.wrap is None:
                    layout.set_axes_rc_labels(ir, ic)
                layout.axes.obj[ir, ic].axis('off')
                layout.axes.visible[ir, ic] = False
                if layout.axes2.obj[ir, ic] is not None:
                    layout.axes2.obj[ir, ic].axis('off')
                continue
        kwargs['timer'].get('ifig=%s | turn off empty subplots' % ifig)

        # Make the subplots
        for ir, ic, df_rc in dd.get_rc_subset():
            if not layout.axes.visible[ir, ic]:
                continue

            # Set the axes colors
            layout.set_axes_colors(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_colors' % (ifig, ir, ic))

            # Add and format gridlines
            layout.set_axes_grid_lines(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_grid_lines' % (ifig, ir, ic))

            # Add horizontal and vertical lines
            layout.add_hvlines(ir, ic, df_rc)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | add_hvlines' % (ifig, ir, ic))

            # Plot the data
            dd = globals()['plot_{}'.format(dd.name)](dd, layout, ir, ic, df_rc, kwargs)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | plot' % (ifig, ir, ic))

            # Set linear or log axes scaling
            layout.set_axes_scale(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_scale' % (ifig, ir, ic))

            # Set axis ranges
            layout.set_axes_ranges(ir, ic, dd.ranges)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_ranges' % (ifig, ir, ic))

            # Add axis labels
            layout.set_axes_labels(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_labels' % (ifig, ir, ic))

            # Add rc labels
            layout.set_axes_rc_labels(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_rc_labels' % (ifig, ir, ic))

            # Adjust tick marks
            layout.set_axes_ticks(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | set_axes_ticks' % (ifig, ir, ic))

            # Add box labels
            if dd.name == 'box':
                layout.add_box_labels(ir, ic, dd)
                kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | add_box_labels' % (ifig, ir, ic))

            # Add arbitrary text
            layout.add_text(ir, ic)
            kwargs['timer'].get('ifig=%s | ir=%s | ic=%s | add_text' % (ifig, ir, ic))

        # Make the legend
        layout.add_legend(dd.legend_vals)
        kwargs['timer'].get('ifig=%s | add_legend' % (ifig))

        # Add a figure title
        layout.set_figure_title()
        kwargs['timer'].get('ifig=%s | set_figure_title' % (ifig))

        # Final adjustments
        layout.set_figure_final_layout(dd, **kwargs)
        kwargs['timer'].get('ifig=%s | set_figure_final_layout' % (ifig))

        # Build the save filename
        filename = utl.set_save_filename(dd.df_fig, ifig, fig_item, fig_cols, layout, kwargs)
        if 'filepath' in kwargs.keys():
            filename = os.path.join(kwargs['filepath'], filename)

        # Optionally save and open
        if kwargs.get('save', False) or kwargs.get('show', False):
            if ifig:
                idx = ifig
            else:
                idx = 0
            layout.save(filename, idx)
            if kwargs.get('return_filename'):
                layout.close()
                if 'filepath' in kwargs.keys():
                    return osjoin(kwargs['filepath'], filename)
                else:
                    return osjoin(os.getcwd(), filename)
            if kwargs.get('print_filename', False):
                print(filename)
            if kwargs.get('show', False):
                utl.show_file(filename)

            # Disable inline unless explicitly called in kwargs
            if not kwargs.get('inline'):
                kwargs['inline'] = False
        kwargs['timer'].get('ifig=%s | save' % (ifig))

        # Return inline plot
        if not kwargs.get('inline', True):
            layout.close()
        else:
            layout.show()
        kwargs['timer'].get('ifig=%s | return inline' % (ifig))

    # Save data used in the figures
    if kwargs.get('save_data', False):
        if isinstance(kwargs['save_data'], str):
            filename = kwargs['save_data']
        else:
            filename = filename.split('.')[0] + '.csv'
        dd.df_all[dd.cols_all].to_csv(filename, index=False)
        kwargs['timer'].get('ifig=%s | save_data' % (ifig))

    kwargs['timer'].get_total()


def set_theme(theme=None, verbose=False):
    """
    Select a "defaults" file and copy to the user directory
    """

    if theme is not None and os.path.exists(theme):
        my_theme_dir = os.sep.join(theme.split(os.sep)[0:-1])
        theme = theme.split(os.sep)[-1]
        ignores = []
    else:
        my_theme_dir = osjoin(user_dir, '.fivecentplots')
        ignores = ['defaults.py', 'defaults_old.py']

    if theme is not None:
        theme = theme.replace('.py', '')

    themes = [f.replace('.py', '') for f in os.listdir(osjoin(cur_dir, 'themes')) if '.py' in f]
    mythemes = [f.replace('.py', '') for f in os.listdir(my_theme_dir) if '.py' in f and f not in ignores]
    if verbose:
        print(f'theme: {theme}\nthemes: {themes}\nmythemes: {mythemes}\nmy_theme_dir: {my_theme_dir}')

    if theme in themes:
        entry = themes.index('%s' % theme) + 1

    elif theme in mythemes:
        entry = mythemes.index('%s' % theme) + 1 + len(themes)

    elif theme is not None:
        print('Theme file not found!  Please try again')
        return

    else:
        if '_test' in themes:
            themes.remove('_test')
        print('Select default styling theme:')
        print('   Built-in theme list:')
        for i, th in enumerate(themes):
            print('      %s) %s' % (i + 1, th))
        if len(themes) > 0:
            print('   User theme list:')
            for i, th in enumerate(mythemes):
                print('      %s) %s' % (i + 1 + len(themes), th))
        entry = input('Entry: ')

        try:
            int(entry)
        except TypeError:
            print('Invalid selection!  Please try again')
            return

        if int(entry) > len(themes) + len(mythemes) or int(entry) <= 0:
            print('Invalid selection!  Please try again')
            return

        if int(entry) <= len(themes):
            print('Copying %s >> %s' % (themes[int(entry) - 1], osjoin(user_dir, '.fivecentplots', 'defaults.py')))
            theme = themes[int(entry) - 1]
        else:
            print('Copying %s >> %s' % (mythemes[int(entry) - 1 - len(themes)],
                                        osjoin(user_dir, '.fivecentplots', 'defaults.py')))
            theme = mythemes[int(entry) - 1 - len(themes)]

    if os.path.exists(osjoin(user_dir, '.fivecentplots', 'defaults.py')):
        print(f'Previous theme file found! Renaming to "defaults_old.py" and copying theme "{theme}"...', end='')
        shutil.copy2(osjoin(user_dir, '.fivecentplots', 'defaults.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults_old.py'))

    if not os.path.exists(osjoin(user_dir, '.fivecentplots')):
        os.makedirs(osjoin(user_dir, '.fivecentplots'))

    if entry is not None and int(entry) <= len(themes):
        shutil.copy2(osjoin(cur_dir, 'themes', themes[int(entry) - 1] + '.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults.py'))
    else:
        shutil.copy2(osjoin(my_theme_dir, mythemes[int(entry) - 1 - len(themes)] + '.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults.py'))

    print('done!')


# functions for docstring purposes only
def axes():
    """Dummy function to return the axes API with ``help()`` (not used directly for plotting).

    Keyword Args:
        ax_edge_alpha (str): Transparency value for axes edge between 0-1. Defaults to 1.
        ax_edge_bottom (boolean): Enable/disable the bottom axes edge (or spine). Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Axes-edges
        ax_edge_color (str): Hex color string for the border edge of the axes region. Defaults to #aaaaaa. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Typical-elements
        ax_edge_left (boolean): Enable/disable the left axes edge (or spine). Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Axes-edges
        ax_edge_right (boolean): Enable/disable the right axes edge (or spine). Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Axes-edges
        ax_edge_top (boolean): Enable/disable the top axes edge (or spine). Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Axes-edges
        ax_edge_width (float): Width of the axes border in pixels. Defaults to 1.
        ax_fill_alpha (str): Transparency value for axes fill between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Alpha
        ax_fill_color (str): Hex color string for the fill color of the axes region. Defaults to #eaeaea. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Alpha
        ax_scale|ax2_scale (str): Set the scale type of the axes {'linear'; 'logx'; 'semilogx'; 'logy'; 'semilogy';
          'loglog'; 'log'; 'symlog'; 'logit'}. Defaults to 'linear'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Log-scale
        ax_size (list of int | str): Axes size [width, height]; note this is not the size of the entire figure but
          just the axes area; for boxplots can enter 'auto' to auto-scale the width. Defaults to [400, 400].
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Time-series
        share_col (boolean): Share the x and y axis ranges of subplots in the same column when grouping. Defaults to
          True. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Share-columns
        share_row (boolean): Share the x and y axis ranges of subplots in the same row when grouping. Defaults to True.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Share-row
        share_x (boolean): Share the x-axis range across grouped plots with multiple axes. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Axes-sharing
        share_x2 (boolean): Share the secondary x-axis range across grouped plots with multiple axes. Defaults to True.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Axes-sharing
        share_y (boolean): Share the y-axis range across grouped plots with multiple axes. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Axes-sharing
        share_y2 (boolean): Share the secondary y-axis range across grouped plots with multiple axes. Defaults to True.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ranges.html#Axes-sharing
        twin_x (boolean): Add a secondary y-axis by "twinning" the x-axis. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Shared-x-axis-(twin_x)
        twin_y (boolean): Add a secondary x-axis by "twinning" the y-axis. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Shared-y-axis-(twin_y)

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], legend=['Die', 'Substrate'],
                     col='Boost Level', twin_x=True,
                     share_y=False, share_y2=False, share_x=True,
                     filter='Target Wavelength==450 & Temperature [C]==25',
                     ax_edge_color='#FF0000', ax_edge_left=False, ax_edge_right=False, ax_edge_width=2,
                     ax_fill_color='#96BEAA', ax_fill_alpha=0.5,
                     ax_scale='logx', ax_size=[400, 300])

            .. figure:: ../_static/images/example_axes.png

               Axes element is shown in olive green with red borders

    """


def cbar():
    """Dummy function to return the colorbar API with `help()` (not used directly for plotting).

    Keyword Args:
        cbar (bool): Toggle colorbar on/off for contour and heatmap plots. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/contour.html#Filled-contour
        size (int): cbar width [height will match the height of the axes]. Defaults to 30.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
        >>> fcp.contour(cc, x='X', y='Y', z='Value', cbar=True, cbar_size=20, xmin=-4, xmax=2, ymin=-4, ymax=2)

            .. figure:: ../_static/images/example_cbar.png

    """


def figure():
    """Dummy function to return the figure API with `help()` (not used directly for plotting).

    Keyword Args:
        dpi (int): Dots per square inch resolution for the figure. Defaults to 100.
        fig_edge_alpha (str): Transparency value for figure edge between 0-1. Defaults to 1.
        fig_edge_color (str): Hex color string for the border edge of the figure region. Defaults to #aaaaaa. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Typical-elements
        fig_edge_width (float): Width of the figure border in pixels. Defaults to 3.
        fig_fill_alpha (str): Transparency value for figure fill between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Alpha
        fig_fill_color (str): Hex color string for the fill color of the figure region. Defaults to #eaeaea. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Alpha

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     fig_edge_color='#000000', fig_edge_width=3, fig_edge_alpha=0.5,
                     fig_fill_color='#96BEAA', fig_fill_alpha=0.8,
                     ax_size=[400, 300]

            .. figure:: ../_static/images/example_figure.png

               Figure element is shown in olive green with red border
    """


def gridlines():
    """Dummy function to return the gridline API with `help()` (not used directly for plotting).

    Keyword Args:
        grid_major (boolean): Enable/disable major x-axis and y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_alpha (str): Transparency value for major gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_color (str): Hex-axis color string for x-axis and y-axis major gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_width (float): Major gridline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x (boolean): Enable/disable major x-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x2 (boolean): Enable/disable secondary-axis major x-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x2_alpha (str): Transparency value for secondary-axis major x-axis gridlines between 0-1. Defaults to
          1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x2_color (str): Hex-axis color string for secondary-axis x-axis major gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x2_width (float): Major secondary x-axis gridline width in pixels (float ok). Defaults to 1.3.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x_alpha (str): Transparency value for major x-axis gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x_color (str): Hex-axis color string for x-axis major gridlines. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_x_width (float): Major x-axis gridline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y (boolean): Enable/disable major y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y2 (boolean): Enable/disable secondary-axis major y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y2_alpha (str): Transparency value for secondary-axis major y-axis gridlines between 0-1. Defaults to
          1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y2_color (str): Hex-axis color string for secondary-axis y-axis major gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y2_width (float): Major secondary y-axis gridline width in pixels (float ok). Defaults to 1.3.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y_alpha (str): Transparency value for major y-axis gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y_color (str): Hex-axis color string for y-axis major gridlines. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_major_y_width (float): Major y-axis gridline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor (boolean): Enable/disable minor x-axis and y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_alpha (str): Transparency value for minor gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_color (str): Hex-axis color string for x-axis and y-axis minor gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_width (float): minor gridline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x (boolean): Enable/disable minor x-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x2 (boolean): Enable/disable secondary-axis minor x-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x2_alpha (str): Transparency value for secondary-axis minor x-axis gridlines between 0-1. Defaults to
          1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x2_color (str): Hex-axis color string for secondary-axis x-axis minor gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x2_width (float): minor secondary x-axis gridline width in pixels (float ok). Defaults to 0.5.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x_alpha (str): Transparency value for minor x-axis gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x_color (str): Hex-axis color string for x-axis minor gridlines. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_x_width (float): minor x-axis gridline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y (boolean): Enable/disable minor y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y2 (boolean): Enable/disable secondary-axis minor y-axis gridlines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y2_alpha (str): Transparency value for secondary-axis minor y-axis gridlines between 0-1. Defaults to
          1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y2_color (str): Hex-axis color string for secondary-axis y-axis minor gridlines. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y2_width (float): minor secondary y-axis gridline width in pixels (float ok). Defaults to 0.5.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y_alpha (str): Transparency value for minor y-axis gridlines between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y_color (str): Hex-axis color string for y-axis minor gridlines. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        grid_minor_y_width (float): Minor y-axis gridline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Grids
        tick_cleanup (str|boolean): Set the tick cleanup style when dealing with overlaping tick labels
          {False -> ignore | "shrink" -> change the font | "remove" -> delete one of the overlapping labels}.
          Defaults to "shirnk".  Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-cleanup

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', lines=False, ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     grid_major_x_style='--', grid_major_y_color='#00EE00',
                     grid_minor=True, grid_minor_color='#BB0000', grid_minor_y_alpha=0.3, grid_minor_x_width=2

            .. figure:: ../_static/images/example_gridlines.png
    """


def grouping():
    """Dummy function to return the grouping API with `help()` (not used directly for plotting).

    Keyword Args:
        col (str): [1] name of DataFrame column for grouping into columns of subplots based on each unique value; or [2]
          col="x" with multiple values defined for "x" creates columns of subplots for each x-value. Defaults to None.
        groups (str): for xy plot = name of DataFrame column that can be used to separate the data into unique groups so
          plot lines do not circle back on themselves. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/grouping.html#xy-plots
        groups (str|list): for boxplot = name or list of names of DataFrame column(s) used to split the data into
          separate boxes. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/grouping.html#boxplots
        row (str): [1] name of DataFrame column for grouping into rows of subplots based on each unique value; or [2]
          row="y" with multiple values defined for "y" creates rows of subplots for each y-value. Defaults to None.
        wrap (str|list): [1] name or list of names of DataFrame column(s) for grouping into a grid of subplots; [2]
          wrap="x" with multiple values defined for "x" creates a grid of subplots for each x-value; or [3] wrap="y"
          with multiple values defined for "y" creates a grid of subplots for each y-value. Defaults to None.

    Examples
    --------
    Row by column style:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', col='Die', row='Substrate',
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost  Level==0.2', ax_size=[300, 250])

            .. figure:: ../_static/images/example_grouping1.png
               :width: 1078px

    Wrap style:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', wrap=['Die', 'Substrate'],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost  Level==0.2', ax_size=[300, 250])

            .. figure:: ../_static/images/example_grouping2.png
                :width: 977px

    Box plot:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, y='Value', groups=['Batch', 'Sample'])

            .. figure:: ../_static/images/example_grouping3.png
               :width: 544px

    """


def labels():
    """Dummy function to return the labels API with `help()` (not used directly for plotting).

    Keyword Args:
        AXES_LABELS:
        label_bg_padding (float): Padding around the label text for the background object behind the text. Defaults to
          2.
        label_q (str): Custom text for a specific axes label [where q = x, y, x2, y2]. Defaults to DataFrame column
          name. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Multiple-y-only
        label_q_edge_alpha (float): Transparency value for the label edge between 0-1 [where q = x, y, x2, y2]. Defaults
          to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-colors
        label_q_edge_color (str): Hex color string for the label edge [where q = x, y, x2, y2]. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-colors
        label_q_edge_width (float): Width of the border edge of a label in pixels [where q = x, y, x2, y2]. Defaults to
          1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-colors
        label_q_fill_alpha (float): Transparency value for the label background fill between 0-1 [where q = x, y, x2,
          y2]. Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fill-colors
        label_q_fill_color (str): Hex color string for the label edge [where q = x, y, x2, y2]. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fill-colors
        label_q_font (str): Font for a specific axes label [where q = x, y, x2, y2]. Defaults to sans-serif. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_color (str): Hex color string for font color of a specific axes label [where q = x, y, x2, y2].
          Defaults to #000000. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_size (str): Font size for a specific axes label [where q = x, y, x2, y2]. Defaults to 14. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_style (str): Font style {'normal'|'italic'|'oblique'} for a specific axes label [where q = x, y,
          x2, y2]. Defaults to italic. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} for a specific
          axes label [where q = x, y, x2, y2]. Defaults to bold. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        RC_LABELS:
        label_q_edge_alpha (float): Transparency value for the label edge between 0-1 [where q = rc, col, row, wrap; rc
          changes all]. Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-
          colors
        label_q_edge_color (str): Hex color string for the label edge [where q = rc, col, row, wrap; rc changes all].
          Defaults to #8c8c8c. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-colors
        label_q_edge_width (float): Width of the border edge of a label in pixels [where q = rc, col, row, wrap; rc
          changes all]. Defaults to 0. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-
          colors
        label_q_fill_alpha (float): Transparency value for the label background fill between 0-1 [where q = rc, col,
          row, wrap; rc changes all]. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fill-colors
        label_q_fill_color (str): Hex color string for the label edge [where q = rc, col, row, wrap; rc changes all].
          Defaults to #8c8c8c. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fill-colors
        label_q_font (str): Font for a specific axes label [where q = rc, col, row, wrap; rc changes all]. Defaults to
          sans-serif. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_color (str): Hex color string for font color of a specific axes label [where q = rc, col, row,
          wrap; rc changes all]. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_size (str): Font size for a specific axes label [where q = rc, col, row, wrap; rc changes all].
          Defaults to 16. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_style (str): Font style {'normal'|'italic'|'oblique'} for a specific axes label [where q = rc, col,
          row, wrap; rc changes all]. Defaults to normal. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} for a specific
          axes label [where q = rc, col, row, wrap; rc changes all]. Defaults to bold. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        label_q_names (boolean): Toggle including the DataFrame column names in the row or column labels [where q = rc,
          col, row; rc changes all]. Defaults to False.  Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Column-plot
        label_q_size (str): Label background rectangle height for an col/wrap label OR width for row label,30,None
          title_wrap_font,str,Font for the wrap title bar text". Defaults to label_wrap_font.
        title_wrap_edge_alpha (float): Transparency value for the wrap title bar edge between 0-1. Defaults to
          label_rc_. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Edge-colors
        title_wrap_edge_color (str): Hex color string for the wrap title bar edge. Defaults to #5f5f5f.
        title_wrap_edge_width (float): Width of the wrap title bar edge in pixels. Defaults to label_wrap_edge_width.
        title_wrap_fill_alpha (float): Transparency value for the wrap title bar background fill between 0-1. Defaults
          to label_wrap_fill_alpha. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fill-
          colors
        title_wrap_fill_color (str): Hex color string for the wrap title bar. Defaults to #5f5f5f.
        title_wrap_font_color (str): Hex color string for the wrap title bar text. Defaults to label_wrap_font_color.
        title_wrap_font_size (str): Font size for the wrap title bar text. Defaults to 16.
        title_wrap_font_style (str): Font style {'normal'|'italic'|'oblique'} for the wrap title bar text. Defaults to
          label_wrap_font_style.
        title_wrap_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} for the
          wrap title bar text. Defaults to label_wrap_font_weight.
        title_wrap_size (str): Label background rectangle height for the wrap title bar. Defaults to label_wrap_size.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', col='Die', row='Substrate',
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2', ax_size=[300, 250],
                     label_rc_edge_color='#000000', label_rc_edge_width=2,
                     label_row_font_color='#AA0000', label_row_fill_alpha=0.5, label_row_fill_color='#00AA00',
                     label_col_fill_color='#0000AA',
                     label_x_font_style='normal', label_y_font_color='#AABBCC', label_x_fill_color='#DDDDDD')

            .. figure:: ../_static/images/example_labels.png
    """


def legend():
    """Dummy function to return the legend API with `help()` (not used directly for plotting).

    Keyword Args:
        legend_edge_color (str): Hex color string for the legend border. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Typical-elements
        legend_edge_width (float): Width of the legend border in pixels. Defaults to 1.
        legend_font_size (float): Font size of the legend text. Defaults to 12.
        legend_location (int): Position of the legend {0 = outside; 1 = upper right; 2 = upper left; 3 = lower left; 4 =
          lower right; 5 = right; 6 = center left; 7 = center right; 8 = lower center; 9 = upper center; 10 = center; 11
          = below}. Defaults to  0. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/grouping.html#Location
        legend_marker_alpha (float): Transparency value for legend markers between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Alpha
        legend_marker_size (float): Marker size in the legend in pixels. Defaults to 7. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-size
        legend_on (boolean): Enable/disable visibility of legend that has been created using the legend kwarg. Defaults
          to True [if legend enabled].
        legend_points (int): Number of points in the legend region for each entry [to enable multiple markers as in
          matplotlib]. Defaults to 1.
        legend_title (str): Custom title for the legend region [default is the column name used for the legend
          grouping].

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2', ax_size=[400, 300],
                     legend_edge_color='#555555', legend_edge_width=2, legend_font_size=10,
                     legend_marker_size=10, legend_marker_alpha=0.5, legend_title='DS9')

            .. figure:: ../_static/images/example_legend.png
    """


def lines():
    """Dummy function to return the lines API with `help()` (not used directly for plotting).

    Keyword Args:
        cmap (str): Color map name (overrides all other color parameters). Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Colormap
        line_alpha (str|list): Transparency value for the line(s) between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_color (str|list): Hex color string or list of hex color strings for the plot lines. Defaults to
          fcp.DEFAULT_COLORS. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_style (str|list): Matplotlib string character for line style {'-'; '--'; '-.' ':'}. Defaults to '-'.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        line_width (int|list): Line width in pixels. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Line-styling
        lines (boolean): Enable/disable plotting of lines. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/plot.html#Scatter

    Examples
    --------
    All lines the same color:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     line_color='#000000', line_width=2)

            .. figure:: ../_static/images/example_lines1.png

    Patterns of alternating custom color and style:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     line_color=['#AA00AA', '#00AA00', '#0000AA'],  line_width=2, line_style=['-', '--'])

            .. figure:: ../_static/images/example_lines2.png

    Use a colormap with alpha:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     cmap='viridis', lines_alpha=0.7)

            .. figure:: ../_static/images/example_lines3.png

    No lines:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     lines=False)

            .. figure:: ../_static/images/example_lines4.png
    """


def markers():
    """Dummy function to return the markers API with `help()` (not used directly for plotting).

    Keyword Args:
        marker_edge_color (str|list): Hex color string for the marker edges. Defaults to fcp.DEFAULT_COLORS. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_edge_width (float): Marker edge line width in pixels. Defaults to 1.
        marker_fill (boolean): Enable/disable color fill in markers. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_fill_color (str|list): Hex color string for the fill color of markers. Defaults to fcp.DEFAULT_COLORS.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-colors
        marker_jitter|jitter (boolean): For boxplots add random noise on x-axis to show separation between markers.
          Defaults to True. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/boxplot.html#Jitter
        marker_size (float|str): Size in pixels of the data point markers or a DataFrame column name with a custom
          marker size on each row. Defaults to 6.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Marker-size
        markers (boolean): Enable/disable data point markers. Defaults to True.

    Examples
    --------
    No markers:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     markers=False)

            .. figure:: ../_static/images/example_markers1.png

    Styled markers:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     marker_size=10, marker_edge_width=2, marker_fill=True)

            .. figure:: ../_static/images/example_markers2.png

    Custom markers types and color (a mix of indices from the ``fcp.DEFAULT_COLORS`` list and a custom hex color):

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     markers=['o', None, '+', '*', 'B', None], marker_edge_color=[3, 0, 6, 1, '#FF0000'])

            .. figure:: ../_static/images/example_markers3.png

    """


def options():
    """Dummy function to return the API for other control options with `help()` (not used directly for plotting).

    Keyword Args:
        BAYER (list): Color scheme for RGGB channel data so lines and markers match CFA type. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#RGB
        DEFAULT_COLORS (list): Default color scheme used for lines and markers (from colors.py). Defaults to None.
        HIST (dict): Shortcut of useful kwargs to format hist plots {'ax_scale': 'logy', 'markers': False, 'line_width':
          2, 'preset': 'HIST'}. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/hist.html#fcp.HIST
        engine (str): Specify the plotting engine {'mpl', 'bokeh'}. Defaults to 'mpl'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/layout.html#engine
        filename (str): Name of the saved image (with or without path and/or extension). Defaults to Automatic name
          based on conditions with extention '.png'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/grouping.html#figure-plots
        filepath (str): Name of the directory to save images (convenient if you want to use the default naming but save
          in a different directory. Defaults to current directory.
        inline (boolean): Flag to display the rendered plot in the native plotting viewer or jupyter notebook
          (convenient to disable if doing automated batch plotting). Defaults to True.
        print_filename (boolean): Print the output filename, if the plot is saved. Defaults to False. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/grouping.html#figure-plots
        return_filename (boolean): Return the output filename, if the plot is saved. Defaults to False.
        save (boolean): Save the plot to disk. Defaults to False.
        save_data (boolean): Save the DataFrame subset that is created and used by a given plot. Defaults to False.
        save_ext (str): Set the file extension of saved plots to determine the format. Defaults to depends on plotting
          engine {'mpl': '.png', 'bokeh': '.html'}.
        show (str): Show the "saved" plot image file using the default image viewer of the host PC.  Setting to "True"
          forces the image to be saved to disk. Defaults to False.
        theme (str): Select a theme file for the current plot only. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#On-the-fly
        timer (boolean): Debug feature to get a time log for each step in the plotting process. Defaults to False.

    """


def tick_labels():
    """Dummy function to return the tick labels API with `help()` (not used directly for plotting).

    Keyword Args:
        tick_labels (boolean): Enable/disable all tick labels. Defaults to True.
        tick_labels_edge_alpha (float): Transparency value for all tick labels background edges between 0-1. Defaults to
          0.
        tick_labels_edge_color (str): Hex color string for all tick label background edges. Defaults to #ffffff.
        tick_labels_edge_width (float): Width of the border edge of all tick labels in pixels. Defaults to 0.
        tick_labels_fill_alpha (float): Transparency value for the background fill of all tick labels between 0-1.
          Defaults to 1.
        tick_labels_fill_color (str): Hex color string for all tick label background edges. Defaults to #ffffff.
        tick_labels_font (str): Font for all tick labels. Defaults to sans-serif.
        tick_labels_font_color (str): Hex color string for font color of all tick labels. Defaults to #000000.
        tick_labels_font_size (str): Font size for all tick labels. Defaults to 13.
        tick_labels_font_style (str): Font style {'normal'|'italic'|'oblique'} for all tick labels. Defaults to normal.
        tick_labels_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} of all
          tick labels. Defaults to normal.
        tick_labels_major (boolean): Enable/disable all major tick labels. Defaults to True.
        tick_labels_major_edge_alpha (float): Transparency value for all major tick labels background edges between 0-1.
          Defaults to 0.
        tick_labels_major_edge_color (str): Hex color string for all major tick label background edges. Defaults to
          #ffffff.
        tick_labels_major_edge_width (float): Width of the border edge of all major tick labels in pixels. Defaults to
          0.
        tick_labels_major_fill_alpha (float): Transparency value for the background fill of all major tick labels
          between 0-1. Defaults to 1.
        tick_labels_major_fill_color (str): Hex color string for all major tick label background edges. Defaults to
          #ffffff.
        tick_labels_major_font (str): Font for all major tick labels. Defaults to sans-serif.
        tick_labels_major_font_color (str): Hex color string for font color of all major tick labels. Defaults to
          #000000.
        tick_labels_major_font_size (str): Font size for all major tick labels. Defaults to 13.
        tick_labels_major_font_style (str): Font style {'normal'|'italic'|'oblique'} for all major tick labels. Defaults
          to normal.
        tick_labels_major_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} of
          all major tick labels. Defaults to normal.
        tick_labels_major_q (boolean): Enable/disable major tick labels of a specific axes [where q = x, y, x2, y2].
          Defaults to True.
        tick_labels_major_q_edge_alpha (float): Transparency value for major tick labels background edges of a specific
          axes between 0-1 [where q = x, y, x2, y2]. Defaults to 0.
        tick_labels_major_q_edge_color (str): Hex color string for major tick label background edges of a specific axis
          [where q = x, y, x2, y2]. Defaults to #ffffff.
        tick_labels_major_q_edge_width (float): Width of the border edge of all major tick labels of a specific axis in
          pixels [where q = x, y, x2, y2]. Defaults to 0.
        tick_labels_major_q_fill_alpha (float): Transparency value for the background fill of all major tick labels of a
          specific axis between 0-1 [where q = x, y, x2, y2]. Defaults to 1.
        tick_labels_major_q_fill_color (str): Hex color string for major tick label background edges of a specific axis
          [where q = x, y, x2, y2]. Defaults to #ffffff.
        tick_labels_major_q_font (str): Font for major tick labels of a specific axes [where q = x, y, x2, y2]. Defaults
          to sans-serif.
        tick_labels_major_q_font_color (str): Hex color string for font color of major tick labels of a specific axes
          [where q = x, y, x2, y2]. Defaults to #000000.
        tick_labels_major_q_font_size (str): Font size for major tick labels of a specific axes label [where q = x, y,
          x2, y2]. Defaults to 13.
        tick_labels_major_q_font_style (str): Font style {'normal'|'italic'|'oblique'} for major tick labels of a
          specific axes [where q = x, y, x2, y2]. Defaults to normal.
        tick_labels_major_q_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} of
          major tick labels of a specific axes [where q = x, y, x2, y2]. Defaults to normal.
        tick_labels_minor (boolean): Enable/disable all minor tick labels. Defaults to False.
        tick_labels_minor_edge_alpha (float): Transparency value for all minor tick labels background edges between 0-1.
          Defaults to 0.
        tick_labels_minor_edge_color (str): Hex color string for all minor tick label background edges. Defaults to
          #ffffff.
        tick_labels_minor_edge_width (float): Width of the border edge of all minor tick labels in pixels. Defaults to
          0.
        tick_labels_minor_fill_alpha (float): Transparency value for the background fill of all minor tick labels
          between 0-1. Defaults to 1.
        tick_labels_minor_fill_color (str): Hex color string for all minor tick label background edges. Defaults to
          #ffffff.
        tick_labels_minor_font (str): Font for all minor tick labels. Defaults to sans-serif.
        tick_labels_minor_font_color (str): Hex color string for font color of all minor tick labels. Defaults to
          #000000.
        tick_labels_minor_font_size (str): Font size of a specific axes label [where q = x, y, x2, y2]. Defaults to 10.
        tick_labels_minor_font_style (str): Font style {'normal'|'italic'|'oblique'} for all minor tick labels. Defaults
          to normal.
        tick_labels_minor_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} of
          all minor tick labels. Defaults to normal.
        tick_labels_minor_q (boolean): Enable/disable minor tick labels of a specific axes [where q = x, y, x2, y2].
          Defaults to False.
        tick_labels_minor_q_edge_alpha (float): Transparency value for the label edge of a specific axes between 0-1
          [where q = x, y, x2, y2]. Defaults to 0.
        tick_labels_minor_q_edge_color (str): Hex color string for minor tick label background edges of a specific axis
          [where q = x, y, x2, y2]. Defaults to #ffffff.
        tick_labels_minor_q_edge_width (float): Width of the border edge of all minor tick labels of a specific axis in
          pixels [where q = x, y, x2, y2]. Defaults to 0.
        tick_labels_minor_q_fill_alpha (float): Transparency value for the background fill of all minor tick labels of a
          specific axis between 0-1 [where q = x, y, x2, y2]. Defaults to 1.
        tick_labels_minor_q_fill_color (str): Hex color string for minor tick label background edges of a specific axes
          [where q = x, y, x2, y2]. Defaults to #ffffff.
        tick_labels_minor_q_font (str): Font for minor tick labels of a specific axes [where q = x, y, x2, y2]. Defaults
          to sans-serif.
        tick_labels_minor_q_font_color (str): Hex color string for font color of minor tick labels of a specific axes
          [where q = x, y, x2, y2]. Defaults to #000000.
        tick_labels_minor_q_font_size (str): Font size for all minor tick labels. Defaults to 10.
        tick_labels_minor_q_font_style (str): Font style {'normal'|'italic'|'oblique'} for minor tick labels of a
          specific axes [where q = x, y, x2, y2]. Defaults to normal.
        tick_labels_minor_q_font_weight (str): Font weight {'normal'|'bold'|'heavy'|'light'|'ultrabold'|'ultralight'} of
          minor tick labels of a specific axes [where q = x, y, x2, y2]. Defaults to normal.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     tick_labels_major_x_font_color='#FF0000', tick_labels_major_x_font_style='italic',
                     tick_labels_major_x_fill_color='#EEEEEE', tick_labels_major_x_edge_color='#0000FF',
                     tick_labels_minor=True, tick_labels_minor_y_font_weight='bold', tick_labels_minor_x_edge_width=1,
                     tick_labels_minor_y_edge_color='#00FF00', tick_labels_minor_y_edge_width=1,
                     tick_labels_minor_y_font_color='#FF00FF', tick_labels_minor_x_rotation=45,
                     tick_labels_minor_y_fill_color='#000000', tick_labels_minor_font_size=6)

            .. figure:: ../_static/images/example_tick_labels.png
    """


def ticks():
    """Dummy function to return the ticks API with `help()` (not used directly for plotting).

    Keyword Args:
        ticks_major (boolean): Enable/disable major x-axis and y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_alpha (str): Transparency-axis value for major tick marks between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_color (str): Hex-axis color string for x-axis and y-axis major tick marks. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_direction (str): Point tick marks 'in' or 'out' from the axes area. Defaults to 'in'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_major_increment (float): Specify the spacing of major tick marks. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-increment
        ticks_major_length (float): Specify the length of the major tick marks in pixels. Defaults to 6.2. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_major_width (float): Major tickline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_width (float): Specify the width of the major tick marks in pixels. Defaults to 2.2. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_major_x (boolean): Enable/disable major x-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x2 (boolean): Enable/disable secondary-axis major x-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x2_alpha (str): Transparency-axis value for secondary-axis major x-axis tick marks between 0-1.
          Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x2_color (str): Hex-axis color string for secondary-axis x-axis major tick marks. Defaults to
          #ffffff. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x2_width (float): Major secondary x-axis tickline width in pixels (float ok). Defaults to 1.3.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x_alpha (str): Transparency-axis value for major x-axis tickslines between 0-1. Defaults to 1.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x_color (str): Hex-axis color string for x-axis major tick marks. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_x_width (float): Major x-axis tickline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y (boolean): Enable/disable major y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y2 (boolean): Enable/disable secondary-axis major y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y2_alpha (str): Transparency-axis value for secondary-axis major y-axis tick marks between 0-1.
          Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y2_color (str): Hex-axis color string for secondary-axis y-axis major tick marks. Defaults to
          #ffffff. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y2_width (float): Major secondary y-axis tickline width in pixels (float ok). Defaults to 1.3.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y_alpha (str): Transparency-axis value for major y-axis tick marks between 0-1. Defaults to 1.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y_color (str): Hex-axis color string for y-axis major tick marks. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_major_y_width (float): Major y-axis tickline width in pixels (float ok). Defaults to 1.3. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor (boolean): Enable/disable minor x-axis and y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_alpha (str): Transparency-axis value for minor tick marks between 0-1. Defaults to 1. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_color (str): Hex-axis color string for x-axis and y-axis minor tick marks. Defaults to #ffffff.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_direction (str): Point tick marks 'in' or 'out' from the axes area. Defaults to 'in'. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_minor_length (float): Specify the length of the minor tick marks in pixels. Defaults to 0.67 *
          ticks_major_length. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_minor_number (float): Specify the number of minor tick marks. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-increment
        ticks_minor_width (float): Specify the width of the minor tick marks in pixels. Defaults to 0.6 *
          ticks_major_width. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-style
        ticks_minor_width (float): minor tickline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x (boolean): Enable/disable minor x-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x2 (boolean): Enable/disable secondary-axis minor x-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x2_alpha (str): Transparency-axis value for secondary-axis minor x-axis tick marks between 0-1.
          Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x2_color (str): Hex-axis color string for secondary-axis x-axis minor tick marks. Defaults to
          #ffffff. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x2_width (float): minor secondary x-axis tickline width in pixels (float ok). Defaults to 0.5.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x_alpha (str): Transparency-axis value for minor x-axis tick marks between 0-1. Defaults to 1.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x_color (str): Hex-axis color string for x-axis minor tick marks. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_x_width (float): minor x-axis tickline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y (boolean): Enable/disable minor y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y2 (boolean): Enable/disable secondary-axis minor y-axis tick marks. Defaults to True. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y2_alpha (str): Transparency-axis value for secondary-axis minor y-axis tick marks between 0-1.
          Defaults to 1. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y2_color (str): Hex-axis color string for secondary-axis y-axis minor tick marks. Defaults to
          #ffffff. Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y2_width (float): minor secondary y-axis tickline width in pixels (float ok). Defaults to 0.5.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y_alpha (str): Transparency-axis value for minor y-axis tick marks between 0-1. Defaults to 1.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y_color (str): Hex-axis color string for y-axis minor tick marks. Defaults to #ffffff. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks
        ticks_minor_y_width (float): minor y-axis tickline width in pixels (float ok). Defaults to 0.5. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/ticks.html#Tick-marks

    Examples
    --------
    Default styling:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2')

            .. figure:: ../_static/images/example_ticks1.png

    Ugly styling:

        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     ticks_major_increment=0.1,
                     ticks_major_x_color='#000000', ticks_major_x_length=15, ticks_major_x_width=4,
                     ticks_major_y_color='#0000FF', ticks_major_y_direction='out',
                     ticks_minor=True, ticks_minor_number=3, ticks_minor_color='#00FF00')

            .. figure:: ../_static/images/example_ticks2.png
    """


def titles():
    """Dummy function to return the figure title API with `help()` (not used directly for plotting).

    Keyword Args:
        title (str): Figure title text. Defaults to None. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        title_edge_alpha (float): Transparency value for the title area edge between 0-1. Defaults to 1.
        title_edge_color (str): Hex color string for the title area edge. Defaults to #ffffff.
        title_edge_width (float): Width of the border edge of a title area in pixels. Defaults to 1.
        title_fill_alpha (float): Transparency value for the title area background fill between 0-1. Defaults to 1.
        title_fill_color (str): Hex color string for the title area edge. Defaults to #ffffff.
        title_font (str): Font for the figure title. Defaults to sans-serif. Example:
          https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        title_font_color (str): Hex color string for font color of the figure title. Defaults to #000000.
        title_font_size (str): Font size for the figure title. Defaults to 18.
        title_font_style (str): Font style {'normal'|'italic'|'oblique'} for the figure title. Defaults to italic.
          Example: https://endangeredoxen.github.io/fivecentplots/0.5.3/styles.html#Fonts
        title_font_weight (str): Font weight {a numeric value in range 0-1000|'ultralight'|'light'|'normal'|'regular'|'b
          ook'|'medium'|'roman'|'semibold'|'demibold'|'demi'|'bold'|'heavy'|'extra bold'|'black'} for the figure title.
          Defaults to bold.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2',
                     title='Vaxis III', title_edge_alpha=0.5, title_edge_color='#333333',
                     title_edge_width=2, title_fill_alpha=0.2, title_fill_color='#00AA88',
                     title_font='monospace', title_font_size=24, title_font_style='oblique',
                     title_font_weight=700)

            .. figure:: ../_static/images/example_title.png
    """


def ws():
    """Dummy function to return the white space API with `help()` (not used directly for plotting).

    Keyword Args:
        ws_ax_box_title (int): White space between axes edge and box plot titles. Defaults to 10.
        ws_ax_cbar (int): White space from right axes edge to left side of cbar. Defaults to 10.
        ws_ax_fig (int): White space right edge of axes to right edge of figure [if no legend present]. Defaults to 10.
        ws_ax_label_xs (int): Extra white space between axes and label when using separate labels. Defaults to 5.
        ws_ax_leg (int): White space from right edge of axes to left edge of legend [if present]. Defaults to 5.
        ws_col (int): White space between column subplots [ignored if tick labels or axes labels are present and wider
          than this]. Defaults to 30.
        ws_fig_ax (int): White space from left figure edge to axes left edge. Defaults to 10.
        ws_fig_label (int): White space between top of figure and x2 label [if present]. Defaults to 10.
        ws_fig_title (int): White space from top of figure to top of title [if present]. Defaults to 10.
        ws_label_col (int): White space from axes and col labels. Defaults to ws_label_rc.
        ws_label_fig (int): White space from bottom of x label to bottom of figure. Defaults to ws_fig_label.
        ws_label_rc (int): White space between axes and row & col labels. Defaults to 10.
        ws_label_row (int): White space from axes to row labels. Defaults to ws_label_rc.
        ws_label_tick (int): White space from edge of axes label to edge of tick labels. Defaults to 10.
        ws_leg_fig (int): White space from right of legend in position 0 and figure right edge. Defaults to 10.
        ws_row (int): White space between row subplots [ignored if tick labels or axes labels are present and wider than
          this]. Defaults to 30.
        ws_tick_minimum (int): Minimum width for tick labels. Defaults to 10.
        ws_ticks_ax (int): White space from tick labels to edge of axes. Defaults to 5.
        ws_title_ax (int): White space bottom of title to top of axes. Defaults to 10.

    Examples
    --------
        >>> import fivecentplots as fcp
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> df = pd.read_csv(Path(fcp.__file__).parent / 'test_data' / 'fake_data.csv')
        >>> fcp.plot(df, x='Voltage', y='I [A]', row='Die', col='Substrate', ax_size=[400, 300],
                     filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2', save=True,
                     fig_edge_color='#000000',
                     ws_col=100, ws_row=0, ws_ax_fig=250, ws_label_rc=30, ws_ticks_ax=25, ws_label_tick=25)

            .. figure:: ../_static/images/example_ws.png
    """
