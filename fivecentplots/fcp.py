############################################################################
# fcp.py
#   Custom library of plot functions based on matplotlib to generate more
#   attractive plots more easily.  Part of the fivecentplots project.
############################################################################

# maybe subclass fig and ax in a new class that contains all the internal
# functions needed for an mpl plot.  then another for bokeh

import os

__author__ = 'Steve Nicholes'
__copyright__ = 'Copyright (C) 2016 Steve Nicholes'
__license__ = 'GPLv3'
with open(os.path.join(os.path.dirname(__file__), r'version.txt'), 'r') as fid:
    __version__ = fid.readlines()[0].replace('\n', '')
__url__ = 'https://github.com/endangeredoxen/fivecentplots'

import numpy as np
import pandas as pd
import pdb
import copy
import shutil
import sys
from . import data
from . colors import *
from . import engines
from . import keywords
from . utilities import *
try:
    # optional import - only used for paste_kwargs to use windows clipboard
    # to directly copy kwargs from ini file
    import fivecentfileio as fileio
    import win32clipboard  # noqa
except (ModuleNotFoundError, ImportError, NameError):
    pass
db = pdb.set_trace

osjoin = os.path.join
cur_dir = os.path.dirname(__file__)
user_dir = os.path.expanduser('~')
if not os.path.exists(osjoin(user_dir, '.fivecentplots')):
    os.makedirs(osjoin(user_dir, '.fivecentplots'))
if not os.path.exists(osjoin(user_dir, '.fivecentplots', 'defaults.py')):
    shutil.copy2(osjoin(cur_dir, 'themes', 'gray.py'),
                 osjoin(user_dir, '.fivecentplots', 'defaults.py'))
sys.path = [osjoin(user_dir, '.fivecentplots')] + sys.path

from defaults import *  # noqa | use local file

# Load the keywords for docstrings from an csv files
kw = keywords.make_docstrings()

# install requirements for other packages beyond what is in setup.py
global INSTALL
INSTALL = {}
INSTALL['bokeh'] = ['bokeh']

# Global kwargs to override anything
global KWARGS
KWARGS = {}


def bar(df, **kwargs):
    """Bar chart.

    Args:
        df (pandas.DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (str): x-axis column name
        y (str): y-axis column name

    Optional Keyword Args:
    """
    return plotter(data.Bar, **dfkwarg(df, kwargs))


def boxplot(df, **kwargs):
    """Box plot modeled after the "Variability Chart" in JMP which provides convenient,
    multi-level group labels automatically along the x-axis.

    Args:
        df (pandas.DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        y (str): y-axis column name contining the box plot data

    Optional Keyword Args:
    """
    return plotter(data.Box, **dfkwarg(df, kwargs))


def contour(df, **kwargs):
    """Contour plot module.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (str): x-axis column name
        y (str): y-axis column name
        z (str): z-axis column name

    Optional Keyword Args:
    """
    return plotter(data.Contour, **dfkwarg(df, kwargs))


def deprecated(kwargs):
    """Automatically fix deprecated keyword args."""

    # leg_groups
    if kwargs.get('leg_groups'):
        kwargs['legend'] = kwargs['leg_groups']
        print('"leg_groups" is deprecated. Please use "legend" instead')

    # labels
    labels = ['x', 'x2', 'y', 'y2', 'z']
    for ilab, lab in enumerate(labels):
        # Deprecated style
        keys = [f for f in kwargs.keys() if '%slabel' % lab in f]
        if len(keys) > 0:
            print('"%slabel" is deprecated. Please use "label_%s" instead'
                  % (lab, lab))
            for k in keys:
                kwargs[k.replace('%slabel' % lab, 'label_%s' %
                                 lab)] = kwargs[k]
                kwargs.pop(k)

    # twin + share
    vals = ['sharex', 'sharey', 'twinx', 'twiny']
    for val in vals:
        if val in kwargs:
            print('"%s" is deprecated.  Please use "%s_%s" instead' %
                  (val, val[0:-1], val[-1]))
            kwargs['%s_%s' % (val[0:-1], val[-1])] = kwargs[val]

    return kwargs


def gantt(df, **kwargs):
    """Gantt chart plotting function.  This plot is built off of a horizontal
       implementation of `fcp.bar`.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (list): two x-axis column names containing Datetime values
            1) the start time for each item in the Gantt chart
            2) the stop time for each item in the Gantt chart
        y (str): y-axis column name
        z (str): z-axis column name

    Optional Keyword Args:
    """

    return plotter(data.Gantt, **dfkwarg(df, kwargs))


def heatmap(df, **kwargs):
    """Heatmap plot.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (str): x-axis column name
        y (str): y-axis column name
        z (str): z-axis column name

    Optional Keyword Args:
    """

    return plotter(data.Heatmap, **dfkwarg(df, kwargs))


def help():
    import webbrowser
    webbrowser.open(
        r'https://endangeredoxen.github.io/fivecentplots/index.html')


def hist(df, **kwargs):
    """Histogram plot.

    Args:
        df (DataFrame | numpy array): DataFrame or numpy array containing data to plot
            [when passing a numpy array it is automatically converted to a DataFrame]

    Required Keyword Args:
        x (str): x-axis column name [the "value" column from which "counts" are calculated]

    Optional Keyword Args:
    """

    return plotter(data.Histogram, **dfkwarg(df, kwargs))


def imshow(df, **kwargs):
    """Image show plotting function.

    Args:
        df (DataFrame | numpy array): DataFrame or numpy array containing 2D row/column
            image data to plot [when passing a numpy array it is automatically converted
            to a DataFrame]

    Required Keyword Args:
        None

    Optional Keyword Args:
    """

    kwargs['tick_labels'] = kwargs.get('tick_labels', True)

    return plotter(data.ImShow, **dfkwarg(df, kwargs))


def nq(df, **kwargs):
    """Plot the normal quantiles of a data set.

    Args:
        df (DataFrame | numpy array): DataFrame containing a column of
            values data or a DataFrame or numpy array containing a set of 2D values
            that can be used to calculate quantiles at various "sigma" intervals

    Required Keyword Args:
        None

    Optional Keyword Args:
    """

    kwargs['tick_labels'] = kwargs.get('tick_labels', True)

    return plotter(data.NQ, **dfkwarg(df, kwargs))


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


def plot(df, **kwargs):
    """XY plot.

    Args:
        df (DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (str): x-axis column name
        y (str): y-axis column name

    Optional Keyword Args:
    """

    return plotter(data.XY, **dfkwarg(df, kwargs))


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
                df_rc.loc[df_rc.index.isin(gg.index), 'Total'] = len(
                    g.groupby(data.legend))
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
            if type(gg) == pd.Series:
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
            if type(row) is not tuple:
                row = [row]
            else:
                row = [str(f) for f in row]
            labels += ['']

            if len(dd.changes.columns) > 1 and \
                    dd.changes[col[0]].iloc[irow] == 1 \
                    and len(kwargs['groups']) > 1:
                dividers += [irow + 0.5]

            # Plot points
            if type(dd.legend_vals) is pd.DataFrame:
                for jj, jrow in dd.legend_vals.iterrows():
                    temp = gg.loc[gg[dd.legend] == jrow['names']][dd.y].dropna()
                    temp['x'] = irow + 1
                    if len(temp) > 0:
                        layout.plot_xy(ir, ic, jj, temp, 'x', dd.y[0],
                                       jrow['names'], False, zorder=10)
            else:
                if len(temp) > 0:
                    layout.plot_xy(ir, ic, irow, temp, 'x', dd.y[0], None, False,
                                   zorder=10)

    else:
        data = [df_rc[dd.y].dropna()]
        labels = ['']
        data[0]['x'] = 1
        if type(dd.legend_vals) is pd.DataFrame:
            for jj, jrow in dd.legend_vals.iterrows():
                temp = data[0].loc[df_rc[dd.legend] == jrow['names']].index

                layout.plot_xy(ir, ic, jj, data[0].loc[temp], 'x', dd.y[0],
                               jrow['names'], False, zorder=10)
        else:
            layout.plot_xy(ir, ic, 0, data[0], 'x', dd.y[0], None, False,
                           zorder=10)

    # Remove lowest divider
    dividers = [f for f in dividers if f > 0.5]

    # Remove temporary 'x' column
    for dat in data:
        del dat['x']

    # Range lines
    if layout.box_range_lines.on:
        for id, dat in enumerate(data):
            kwargs = layout.box_range_lines.kwargs.copy()
            layout.plot_line(ir, ic, id + 1 - 0.2, dat.max().iloc[0],
                             x1=id + 1 + 0.2, y1=dat.max().iloc[0], **kwargs)
            layout.plot_line(ir, ic, id + 1 - 0.2, dat.min().iloc[0],
                             x1=id + 1 + 0.2, y1=dat.min().iloc[0], **kwargs)
            kwargs['style'] = kwargs['style2']
            layout.plot_line(ir, ic, id + 1, dat.min().iloc[0],
                             x1=id + 1, y1=dat.max().iloc[0], **kwargs)

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
        layout.plot_line(ir, ic, x, stats, **layout.box_stat_line.kwargs,)

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
            layout.plot_line(ir, ic, [x, x2], y, **layout.box_group_means.kwargs,)
            x = x2

    # add grand mean
    if layout.box_grand_mean.on is True:
        x = np.linspace(0.5, dd.ngroups + 0.5, dd.ngroups)
        mm = df_rc[dd.y[0]].mean()
        y = [mm for f in x]
        layout.plot_line(ir, ic, x, y, **layout.box_grand_mean.kwargs,)

    # add grand mean
    if layout.box_grand_median.on:
        x = np.linspace(0.5, dd.ngroups + 0.5, dd.ngroups)
        mm = df_rc[dd.y[0]].median()
        y = [mm for f in x]
        layout.plot_line(ir, ic, x, y, **layout.box_grand_median.kwargs,)

    # add mean confidence diamonds
    if layout.box_mean_diamonds.on:
        mgroups = df_rc.groupby(dd.groups)
        for ii, (nn, mm) in enumerate(mgroups):
            low, high = ci(mm[dd.y[0]], layout.box_mean_diamonds.conf_coeff)
            mm = mm[dd.y[0]].mean()
            x1 = -layout.box_mean_diamonds.width[0] / 2
            x2 = layout.box_mean_diamonds.width[0] / 2
            points = [[ii + 1 + x1, mm],
                      [ii + 1, high],
                      [ii + 1 + x2, mm],
                      [ii + 1, low],
                      [ii + 1 + x1, mm],
                      [ii + 1 + x2, mm]]
            layout.plot_polygon(ir, ic, points,
                                **layout.box_mean_diamonds.kwargs)

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


def plot_control_limit(ir: int, ic: int, iline: int, layout: 'Layout', data: 'Data'):
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
        cols += [f for f in validate_list(data.legend)
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
        if data.stat is not None:
            layout.lines.on = False
        if kwargs.get('groups', False):
            for nn, gg in df.groupby(validate_list(kwargs['groups'])):
                hist, data = layout.plot_hist(ir, ic, iline, gg, x, y, leg_name, data)

        else:
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
    if layout.legend.on:
        if layout.interval.type == 'nq':
            leg_name = f'nq = [{layout.interval.value[0]}, {layout.interval.value[1]}]'
        elif layout.interval.type == 'percentile':
            leg_name = f'q = [{layout.interval.value[0]}, {layout.interval.value[1]}]'
        else:
            leg_name = f'ci = {layout.interval.value[0]}'

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
        x = df[data.x[0]].drop_duplicates()
    y = df.groupby(data.x[0]).sum().loc[x][data.y[0]].values

    if any(y < 0):
        print('Pie plot had negative values.  Skipping...')
        return data

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
    layout.plot_xy(ir, ic, iline, df_stat, x, y,
                   leg_name, twin, marker_disable=True)

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
            for nn, gg in df.groupby(validate_list(kwargs['groups']), sort=data.sort):
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
        UPDATE

    Returns:
        plots
    """

    # Check for deprecated kwargs
    kwargs = deprecated(kwargs)

    # Apply globals if they don't exist
    for k, v in KWARGS.items():
        if k not in kwargs.keys():
            kwargs[k] = v

    # Set the plotting engine
    theme = kwargs.get('theme', None)
    engine = kwget(kwargs, reload_defaults(theme)[0], 'engine', 'mpl')
    if not hasattr(engines, engine):
        if engine in INSTALL.keys():
            installs = '\npip install '.join(INSTALL[engine])
            print('Plotting engine "%s" is not installed! Please run the '
                  'following:\npip install %s' % (engine, installs))
        else:
            print('Plotting engine "%s" is not supported' % (engine))
        return
    else:
        engine = getattr(engines, engine)

    # Timer
    kwargs['timer'] = Timer(print=kwargs.get('timer', False), start=True,
                            units='ms')

    # Build the data object and update kwargs
    dd = dobj(**kwargs)
    for k, v in kwargs.items():
        if k in dd.__dict__.keys():
            kwargs[k] = getattr(dd, k)
    kwargs['timer'].get('Data obj')

    # Iterate over discrete figures
    for ifig, fig_item, fig_cols, dd in dd.get_df_figure():
        kwargs['timer'].get('dd.get_df_figure')
        # Create a layout object
        layout = engine.Layout(dd, **kwargs)
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
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_colors' % (ifig, ir, ic))

            # Add and format gridlines
            layout.set_axes_grid_lines(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_grid_lines' % (ifig, ir, ic))

            # Add horizontal and vertical lines
            layout.add_hvlines(ir, ic, df_rc)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | add_hvlines' % (ifig, ir, ic))

            # Plot the data
            dd = globals()['plot_{}'.format(dd.name)](
                dd, layout, ir, ic, df_rc, kwargs)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | plot' % (ifig, ir, ic))

            # Set linear or log axes scaling
            layout.set_axes_scale(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_scale' % (ifig, ir, ic))

            # Set axis ranges
            layout.set_axes_ranges(ir, ic, dd.ranges)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_ranges' % (ifig, ir, ic))

            # Add axis labels
            layout.set_axes_labels(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_labels' % (ifig, ir, ic))

            # Add rc labels
            layout.set_axes_rc_labels(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_rc_labels' % (ifig, ir, ic))

            # Adjust tick marks
            layout.set_axes_ticks(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | set_axes_ticks' % (ifig, ir, ic))

            # Add box labels
            if dd.name == 'box':
                layout.add_box_labels(ir, ic, dd)
                kwargs['timer'].get(
                    'ifig=%s | ir=%s | ic=%s | add_box_labels' % (ifig, ir, ic))

            # Add arbitrary text
            layout.add_text(ir, ic)
            kwargs['timer'].get(
                'ifig=%s | ir=%s | ic=%s | add_text' % (ifig, ir, ic))

        # Make the legend
        layout.add_legend()
        kwargs['timer'].get('ifig=%s | add_legend' % (ifig))

        # Add a figure title
        layout.set_figure_title()
        kwargs['timer'].get('ifig=%s | set_figure_title' % (ifig))

        # Final adjustments
        layout.set_figure_final_layout(dd, **kwargs)
        kwargs['timer'].get('ifig=%s | set_figure_final_layout' % (ifig))

        # Build the save filename
        filename = set_save_filename(dd.df_fig, ifig, fig_item, fig_cols,
                                     layout, kwargs)
        if 'filepath' in kwargs.keys():
            filename = os.path.join(kwargs['filepath'], filename)

        # Optionally save and open
        if kwargs.get('save', False) or kwargs.get('show', False):
            if ifig:
                idx = ifig
            else:
                idx = 0
            layout.save(filename, idx)

            if kwargs.get('show', False):
                show_file(filename)
        kwargs['timer'].get('ifig=%s | save' % (ifig))

        # Return inline
        if kwargs.get('return_filename'):
            layout.close()
            if 'filepath' in kwargs.keys():
                return osjoin(kwargs['filepath'], filename)
            else:
                return osjoin(os.getcwd(), filename)
        elif not kwargs.get('inline', True):
            layout.close()
        else:
            if kwargs.get('print_filename', False):
                print(filename)
            out = layout.show(filename)
            if out is not None:
                return out
        kwargs['timer'].get('ifig=%s | return inline' % (ifig))

    # Save data used in the figures
    if kwargs.get('save_data', False):
        if type(kwargs['save_data']) is str:
            filename = kwargs['save_data']
        else:
            filename = filename.split('.')[0] + '.csv'
        dd.df_all[dd.cols_all].to_csv(filename, index=False)
        kwargs['timer'].get('save_data' % (ifig))

    # Restore plotting engine settings
    layout.restore()

    kwargs['timer'].get_total()


def pie(df, **kwargs):
    """Pie chart

    Args:
        df (DataFrame): DataFrame containing data to plot

    Required Keyword Args:
        x (str): x-axis column name with categorical data
        y (str): y-axis column name with values

    Optional Keyword Args:
    """

    return plotter(data.Pie, **dfkwarg(df, kwargs))


def set_theme(theme=None):
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

    themes = [f.replace('.py', '')
              for f in os.listdir(osjoin(cur_dir, 'themes')) if '.py' in f]
    mythemes = [f.replace('.py', '') for f in os.listdir(my_theme_dir)
                if '.py' in f and f not in ignores]

    if theme in themes:
        entry = themes.index('%s' % theme) + 1

    elif theme in mythemes:
        entry = mythemes.index('%s' % theme) + 1 + len(themes)

    elif theme is not None:
        print('Theme file not found!  Please try again')
        return

    else:
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
            print('Copying %s >> %s' %
                  (themes[int(entry) - 1],
                   osjoin(user_dir, '.fivecentplots', 'defaults.py')))
        else:
            print('Copying %s >> %s' %
                  (mythemes[int(entry) - 1 - len(themes)],
                   osjoin(user_dir, '.fivecentplots', 'defaults.py')))

    if os.path.exists(osjoin(user_dir, '.fivecentplots', 'defaults.py')):
        print('Previous theme file found! Renaming to "defaults_old.py" and '
              'copying new theme...', end='')
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
    """Axes."""

    pass


def cbar():
    """
    Color bar
    """

    pass


def figure():
    """
    Figure
    """

    pass


def fit():
    """
    Fit
    """

    pass


def gridlines():
    """
    Gridlines
    """

    pass


def labels():
    """
    Labels
    """

    pass


def legend():
    """
    Legend
    """

    pass


def lines():
    """
    Lines
    """

    pass


def markers():
    """
    Markers
    """

    pass


def ref_line():
    """
    Reference line
    """

    pass


def ticks():
    """
    Ticks
    """

    pass


def tick_labels():
    """
    Tick labels
    """

    pass


def ws():
    """"""
    pass


# Update the docstrings for the primary plot types with keywords
# found in an csv files
bar.__doc__ += keywords.kw_print(kw['bar'])


boxplot.__doc__ += \
    keywords.kw_header('Basic:', indent='   ') + \
    keywords.kw_print(kw['box']) + \
    keywords.kw_header('Grouping text:') + \
    keywords.kw_print(kw['box_label']) + \
    keywords.kw_header('Stat Lines:') + \
    keywords.kw_print(kw['box_stat']) + \
    keywords.kw_header('Diamonds:') + \
    keywords.kw_print(kw['box_diamond']) + \
    keywords.kw_header('Violins:') + \
    keywords.kw_print(kw['box_violin'])


contour.__doc__ += \
    keywords.kw_header('Basic:', indent='   ') + \
    keywords.kw_print(kw['contour']) + \
    keywords.kw_header('Color bar:', indent='   ') + \
    keywords.kw_print(kw['cbar'])


gantt.__doc__ += keywords.kw_print(kw['gantt'])


heatmap.__doc__ += \
    keywords.kw_header('Basic:', indent='   ') + \
    keywords.kw_print(kw['heatmap']) + \
    keywords.kw_header('Color bar:', indent='   ') + \
    keywords.kw_print(kw['cbar'])


hist.__doc__ += keywords.kw_print(kw['hist'])


imshow.__doc__ += keywords.kw_print(kw['imshow'])


pie.__doc__ += keywords.kw_print(kw['pie'])


nq.__doc__ += \
    keywords.kw_header('Basic:', indent='   ') + \
    keywords.kw_print(kw['nq']) + \
    keywords.kw_header('Calculation:', indent='   ') + \
    keywords.kw_print(kw['nq_calc'])
# add line stuff from plot here


plot.__doc__ += ''


# axes.__doc__ = \
#     keywords.kw_print(kw['Axes'])


# cbar.__doc__ = \
#     keywords.kw_print(kw['Cbar'])


# figure.__doc__ = \
#     keywords.kw_print(kw['Figure'])


# fit.__doc__ = \
#     keywords.kw_print(kw['Fit'])


# gridlines.__doc__ = ''


# labels.__doc__ = \
#     keywords.kw_print(kw['Label'])


# legend.__doc__ = \
#     keywords.kw_print(kw['Legend'])


# lines.__doc__ = \
#     keywords.kw_print(kw['Lines'])


# markers.__doc__ = \
#     keywords.kw_print(kw['Markers'])


# ref_line.__doc__ = \
#     keywords.kw_print(kw['Ref Line'])


# ticks.__doc__ = \
#     keywords.kw_print(kw['Ticks'])


# tick_labels.__doc__ = ''


# ws.__doc__ = \
#     keywords.kw_print(kw['WS'])
