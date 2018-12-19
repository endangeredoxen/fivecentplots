############################################################################
# fcp.py
#   Custom library of plot functions based on matplotlib to generate more
#   attractive plots more easily.  Part of the fivecentplots project.
############################################################################

# maybe subclass fig and ax in a new class that contains all the internal
# functions needed for an mpl plot.  then another for bokeh

__author__    = 'Steve Nicholes'
__copyright__ = 'Copyright (C) 2016 Steve Nicholes'
__license__   = 'GPLv3'
__version__   = '0.3.0'
__url__       = 'https://github.com/endangeredoxen/fivecentplots'
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import numpy as np
import scipy.stats as ss
import pandas as pd
import pdb
import re
import copy
import importlib
import itertools
import shutil
import datetime
import sys
import textwrap
from . data import Data
from . colors import *
from . layout import LayoutMPL, LayoutBokeh, RepeatedList
from . keywords import *
from . utilities import dfkwarg, set_save_filename, validate_list
import warnings
try:
    import fileio
except:
    pass
try:
    import win32clipboard
except:
    print('Could not import win32clipboard.  Make sure pywin32 is installed '
          'to use the paste from clipboard option.')
try:
    from natsort import natsorted
except:
    natsorted = sorted
st = pdb.set_trace

osjoin = os.path.join
cur_dir = os.path.dirname(__file__)
user_dir = os.path.expanduser('~')
if not os.path.exists(osjoin(user_dir, '.fivecentplots')):
    os.makedirs(osjoin(user_dir, '.fivecentplots'))
if not os.path.exists(osjoin(user_dir, '.fivecentplots', 'defaults.py')):
    shutil.copy2(osjoin(cur_dir, 'themes', 'gray.py'),
                 osjoin(user_dir, '.fivecentplots', 'defaults.py'))
sys.path = [osjoin(user_dir, '.fivecentplots')] + sys.path

from defaults import *  # use local file

LAYOUT = {'mpl': LayoutMPL,
          'bokeh': LayoutBokeh}


def bar(**kwargs):
    """ Main bar chart plotting function
    At minimum, it requires a pandas DataFrame with at
    least one column for the y axis.  Plots can be customized and enhanced by
    passing keyword arguments.  Default values that must be defined in order to
    generate the plot are pulled from the fcp_params default dictionary
    Args:
        df (DataFrame): DataFrame containing data to plot
        x (str):        name of x column in df
        y (str|list):   name or list of names of y column(s) in df
    Keyword Args:
        see online docs
    Returns:
        plots
    """

    return plotter('plot_bar', **kwargs)


def boxplot(*args, **kwargs):
    """ Main boxplot plotting function
    At minimum, it requires a pandas DataFrame with at
    least one column for the y axis.  Plots can be customized and enhanced by
    passing keyword arguments.  Default values that must be defined in order to
    generate the plot are pulled from the fcp_params default dictionary

    Args:
        df (DataFrame): DataFrame containing data to plot
        y (str|list):   column name in df to use for the box(es)

    Keywords:
    """

    return plotter('plot_box', **dfkwarg(args, kwargs))


def contour(*args, **kwargs):
    """ Main contour plotting function
    At minimum, it requires a pandas DataFrame with at
    least three columns and three column names for the x, y, and z axis.
    Plots can be customized and enhanced by passing keyword arguments as
    defined below. Default values that must be defined in order to
    generate the plot are pulled from the fcp_params default dictionary
    Args:
        df (DataFrame): DataFrame containing data to plot
        x (str|list):   name of list of names of x column in df
        y (str|list):   name or list of names of y column(s) in df
        z (str):   name of z column(s) in df
    Keyword Args:
        see online docs
    Returns:
        plots
    """

    return plotter('plot_contour', **dfkwarg(args, kwargs))


def deprecated(kwargs):
    """
    Fix deprecated keyword args
    """

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
                kwargs[k.replace('%slabel' % lab, 'label_%s' % lab)] = kwargs[k]
                kwargs.pop(k)

    # twin + share
    vals = ['sharex', 'sharey', 'twinx', 'twiny']
    for val in vals:
        if val in kwargs:
            print('"%s" is deprecated.  Please use "%s_%s" instead' % \
                  (val, val[0:-1], val[-1]))
            kwargs['%s_%s' % (val[0:-1], val[-1])] = kwargs[val]

    return kwargs


def heatmap(*args, **kwargs):
    """ Main heatmap plotting function
    At minimum, it requires a pandas DataFrame with at
    least three columns and three column names for the x, y, and z axis.
    Plots can be customized and enhanced by passing keyword arguments as
    defined below. Default values that must be defined in order to
    generate the plot are pulled from the fcp_params default dictionary
    Args:
        df (DataFrame): DataFrame containing data to plot
        x (str|list):   name of list of names of x column in df
        y (str|list):   name or list of names of y column(s) in df
        z (str):   name of z column(s) in df
    Keyword Args:
        see online docs
    Returns:
        plots
    """

    return plotter('plot_heatmap', **dfkwarg(args, kwargs))


def help():
    import webbrowser
    webbrowser.open(r'https://endangeredoxen.github.io/fivecentplots/index.html')


def hist(*args, **kwargs):

    return plotter('plot_hist', **dfkwarg(args, kwargs))


def nq(*args, **kwargs):
    """
    Plot normal quantiles of a data set
    """

    return plotter('plot_nq', **dfkwarg(args, kwargs))


def paste_kwargs(kwargs):
    """
    Get the kwargs from contents of the clipboard in ini file format

    Args:
        kwargs (dict): originally inputted kwargs

    Returns:
        kwargs
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

    except:
        print('This feature requires the fileio package '
              '(download @ https://github.com/endangeredoxen/fileio)')


def plot(*args, **kwargs):

    return plotter('plot_xy', **dfkwarg(args, kwargs))


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

    pass


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
    medians = []

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
                    stats += [temp.quantile(float(ss.strip('q'))/100).iloc[0]]
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
                    temp = gg.loc[gg[dd.legend]==jrow['names']][dd.y].dropna()
                    temp['x'] = irow + 1
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
                temp = data[0].loc[df_rc[dd.legend]==jrow['names']].index

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
            layout.plot_line(ir, ic, id+1-0.2, dat.max().iloc[0],
                            x1=id+1+0.2, y1=dat.max().iloc[0], **kwargs)
            layout.plot_line(ir, ic, id+1-0.2, dat.min().iloc[0],
                            x1=id+1+0.2, y1=dat.min().iloc[0], **kwargs)
            kwargs['style'] = kwargs['style2']
            layout.plot_line(ir, ic, id+1, dat.min().iloc[0],
                            x1=id+1, y1=dat.max().iloc[0], **kwargs)

    # Add boxes
    for ival, val in enumerate(data):
        data[ival] = val[dd.y[0]].values
    bp = layout.plot_box(ir, ic, data, **kwargs)

    # Add divider lines
    if layout.box_divider.on and len(dividers) > 0:
        layout.ax_vlines = copy.deepcopy(layout.box_divider)
        layout.ax_vlines.values = dividers
        layout.ax_vlines.style = copy.copy(layout.box_divider.style)
        layout.ax_vlines.width = copy.copy(layout.box_divider.width)
        layout.add_hvlines(ir, ic)
        layout.ax_vlines.values = []

    # Add mean/median connecting lines
    if layout.box_stat_line.on and len(stats) > 0:
        x = np.linspace(1, dd.ngroups, dd.ngroups)
        layout.plot_line(ir, ic, x, stats, **layout.box_stat_line.kwargs,)

    return dd


def plot_conf_int(ir, ic, iline, data, layout, df, x, y, twin):
    """
    """

    if not layout.conf_int.on:
        return

    data.get_conf_int(df, x, y)

    layout.fill_between_lines(ir, ic, iline, data.stat_idx, data.lcl, data.ucl,
                              'conf_int', twin)

    return data


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
        layout.plot_contour(layout.axes.obj[ir, ic], df, x, y, z, data.ranges[ir, ic])

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
        x (str): x-column name
        y (str): y-column name
        twin (bool): denote twin axis
        leg_name (str): legend value
        ngroups (int): number of groups in this data set

    """

    if not layout.fit.on:
        return

    df, coeffs, rsq = data.get_fit_data(ir, ic, df, x, y)
    if layout.legend.on:
        if (data.wrap_vals is not None and ngroups / data.nwrap > 1 \
                or ngroups / (data.nrow * data.ncol) > 1 \
                or len(np.unique(layout.fit.color.values)) > 1) \
                and data.legend_vals is not None:
            leg_name = '%s [Fit]' % leg_name
        else:
            leg_name = 'Fit'
    else:
        leg_name = None
    layout.plot_xy(ir, ic, iline, df, '%s Fit' % x, '%s Fit' %y,
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
            eqn += '%s*x%s' % (round(coeff,3), power)
        if coeffs[-1] > 0:
            eqn += '+'
        eqn += '%s' % round(coeffs[-1], 3)

        layout.add_text(ir, ic, eqn, 'fit')

    if layout.fit.rsq:
        offsety = (2.2*layout.fit.font_size) / layout.axes.size[1]
        layout.add_text(ir, ic, 'R^2=%s' % round(rsq, 4), 'fit',
                        offsety=-offsety)

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
        layout.plot_heatmap(layout.axes.obj[ir, ic], df, x, y, z, data.ranges[ir, ic])

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


def plot_ref(ir, ic, iline, data, layout, df, x, y):
    """
    Plot a reference line
    """

    if not layout.ref_line.on:
        return

    layout.plot_xy(ir, ic, iline, df, x, layout.ref_line.column, 'ref_line',
                   False, line_type='ref_line', marker_disable=True)

    return data


def plot_stat(ir, ic, iline, data, layout, df, x, y, leg_name=None, twin=False):
    """
    Plot a line calculated by stats
    """

    df_stat = data.get_stat_data(df, x, y)

    if df_stat is None or len(df_stat) == 0:
        return

    layout.lines.on = True
    layout.plot_xy(ir, ic, iline, df_stat, x, y, leg_name, twin)

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
            for nn, gg in df.groupby(validate_list(kwargs['groups'])):
                layout.plot_xy(ir, ic, iline, gg, x, y, leg_name, twin)
                plot_fit(data, layout, ir, ic, iline, gg, x, y, twin, leg_name, ngroups)

        else:
            layout.plot_xy(ir, ic, iline, df, x, y, leg_name, twin)
            plot_fit(data, layout, ir, ic, iline, df, x, y, twin, leg_name, ngroups)

        plot_ref(ir, ic, iline, data, layout, df, x, y)
        if not layout.lines.on and not layout.markers.on:
            plot_stat(ir, ic, iline, data, layout, df, x, y, leg_name, twin)
        else:
            plot_stat(ir, ic, iline, data, layout, df, x, y, twin=twin)
        plot_conf_int(ir, ic, iline, data, layout, df, x, y, twin)

    return data


def plotter(plot_func, **kwargs):
    """ Main plotting function

    UPDATE At minimum, it requires a pandas DataFrame with at
    least two columns and two column names for the x and y axis.  Plots can be
    customized and enhanced by passing keyword arguments as defined below.
    Default values that must be defined in order to generate the plot are
    pulled from the fcp_params default dictionary

    Args:
        df (DataFrame): DataFrame containing data to plot
        x (str):        name of x column in df
        y (str|list):   name or list of names of y column(s) in df

    Keyword Args:
        UPDATE

    Returns:
        plots
    """

    # Check for deprecated kwargs
    kwargs = deprecated(kwargs)

    # Set the plotting engine
    engine = kwargs.get('engine', 'mpl').lower()

    # Build the data object and update kwargs
    dd = Data(plot_func, **kwargs)
    for k, v in kwargs.items():
        if k in dd.__dict__.keys():
            kwargs[k] = getattr(dd, k)

    # Iterate over discrete figures
    for ifig, fig_item, fig_cols, df_fig in dd.get_df_figure():
        # Create a layout object
        layout = LAYOUT[engine](plot_func, **kwargs)

        # Make the figure
        layout.make_figure(dd, **kwargs)

        # Turn off empty subplots
        for ir, ic, df_rc in dd.get_rc_subset(df_fig):

            if len(df_rc) == 0:
                if dd.wrap is None:
                    layout.set_axes_rc_labels(ir, ic)
                layout.axes.obj[ir, ic].axis('off')
                layout.axes.visible[ir, ic] = False
                if layout.axes2.obj[ir, ic] is not None:
                    layout.axes2.obj[ir, ic].axis('off')
                continue

        # Make the subplots
        for ir, ic, df_rc in dd.get_rc_subset(df_fig):
            if not layout.axes.visible[ir, ic]:
                continue

            # Set the axes colors
            layout.set_axes_colors(ir, ic)

            # Add and format gridlines
            layout.set_axes_grid_lines(ir, ic)

            # Add horizontal and vertical lines
            layout.add_hvlines(ir, ic)

            # Plot the data
            dd = globals()[plot_func](dd, layout, ir, ic, df_rc, kwargs)

            # Set linear or log axes scaling
            layout.set_axes_scale(ir, ic)

            # Set axis ranges
            layout.set_axes_ranges(ir, ic, dd.ranges)

            # Add axis labels
            layout.set_axes_labels(ir, ic)

            # Add rc labels
            layout.set_axes_rc_labels(ir, ic)

            # Adjust tick marks
            layout.set_axes_ticks(ir, ic)

            # Add box labels
            layout.add_box_labels(ir, ic, dd)

        # Make the legend
        layout.add_legend()

        # Add a figure title
        layout.set_figure_title()

        # Build the save filename
        filename = set_save_filename(df_fig, ifig, fig_item, fig_cols,
                                     layout, kwargs)
        if 'filepath' in kwargs.keys():
            filename = os.path.join(kwargs['filepath'], filename)

        # Save and optionally open
        if kwargs.get('save', True):
            if ifig:
                idx = ifig
            else:
                idx = 0
            layout.fig.obj.savefig(filename,
                                   edgecolor=layout.fig.edge_color.get(idx),
                                   facecolor=layout.fig.fill_color.get(idx),
                                   linewidth=layout.fig.edge_width)

            if kwargs.get('show', False):
                os.startfile(filename)

        # Return inline
        if kwargs.get('return_filename'):
            plt.close('all')
            if 'filepath' in kwargs.keys():
                return osjoin(kwargs['filepath'], filename)
            else:
                return osjoin(os.getcwd(), filename)
        elif not kwargs.get('inline', True):
            plt.close('all')
        else:
            if kwargs.get('print_filename', False):
                print(filename)
            plt.show()
            plt.close('all')


def save(fig, filename, kw):
    """
    Save the figure

    Args:
        fig (mpl.Figure): current figure
        filename (str): output filename
        kw (dict): kwargs dict

    """

    try:

        if kw['save_ext'] == 'html':
            import mpld3
            mpld3.save_html(fig, filename)

        else:
            fig.savefig(filename)

        if kw['show']:
            os.startfile(filename)

    except:
        if kwargs.get('show_filename', False):
            print(filename)
        raise NameError('%s is not a valid filename!' % filename)


def set_theme(theme=None):
    """
    Select a "defaults" file and copy to the user directory
    """

    if theme is not None:
        theme = theme.replace('.py', '')
    themes = [f.replace('.py', '')
              for f in os.listdir(osjoin(cur_dir, 'themes')) if '.py' in f]
    mythemes = [f.replace('.py', '')
                for f in os.listdir(osjoin(user_dir, '.fivecentplots'))
                if '.py' in f and 'defaults' not in f]

    if theme in themes:
        entry = themes.index('%s' % theme) + 1

    elif theme in mythemes:
        entry = mythemes.index('%s' % theme) + 1 + len(themes)

    else:
        print('Select default styling theme:')
        print('   Built-in theme list:')
        for i, th in enumerate(themes):
            print('      %s) %s' % (i+1, th))
        if len(themes) > 0:
            print('   User theme list:')
            for i, th in enumerate(mythemes):
                print('      %s) %s' % (i + 1 + len(themes), th))
        entry = input('Entry: ')

        try:
            int(entry)
        except:
            print('Invalid selection!  Please try again')
            return

        if int(entry) > len(themes) + len(mythemes) or int(entry) <= 0:
            print('Invalid selection!  Please try again')
            return

        if int(entry) <= len(themes):
            print('Copying %s >> %s' %
                (themes[int(entry)-1],
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
        shutil.copy2(osjoin(cur_dir, 'themes', themes[int(entry)-1] + '.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults.py'))
    else:
        shutil.copy2(osjoin(user_dir, '.fivecentplots', mythemes[int(entry)-1-len(themes)] + '.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults.py'))

    print('done!')


def kw_header(val):
    """
    Indent header names
    """

    return '   %s\n' % val


def kw_print(kw):
    """
    Print friendly version of kw dicts
    """

    indent = '          '
    kwstr = ''

    for irow, row in kw.iterrows():
        line = row['Keyword'] + ' (%s)' % row['Data Type'] + ': ' +\
               row['Description'] + '; default: %s' % row['Default'] + \
               '; ex: %s' % row['Example']

        kwstr += textwrap.fill(line, 80, initial_indent=indent,
                               subsequent_indent=indent + '  ')
        kwstr += '\n'

    return kwstr


boxplot.__doc__ = \
    boxplot.__doc__ + \
    kw_header('Markers:') + \
    kw_print(kw_markers)