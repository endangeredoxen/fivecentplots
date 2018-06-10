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
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import numpy as np
import scipy.stats as ss
import pandas as pd
import pdb
import re
import importlib
import itertools
import shutil
import datetime
import sys
from . data import Data
from . layout import LayoutMPL, LayoutBokeh
import utilities as utl
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


def boxplot(**kwargs):
    """ Main boxplot plotting function

    At minimum, it requires a pandas DataFrame with at
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

    return plotter('plot_box', **kwargs)




def boxplot_old(**kwargs):
    """ Main boxplot function
    This function wraps the boxplot function from the matplotlib
    library.  At minimum, it requires a pandas DataFrame with at least one
    column and a column name for the y axis.  Plots can be
    customized and enhanced by passing keyword arguments as defined below.
    Default values that must be defined in order to generate the plot are
    pulled from the fcp_params dictionary defined in defaults.py.
    Required Keyword Args:
        df (DataFrame): DataFrame containing data to plot
        y (str|list):   name or list of names of y column in df
    Keyword Args:
        see get_defaults for definitions

    Returns:
        layout (LayoutMPL obj):  contains all the spacing information used to
            construct the figure
    """

    def add_points(x, y, ax, color=palette[1], **kw):

        if kw['jitter']:
            x = np.random.normal(x+1, 0.04, size=len(y))
        else:
            x = np.array([x+1]*len(y))
        if len(x) > 0 and len(y) > 0:
            pts = ax.plot(x, y,
                          color=color,
                          markersize=kw['marker_size'],
                          marker=kw['marker_type'],
                          markeredgecolor=color,
                          markerfacecolor='none',
                          markeredgewidth=1.5,
                          linestyle='none',
                          zorder=2)
            return pts

    def index_changes(df, num_groups):
        """
        Make a DataFrame that shows when groups vals change; used for grouping labels

        Args:
            df (pd.DataFrame): grouping values
            num_groups (int): number of unique groups

        Returns:
            new DataFrame with 1's showing where group levels change for each row of df
        """

        changes = df.copy()
        # Set initial level to 1
        for col in df.columns:
            changes.loc[0, col] = 1
        # Determines values for all other rows
        for i in range(1, num_groups):
            for col in df.columns:
                if df[col].iloc[i-1] == df[col].iloc[i]:
                    changes.loc[i, col] = 0
                else:
                    changes.loc[i, col] = 1

        return changes


    plt.close('all')

    # Init plot
    df, x, y, z, kw = init('boxplot', kwargs)
    kw['ptype'] = 'boxplot'
    if type(df) is bool and not df:
        return

    if 'ax_fig_ws' not in kwargs.keys():
        kw['ax_fig_ws'] = 15  # no standard y-axis ticks/label

    # Default overrides
    if kw['marker_type'] is None:
        kw['marker_type'] = 'o'
    if 'sharex' not in kwargs.keys():
        kw['sharex'] = False

    # Backup ax size
    kw['ax_size_orig'] = [kw['ax_size'][0], kw['ax_size'][1]]

    # Get initial values
    ax_fig_ws0 = kw['ax_fig_ws']
    ws_col0 = kw['ws_col']
    ws_row0 = kw['ws_row']

    # Iterate over discrete figures
    for ifig, fig_item, fig_cols in enumerate(kw['fig_items']):

        # Reset ax size
        kw['ax_size'] = [kw['ax_size_orig'][0], kw['ax_size_orig'][1]]

        # Make a data subset and filter
        df_fig = get_df_figure(df, fig_item, kw)

        # Set up the row grouping
        kw = get_rc_groupings(df_fig, kw)

        # Special boxplot spacing for labels
        if kw['bp_labels_on'] and kw['groups'] is not None:
            # Get the changes df
            kw['groups'] = validate_columns(df_fig, kw['groups'])
            if kw['groups'] is None:
                groups = df_fig.copy()
            else:
                groups = df_fig.groupby(kw['groups'])

            # Order the group labels with natsorting
            gidx = []
            for i, (nn, g) in enumerate(groups):
                gidx += [nn]
            gidx = natsorted(gidx)

            # Make indices df
            indices = pd.DataFrame(gidx)
            num_groups = groups.ngroups
            changes = index_changes(indices, num_groups)

            ## LAYOUT
            # Determine if label should be aligned vertically or horizontally
            align = {}
            xs_height = 0
            for ii, cc in enumerate(indices.columns):
                align[ii] = 0
                vals = [str(f) for f in indices[cc].unique()]
                longest = max(vals, key=len)
                uniq_vals = len(changes[changes[cc]==1])
                label_width = kw['ax_size'][0]/uniq_vals
                val_width = get_font_to_px(longest, kw['bp_label_font_size'],
                                           kw['dpi'])[1]
                if val_width > label_width:
                    align[ii] = max(align[ii], val_width + 10)
                    xs_height += align[ii] - kw['bp_label_size']

            # Set padding and new sizes
            bp_labels = kw['bp_label_size']*len(kw['groups'])#+0.5)
            kw['ax_fig_ws'] =  ax_fig_ws0 + bp_labels + xs_height
            kw['ws_leg_ax'] = 0
            kw['ws_fig_leg'] = max([len(gr) for gr in kw['groups']]) * \
                                  kw['bp_label_font_size'] + \
                                  kw['ws_bp_name']
            if kw['wrap'] is None:
                kw['ws_col'] = ws_col0 + kw['ws_fig_leg']
            kw['ws_row'] = ws_row0 + bp_labels + xs_height

            ylabel_height = get_font_to_px(kw['ylabel'],
                                          kw['label_font_size'],
                                          kw['dpi'], kw['label_weight'],
                                          kw['label_style'],
                                          'vertical')[0]
            if ylabel_height > kw['ax_size'][1] + kw['ws_row'] + bp_labels:
                kw['ws_row'] = \
                    (kw['bp_label_font_size']*len(kw['ylabel']) - \
                    kw['ax_size'][1])/2 + kw['ws_row']/2
            if kw['rows'] is not None and kw['rows'][0] is not None:
                kw['ws_fig_leg'] -= kw['row_label_size'] + kw['ws_row_label']
            kw['group_labels'] = len(max(kw['groups']))*kw['bp_label_font_size']

        # Set up the legend grouping
        kw = get_legend_groupings(df_fig, y, kw)

        # Make the plot figure and axes
        layout, fig, axes, kw = make_fig_and_ax(kw)

        for ir in range(0, kw['nrow']):
            for ic in range(0, kw['ncol']):

                # Handle missing wrap plots
                if kw['wrap'] is not None:
                    if ir*kw['ncol'] + ic > len(kw['wrap'])-1:
                        axes[ir, ic].axis('off')
                        continue

                # Init arrays
                data = []
                labels = []
                dividers = []
                means = []
                medians = []

                # Set colors
                axes[ir, ic] = set_axes_colors(axes[ir, ic], kw)

                # Style gridlines
                axes[ir, ic] = set_axes_grid_lines(axes[ir, ic], kw)

                # Subset the data
                df_sub = get_rc_subset(df_fig, ir, ic, kw)

                # Get the changes df
                kw['groups'] = validate_columns(df_sub, kw['groups'])
                if kw['groups'] is not None:
                    groups = df_sub.groupby(kw['groups'])
                else:
                    groups = df_sub.copy()

                # Order the group labels with natsorting
                gidx = []
                for i, (nn, g) in enumerate(groups):
                    gidx += [nn]
                gidx = natsorted(gidx)

                # Make indices df
                indices = pd.DataFrame(gidx)
                num_groups = groups.ngroups
                changes = index_changes(indices, num_groups)
                num_groups = 0
                if kw['groups'] is not None:
                    groups = df_sub.groupby(kw['groups'])
                    num_groups = groups.ngroups
                    num_groupby = len(kw['groups'])
                    col = changes.columns

                    # Plot the groups
                    for i, nn in enumerate(gidx):
                        g = df_sub.copy().sort_values(by=kw['groups'])
                        g = g.set_index(kw['groups'])
                        if len(g) > 1:
                            g = g.loc[nn]
                        if type(g) == pd.Series:
                            g = pd.DataFrame(g).T
                        else:
                            g = g.reset_index()
                        temp = g[y].dropna()
                        data += [temp]
                        means += [temp.mean()]
                        medians += [temp.median()]
                        if type(nn) is not tuple:
                            nn = [nn]
                        else:
                            nn = [str(f) for f in nn]
                        labels += ['']

                        if len(changes.columns) > 1 and changes[col[-2]].iloc[i] == 1 \
                                and len(kw['groups']) > 1:
                            dividers += [i+0.5]

                        point_curves = []
                        if kw['points'] and len(kw['leg_items'])==0:
                            add_points(i, temp, axes[ir, ic], palette[1], **kw)
                        else:
                            for ileg, leg_item in enumerate(kw['leg_items']):
                                temp = g.loc[g[kw['leg_groups']]==leg_item][y].dropna()
                                kw['marker_type'] = markers[ileg]
                                point_curves += add_points(i, temp, axes[ir, ic],
                                                            palette[ileg+1], **kw)

                else:
                    data = df_sub[y].dropna()
                    labels = ['']
                    if kw['points']:
                        add_points(0, data, axes[ir, ic], palette[1], **kw)

                # Plot
                if kw['points']:
                    showfliers = False
                else:
                    showfliers = True

                if type(data) is pd.Series:
                    data = data.values
                elif type(data) is pd.DataFrame and len(data.columns) == 1:
                    data = data.values

                if len(data) > 0:
                    if kw['range_lines']:
                        for id, dd in enumerate(data):
                            axes[ir, ic].plot([id+1, id+1],
                                [dd.min().iloc[0], dd.max().iloc[0]],
                                linestyle='--', color='#dddddd',
                                zorder=0)
                            axes[ir, ic].plot([id+1-0.2, id+1+0.2],
                                [dd.min().iloc[0], dd.min().iloc[0]],
                                linestyle='-', color='#dddddd', zorder=0)
                            axes[ir, ic].plot([id+1-0.2, id+1+0.2],
                                [dd.max().iloc[0], dd.max().iloc[0]],
                                linestyle='-', color='#dddddd', zorder=0)
                    try:
                        bp = axes[ir,ic].boxplot(data, labels=labels,
                                            showfliers=showfliers,
                                            boxprops={'color': palette[0]},
                                            whiskerprops={'color': palette[0]},
                                            capprops={'color': palette[0]},
                                            medianprops={'color': palette[1]},
                                            patch_artist=True,
                                            zorder=1,
                                            )
                    except:
                        bp = axes[ir,ic].boxplot(data, labels=labels,
                                            showfliers=showfliers,
                                            boxprops={'color': palette[0]},
                                            whiskerprops={'color': palette[0]},
                                            capprops={'color': palette[0]},
                                            medianprops={'color': palette[1]},
                                            patch_artist=True,
                                            )
                    axes[ir,ic].xaxis.grid(False)
                    for patch in bp['boxes']:
                        patch.set_facecolor(kw['bp_fill_color'])
                    for flier in bp['fliers']:
                        flier.set(marker='+', markeredgecolor=palette[0])

                    if str(kw['ax_scale']).lower() in ['logy', 'semilogy', 'loglog']:
                        axes[ir, ic].set_yscale('log')

                    # Add divider lines
                    if kw['dividers']:
                        for d in dividers:
                            axes[ir,ic].axvline(d, linewidth=1,
                                            color=kw['bp_divider_color'])

                    # Add mean/median connecting lines
                    if kw['connect_means']:
                        x = np.linspace(1, num_groups, num_groups)
                        axes[ir,ic].plot(x, means, color=palette[2])

                # Add y-axis label
                if kw['ylabel'] is not None and not(kw['wrap'] is not None
                                                    and ic != 0):
                    axes[ir,ic].set_ylabel(r'%s' % kw['ylabel'],
                                  fontsize=kw['label_font_size'],
                                  weight=kw['label_weight'],
                                  style=kw['label_style'])

                # Add axh/axv lines
                axes[ir, ic] = add_lines(axes[ir, ic], kw)

                # Format the subplot spacing
                fig.subplots_adjust(
                    left=layout.left,
                    right=layout.right,
                    bottom=layout.bottom,
                    top=layout.top,
                    hspace=(1.0*layout.ws_row)/layout.ax_h,
                    wspace=1.0*layout.ws_col/layout.ax_w
                )

                # Add the x-axis grouping labels
                if kw['bp_labels_on'] and changes is not None:
                    num_cols = len(changes.columns)
                    bottom = 0
                    for i in range(0, num_cols):
                        if i > 0:
                            bottom -= height
                        k = num_cols-1-i
                        sub = changes[num_cols-1-i][changes[num_cols-1-i]==1]
                        if len(sub) == 0:
                            sub = changes[num_cols-1-i]
                        for j in range(0, len(sub)):
                            if j == len(sub) - 1:
                                width = len(changes) - sub.index[j]
                            else:
                                width = sub.index[j+1] - sub.index[j]
                            label = indices.loc[sub.index[j], num_cols-1-i]
                            if len(align.keys()) > 0 \
                                    and k in align.keys() \
                                    and align[k] > 0:
                                orient = 90
                                height = align[k]/ kw['ax_size'][1]
                            else:
                                orient = 0
                                height = kw['bp_label_size']/kw['ax_size'][1]
                            add_label(label, (sub.index[j]/len(changes),
                                      bottom-height, width/len(changes),
                                      height),
                                      axes[ir,ic],
                                      orient,
                                      layout,
                                      edgecolor=kw['bp_label_edge_color'],
                                      fillcolor=kw['bp_label_fill_color'],
                                      color=kw['bp_label_text_color'],
                                      fontsize=kw['bp_label_font_size'],
                                      weight=kw['bp_label_text_style'])

                        # Add the grouping label names
                        if not(kw['wrap'] is not None and ic != kw['ncol']-1) \
                                or (ir*kw['ncol'] + ic == len(kw['wrap'])-1):
                            offset = kw['bp_label_font_size']/layout.fig_h_px
                            axes[ir,ic].text(1+kw['ws_bp_name']/kw['ax_size'][0],
                                    bottom-height/2-offset, kw['groups'][k],
                                    fontsize=kw['bp_name_font_size'],
                                    color=kw['bp_name_text_color'],
                                    style=kw['bp_name_text_style'],
                                    weight=kw['bp_name_text_weight'],
                                    transform=axes[ir,ic].transAxes)

                # Adjust the tick marks
                axes[ir, ic] = set_axes_ticks(axes[ir, ic], kw, True)

                # Add row/column labels
                axes[ir, ic] = \
                    set_axes_rc_labels(axes[ir, ic], ir, ic, kw, layout)

                # Axis ranges
                axes[ir, ic], ax = set_axes_ranges(df_fig, df_sub, None, y,
                                                   axes[ir, ic], None, kw)

        # Add the legend
        fig = add_legend(fig, point_curves, kw, layout)

        # Add a figure title
        set_figure_title(df_fig, axes[0,0], kw, layout)

        # Build the save filename
        filename = utl.set_save_filename(df_fig, x, y, kw, ifig)

        # Save and optionally open
        save(fig, filename, kw)

    if not kw['inline']:
        plt.close('all')

    else:
        return fig


def conf_int(df, x, y, ax, color, kw):
    """
    Calculate and draw confidence intervals around a curve

    Args:
        df:
        x:
        y:
        ax:
        color:
        kw:

    Returns:

    """

    if kw['conf_int'] is None:
        return

    if kw['conf_int_fill_color'] is None:
        color = color
    else:
        color = kw['conf_int_fill_color']

    if str(kw['conf_int']).lower() == 'range':
        ymin = df.groupby(x).min()[y]
        xx = ymin.index
        ymin = ymin.reset_index(drop=True)
        ymax = df.groupby(x).max()[y].reset_index(drop=True)
        ax.fill_between(xx, ymin, ymax, facecolor=color,
                        alpha=kw['conf_int_fill_alpha'])

    else:
        if float(kw['conf_int']) > 1:
            kw['conf_int'] = float(kw['conf_int'])/100
        stat = pd.DataFrame()
        stat['mean'] = df[[x,y]].groupby(x).mean().reset_index()[y]
        stat['count'] = df[[x,y]].groupby(x).count().reset_index()[y]
        stat['std'] = df[[x,y]].groupby(x).std().reset_index()[y]
        stat['sderr'] = stat['std'] / np.sqrt(stat['count'])
        stat['ucl'] = np.nan
        stat['lcl'] = np.nan
        for irow, row in stat.iterrows():
            if row['std'] == 0:
                conf = [0, 0]
            else:
                conf = ss.t.interval(kw['conf_int'], int(row['count'])-1,
                                     loc=row['mean'], scale=row['sderr'])
            stat.loc[irow, 'ucl'] = conf[1]
            stat.loc[irow, 'lcl'] = conf[0]

        ax.fill_between(df.groupby(x).mean().index, stat['lcl'], stat['ucl'],
                        facecolor=color, alpha=kw['conf_int_fill_alpha'])


def contour(**kwargs):
    """

    Args:
        **kwargs:

    Returns:

    """

    plt.close('all')

    # Override some defaults
    if kwargs.get('grid_major', None):
        kwargs['grid_major'] = False

    # Init plot
    df, x, y, z, kw = init('plot', kwargs)
    kw['ptype'] = 'contour'
    if type(df) is bool and not df:
        return

    # Iterate over discrete figures
    for ifig, fig_item in enumerate(kw['fig_items']):

        # Make a data subset and filter
        df_fig = get_df_figure(df, fig_item, kw)

        # Set up the row grouping
        kw = get_rc_groupings(df_fig, kw)

        # Set up the legend grouping
        kw = get_legend_groupings(df, y, kw)

        # Make the plot figure and axes
        layout, fig, axes, kw = make_fig_and_ax(kw)
        ax2 = None

        # Handle colormaps
        if kw['cmap']=='jmp_spectral':
            cmap = jmp_spectral
        elif kw['cmap'] is not None:
            cmap = plt.get_cmap(kw['cmap'])


        # Make the plots by row and by column
        for ir in range(0, kw['nrow']):
            for ic in range(0, kw['ncol']):

                # Handle missing wrap plots
                if kw['wrap'] is not None:
                    if ir*kw['ncol'] + ic > len(kw['wrap']) - 1:
                        axes[ir, ic].axis('off')
                        continue

                # Set colors
                axes[ir, ic] = set_axes_colors(axes[ir, ic], kw)

                # Style gridlines
                axes[ir, ic] = set_axes_grid_lines(axes[ir, ic], kw)

                # Subset the data
                df_sub = get_rc_subset(df, ir, ic, kw)

                # Apply any data transformations
                df_sub = set_data_transformation(df_sub, x, y, z, kw)

                # Select the contour type
                if kw['filled']:
                    contour = axes[ir, ic].contourf
                else:
                    contour = axes[ir, ic].contour

                # Change data type
                xx = np.array(df_sub[x])
                yy = np.array(df_sub[y[0]])
                zz = np.array(df_sub[z])

                # Handle colormaps
                if kw['cmap'] == 'jmp_spectral':
                    cmap = jmp_spectral
                elif kw['cmap'] is not None:
                    cmap = plt.get_cmap(kw['cmap'])
                elif fcp_params['cmap'] == 'jmp_spectral':
                    cmap = jmp_spectral
                elif fcp_params['cmap'] is not None:
                    cmap = plt.get_cmap(fcp_params['cmap'])
                else:
                    cmap = plt.get_cmap('inferno')

                # Make the grid
                xi = np.linspace(min(xx), max(xx))
                yi = np.linspace(min(yy), max(yy))
                zi = mlab.griddata(xx, yy, zz, xi, yi, interp='linear')
                cc = contour(xi, yi, zi, kw['levels'],
                             line_width=kw['line_width'], cmap=cmap)

                # Set the axes ranges
                if kw['xmin'] is None:
                    kw['xmin'] = min(xi)
                if kw['xmax'] is None:
                    kw['xmax'] = max(xi)
                if kw['ymin'] is None:
                    kw['ymin'] = min(yi)
                if kw['ymax'] is None:
                    kw['ymax'] = max(yi)

                # Adjust the tick marks
                axes[ir, ic] = set_axes_ticks(axes[ir, ic], kw)

                # Axis ranges
                axes[ir, ic], ax = set_axes_ranges(df_fig, df_sub, x, y,
                                                   axes[ir, ic], ax2, kw)

                # Add labels
                axes[ir, ic], ax2 = \
                    set_axes_labels(axes[ir, ic], ax2, ir, ic, kw)

                # Add row/column labels
                axes[ir, ic] = \
                    set_axes_rc_labels(axes[ir, ic], ir, ic, kw, layout)

                # Add the colorbar
                if kw['cbar']:
                    # Define colorbar position
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    divider = make_axes_locatable(axes[ir, ic])
                    size = '%s%%' % (100*kw['cbar_width']/layout.ax_size[0])
                    pad = kw['ws_cbar_ax']/100
                    cax = divider.append_axes("right", size=size, pad=pad)

                    # Add the colorbar and label
                    cbar = plt.colorbar(cc, cax=cax)
                    cbar.ax.set_ylabel(r'%s' % kw['cbar_label'], rotation=270,
                                       labelpad=kw['label_font_size'],
                                       style=kw['label_style'],
                                       fontsize=kw['label_font_size'],
                                       weight=kw['label_weight'],
                                       color=kw['ylabel_color'])

        # Add a figure title
        set_figure_title(df_fig, axes[0,0], kw, layout)

        # Build the save filename
        filename = set_save_filename(df_fig, x, y, kw, ifig)

        # Save and optionally open
        save(fig, filename, kw)

        # Reset values for next loop
        kw['leg_items'] = []

    if not kw['inline']:
        plt.close('all')

    else:
        return fig


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

    return kwargs


def index_changes(df, num_groups):
    """
    Make a DataFrame that shows when groups vals change; used for grouping labels

    Args:
        df (pd.DataFrame): grouping values
        num_groups (int): number of unique groups

    Returns:
        new DataFrame with 1's showing where group levels change for each row of df
    """

    changes = df.copy()
    # Set initial level to 1
    for col in df.columns:
        changes.loc[0, col] = 1
    # Determines values for all other rows
    for i in range(1, num_groups):
        for col in df.columns:
            if df[col].iloc[i-1] == df[col].iloc[i]:
                changes.loc[i, col] = 0
            else:
                changes.loc[i, col] = 1

    return changes

def init(plot, kwargs):
    """
    Initialize plotting
        * init matplotlib
        * Extract params from kwargs
        * Convert data column types
        * Filter data

    Args:
        plot (str): plot type ('boxplot', 'contour', or 'plot')
        kwargs (dict): keyword args

    Keyword Args:
        ax_edge_color (str):  hex color code for axis edge border color
            (default from fcp_params)
        ax_face_color (str):  hex color code for axis fill color (default from
            fcp_params)
        ax_hlines (list):  list of y-values at which to add horizontal lines
            (default=[])
        ax_lim (list):  axes range values [xmin, xmax, ymin, ymax] (default=[])
        ax_lim_pad (float):  in place of discrete axes limits, adds whitespace
            to the bottom|top and left|right of the axis by this amount as a
            percent/100 of the total axis range (default=0.05)
        ax_scale (None|str):  set the scale of an axis as linear or log
            (default=None --> linear; options include 'logx'|'semilogx' for the
            x-axis, 'logy'|'semilogy' for the y-axis or 'loglog' for both axes
        ax_scale2 (None|str): same as ax_scale but for secondary y-axis
            when twinx is True
        ax_size (list):  [width, height] of each plot window in pixels
            (default defined by fcp_params)
        ax_vlines (list):  list of x-values at which to add vertical lines
            (default=[])
        bp_divider_color (str): hex color code for divider line color between
            groups (default from fcp_params)
        bp_fill_color (str):  hex color code for boxplot fill color (default
            from fcp_params)
        bp_labels_on (bool):  toggle on|off boxplot grouping labels
            (default=True)
        bp_label_size (int):  grouping label size (default from fcp_params)
        bp_label_edge_color (str):  hex color code for boxplot grouping label
            regions (default from fcp_params)
        bp_label_fill_color (str):  hex color code for grouping labels (default
            from fcp_params)
        bp_label_font_size (int):  grouping label font size (default from
            fcp_params)
        bp_label_text_color (str):  hex color code for grouping label text
            (default from fcp_params)
        bp_label_text_style (str):  grouping label text style; use standard
            mpl styles like 'italic' (default from fcp_params)
        bp_label_text_weight (str):  grouping label text weight; use standard
            mpl weights like 'bold' (default from fcp_params)
        bp_name_font_size (str):  grouping label summary name font size
            (default from fcp_params)
        bp_name_text_color (str):  hex color code for grouping label summary
            name (default from fcp_params)
        bp_name_text_style (str):  grouping label summary name text style; use
            standard mpl styles like 'italic' (default from fcp_params)
        bp_name_text_weight (str):  grouping label summary name text weight;
            use standard mpl weights like 'bold' (default from fcp_params)
        cmap (str): name of color map to use for plotting (default=None -->
            use color order defined in fcp_params
        col (str):  name of a column in df to use as a grouping variable for
            facet grid style plotting; each unique value from df[col] will
            represent a column of plots in the final figure (default=None -->
            no column grouping)
        col_label (str):  a custom label to display for each column in the
            facet grid plot (default=None --> use the value of kwargs['col']
        col_labels_on (bool):  toggle on|off column labels in facet grid plot
            (default=True)
        col_label_size (int):  label height in pixel for col labels in pixels
            (default from fcp_params)
        ws_col_label (int):  whitespace in pixels between axis and col labels
            in pixels (default from fcp_params)
        colors (list):  list of tuple values used to define colors for plotting
            using RGB/255 mpl style (default is palette from defaults.py)
        cols (list):  list used to manually define the columns to use in the
            facet grid (default=None). These values must actually be in df[col]
        connect_means (bool):  draw a line connecting the mean value of each
            group in the boxplot (default=False)
        dividers (bool):  show vertical diveider lines between groups
            (default=True)
        fig_ax_ws (int):  number of pixels between the left figure edge and the
            left axis edge (default from fcp_params)
        fig_edge_color (str):  hex color code for figure edge color (default
            from fcp_params)
        fig_face_color (str):  hex color code for figure fill color (default
            from fcp_params)
        fig_groups (list|None):  name of df column by which to group the data
            and make discrete figures based on the unique values in this column
            (default=None)
        fig_group_path (bool):  use the unique grouping item from fig_groups
            in the plot filename (default=False)
        filename (str):  name of saved figure (default=None--> custom filename
            will be built based on the data that is plotted
        filter (str):  str to use in df.query to include or exclude certain
            data from df (default=None).
        grid_major (bool):  toggle major gridlines (default=True)
        grid_major_color (str):  hex color code for major gridline color
            (default from fcp_params)
        grid_major_line_style (str):  matplotlib str code for linestyle (default
            from fcp_params)
        grid_major_line_width (float): major grid line width (default
            from fcp_params)
        grid_minor (bool):  toggle minor gridlines (default=False)
        grid_minor_color (str):  hex color code for minor gridline color
            (default from fcp_params)
        grid_minor_line_style (str):  matplotlib str code for linestyle (default
            from fcp_params)
        grid_minor_line_width (float):  minor grid line width (default
            from fcp_params)
        groups (list):  (default=None)
        jitter (bool):  jitter the data points (default=False)
        label_font_size (int):  font size in pixels for the x and y labels
            (default from fcp_params)
        label_style (str):  define the label style (default from fcp_params).
            Use standard mpl styles like 'italic'
        label_weight (str):  define the label weight (default from fcp_params).
            Use standard mpl weights like 'bold'
        leg_bkgrd (str):  hex color code for legend background (default from
            fcp_params)
        leg_border (str):  hex color code for legend border (default from
            fcp_params)
        leg_groups (str):  name of df column by which to legend the data
            (default=None)
        leg_items (str):  explicit definition of the legend items; ignores any
            values that are not listed (default=None)
        leg_on (bool):  toggle legend visibility (default=True)
        leg_title (str):  title for the legend (default=None)
        line_color (str):  hex color code to override the built-in line color
            cycle (default=None)
        line_fit (None|int):  adds a line fit to the data of polynomial order
            ``line_fit`` (default=None)
        line_style (str):  set the default line style (default="-")
        line_width (int):  set the linewidth of curves (default=1)
        lines (bool):  turn plot lines on|off (default=True)
        marker_size (int):  set marker size (default from fcp_params)
        marker_type (str):  set marker type (default==None)
        points (bool):  turn markers on|off (default=True)
        range_lines(bool):  add range lines on boxplot
        rc_label_edge_color (str):  hex color code for row/column labels border
            edges (default from fcp_params)
        rc_label_fill_color (str):  hex color code for row/column labels
            background (default from fcp_params)
        rc_label_font_size (int):  font size for row/column labels (default
            from fcp_params)
        rc_label_text_color (str):  hex color code for row/column label text
            (default from fcp_params)
        rc_label_text_style (str):  define the style row/column label text. Use
            standard mpl styles like 'italic'" (default from fcp_params)
        row (str):  name of a column in df to use as a grouping variable for
            facet grid style plotting; each unique value from df[row] will
            represent a row of plots in the final figure (default=None --> no
            column grouping)
        row_label (str):  a custom label to display for each row in the facet
            grid plot (default=None --> use the value of kwargs['row'])
        row_labels_on (bool):  toggle visibility of row grouping labels in
            facet grid (default=True)
        row_label_size (int):  label height in pixel for row labels in pixels
            (default from fcp_params)
        ws_row_label (int):  whitespace in pixels between axis and row labels
            in pixels (default from fcp_params)
        rows (list):  list used to manually define the rows to use in the facet
            grid. These values must actually be in df[row] (default=None)
        save_ext (str):  filename extension for saved figures (default='png')
        save_path (str):  destination file folder for plots (default=None)
        scalar_x (bool):  force scalar tick labels on the x-axis instead of
            powers of ten (default=False)
        scalar_y (bool):  force scalar tick labels on the y-axis instead of
            powers of ten (default=False)
        sci_x (bool):  force scientific notation for tick labels on x-axis
            (default=False)
        sci_y (bool):  force scientific notation for tick labels on y-axis
            (default=False)
        separate_labels (bool):  give each plot its own axes labels
        sharex (bool):  share plot range for x-axis (default=True)
        sharey (bool):  share plot range for y-axis (default=True)
        show (bool):  pop open plot after it is generated.  Otherwise, the plot
            is just saved (default=False)
        stat (str):  statistic to apply to the y-data being plotted (default=
            None)
        stat_val (str):  column name to which the statistic should be applied.
            Options include: 'median', 'median_only', 'mean_only', and 'mean'.
            When 'only' is in the stat value, median lines and median/mean data
            points are plotted.  Without 'only', median/mean lines and all raw
            data points are plotted (default=kwargs['x'])
        tick_color_major (str):  hex color code for major tick marks (default
            from fcp_params)
        tick_color_minor (str):  hex color code for minor tick marks (default
            from fcp_params)
        tick_font_size (int):  font size for tick labels (default from
            fcp_params)
        tick_label_color (str):  hex color code for tick labels (default
            from fcp_params)
        tick_length (int):  length of the tick marks (default from fcp_params)
        tick_width (int):  line width of the tick marks (default from
            fcp_params)
        title (str):  plot title (default=None)
        title_edge_color (str):  hex color code for edge of a box around the
            plot title (default from fcp_params)
        title_fill_color (str):  hex color code for fill of a box around the
            plot title (default from fcp_params)
        title_text_color (str):  hex color code for title color (default from
            fcp_params)
        title_font_size (int):  font size of plot title (default from
            fcp_params)
        title_text_style (str):  define the title style. Use standard mpl
            styles like 'italic' (default from fcp_params)
        twinx (bool):  allow for twinning of x-axis for secondary y-axis
            (default=False)
        twiny (bool):  allow for twinning of y-axis for secondary x-axis
            (not currently supported) (default=False)
        ws_bp_name (int):  whitespace in pixels between grouping labels under
            the boxplot and the summary names (default from fcp_params)
        ws_col (int):  whitespace between facet columns in pixels (default
            in fcp_params)
        ws_row (int):  whitespace between facet rows in pixels (default
            from fcp_params
        ws_tick_ax (int):  label offset padding in pixels (default=0)
        xlabel (str):  label for x-axis (default=kwargs['x'])
        xmax (float):  maximum x-value on x-axis (default=None --> use mpl
            defaults)
        xmin (float):  minimum x-value of x-axis (default=None --> use mpl
            defaults)
        xticks (int):  specify the number of xticks (not currently supported)
            (default=None --> use mpl defaults)
        xtrans (str):  translate the x-axis data.  Options include: 'abs'
            (absolute value), 'neg'/'negative' (negative value),
            'inv'/'inverse' (1/data), or tuple of ('pow', int) (to raise to a
            power) (default = None)
        ylabel (str):  label for primary y-axis (default=kwargs['y'] or
            kwargs['y'][0])
        ymax (float):  maximum y-value on y-axis (default=None --> use mpl
            defaults)
        ymin (float):  minimum y-value of y-axis (default=None --> use mpl
            defaults)
        yticks (int):  specify the number of yticks (not currently supported)
            (default=None --> use mpl defaults)
        ytrans (str):  translate the y-axis data.  Options include: 'abs'
            (absolute value), 'neg'/'negative' (negative value),
            'inv'/'inverse' (1/data), or tuple of ('pow', int) (to raise to a
            power) [won't work with ``twinx``] (default=None)
        ylabel2 (str):  label for secondary y-axis (default=kwargs['y'][1])
        ymax2 (float):  maximum y-value on secondary y-axis (default=None -->
            use mpl defaults)
        ymin2 (float):  minimum y-value on secondary y-axis (default=None -->
            use mpl defaults)
        yticks2 (int):  specify the number of yticks2 (not currently supported)
            (default=None --> use mpl defaults)

    Returns:
        dictionary of keyword arguments
    """

    # Reload defaults
    fcp_params, palette, markers = reload_defaults()

    # Check for pasted kwargs
    if kwargs.get('paste'):
        kwargs = paste_kwargs(kwargs)

    # Extract keywords
    df = kwargs.get('df').copy()
    if df is None:
        raise ValueError('Must provide a DataFrame called "df" for analysis!')

    x = kwargs.get('x')
    if plot in ['contour', 'plot']:
        if x is None:
            raise ValueError('Must provide a column name for "x"')
        if x not in df.columns:
            raise ValueError('Column "%s" not found in DataFrame!' % x)
        try:
            df[x] = df[x].astype(float)
        except:
            raise ValueError('Could not convert x-column "%s" to float!' % x)

    y = kwargs.get('y')
    y = validate_list(y)
    if plot in ['boxplot', 'contour', 'plot']:
        if y is None:
            raise ValueError('Must provide a column name for "y"')
        for yy in y:
            if yy not in df.columns:
                raise ValueError('Column "%s" not found in DataFrame!' % yy)
            try:
                df[yy] = df[yy].astype(float)
            except:
                raise ValueError('Could not convert y-column "%s" to float!'
                                 % y)

    z = kwargs.get('z')
    if plot in ['contour']:
        if z is None:
            raise ValueError('Must provide a column name for "z"')
        if z not in df.columns:
            raise ValueError('Column "%s" not found in DataFrame!' % z)
        try:
            df[z] = df[z].astype(float)
        except:
            raise ValueError('Could not convert z-column to float!')

    kw = dict()
    try:
        kw['rc_label_size'] = \
            kwargs.get('rc_label_size', fcp_params['rc_label_size'])
        kw['alpha'] = kwargs.get('alpha', 1)
        kw['ax_edge_color'] = kwargs.get('ax_edge_color',
                                         fcp_params['ax_edge_color'])
        kw['ax_face_color'] = kwargs.get('ax_face_color',
                                         fcp_params['ax_face_color'])
        kw['ax_hlines'] = validate_list(kwargs.get('ax_hlines', []))
        kw['ax_lim'] = kwargs.get('ax_lim', [])
        kw['ax_lim_pad'] = kwargs.get('ax_lim_pad', 0.05)
        kw['ax_scale'] = kwargs.get('ax_scale', None)
        kw['ax_scale2'] = kwargs.get('ax_scale2', None)
        kw['ax_size'] = kwargs.get('ax_size', fcp_params['ax_size'])
        kw['ax_vlines'] = validate_list(kwargs.get('ax_vlines', []))
        kw['bp_divider_color'] = kwargs.get('bp_divider_color',
                                            fcp_params['bp_divider_color'])
        kw['bp_fill_color'] = kwargs.get('bp_fill_color',
                                            fcp_params['bp_fill_color'])
        kw['bp_labels_on'] = kwargs.get('bp_labels_on', True)
        kw['bp_label_size'] = kwargs.get('bp_label_size',
                                         fcp_params['bp_label_size'])
        kw['bp_label_edge_color'] = kwargs.get('bp_label_edge_color',
                                               fcp_params['bp_label_edge_color'])
        kw['bp_label_fill_color'] = kwargs.get('bp_label_fill_color',
                                               fcp_params['bp_label_fill_color'])
        kw['bp_label_font_size'] = kwargs.get('bp_label_font_size',
                                              fcp_params['bp_label_font_size'])
        kw['bp_label_text_color'] = kwargs.get('bp_label_text_color',
                                               fcp_params['bp_label_text_color'])
        kw['bp_label_text_style'] = kwargs.get('bp_label_text_style',
                                               fcp_params['bp_label_text_style'])
        kw['bp_label_text_weight'] = kwargs.get('bp_label_text_weight',
                                                fcp_params['bp_label_text_weight'])
        kw['bp_name_font_size'] = kwargs.get('bp_name_font_size',
                                              fcp_params['bp_name_font_size'])
        kw['bp_name_text_color'] = kwargs.get('bp_name_text_color',
                                               fcp_params['bp_name_text_color'])
        kw['bp_name_text_style'] = kwargs.get('bp_name_text_style',
                                               fcp_params['bp_name_text_style'])
        kw['bp_name_text_weight'] = kwargs.get('bp_name_text_weight',
                                                fcp_params['bp_name_text_weight'])
        kw['cbar'] = kwargs.get('cbar', False)
        kw['cbar_label'] = kwargs.get('cbar_label', z)
        kw['cbar_width'] = kwargs.get('cbar_width', fcp_params['cbar_width'])
        kw['cmap'] = kwargs.get('cmap', None)
        kw['col'] = kwargs.get('col', None)
        kw['col_label'] = kwargs.get('col_label', None)
        kw['col_labels_on'] = kwargs.get('col_labels_on', True)
        kw['col_label_size'] = kwargs.get('col_label_size',
                                          kw['rc_label_size'])
        kw['colors'] = kwargs.get('colors', palette)
        kw['cols'] = kwargs.get('cols', None)
        kw['cols_orig'] = kw['cols']
        kw['conf_int'] = kwargs.get('conf_int', None)
        kw['conf_int_fill_alpha'] = kwargs.get('conf_int_fill_alpha', 0.2)
        kw['conf_int_fill_color'] = kwargs.get('conf_int_fill_color', None)
        kw['connect_means'] = kwargs.get('connect_means', False)
        kw['dividers'] = kwargs.get('dividers', True)
        kw['dpi'] = kwargs.get('dpi', fcp_params['dpi'])
        kw['fig_edge_color'] = kwargs.get('fig_edge_color',
                                         fcp_params['fig_edge_color'])
        kw['fig_face_color'] = kwargs.get('fig_face_color',
                                         fcp_params['fig_face_color'])
        kw['fig_groups'] = kwargs.get('fig_groups', None)
        kw['fig_group_path'] = kwargs.get('fig_group_path', False)
        kw['fig_label'] = kwargs.get('fig_label', True)
        kw['filename'] = kwargs.get('filename', None)
        kw['filename_orig'] = kwargs.get('filename', None)
        kw['filled'] = kwargs.get('filled', True)
        kw['filter'] = kwargs.get('filter', None)
        kw['grid_major_color'] = kwargs.get('grid_major_color',
                                            fcp_params['grid_major_color'])
        kw['grid_major_line_style'] = kwargs.get('grid_major_line_style',
                                          fcp_params['grid_major_line_style'])
        kw['grid_major_line_width'] = kwargs.get('grid_major_line_width',
                                          fcp_params['grid_major_line_width'])
        kw['grid_minor_color'] = kwargs.get('grid_minor_color',
                                          fcp_params['grid_minor_color'])
        kw['grid_minor_line_style'] = kwargs.get('grid_minor_line_style',
                                          fcp_params['grid_minor_line_style'])
        kw['grid_minor_line_width'] = kwargs.get('grid_minor_line_width',
                                          fcp_params['grid_minor_line_width'])
        kw['grid_major'] = kwargs.get('grid_major', True)
        kw['grid_minor'] = kwargs.get('grid_minor', False)
        kw['groups'] = kwargs.get('groups', None)
        kw['inline'] = kwargs.get('inline', fcp_params['inline'])
        kw['jitter'] = kwargs.get('jitter', False)
        kw['label_font_size'] = kwargs.get('label_font_size',
                                           fcp_params['label_font_size'])
        kw['label_style'] = kwargs.get('label_style',
                                       fcp_params['label_style'])
        kw['label_weight'] = kwargs.get('label_weight',
                                        fcp_params['label_weight'])
        kw['leg_bkgrd'] = kwargs.get('leg_bkgrd', fcp_params['leg_bkgrd'])
        kw['leg_border'] = kwargs.get('leg_border', fcp_params['leg_border'])
        kw['leg_groups'] = kwargs.get('leg_groups', None)
        kw['leg_items'] = kwargs.get('leg_items', [])
        kw['leg_on'] = kwargs.get('leg_on', True)
        kw['leg_title'] = kwargs.get('leg_title', None)
        kw['levels'] = kwargs.get('levels', 20)
        kw['line_color'] = kwargs.get('line_color', None)
        kw['line_fit'] = kwargs.get('line_fit', None)
        kw['line_style'] = kwargs.get('line_style', '-')
        kw['line_width'] = kwargs.get('line_width', fcp_params['line_width'])
        kw['lines'] = kwargs.get('lines', True)
        kw['marker_size'] = kwargs.get('marker_size', fcp_params['marker_size'])
        kw['marker_type'] = kwargs.get('marker_type', None)
        kw['normalize'] = kwargs.get('normalize', False)
        kw['points'] = kwargs.get('points', True)
        kw['range_lines'] = kwargs.get('range_lines', True)
        kw['rc_label_edge_color'] = kwargs.get('rc_label_edge_color',
                                               fcp_params['rc_label_edge_color'])
        kw['rc_label_fill_color'] = kwargs.get('rc_label_fill_color',
                                               fcp_params['rc_label_fill_color'])
        kw['rc_label_font_size'] = kwargs.get('rc_label_font_size',
                                              fcp_params['rc_label_font_size'])
        kw['rc_label_text_color'] = kwargs.get('rc_label_text_color',
                                               fcp_params['rc_label_text_color'])
        kw['rc_label_text_style'] = kwargs.get('rc_label_text_style',
                                               fcp_params['rc_label_text_style'])
        kw['row'] = kwargs.get('row', None)
        kw['row_label'] = kwargs.get('row_label', None)
        kw['row_labels_on'] = kwargs.get('row_labels_on', True)
        kw['row_label_size'] = kwargs.get('row_label_size',
                                          kw['rc_label_size'])
        kw['rows'] = kwargs.get('rows', None)
        kw['rows_orig'] = kw['rows']
        kw['save_ext'] = kwargs.get('save_ext', 'png')
        kw['save_name'] = kwargs.get('save_name', None)
        kw['save_path'] = kwargs.get('save_path', None)
        kw['scalar_x'] = kwargs.get('scalar_x', False)
        kw['scalar_y'] = kwargs.get('scalar_y', False)
        kw['sci_x'] = kwargs.get('sci_x', False)
        kw['sci_y'] = kwargs.get('sci_y', False)
        kw['separate_labels'] = kwargs.get('separate_labels', False)
        kw['sharex'] = kwargs.get('sharex', True)
        kw['sharey'] = kwargs.get('sharey', True)
        kw['show'] = kwargs.get('show', False)
        kw['stat'] = kwargs.get('stat', None)
        kw['stat_val'] = kwargs.get('stat_val', x)
        kw['ticks'] = kwargs.get('ticks', True)
        kw['ticks_major'] = kwargs.get('ticks_major', kw['ticks'])
        kw['tick_color_major'] = kwargs.get('tick_color_major',
                                            fcp_params['tick_color_major'])
        kw['ticks_minor'] = kwargs.get('ticks_minor', kw['ticks'])
        kw['tick_color_minor'] = kwargs.get('tick_color_minor',
                                            fcp_params['tick_color_minor'])
        kw['tick_font_size'] = kwargs.get('tick_font_size',
                                          fcp_params['tick_font_size'])
        kw['tick_label_color'] = kwargs.get('tick_label_color',
                                          fcp_params['tick_label_color'])
        kw['tick_length'] = kwargs.get('tick_length',
                                       fcp_params['tick_length'])
        kw['tick_length_major'] = kwargs.get('tick_length_major',
                                             kw['tick_length'])
        kw['tick_length_minor'] = kwargs.get('tick_length_minor',
                                             0.8*kw['tick_length'])
        kw['tick_width'] = kwargs.get('tick_width', fcp_params['tick_width'])
        kw['title'] = kwargs.get('title', None)
        kw['title_orig'] = kwargs.get('title', None)
        kw['title_edge_color'] = kwargs.get('title_edge_color',
                                            fcp_params['title_edge_color'])
        kw['title_fill_color'] = kwargs.get('title_fill_color',
                                            fcp_params['title_fill_color'])
        kw['title_text_color'] = kwargs.get('title_text_color',
                                            fcp_params['title_text_color'])
        kw['title_font_size'] = kwargs.get('title_font_size',
                                            fcp_params['title_font_size'])
        kw['title_text_style'] = kwargs.get('title_text_style',
                                            fcp_params['title_text_style'])
        kw['twinx'] = kwargs.get('twinx', False)
        kw['twiny'] = kwargs.get('twiny', False)
        kw['wrap'] = kwargs.get('wrap', None)
        if kw['wrap']:
            kw['wrap'] = validate_list(kw['wrap'])
        kw['wrap_orig'] = kw['wrap']
        kw['wrap_title'] = kwargs.get('wrap_title', True)
        kw['wrap_title_edge_color'] = \
            kwargs.get('wrap_title_edge_color',
                       fcp_params['wrap_title_edge_color'])
        kw['wrap_title_fill_color'] = \
            kwargs.get('wrap_title_fill_color',
                       fcp_params['wrap_title_fill_color'])
        kw['wrap_title_font_size'] = \
            kwargs.get('wrap_title_font_size',
                       fcp_params['wrap_title_font_size'])
        kw['wrap_title_text_color'] = \
            kwargs.get('wrap_title_text_color',
                       fcp_params['wrap_title_text_color'])
        kw['wrap_title_text_style'] = \
            kwargs.get('wrap_title_text_style',
                       fcp_params['wrap_title_text_style'])
        kw['wrap_title_text_weight'] = \
            kwargs.get('wrap_title_text_weight',
                       fcp_params['wrap_title_text_weight'])
        kw['wrap_title_size'] = kwargs.get('wrap_title_size', fcp_params['wrap_title_size'])
        kw['ws_bp_name'] = kwargs.get('ws_bp_name', fcp_params['ws_bp_name'])
        kw['ws_cbar_ax'] = kwargs.get('ws_cbar_ax', fcp_params['ws_cbar_ax'])
        kw['ws_col'] = kwargs.get('ws_col', fcp_params['ws_col'])
        kw['ws_col_label'] = kwargs.get('ws_col_label',
                                        fcp_params['ws_rc_label'])
        kw['ws_row'] = kwargs.get('ws_row', fcp_params['ws_row'])
        kw['ws_row_label'] = kwargs.get('ws_row_label',
                                        fcp_params['ws_rc_label'])
        kw['ws_tick_ax'] = kwargs.get('ws_tick_ax', fcp_params['ws_tick_ax'])
        kw['ws_wrap_title'] = kwargs.get('ws_wrap_title',
                                         fcp_params['ws_wrap_title'])
        kw['xlabel'] = kwargs.get('xlabel', x)
        kw['xlabel_color'] = kwargs.get('xlabel_color',
                                        fcp_params['label_color'])
        kw['xmax'] = kwargs.get('xmax', None)
        kw['xmin'] = kwargs.get('xmin', None)
        kw['xticks'] = kwargs.get('xticks', None)
        kw['xtrans'] = kwargs.get('xtrans', None)
        kw['ylabel'] = kwargs.get('ylabel', ' + '.join(y))
        kw['ylabel_color'] = kwargs.get('ylabel_color',
                                        fcp_params['label_color'])
        kw['yline'] = kwargs.get('yline', None)
        kw['ymax'] = kwargs.get('ymax', None)
        kw['ymin'] = kwargs.get('ymin', None)
        kw['yticks'] = kwargs.get('yticks', None)
        kw['ytrans'] = kwargs.get('ytrans', None)
        kw['ylabel2'] = kwargs.get('ylabel2', y)
        kw['ylabel2_color'] = kwargs.get('ylabel2_color',
                                         fcp_params['label_color'])
        kw['ymax2'] = kwargs.get('ymax2', None)
        kw['ymin2'] = kwargs.get('ymin2', None)
        kw['yticks2'] = kwargs.get('yticks2', None)
        kw['ytrans2'] = kwargs.get('ytrans2', None)
        kw['ztrans'] = kwargs.get('ztrans', None)

    except KeyError as e:
        print('fcp Param Error!\n'
              '   Keyword %s was not found in your defaults.py file!\n'
              '   Either manually add the value or run fcp.set_theme() to'
              ' replace the entire file'
              % e)
        return False, False, False, False, False

    # Make lists
    vals = ['groups']
    for v in vals:
        kw[v] = validate_list(kw[v])

    # Dummy-proof colors
    if type(kw['colors'][0]) is not tuple:
        kw['colors'] = [kw['colors']]
    else:
        kw['colors'] = list(kw['colors'])
    kw['colors'] += [f for f in palette if f not in kw['colors']]

    # Filter the dataframe
    if kw['filter']:
        df = df_filter(df, kw['filter'])
    if len(df) == 0:
        raise ValueError('No data remains after filter.  Killing plot.')

    # Eliminate title buffer if no title is provided
    if not kw['title']:
        kw['title_h'] = 0
        kw['ws_title_ax'] = 0

    # Set up the figure grouping and iterate (each value corresponds to a
    #  separate figure)
    if kw['fig_groups'] is not None:
        kw['fig_groups'] = validate_columns(df, kw['fig_groups'])
        if kw['fig_groups'] is not None and type(kw['fig_groups']) is list:
            kw['fig_items'] = list(df.groupby(kw['fig_groups']).groups.keys())
        elif kw['fig_groups'] is not None:
            kw['fig_items'] = list(df[kw['fig_groups']].unique())
    else:
        kw['fig_items'] = [None]
    if kw['fig_group_path'] is not None and type(kw['fig_group_path']) is str:
        # needs groupby error handling
        temp = list(df.groupby([kw['fig_groups'],
                                kw['fig_group_path']]).groups.keys())
        kw['fig_path_items'] = [f[1] for f in temp]
    else:
        kw['fig_path_items'] = kw['fig_items']

    # Add padding if sharex|sharey or using separate labels on all facet plots
    if not kw['sharex'] or not kw['sharey']:
        if 'ws_row' not in kwargs.keys():
            kw['ws_row'] += kw['tick_font_size']
        if 'ws_col' not in kwargs.keys():
            kw['ws_col'] += kw['tick_font_size']
    ### HELP
    # if kw['separate_labels']:
    #     kw['ws_col'] = max(kw['fig_ax_ws'], kw['ws_col'])
    #     if plot != 'boxplot':
    #         kw['ws_row'] = max(kw['ax_fig_ws'], kw['ws_row']) + 10

    # Defaults for wrap
    if kw['wrap']:
        if 'separate_labels' not in kwargs.keys():
            kw['separate_labels'] = False
        if 'ws_row' not in kwargs.keys():
            kw['ws_row'] = 0
        if 'ws_col' not in kwargs.keys():
            kw['ws_col'] = 0
        if 'ws_col_label' not in kwargs.keys():
            kw['ws_col_label'] = 0
        kw['sharex'] = True
        kw['sharey'] = True

    # Account for scalar formatted axes
    if kw['scalar_y']:
        max_y = df[y].values.max()
        max_y = int(10**(np.ceil(np.log10(df[y].values.max()))-3))
        kw['fig_ax_ws'] += 10*len(str(max_y))

    # Turn off interactive plotting
    plt.ioff()
    plt.close('all')

    return df.copy(), x, y, z, kw


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


def plot(**kwargs):

    return plotter('plot_xy', **kwargs)


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

    """

    # Init arrays
    data = []
    labels = []
    dividers = []
    means = []
    medians = []

    # Get the changes df
    kwargs['groups'] = kwargs.get('groups', None)
    if kwargs['groups'] is not None:
        kwargs['groups'] = utl.validate_list(kwargs['groups'])
        groups = df_rc.groupby(kwargs['groups'])
    else:
        groups = df_rc.copy()

    # Order the group labels with natsorting
    gidx = []
    for i, (nn, g) in enumerate(groups):
        gidx += [nn]
    gidx = natsorted(gidx)

    # Make indices df
    indices = pd.DataFrame(gidx)
    num_groups = groups.ngroups
    changes = index_changes(indices, num_groups)
    num_groups = 0
    if kwargs['groups'] is not None:
        groups = df_rc.groupby(kwargs['groups'])
        num_groups = groups.ngroups
        num_groupby = len(kwargs['groups'])
        col = changes.columns

        # Plot the groups
        for i, nn in enumerate(gidx):
            gg = df_rc.copy().sort_values(by=kwargs['groups'])
            gg = gg.set_index(kwargs    ['groups'])
            if len(gg) > 1:
                gg = gg.loc[nn]
            if type(gg) == pd.Series:
                gg = pd.DataFrame(gg).T
            else:
                gg = gg.reset_index()
            temp = gg[dd.y].dropna()
            temp['x'] = i + 1
            data += [temp]
            means += [temp.mean()]
            medians += [temp.median()]
            if type(nn) is not tuple:
                nn = [nn]
            else:
                nn = [str(f) for f in nn]
            labels += ['']

            if len(changes.columns) > 1 and changes[col[-2]].iloc[i] == 1 \
                    and len(kwargs['groups']) > 1:
                dividers += [i+0.5]

            # Plot points
            if dd.legend_vals:
                for ileg, leg_name in enumerate(dd.legend_vals):
                    temp = gg.loc[gg[dd.legend==leg_item]][dd.y].dropna()
                    temp['x'] = i + 1
                    layout.plot_xy(ir, ic, ileg, temp, 'x', dd.y[0], leg_name, False)
            else:
                layout.plot_xy(ir, ic, 0, temp, 'x', dd.y[0], None, False)

            # point_curves = []
            # if kw['points'] and len(kw['leg_items'])==0:
            #     layout.add_points(ir, ic, x, y)

            #     add_points(ir, ic, i, temp)
            # else:
            #     for ileg, leg_item in enumerate(kw['leg_items']):
            #         temp = g.loc[g[kw['leg_groups']]==leg_item][y].dropna()
            #         kw['marker_type'] = markers[ileg]
            #         point_curves += add_points(i, temp, axes[ir, ic],
            #                                     palette[ileg+1], **kw)

    else:
        data = df_rc[dd.y].dropna()
        labels = ['']
        data['x'] = 1
        layout.plot_xy(ir, ic, 0, data, 'x', dd.y[0], None, False)

    if type(data) is pd.Series:
        data = data.values
    elif type(data) is pd.DataFrame and len(data.columns) == 1:
        data = data.values

    if len(data) > 0:
        st()


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

    for iline, df, x, y, leg_name, twin in data.get_plot_data(df_rc):
        if kwargs.get('groups', False):
            for nn, gg in df.groupby(utl.validate_list(kwargs['groups'])):
                layout.plot_xy(ir, ic, iline, gg, x, y, leg_name, twin)
        else:
            layout.plot_xy(ir, ic, iline, df, x, y, leg_name, twin)


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

    # Build the data object
    dd = Data(plot_func, **kwargs)

    # Iterate over discrete figures
    for ifig, fig_item, fig_cols, df_fig in dd.get_df_figure():
        # Create a layout object
        layout = LAYOUT[engine](plot_func, **kwargs)

        # Make the figure
        layout.make_figure(dd, **kwargs)

        # Make the subplots
        for ir, ic, df_rc in dd.get_rc_subset(df_fig):

            if len(df_rc) == 0:
                if dd.wrap is None:
                    layout.set_axes_rc_labels(ir, ic)
                layout.axes.obj[ir, ic].axis('off')
                continue

            # Set the axes colors
            layout.set_axes_colors(ir, ic)

            # Add and format gridlines
            layout.set_axes_grid_lines(ir, ic)

            # Add horizontal and vertical lines
            layout.add_hvlines(ir, ic)

            # Plot the data
            globals()[plot_func](dd, layout, ir, ic, df_rc, kwargs)

            # Set linear or log axes scaling
            layout.set_axes_scale(ir, ic)

            # Set axis ranges
            layout.set_axes_ranges(ir, ic, dd.ranges[ir, ic])

            # Add axis labels
            layout.set_axes_labels(ir, ic)

            # Add rc labels
            layout.set_axes_rc_labels(ir, ic)

            # Adjust tick marks
            layout.set_axes_ticks(ir, ic)

        # Make the legend
        layout.add_legend()

        # Add a figure title
        layout.set_figure_title()

        # Build the save filename
        filename = utl.set_save_filename(df_fig, fig_item, fig_cols,
                                         layout, kwargs)

        # Save and optionally open
        if kwargs.get('save', True):
            layout.fig.obj.savefig(filename)

            if kwargs.get('show', False):
                os.startfile(filename)

        # Return inline
        if not kwargs.get('inline', True):
            plt.close('all')
        else:
            print(filename)
            plt.show()#return layout.fig.obj
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
        print(filename)
        raise NameError('%s is not a valid filename!' % filename)


def set_theme(theme=None):
    """
    Select a "defaults" file and copy to the user directory
    """

    themes = [f for f in os.listdir(osjoin(cur_dir, 'themes')) if '.py' in f]

    if theme and '%s.py' % theme in themes:
        entry = themes.index('%s.py' % theme) + 1

    else:
        print('Select default styling theme:')
        for i, th in enumerate(themes):
            print('   %s) %s' % (i+1, th))
        entry = input('Entry: ')


        print('Copying %s >> %s...' %
              (themes[int(entry)-1],
               osjoin(user_dir, '.fivecentplots', 'defaults.py')), end='')

    if os.path.exists(osjoin(user_dir, '.fivecentplots', 'defaults.py')):
        print('Previous theme file found! Renaming to "defaults_old.py" and '
              'copying new theme...', end='')
        shutil.copy2(osjoin(user_dir, '.fivecentplots', 'defaults.py'),
                     osjoin(user_dir, '.fivecentplots', 'defaults_old.py'))

    if not os.path.exists(osjoin(user_dir, '.fivecentplots')):
        os.makedirs(osjoin(user_dir, '.fivecentplots'))
    shutil.copy2(osjoin(cur_dir, 'themes', themes[int(entry)-1]),
                 osjoin(user_dir, '.fivecentplots', 'defaults.py'))

    print('done!')


