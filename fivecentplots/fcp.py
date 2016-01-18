import matplotlib as mpl
import matplotlib.patches as patches
from fivecentplots.defaults import *
from fivecentplots.design import FigDesign
import os
import numpy as np
import pdb
import re
import itertools
st = pdb.set_trace


def add_curves(plotter, x, y, color='k', marker='o', points=False, line=True,
               **kwargs):
    """

    Args:
        plotter:
        x:
        y:
        color:
        marker:
        points:
        line:
        **kwargs:

    Returns:

    """

    if points:
        points = plotter(x, y, color=color, marker=marker,
                         markerfacecolor='none', markeredgecolor=color,
                         markeredgewidth=1.5, **kwargs)
    if line:
        lines = plotter(x, y, color=color, **kwargs)
    if points:
        return points
    else:
        return lines


def add_label(label, pos, axis, rotation, **kwargs):
    """ Add a label to the plot

    This function can be used for title labels or for group labels applied
    to rows and columns when plotting facet grid style plots.

    Args:
        label (str):
        pos (tuple): left, bottom, width, height
        axis (matplotlib.axes):
        rotation (int):
        **kwargs:  'rc_label_' kwargs values

    Returns:
        None
    """
    if kwargs['rc_label_fill_color']:
        filled = True

    # Define the label background
    rect = patches.Rectangle((pos[0], pos[1]), pos[2], pos[3],
                             fill=filled, transform=axis.transAxes,
                             facecolor=kwargs['rc_label_fill_color'],
                             edgecolor=kwargs['rc_label_edge_color'],
                             clip_on=False, zorder=-1)
    axis.add_patch(rect)

    # Add the label text
    axis.text(pos[0]+pos[2]/2, pos[1]+pos[3]/2, label,
              transform=axis.transAxes,
              horizontalalignment='center',
              verticalalignment='center',
              rotation=rotation,
              color=kwargs['rc_label_text_color'],
              weight=kwargs['rc_label_text_style'],
              fontsize=kwargs['rc_label_font_size'],
              )


def df_filter(df, filt):
    """  Filter a DataFrame

    Due to limitations in pd.query, column names must not have spaces.  This
    function will temporarily replace spaces in the column names with
    underscores, but the supplied query string must contain column names
    without any spaces

    Args:
        df (DataFrame):  data set to be filtered
        filt (str):  query expression for filtering

    Returns:
        filtered DataFrame
    """

    # Remove spaces from
    cols_orig = df.columns
    cols_new = [f.replace(' ', '_')
                 .replace('[', '')
                 .replace(']', '')
                 .replace('(', '')
                 .replace(')', '')
                for f in cols_orig]
    df.columns = cols_new

    # Apply the filter
    df = df.query(filt)

    # Reset the columns
    df.columns = cols_orig

    return df


def filename_label(label):
    """

    Args:
        label:

    Returns:

    """

    label = re.sub('\(.*?\)','', label)
    label = re.sub('\[.*?\]','', label)
    label = label.replace(' ','')

    return label


def plot(df, x, y, **kwargs):
    '''
    New plotting function
    :param df: a pandas DataFrame
    :param x: str, name of x-axis values
    :param y: str, name of y-axis values
    :param kwargs:  see below
    :Keyword Arguments:
        * ax_leg_ws:
        * ax_hlines:
        * ax_lim:
        * ax_vlines:
        * cmap:
        * fig_groups:
        * fs_ax_label:
        * fs_rc_groups:
        * fs_ticks:
        * grid_major:
        * grid_minor:
        * leg_groups:
        * leg_items:
        * leg_on:
        * leg_title:
        * line_color:
        * line_style:
        * line_width:
        * marker_size:
        * marker_type:
        * points:
        * prunex:
        * pruney:
        * rc_groups:
        * rc_groups_label_padding:
        * rc_groups_label_size:
        * rc_labels:
        * save_path:
        * sci_x:
        * sci_y:
        * twinx:
        * xlabel:
        * xmax:
        * xmin:
        * xticks:
        * ylabel:
        * ymax:
        * ymin:
        * yticks:
    :return: None
    '''

    # Keyword-argument definition
    kw = {}
    kw['ax_hlines'] = kwargs.get('ax_hlines', [])
    kw['ax_lim'] = kwargs.get('ax_lim', [])
    kw['ax_scale'] = kwargs.get('ax_scale', None)
    kw['ax_size'] = kwargs.get('ax_size', fcp_params['ax_size'])
    kw['ax_vlines'] = kwargs.get('ax_vlines', [])
    kw['cmap'] = kwargs.get('cmap', None)
    kw['col'] = kwargs.get('col', None)
    kw['col_label'] = kwargs.get('col_label', None)
    kw['col_labels_on'] = kwargs.get('col_labels_on', True)
    kw['col_label_size'] = kwargs.get('col_label_size',
                                      fcp_params['rc_label_size'])
    kw['col_label_ws'] = kwargs.get('col_label_ws', fcp_params['rc_label_ws'])
    kw['col_padding'] = kwargs.get('row_padding', fcp_params['col_padding'])
    kw['fig_groups'] = kwargs.get('fig_groups', None)
    kw['filter'] = kwargs.get('filter', None)
    kw['fs_ax_label'] = kwargs.get('fs_ax_label', fcp_params['fs_ax_label'])
    kw['fs_rc_groups'] = kwargs.get('fs_rc_groups', fcp_params['fs_rc_groups'])
    kw['fs_ticks'] = kwargs.get('fs_ticks', fcp_params['fs_ticks'])
    kw['grid_major'] = kwargs.get('grid_major', True)
    kw['grid_minor'] = kwargs.get('grid_minor', False)
    kw['label_style'] = kwargs.get('label_style', 'italic')
    kw['leg_bkgrd'] = kwargs.get('leg_bkgrd', fcp_params['leg_bkgrd'])
    kw['leg_border'] = kwargs.get('leg_border', fcp_params['leg_border'])
    kw['leg_groups'] = kwargs.get('leg_groups', None)
    kw['leg_items'] = kwargs.get('leg_items', [])
    kw['leg_on'] = kwargs.get('leg_on', True)
    kw['leg_title'] = kwargs.get('leg_title', None)
    kw['line_color'] = kwargs.get('line_color', None)
    kw['line_style'] = kwargs.get('line_style', '-')
    kw['line_width'] = kwargs.get('line_width', 1)
    kw['marker_size'] = kwargs.get('marker_size', fcp_params['marker_size'])
    kw['marker_type'] = kwargs.get('marker_type')
    kw['points'] = kwargs.get('points', True)
    kw['prunex'] = kwargs.get('prunex', True)
    kw['pruney'] = kwargs.get('pruney', True)
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
                                     fcp_params['rc_label_size'])
    kw['row_label_ws'] = kwargs.get('row_label_ws', fcp_params['rc_label_ws'])
    kw['row_padding'] = kwargs.get('row_padding', fcp_params['row_padding'])
    kw['save_ext'] = kwargs.get('save_ext', 'png')
    kw['save_name'] = kwargs.get('save_name', None)
    kw['save_path'] = kwargs.get('save_path', None)
    kw['sci_x'] = kwargs.get('sci_x', False)
    kw['sci_y'] = kwargs.get('sci_y', False)
    kw['sharex'] = kwargs.get('sharex', True)
    kw['sharey'] = kwargs.get('sharey', True)
    kw['show'] = kwargs.get('show', True)
    kw['tick_font_size'] = kwargs.get('tick_font_size',
                                      fcp_params['tick_font_size'])
    kw['twinx'] = kwargs.get('twinx', False)
    kw['twiny'] = kwargs.get('twiny', False)
    kw['xlabel'] = kwargs.get('xlabel', x)
    kw['xmax'] = kwargs.get('xmax', None)
    kw['xmin'] = kwargs.get('xmin', None)
    kw['xticks'] = kwargs.get('xticks', None)
    kw['xtrans'] = kwargs.get('xtrans', None)
    kw['ylabel'] = kwargs.get('ylabel', y)
    kw['ymax'] = kwargs.get('ymax', None)
    kw['ymin'] = kwargs.get('ymin', None)
    kw['yticks'] = kwargs.get('yticks', None)
    kw['ytrans'] = kwargs.get('ytrans', None)
    kw['ylabel2'] = kwargs.get('ylabel2', y)
    kw['ymax2'] = kwargs.get('ymax2', None)
    kw['ymin2'] = kwargs.get('ymin2', None)
    kw['yticks2'] = kwargs.get('yticks2', None)
    kw['ytrans2'] = kwargs.get('ytrans2', None)

    # Filter the dataframe
    if kw['filter']:
        df = df_filter(df, kw['filter'])

    # Handle multiple y-values
    if type(y) is not list:
        y = list(y)
    if kw['twinx']:
        ax2 = []

    # Clear out existing plot buffer
    mpl.pyplot.close('all')

    # Set up the figure grouping and iterate (each value corresponds to a
    #  separate figure)
    if kw['fig_groups'] is not None:
        if type(kw['fig_groups']) is list:
            kw['fig_items'] = list(df.groupby(kw['fig_groups']).groups.keys())
        else:
            kw['fig_items'] = list(df[kw['fig_groups']].unique())
    else:
        kw['fig_items'] = [None]

    # Iterate over discrete figures
    for ifig, fig_item in enumerate(kw['fig_items']):

        # Make a data subset and filter
        df_fig = df.copy()

        if type(fig_item) is tuple:
            for ig, g in enumerate(fig_item):
                df_fig = df_fig[df_fig[kw['fig_groups'][ig]]==g]
        elif kw['fig_groups'] is not None:
            df_fig = df_fig[df_fig[kw['fig_groups']]==fig_item]

        # Set up the row grouping
        if kw['row'] is not None:
            rows = list(df_fig[kw['row']].unique())
            nrow = len(rows)
        else:
            rows = [None]
            kw['row_labels_on'] = False
            kw['row_label_size'] = 0
            nrow = 1
        kw['rows'] = nrow

        # Set up the column grouping
        if kw['col'] is not None:
            cols = list(df_fig[kw['col']].unique())
            ncol = len(cols)
        else:
            cols = [None]
            kw['col_labels_on'] = False
            kw['col_label_size'] = 0
            ncol = 1
        kw['cols'] = ncol

        # Set up the legend grouping
        if kw['leg_groups'] is not None and len(kw['leg_items'])==0:
            kw['leg_items'] = df_fig[kw['leg_groups']].unique()
            if len(y) > 1 and not kw['twinx']:
                temp = list(itertools.product(kw['leg_items'], y))
                kw['leg_items'] = ['%s: %s' % (f[0], f[1]) for f in temp]
        elif not kw['leg_groups']:
            kw['leg_items'] = []
        if not kw['leg_title']:
            kw['leg_title'] = kw['leg_groups']

        # Add padding if sharex|sharey
        if not kw['sharex'] or not kw['sharey']:
            if 'row_padding' not in kwargs.keys():
                kw['row_padding'] += kw['tick_font_size']
            if 'col_padding' not in kwargs.keys():
                kw['col_padding'] += kw['tick_font_size']

        # Format the figure dimensions
        design = FigDesign(**kw)

        # Make the figure and axes
        fig, axes = mpl.pyplot.subplots(nrow, ncol,
                                        figsize=[design.fig_w, design.fig_h],
                                        sharex=kw['sharex'],
                                        sharey=kw['sharey'],
                                        dpi=design.dpi)

        # Reformat the axes variable if it is only one plot
        if not type(axes) is np.ndarray:
            axes = np.array([axes])
        if len(axes.shape) == 1:
            if nrow == 1:
                axes = np.reshape(axes, (1, -1))
            else:
                axes = np.reshape(axes, (-1, 1))

        # Format the subplot spacing
        fig.subplots_adjust(
            left=design.left,
            right=design.right,
            bottom=design.bottom,
            top=design.top,
            hspace=1.0*design.row_padding/design.ax_h,
            wspace=1.0*design.col_padding/design.ax_w
        )

        # Handle colormaps
        if kw['cmap']=='jmp_spectral':
            cmap = jmp_spectral
        elif kw['cmap'] is not None:
            cmap = mpl.pyplot.get_cmap(kw['cmap'])

        # Make the plots by row and by column
        curves = []

        for ir, r in enumerate(rows):
            for ic, c in enumerate(cols):

                # Build the row/col filename labels
                if kw['row']:
                    fnrow = filename_label(kw['row'])
                if kw['col']:
                    fncol = filename_label(kw['col'])

                # Subset the data
                if r is not None and c is not None:
                    df_sub = df_fig[(df_fig[kw['row']]==r)&\
                                    (df_fig[kw['col']]==c)].copy()
                    rc_name = '_%s_by_%s' % (fnrow, fncol)
                elif r and not c:
                    df_sub = df_fig[(df_fig[kw['row']]==r)].copy()
                    rc_name = '_by_%s' % (fnrow)
                elif c and not r:
                    df_sub = df_fig[(df_fig[kw['col']]==c)].copy()
                    rc_name = '_by_%s' % (fncol)
                else:
                    df_sub = df_fig.copy()
                    rc_name = ''

                # Set the axes scale
                if kw['ax_scale'] == 'loglog':
                    plotter = axes[ir, ic].loglog
                elif kw['ax_scale'] == 'semilogx' or kw['ax_scale'] == 'logx':
                    plotter = axes[ir, ic].semilogx
                elif kw['ax_scale'] == 'semilogy' or kw['ax_scale'] == 'logy':
                    plotter = axes[ir, ic].semilogy
                else:
                    plotter = axes[ir, ic].plot

                # Apply any data transformations
                if kw['xtrans'] == 'abs':
                    df_sub.loc[:, x] = abs(df_sub[x])
                elif kw['xtrans'] == 'negative' or kw['xtrans'] == 'neg':
                    df_sub.loc[:, x] = -df_sub[x]
                elif kw['xtrans'] == 'inverse' or kw['xtrans'] == 'inv':
                    df_sub.loc[:, x] = 1/df_sub[x]
                elif type(kw['xtrans']) is tuple and kw['xtrans'][0] == 'pow':
                    df_sub.loc[:, x] = df_sub[x]**kw['xtrans'][1]
                ### these will fail if multiple y provided...need handling
                if kw['ytrans'] == 'abs':
                    df_sub.loc[:, y] = abs(df_sub[y])
                elif kw['ytrans'] == 'negative' or kw['ytrans'] == 'neg':
                    df_sub.loc[:, y] = -df_sub[y]
                elif kw['ytrans'] == 'inverse' or kw['ytrans'] == 'inv':
                    df_sub.loc[:, y] = 1/df_sub[y]
                elif type(kw['ytrans']) is tuple and kw['ytrans'][0] == 'pow':
                    df_sub.loc[:, y] = df_sub[y]**kw['ytrans'][1]

                # Add horizontal and vertical lines
                for h in kw['ax_hlines']:
                    if type(h) is tuple and len(h)==3:
                        axes[ir,ic].axhline(h[0], color=h[1], linestyle=h[2])
                    else:
                        axes[ir,ic].axhline(h, color='k', linewidth=0.5)

                for v in kw['ax_vlines']:
                    if type(v) is tuple and len(v)==3:
                        axes[ir,ic].axvline(v[0], color=v[1], linestyle=v[2])
                    else:
                        axes[ir,ic].axvline(v, color='k', linewidth=0.5)

                # Legend grouping
                if kw['leg_groups'] is None and not kw['twinx']:
                    for iy, yy in enumerate(y):

                        # Define color and marker types
                        color = \
                            kw['line_color'] if kw['line_color'] is not None \
                                             else palette[iy]
                        marker = \
                            kw['marker_type'] if kw['marker_type'] is not None\
                                              else markers[iy]
                        # Plot
                        curves += add_curves(plotter,
                                             df_sub[x][subset],
                                             df_sub[yy][subset],
                                             color,
                                             marker,
                                             True,
                                             markersize=kw['marker_size'],
                                             linestyle=kw['line_style'],
                                             linewidth=kw['line_width'])

                elif kw['leg_groups'] is None and kw['twinx']:

                    # Define color and marker types
                    color = \
                        kw['line_color'] if kw['line_color'] is not None \
                                         else palette[0:2]
                    marker = \
                        kw['marker_type'] if kw['marker_type'] is not None\
                                          else markers[0:2]
                    # Plot
                    curves += add_curves(plotter,
                                         df_sub[x][subset],
                                         df_sub[y[0]][subset],
                                         color[0],
                                         marker[0],
                                         True,
                                         markersize=kw['marker_size'],
                                         linestyle=kw['line_style'],
                                         linewidth=kw['line_width'])
                    ax2 += axes[ir, ic].twinx()
                    curves += add_curves(plotter,
                                         df_sub[x][subset],
                                         df_sub[y[1]][subset],
                                         color[1],
                                         marker[1],
                                         True,
                                         markersize=kw['marker_size'],
                                         linestyle=kw['line_style'],
                                         linewidth=kw['line_width'])

                # NEED TO HANDLE BELOW FOR TWINX
                # then need to add code to handle multiple labels, label
                # colors, etc.
                else:
                    for ileg, leg_group in enumerate(sorted(kw['leg_items'])):

                        # Define color and marker types
                        if kw['cmap']:
                            color = cmap((ileg+1)/(len(kw['leg_items'])+1))
                        else:
                            color = kw['line_color'] \
                                if kw['line_color'] is not None \
                                else palette[ileg]
                        marker = kw['marker_type'][ileg] \
                                 if kw['marker_type'] is not None \
                                 else markers[ileg]

                        # Subset the data by legend group and plot
                        group = leg_group.split(': ')[0]
                        yy = leg_group.split(': ')[1]
                        subset = df_sub[kw['leg_groups']]==group
                        curves += add_curves(plotter,
                                             df_sub[x][subset],
                                             df_sub[yy][subset],
                                             color,
                                             marker,
                                             True,
                                             markersize=kw['marker_size'],
                                             linestyle=kw['line_style'],
                                             linewidth=kw['line_width'])

                    # Add the legend (wrong indent??)
                    leg = fig.legend(curves,
                                     kw['leg_items'],
                                     loc='upper right',
                                     title=kw['leg_title'],
                                     bbox_to_anchor=(design.leg_right,
                                                     design.leg_top),
                                     numpoints=1,
                                     prop={'size':12})
                    leg.get_frame().set_facecolor(kw['leg_bkgrd'])
                    leg.get_frame().set_edgecolor(kw['leg_border'])

                # Adjust the tick marks
                axes[ir, ic].tick_params(axis='both', which='major',
                                         labelsize=kw['tick_font_size'])

                if not kw['sharex'] or not kw['sharey']:
                    # Turn off first and last tick marks if not sharing
                    xticks = axes[ir, ic].xaxis.get_major_ticks()
                    yticks = axes[ir, ic].yaxis.get_major_ticks()
                    xticks[0].label1.set_visible(False)
                    yticks[0].label1.set_visible(False)
                    xticks[-1].label1.set_visible(False)
                    yticks[-1].label1.set_visible(False)

                else:
                    if kw['prunex']:
                        xticks = axes[ir, ic].xaxis.get_major_ticks()
                        xticks[0].label1.set_visible(False)
                    if kw['pruney']:
                        yticks = axes[ir, ic].yaxis.get_major_ticks()
                        yticks[0].label1.set_visible(False)

                # Axis ranges
                if kw['xmin']:
                    axes[ir, ic].set_xlim(left=kw['xmin'])
                if kw['xmax']:
                    axes[ir, ic].set_xlim(right=kw['xmax'])
                if kw['ymin']:
                    axes[ir, ic].set_ylim(bottom=kw['ymin'])
                if kw['ymax']:
                    axes[ir, ic].set_ylim(top=kw['ymax'])

                # Try to deal with overlapping labels (needs debug)
                if kw['xticks']:
                    x_tick_max = kw['xticks']
                else:
                    x_tick_max = round(kw['ax_size'][0]/kw['tick_font_size']/3)
                xloc = mpl.pyplot.MaxNLocator(x_tick_max)
                axes[ir, ic].xaxis.set_major_locator(xloc)

                if kw['yticks']:
                    y_tick_max = kw['yticks']
                else:
                    y_tick_max = round(kw['ax_size'][1]/kw['tick_font_size']/3)
                yloc = mpl.pyplot.MaxNLocator(y_tick_max)
                axes[ir, ic].yaxis.set_major_locator(yloc)

                # Add labels
                if kw['xlabel'] is not None and ir == len(rows)-1:
                    axes[ir, ic].set_xlabel(kw['xlabel'],
                                            style=kw['label_style'])
                if kw['ylabel'] is not None and ic == 0:
                    axes[ir, ic].set_ylabel(kw['ylabel'],
                                            style=kw['label_style'])

                # Add row/column labels
                if ic == len(cols)-1 and kw['row_labels_on']:
                    if not kw['row_label']:
                        kw['row_label'] = kw['row']
                    add_label('%s=%s' % (kw['row_label'], r),
                              (design.row_label_left, 0,
                               design.row_label_width, 1),
                              axes[ir, ic], 270, **kw)

                if ir == 0 and kw['col_labels_on']:
                    if not kw['col_label']:
                        kw['col_label'] = kw['col']
                    add_label('%s=%s' % (kw['col_label'], c),
                              (0, design.col_label_bottom,
                               1, design.col_label_height),
                              axes[ir, ic], 0, **kw)

        if not kw['save_path']:
            save_path = 'fig' + '.' + kw['save_ext']

        else:
            save_path = filename_label(x) + '_vs_' + filename_label(y) + \
                        rc_name + '.' + kw['save_ext']


        fig.savefig(save_path)
        if kw['show']:
            # mpl.pyplot.show()
            os.startfile(save_path)

    return design
