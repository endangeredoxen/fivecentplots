import matplotlib as mpl
import matplotlib.patches as patches
from fivecentplots.defaults import *
from fivecentplots.design import FigDesign
import os
import numpy as np
import pdb
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
    kw['ax_vlines'] = kwargs.get('ax_vlines', [])
    kw['cmap'] = kwargs.get('cmap', None)
    kw['col'] = kwargs.get('col', None)
    kw['col_label'] = kwargs.get('col_label', None)
    kw['col_labels_on'] = kwargs.get('col_labels_on', True)
    kw['col_label_size'] = kwargs.get('col_label_size',
                                      fcp_params['rc_label_size'])
    kw['col_label_ws'] = kwargs.get('col_label_ws', fcp_params['rc_label_ws'])
    kw['fig_groups'] = kwargs.get('fig_groups', None)
    kw['fs_ax_label'] = kwargs.get('fs_ax_label', fcp_params['fs_ax_label'])
    kw['fs_rc_groups'] = kwargs.get('fs_rc_groups', fcp_params['fs_rc_groups'])
    kw['fs_ticks'] = kwargs.get('fs_ticks', fcp_params['fs_ticks'])
    kw['grid_major'] = kwargs.get('grid_major', True)
    kw['grid_minor'] = kwargs.get('grid_minor', False)
    kw['label_style'] = kwargs.get('label_style', 'italic')
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
    kw['prunex'] = kwargs.get('prunex', False)
    kw['pruney'] = kwargs.get('pruney', False)
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
    kw['save_path'] = kwargs.get('save_path', 'fig.png')
    kw['sci_x'] = kwargs.get('sci_x', False)
    kw['sci_y'] = kwargs.get('sci_y', False)
    kw['sharex'] = kwargs.get('sharex', True)
    kw['sharey'] = kwargs.get('sharey', True)
    kw['show'] = kwargs.get('show', True)
    kw['twinx'] = kwargs.get('twinx', False)
    kw['xlabel'] = kwargs.get('xlabel', x)
    kw['xmax'] = kwargs.get('xmax', None)
    kw['xmin'] = kwargs.get('xmin', None)
    kw['xticks'] = kwargs.get('xticks', None)
    kw['ylabel'] = kwargs.get('ylabel', y)
    kw['ymax'] = kwargs.get('ymax', None)
    kw['ymin'] = kwargs.get('ymin', None)
    kw['yticks'] = kwargs.get('yticks', None)

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
        elif not kw['leg_groups']:
            kw['leg_items'] = []
        if not kw['leg_title']:
            kw['leg_title'] = kw['leg_groups']

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

                # Subset the data
                if r is not None and c is not None:
                    df_sub = df_fig[(df_fig[kw['row']]==r)&\
                                    (df_fig[kw['col']]==c)]
                elif r and not c:
                    df_sub = df_fig[(df_fig[kw['row']]==r)]
                elif c and not r:
                    df_sub = df_fig[(df_fig[kw['col']]==c)]
                else:
                    df_sub = df_fig.copy()
                
                # Set the axes scale
                if kw['ax_scale'] == 'loglog':
                    plotter = axes[ir, ic].loglog
                elif kw['ax_scale'] == 'semilogx' or kw['ax_scale'] == 'logx':
                    plotter = axes[ir, ic].semilogx
                elif kw['ax_scale'] == 'semilogy' or kw['ax_scale'] == 'logy':
                    plotter = axes[ir, ic].semilogy
                else:
                    plotter = axes[ir, ic].plot

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
                if kw['leg_groups'] is None:

                    # Define color and marker types
                    color = kw['line_color'] if kw['line_color'] is not None \
                                             else palette[0]
                    marker = kw['marker_type'] if kw['marker_type'] is not \
                                                  None else markers[0]
                    # Plot
                    curves += add_curves(plotter,
                                         df_sub[x][subset],
                                         df_sub[y][subset],
                                         color,
                                         marker,
                                         True,
                                         markersize=kw['marker_size'],
                                         linestyle=kw['line_style'],
                                         linewidth=kw['line_width'])
                else:
                    leg_group_items = df_fig[kw['leg_groups']].unique()
                    for ileg, leg_group in enumerate(leg_group_items):

                        # Define color and marker types
                        if kw['cmap']:
                            color = cmap((ileg+1)/(len(leg_group_items)+1))
                        else:
                            color = kw['line_color'] \
                                if kw['line_color'] is not None \
                                else palette[ileg]
                        marker = kw['marker_type'][ileg] \
                                 if kw['marker_type'] is not None \
                                 else markers[ileg]

                        # Subset the data by legend group and plot
                        subset = df_sub[kw['leg_groups']]==leg_group
                        curves += add_curves(plotter,
                                             df_sub[x][subset],
                                             df_sub[y][subset],
                                             color,
                                             marker,
                                             True,
                                             markersize=kw['marker_size'],
                                             linestyle=kw['line_style'],
                                             linewidth=kw['line_width'])

                        # Add the legend
                        leg = fig.legend(curves,
                                         leg_group_items,
                                         loc='upper right',
                                         title=kw['leg_title'],
                                         bbox_to_anchor=(design.leg_right,
                                                         design.leg_top),
                                         numpoints=1,
                                         prop={'size':12})
                        leg.get_frame().set_facecolor('white')
                        leg.get_frame().set_edgecolor('black')

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




    fig.savefig(kw['save_path'])
    if kw['show']:
        # mpl.pyplot.show()
        os.startfile(kw['save_path'])
    return design
