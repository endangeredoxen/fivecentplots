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
import copy
import importlib
import itertools
import shutil
import datetime
import sys
from . data import Data
from . layout import LayoutMPL, LayoutBokeh
import utilities as utl
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


def contour(**kwargs):
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

    return plotter('plot_contour', **kwargs)


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
            ss = layout.box_connect.stat.lower()
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
                layout.plot_xy(ir, ic, 0, temp, 'x', dd.y[0], None, False,
                               zorder=10)

    else:
        data = [df_rc[dd.y].dropna()]
        labels = ['']
        data[0]['x'] = 1
        layout.plot_xy(ir, ic, 0, data[0], 'x', dd.y[0], None, False, zorder=10)

    # Remove temporary 'x' column
    for dat in data:
        del dat['x']

    if type(data) is pd.Series:
        data = data.values
    elif type(data) is pd.DataFrame and len(data.columns) == 1:
        data = data.values

    if len(data) > 0:  # needed?
        for id, dat in enumerate(data):
            # Range lines
            kwargs = layout.box_range_lines.kwargs.copy()
            layout.plot_line(ir, ic, id+1-0.2, dat.max().iloc[0],
                             x1=id+1+0.2, y1=dat.max().iloc[0], **kwargs)
            layout.plot_line(ir, ic, id+1-0.2, dat.min().iloc[0],
                             x1=id+1+0.2, y1=dat.min().iloc[0], **kwargs)
            kwargs['style'] = kwargs['style2']
            layout.plot_line(ir, ic, id+1, dat.min().iloc[0],
                             x1=id+1, y1=dat.max().iloc[0], **kwargs)

        # Add boxes
        bp = layout.plot_box(ir, ic, data, **kwargs)

        # Add divider lines
        if layout.box_dividers.on and len(dividers) > 0:
            layout.ax_vlines = copy.deepcopy(layout.box_dividers)
            layout.ax_vlines.values = dividers
            layout.ax_vlines.color = [layout.box_dividers.color] * len(dividers)
            layout.ax_vlines.style = [layout.box_dividers.style] * len(dividers)
            layout.ax_vlines.width = [layout.box_dividers.width] * len(dividers)
            layout.ax_vlines.alpha = [layout.box_dividers.alpha] * len(dividers)
            layout.add_hvlines(ir, ic)

        # Add mean/median connecting lines
        if layout.box_connect.on and len(stats) > 0:
            x = np.linspace(1, dd.ngroups, dd.ngroups)
            layout.plot_line(ir, ic, x, stats, **layout.box_connect.kwargs,)


def plot_conf_int(data, layout, df, x, y):
    """
    """

    if not layout.conf_int.on:
        return

    data.get_conf_int(df, x, y)

    # need to add the fill between


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

    for iline, df, x, y, z, leg_name, twin in data.get_plot_data(df_rc):
        layout.plot_contour(layout.axes.obj[ir, ic], df, x, y, z)


def plot_fit(data, layout, ir, ic, iline, df, x, y, twin):
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

    """

    if not layout.line_fit.on:
        return

    df, coeffs, rsq = data.get_fit_data(df, x, y)
    layout.plot_xy(ir, ic, iline, df, '%s Fit' % x, '%s Fit' %y,
                    None, twin, line_obj=layout.line_fit,
                    marker_disable=True)

    if layout.line_fit.eqn:
        eqn = 'y='
        for ico, coeff in enumerate(coeffs[0:-1]):
            if coeff > 0 and ico > 0:
                eqn += '+'
            if len(coeffs)-1 > 1:
                power = '^%s' % str(len(coeffs)-1)
            else:
                power = ''
            eqn += '%s*x%s' % (round(coeff,3), power)
        if coeffs[-1] > 0:
            eqn += '+'
        eqn += '%s' % round(coeffs[-1], 3)

        layout.add_text(ir, ic, eqn, 'line_fit')

    if layout.line_fit.rsq:
        offsety = (5 + layout.line_fit.font_size) / layout.axes.size[1]
        layout.add_text(ir, ic, 'R^2=%s' % round(rsq, 4), 'line_fit',
                        offsety=-offsety)


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

    for iline, df, x, y, z, leg_name, twin in data.get_plot_data(df_rc):
        if kwargs.get('groups', False):
            for nn, gg in df.groupby(utl.validate_list(kwargs['groups'])):
                layout.plot_xy(ir, ic, iline, gg, x, y, leg_name, twin)
                plot_fit(data, layout, ir, ic, iline, gg, x, y, twin)

        else:
            layout.plot_xy(ir, ic, iline, df, x, y, leg_name, twin)
            plot_fit(data, layout, ir, ic, iline, df, x, y, twin)

        plot_conf_int(data, layout, df, x, y)

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

            # Add box labels
            layout.add_box_labels(ir, ic, dd)

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


