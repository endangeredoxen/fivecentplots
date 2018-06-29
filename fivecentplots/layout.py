from . import fcp
import matplotlib as mpl
import matplotlib.pyplot as mplp
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator, LogLocator
import matplotlib.mlab as mlab
import importlib
import os, sys
import pandas as pd
import pdb
import datetime
import numpy as np
import copy
import fivecentplots.utilities as utl
from collections import defaultdict
import warnings
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return 'Warning: ' + str(msg) + '\n'

warnings.formatwarning = custom_formatwarning
warnings.filterwarnings("ignore", "invalid value encountered in double_scalars")  # weird error in boxplot with no groups

try:
    from natsort import natsorted
except:
    natsorted = sorted

st = pdb.set_trace

DEFAULT_COLORS = ['#4b72b0', '#c34e52', '#54a767', '#8172b1', '#64b4cc',
                  '#ccb973', '#fa8d62', '#8d9fca', '#a6d753', '#fed82f',
                  '#b3b2b2', '#8cd3c6', '#bfbbd9', '#f98174', '#b07b9d',
                  '#f2be5b', '#326438', '#3c5477', '#de7426', '#588281',
                  '#c22a64', '#57324e', '#948974', '#9dcbdf', '#6f6f6f',
                  '#b07b9d', '#f2be5b', '#326438', '#3c5477', '#de7426',
                  '#588281', '#c22a64', '#57324e', '#948974', '#9dcbdf',
                  '#6f6f6f', ]

DEFAULT_MARKERS = ['o', '+', 's', 'x', 'd', 'Z', '^', 'Y', 'v', '\infty',
                   '\#', '<', u'\u2B21', u'\u263A', '>', u'\u29C6', '\$',
                   u'\u2B14', u'\u2B1A', u'\u25A6', u'\u229E', u'\u22A0',
                   u'\u22A1', u'\u20DF', '\gamma', '\sigma', '\star',]

LEGEND_LOCATION = defaultdict(int,
                  {'outside': 0, # always "best" outside of the damn plot
                   'upper right': 1, 1: 1,
                   'upper left': 2, 2: 2,
                   'lower left': 3, 3: 3,
                   'lower right': 4, 4: 4,
                   'right': 5, 5: 5,
                   'center left': 6, 6: 6,
                   'center right': 7, 7: 7,
                   'lower center': 8, 8: 8,
                   'upper center': 9, 9: 9,
                   'center': 10, 10: 10})


def mplc_to_hex(color, alpha=True):
    """
    Convert mpl color to hex

    Args:
        color (tuple): matplotlib style color code
        alpha (boolean): include or exclude the alpha value
    """

    hexc = '#'
    for ic, cc in enumerate(color):
        if not alpha and ic == 3:
            continue
        hexc += '%s' % hex(int(cc * 255))[2:].zfill(2)

    return hexc


def mpl_get_ticks(ax, xon=True, yon=True):
    """
    Divine a bunch of tick and label parameters for mpl layouts

    Args:
        ax (mpl.axes)

    Returns:
        dict of x and y ax tick parameters

    """

    tp = {}
    xy = []
    if xon:
        xy += ['x']
    if yon:
        xy += ['y']

    for vv in xy:
        tp[vv] = {}
        tp[vv]['min'], tp[vv]['max'] = getattr(ax, 'get_%slim' % vv)()
        tp[vv]['ticks'] = getattr(ax, 'get_%sticks' % vv)()
        tp[vv]['labels'] = [f for f in getattr(ax, '%saxis' % vv).iter_ticks()]
        tp[vv]['label_text'] = [f[2] for f in tp[vv]['labels']]
        try:
            tp[vv]['first'] = [i for i, f in enumerate(tp[vv]['labels'])
                               if f[1] >= tp[vv]['min'] and f[2] != ''][0]
        except:
            tp[vv]['first'] = -999
        try:
            tp[vv]['last'] = [i for i, f in enumerate(tp[vv]['labels'])
                              if f[1] <= tp[vv]['max'] and f[2] != ''][-1]
        except:
            tp[vv]['last'] = -999

    return tp


class BaseLayout:
    def __init__(self, plot_func, **kwargs):
        """
        Generic layout properties class

        Args:
            **kwargs: styling, spacing kwargs

        """

        self.plot_func = plot_func

        # Reload default file
        self.fcpp, color_list, marker_list = utl.reload_defaults()

        # Figure
        self.fig = Element('fig', self.fcpp, kwargs)

        # Color list
        if 'line_color' in kwargs.keys():
            color_list = kwargs['line_color']
        elif kwargs.get('colors'):
            colors = utl.validate_list(kwargs.get('colors'))
            for icolor, color in enumerate(colors):
                if type(color) is int:
                    colors[icolor] = DEFAULT_COLORS[icolor]
            color_list = colors
        else:
            color_list = DEFAULT_COLORS
        self.cmap = kwargs.get('cmap', None)
        if 'contour' in self.plot_func:
            self.cmap = utl.kwget(kwargs, self.fcpp, 'cmap', None)

        # Axis
        self.ax = ['x', 'y', 'x2', 'y2']
        self.axes = Element('ax', self.fcpp, kwargs,
                            size=utl.kwget(kwargs, self.fcpp,
                                       'ax_size', [400, 400]),
                            edge_color='#aaaaaa',
                            fill_color='#eaeaea',
                            primary=True,
                            scale=kwargs.get('ax_scale', None),
                            share_x=kwargs.get('share_x', True),
                            share_y=kwargs.get('share_y', True),
                            share_z=kwargs.get('share_z', True),
                            share_x2=kwargs.get('share_x2', True),
                            share_y2=kwargs.get('share_y2', True),
                            share_col = kwargs.get('share_col', False),
                            share_row = kwargs.get('share_row', False),
                            twin_x=kwargs.get('twin_x', False),
                            twin_y=kwargs.get('twin_y', False),
                            )
        if self.axes.scale:
            self.axes.scale = self.axes.scale.lower()
        if self.axes.share_row or self.axes.share_col:
            self.axes.share_x = False
            self.axes.share_y = False

        twinned = kwargs.get('twin_x', False) or kwargs.get('twin_y', False)
        self.axes2 = Element('ax', self.fcpp, kwargs,
                             on=True if twinned else False,
                             edge_color=self.axes.edge_color,
                             fill_color=self.axes.fill_color,
                             primary=False,
                             scale=kwargs.get('ax2_scale', self.axes.scale),
                             xmin=kwargs.get('x2min', None),
                             xmax=kwargs.get('x2max', None),
                             ymin=kwargs.get('y2min', None),
                             ymax=kwargs.get('y2max', None),
                             )
        if self.axes2.scale:
            self.axes2.scale = self.axes2.scale.lower()

        # Axes labels
        label = Element('label', self.fcpp, kwargs,
                        font_style='italic',
                        font_weight='bold',
                        )
        labels = ['x', 'x2', 'y', 'y2', 'z']
        rotations = [0, 0, 90, 270, 270]
        for ilab, lab in enumerate(labels):
            # Copy base label object and set default rotation
            setattr(self, 'label_%s' % lab, copy.deepcopy(label))
            getattr(self, 'label_%s' % lab).rotation = rotations[ilab]

            # Override params
            keys = [f for f in kwargs.keys() if 'label_%s' % lab in f]
            for k in keys:
                setattr(getattr(self, 'label_%s' % lab),
                        k.replace('label_%s_' % lab, ''), kwargs[k])

        # Turn off secondary labels
        if not self.axes.twin_y:
            self.label_x2.on = False
        if not self.axes.twin_x:
            self.label_y2.on = False

        # Twinned label colors
        if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
            self.label_y.font_color = color_list[0]
        if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
            self.label_y2.font_color = color_list[1]
        if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
            self.label_x.font_color = color_list[0]
        if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
            self.label_x2.font_color = color_list[1]

        # Figure title
        title = kwargs.get('title', None)
        self.title = Element('title', self.fcpp, kwargs,
                             on=True if title else False,
                             text=title if title is not None else None,
                             font_color='#333333',
                             font_size=18,
                             font_weight='bold',
                             )
        if type(self.title.size) is not list:
            self.title.size = [self.axes.size[0], self.title.size]

        # Ticks
        if 'ticks' in kwargs.keys() and 'ticks_major' not in kwargs.keys():
            kwargs['ticks_major'] = kwargs['ticks']
        ticks_length = utl.kwget(kwargs, self.fcpp, 'ticks_length', 6)
        ticks_width = utl.kwget(kwargs, self.fcpp, 'ticks_width', 2.5)
        self.ticks_major = Element('ticks_major', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major', True),
                                   color=utl.kwget(kwargs, self.fcpp,
                                                   'ticks_major_color',
                                                   '#ffffff'),
                                   increment=utl.kwget(kwargs, self.fcpp,
                                                       'ticks_major_increment',
                                                       None),
                                   padding=utl.kwget(kwargs, self.fcpp,
                                                     'ticks_major_padding',
                                                     4),
                                   size=[utl.kwget(kwargs, self.fcpp,
                                                   'ticks_major_length',
                                                   ticks_length),
                                         utl.kwget(kwargs, self.fcpp,
                                                   'ticks_major_width',
                                                   ticks_width)],
                                   )
        kwargs = self.from_list(self.ticks_major,
                                ['color', 'increment', 'padding'],
                                'ticks_major', kwargs)
        for ia, ax in enumerate(self.ax):
            setattr(self, 'ticks_major_%s' %ax,
                    Element('ticks_major_%s' %ax, self.fcpp, kwargs,
                            on=utl.kwget(kwargs, self.fcpp,
                                         'ticks_major_%s' % ax, self.ticks_major.on),
                            color=utl.kwget(kwargs, self.fcpp,
                                            'ticks_major_color_%s' % ax,
                                            self.ticks_major.color),
                            increment=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major_increment_%s' % ax,
                                                self.ticks_major.increment),
                            padding=utl.kwget(kwargs, self.fcpp,
                                              'ticks_major_padding_%s' % ax,
                                              self.ticks_major.padding),
                            size=self.ticks_major.size,
                            ))

        if 'tick_labels' in kwargs.keys() \
                and 'tick_labels_major' not in kwargs.keys():
            kwargs['tick_labels_major'] = kwargs['tick_labels']
        self.tick_labels_major = \
            Element('tick_labels_major', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_major',
                                 kwargs.get('tick_labels', True)),
                    font_size=13,
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_major_padding', 4),
                    )
        kwargs = self.from_list(self.tick_labels_major,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_major', kwargs)
        for ax in self.ax + ['z']:
            setattr(self, 'tick_labels_major_%s' %ax,
                    Element('tick_labels_major_%s' %ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_major_%s' % ax,
                                     self.tick_labels_major.on),
                        font=self.tick_labels_major.font,
                        font_color=self.tick_labels_major.font_color,
                        font_size=self.tick_labels_major.font_size,
                        font_style=self.tick_labels_major.font_style,
                        font_weight=self.tick_labels_major.font_weight,
                        padding=utl.kwget(kwargs, self.fcpp,
                                          'tick_labels_major_padding_%s' % ax,
                                          self.tick_labels_major.padding),
                        rotation=self.tick_labels_major.rotation,
                        size=[0, 0],
                        sci=utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, False) or \
                            utl.kwget(kwargs, self.fcpp, 'ticks_major_sci_%s' % ax, False),
                        ))

        if 'ticks' in kwargs.keys() and 'ticks_minor' not in kwargs.keys():
            kwargs['ticks_minor'] = kwargs['ticks']
        self.ticks_minor = Element('ticks_minor', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp,
                                                'ticks_minor', False),
                                   color=utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_color',
                                                   '#ffffff'),
                                   number=utl.kwget(kwargs, self.fcpp,
                                                    'ticks_minor_number',
                                                    3),
                                   padding=utl.kwget(kwargs, self.fcpp,
                                                     'ticks_minor_padding',
                                                     4),
                                   size=[utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_length_minor',
                                                   ticks_length*0.67),
                                         utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_width',
                                                   ticks_width*0.7)],
                                   )
        kwargs = self.from_list(self.ticks_minor,
                                ['color', 'number', 'padding'],
                                'ticks_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'ticks_minor_%s' % ax,
                    Element('ticks_minor_%s' % ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'ticks_minor_%s' % ax, self.ticks_minor.on),
                        color=utl.kwget(kwargs, self.fcpp,
                                        'ticks_minor_color_%s' % ax,
                                        self.ticks_minor.color),
                        number=utl.kwget(kwargs, self.fcpp,
                                         'ticks_minor_number_%s' % ax,
                                         self.ticks_minor.number),
                        padding=utl.kwget(kwargs, self.fcpp,
                                          'ticks_minor_padding_%s' % ax,
                                          self.ticks_minor.padding),
                        size=self.ticks_minor._size,
                        ))

        if 'tick_labels' in kwargs.keys() and 'tick_labels_minor' not in kwargs.keys():
            kwargs['tick_labels_minor'] = kwargs['tick_labels']
        self.tick_labels_minor = \
            Element('tick_labels_minor', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_minor', False),
                    font_size=10,
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_minor_padding', 3),
                    )
        kwargs = self.from_list(self.tick_labels_minor,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'tick_labels_minor_%s' %ax,
                    Element('tick_labels_minor_%s' %ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_minor_%s' % ax,
                                     self.tick_labels_minor.on),
                        font=self.tick_labels_minor.font,
                        font_color=self.tick_labels_minor.font_color,
                        font_size=self.tick_labels_minor.font_size,
                        font_style=self.tick_labels_minor.font_style,
                        font_weight=self.tick_labels_minor.font_weight,
                        padding=self.tick_labels_minor.padding,
                        rotation=self.tick_labels_minor.rotation,
                        size=[0, 0],
                        ))

        # Boxplot labels
        self.box_group_title = Element('box_group_title', self.fcpp, kwargs,
                                      on=True if 'box' in self.plot_func and kwargs.get('box_labels_on', True) else False,
                                      font_color='#666666',
                                      font_size=12,
                                      )
        self.box_group_label = Element('box_group_label', self.fcpp, kwargs,
                                       align={},
                                       on=True if 'box' in self.plot_func and kwargs.get('box_labels_on', True) else False,
                                       edge_color='#aaaaaa',
                                       font_color='#666666',
                                       font_size=13,
                                       padding=15,  # percent
                                       rotation=0,
                                       )

        # Other boxplot elements
        self.box = Element('box', self.fcpp, kwargs,
                           on=True if 'box' in self.plot_func and kwargs.get('box_on', True) else False,
                           edge_color='#4b72b0',
                           median_color=utl.kwget(kwargs, self.fcpp,
                                                 'box_median_line_color',
                                                 '#ff7f0e'),
                           notch=utl.kwget(kwargs, self.fcpp, 'box_notch', False),
                           )
        self.box_stat_line = \
            Element('box_stat_line', self.fcpp, kwargs,
                    on=True if 'box' in self.plot_func and \
                        kwargs.get('box_stat_line', True) else False,
                    color='#666666',
                    stat=kwargs.get('box_stat_line', 'mean'),
                    zorder=utl.kwget(kwargs, self.fcpp,
                                     'box_stat_line_zorder', 7),
                    )

        self.box_divider = Element('box_divider', self.fcpp, kwargs,
                                   on=kwargs.get('box_divider', kwargs.get('box', True)),
                                   color='#bbbbbb',
                                   zorder=2,
                                   )
        self.box_range_lines = Element('box_range_lines', self.fcpp, kwargs,
                                       on=kwargs.get('box_range_lines', kwargs.get('box', True)),
                                       color='#cccccc',
                                       style='-',
                                       style2='--',
                                       zorder=utl.kwget(kwargs, self.fcpp,
                                                        'box_range_lines',
                                                        3),
                                       )

        # Legend
        kwargs['legend'] = kwargs.get('legend', None)
        if type(kwargs['legend']) is str:
            kwargs['legend'] = ' | '.join(utl.validate_list(kwargs['legend']))

        self.legend = DF_Element('legend', self.fcpp, kwargs,
                                 on=True if (kwargs.get('legend') or
                                    kwargs.get('legend_on', True)) else False,
                                 column=kwargs['legend'],
                                 font_size=12,
                                 location=LEGEND_LOCATION[utl.kwget(kwargs,
                                          self.fcpp, 'legend_location', 0)],
                                 marker_size=utl.kwget(kwargs, self.fcpp,
                                                       'legend_marker_size',
                                                       None),
                                 points=utl.kwget(kwargs, self.fcpp,
                                                  'legend_points', 1),
                                 overflow=0,
                                 text=kwargs.get('legend_title',
                                                 kwargs.get('legend') if kwargs.get('legend') != True else ''),
                                 values={} if not kwargs.get('legend') else {'NaN': None},
                                 )

        # Color bar
        cbar_size = utl.kwget(kwargs, self.fcpp, 'cbar_size', 30)
        self.cbar = Element('cbar', self.fcpp, kwargs,
                            on=kwargs.get('cbar', False),
                            size=[cbar_size if type(cbar_size) is not list else cbar_size[0],
                                  self.axes.size[1]],
                            title='',
                            )
        if not self.cbar.on:
            self.label_z.on = False
            self.tick_labels_major_z.on = False

        # Contours
        self.contour = Element('contour', self.fcpp, kwargs,
                               on=True,
                               cmap=utl.kwget(kwargs, self.fcpp,
                                              'cmap', 'inferno'),
                               filled=utl.kwget(kwargs, self.fcpp,
                                                'filled', True),
                               levels=utl.kwget(kwargs, self.fcpp,
                                                'levels', 20),
                              )

        # Line fit
        self.line_fit = Element('line_fit', self.fcpp, kwargs,
                                on=True if kwargs.get('fit', False) else False,
                                color=kwargs.get('fit_color', RepeatedList(['k'], 'line_colors')),
                                eqn=kwargs.get('fit_eqn', False),
                                font_size=12,
                                padding=kwargs.get('fit_padding', 10),
                                rsq=kwargs.get('fit_rsq', False),
                                size=[0,0],
                                )
        self.line_fit.position[0] = self.line_fit.padding / self.axes.size[0]
        self.line_fit.position[3] = 1 - (self.line_fit.padding + \
                                    self.line_fit.font_size)/ self.axes.size[1]

        # Lines
        line_colors = RepeatedList(color_list, 'line_colors')
        self.lines = Element('lines', self.fcpp, kwargs,
                             on=kwargs.get('lines', True),
                             color=copy.copy(line_colors),
                             values=[],
                             )

        # Markers/points
        if 'marker_type' in kwargs.keys():
            marker_list = kwargs['marker_type']
        else:
            marker_list = utl.validate_list(kwargs.get('markers', DEFAULT_MARKERS))
        markers = RepeatedList(marker_list, 'markers')
        marker_colors = utl.validate_list(kwargs.get('marker_colors', color_list))
        marker_edge_color = utl.kwget(kwargs, self.fcpp, 'marker_edge_color', None)
        marker_edge_color = marker_colors if marker_edge_color is None else marker_edge_color
        marker_edge_color = RepeatedList(marker_edge_color, 'marker_edge_color')
        marker_fill_color = utl.kwget(kwargs, self.fcpp, 'marker_fill_color', None)
        marker_fill_color = marker_colors if marker_fill_color is None else marker_fill_color
        marker_fill_color = RepeatedList(marker_fill_color, 'marker_fill_color')
        self.markers = Element('marker', self.fcpp, kwargs,
                               on=utl.kwget(kwargs, self.fcpp,
                                            'markers', True),
                               filled=utl.kwget(kwargs, self.fcpp,
                                                'marker_fill', False),
                               edge_color=marker_edge_color,
                               edge_width=utl.kwget(kwargs, self.fcpp,
                                                    'marker_edge_width',
                                                     1.5),
                               fill_color=marker_fill_color,
                               jitter=utl.kwget(kwargs, self.fcpp,
                                                'jitter', False),
                               size=utl.kwget(kwargs, self.fcpp,
                                              'marker_size', 7),
                               type=markers,
                               zorder=utl.kwget(kwargs, self.fcpp,
                                                'zorder', 2),
                               )
        if 'box' in self.plot_func:
            self.lines.on = False
            marker_edge_color = utl.kwget(kwargs, self.fcpp, 'box_marker_edge_color', None)
            marker_edge_color = color_list if marker_edge_color is None else marker_edge_color
            marker_edge_color = RepeatedList(marker_edge_color, 'marker_edge_color')
            if not kwargs.get('colors'):
                marker_edge_color.shift = 1
            marker_fill_color = utl.kwget(kwargs, self.fcpp, 'marker_fill_color', None)
            marker_fill_color = color_list if marker_fill_color is None else marker_fill_color
            marker_fill_color = RepeatedList(marker_fill_color, 'marker_fill_color')
            if not kwargs.get('colors'):
                marker_fill_color.shift = 1
            self.markers.filled = utl.kwget(kwargs, self.fcpp,
                                            'box_marker_fill',
                                            self.markers.filled)
            self.markers.edge_color = marker_edge_color
            self.markers.edge_alpha = utl.kwget(kwargs, self.fcpp,
                                                'box_edge_alpha',
                                                self.markers.edge_alpha)
            self.markers.edge_width = utl.kwget(kwargs, self.fcpp,
                                                'box_edge_width',
                                                self.markers.edge_width)
            self.markers.fill_color = marker_fill_color
            self.markers.fill_alpha = utl.kwget(kwargs, self.fcpp,
                                                'box_marker_fill_alpha',
                                                self.markers.fill_alpha)
            self.markers.jitter = utl.kwget(kwargs, self.fcpp,
                                            'jitter', True)
            self.markers.size = utl.kwget(kwargs, self.fcpp,
                                          'box_marker_size',
                                          kwargs.get('marker_size', 4))
            self.markers.type = utl.kwget(kwargs, self.fcpp,
                                          'box_marker_type', self.markers.type)
            self.markers.zorder = utl.kwget(kwargs, self.fcpp,
                                            'box_marker_zorder',
                                            self.markers.zorder)

        # Axhlines/axvlines
        axlines = ['ax_hlines', 'ax_vlines', 'yline',
                   'ax2_hlines', 'ax2_vlines', 'y2line']
        # Todo: list
        for axline in axlines:
            val = kwargs.get(axline, False)
            vals = utl.validate_list(val)
            colors = []
            styles = []
            widths = []
            alphas = []
            for ival, val in enumerate(vals):
                if type(val) is list or type(val) is tuple and len(val) > 1:
                    colors += [val[1]]
                else:
                    colors += [utl.kwget(kwargs, self.fcpp, '%s_color' % axline, '#000000')]
                if type(val) is list or type(val) is tuple and len(val) > 2:
                    styles += [val[2]]
                else:
                    styles += [utl.kwget(kwargs, self.fcpp, '%s_style' % axline, '-')]
                if type(val) is list or type(val) is tuple and len(val) > 3:
                    widths += [val[3]]
                else:
                    widths += [utl.kwget(kwargs, self.fcpp, '%s_width' % axline, 1)]
                if type(val) is list or type(val) is tuple and len(val) > 4:
                    alphas += [val[4]]
                else:
                    alphas += [utl.kwget(kwargs, self.fcpp, '%s_alpha' % axline, 1)]

            setattr(self, axline,
                    Element(axline, self.fcpp, kwargs,
                            on=True if axline in kwargs.keys() else False,
                            values=vals, color=colors, style=styles,
                            width=widths, alpha=alphas,
                            zorder=utl.kwget(kwargs, self.fcpp, '%s_zorder' % axline, 1),
                            ))

        # Gridlines
        self.grid_major = Element('grid_major', self.fcpp, kwargs,
                                  on=kwargs.get('grid_major', True),
                                  color=utl.kwget(kwargs, self.fcpp,
                                                  'grid_major_color',
                                                  '#ffffff'),
                                  width=1.3,
                                  )

        for ax in ['x', 'y']:
            # secondary axes cannot get the grid
            setattr(self, 'grid_major_%s' %ax,
                    Element('grid_major_%s' %ax, self.fcpp, kwargs,
                            on=kwargs.get('grid_major_%s' % ax,
                                          self.grid_major.on),
                            color=self.grid_major.color,
                            style=self.grid_major.style,
                            width=self.grid_major.width,
                            ))
            if getattr(getattr(self, 'grid_major_%s' % ax), 'on') and \
                    ('ticks' not in kwargs.keys() or kwargs['ticks'] != False) and \
                    ('ticks_%s' % ax not in kwargs.keys() or
                     kwargs['ticks_%s' % ax] != False) and \
                    ('ticks_major' not in kwargs.keys() or
                     kwargs['ticks_major'] != False) and \
                    ('ticks_major_%s' % ax not in kwargs.keys() or \
                     kwargs['ticks_major_%s' % ax] != False):
                setattr(getattr(self, 'ticks_major_%s' % ax), 'on', True)

        self.grid_minor = Element('grid_minor', self.fcpp, kwargs,
                                  on=kwargs.get('grid_minor', False),
                                  color='#ffffff',
                                  width=0.5,
                                  )
        if self.grid_minor.on and \
                ('ticks' not in kwargs.keys() or kwargs['ticks'] != False) and \
                ('ticks_minor' not in kwargs.keys() or kwargs['ticks_minor'] != False):
            self.ticks_minor.on = True
        for ax in ['x', 'y']:
            # secondary axes cannot get the grid
            setattr(self, 'grid_minor_%s' %ax,
                    Element('grid_minor_%s' %ax, self.fcpp, kwargs,
                            on=kwargs.get('grid_minor_%s' % ax,
                                          self.grid_minor.on),
                            color=utl.kwget(kwargs, self.fcpp,
                                            'grid_minor_color_%s' % ax,
                                            self.grid_minor.color),
                            style=self.grid_minor.style,
                            width=self.grid_minor.width,
                            ))


        # Row column label
        rc_label = DF_Element('rc_label', self.fcpp, kwargs,
                              on=True,
                              size=utl.kwget(kwargs, self.fcpp,
                                             'rc_label_size', 30),
                              edge_color='#8c8c8c',
                              fill_color='#8c8c8c',
                              font_color='#ffffff',
                              font_size=16,
                              font_weight='bold',
                              )
        self.label_row = copy.deepcopy(rc_label)
        self.label_row.on = \
            utl.kwget(kwargs, self.fcpp, 'row_label_on', True) if kwargs.get('row') else False
        self.label_row.column = kwargs.get('row')
        self.label_row.size = [utl.kwget(kwargs, self.fcpp,
                                         'row_label_size', rc_label._size),
                               self.axes.size[1]]
        self.label_row.text_size = None
        self.label_row.edge_color = utl.kwget(kwargs, self.fcpp,
                                              'row_label_edge_color',
                                              rc_label.edge_color)
        self.label_row.edge_alpha = utl.kwget(kwargs, self.fcpp,
                                              'row_label_edge_alpha',
                                              rc_label.edge_alpha)
        self.label_row.edge_width = utl.kwget(kwargs, self.fcpp,
                                              'row_label_edge_width',
                                              rc_label.edge_width)
        self.label_row.fill_color = utl.kwget(kwargs, self.fcpp,
                                              'row_label_fill_color',
                                              rc_label.fill_color)
        self.label_row.font_color = utl.kwget(kwargs, self.fcpp,
                                              'row_label_font_color',
                                              rc_label.font_color)
        self.label_row.rotation = 270

        self.label_col = copy.deepcopy(rc_label)
        self.label_col.on = \
            utl.kwget(kwargs, self.fcpp, 'col_label_on', True) if kwargs.get('col') else False
        self.label_row.column = kwargs.get('col')
        self.label_col.size = [self.axes.size[0],
                               utl.kwget(kwargs, self.fcpp,
                                         'col_label_size', rc_label._size)]
        self.label_col.text_size = None
        self.label_col.edge_color = utl.kwget(kwargs, self.fcpp,
                                              'col_label_edge_color',
                                              rc_label.edge_color)
        self.label_col.edge_width = utl.kwget(kwargs, self.fcpp,
                                              'col_label_edge_width',
                                              rc_label.edge_width)
        self.label_col.edge_alpha = utl.kwget(kwargs, self.fcpp,
                                              'col_label_edge_alpha',
                                              rc_label.edge_alpha)
        self.label_col.fill_color = utl.kwget(kwargs, self.fcpp,
                                              'col_label_fill_color',
                                              rc_label.fill_color)
        self.label_col.font_color = utl.kwget(kwargs, self.fcpp,
                                              'col_label_font_color',
                                              rc_label.font_color)
        # Wrap label
        self.label_wrap = DF_Element('wrap_label', self.fcpp, kwargs,
                                     on=utl.kwget(kwargs, self.fcpp,
                                                  'wrap_label_on', True)
                                                   if kwargs.get('wrap') else False,
                                     column=kwargs.get('wrap'),
                                     size=[self.axes.size[0],
                                           utl.kwget(kwargs, self.fcpp,
                                           'wrap_label_size', 30)],
                                     edge_color=rc_label.edge_color,
                                     edge_width=rc_label.edge_width,
                                     edge_alpha=rc_label.edge_alpha,
                                     fill_color=rc_label.fill_color,
                                     fill_alpha=rc_label.fill_alpha,
                                     font=rc_label.font,
                                     font_color=rc_label.font_color,
                                     font_size=rc_label.font_size,
                                     font_style=rc_label.font_style,
                                     font_weight=rc_label.font_weight,
                                     text_size=None,
                                     )

        if type(self.label_wrap.size) is not list:
            self.label_wrap.size = [self.label_wrap.size, self.axes.size[1]]

        self.title_wrap = Element('wrap_title', self.fcpp, kwargs,
                                  on=utl.kwget(kwargs, self.fcpp,
                                               'wrap_title_on', True)
                                               if kwargs.get('wrap') else False,
                                  size=utl.kwget(kwargs, self.fcpp,
                                                 'wrap_title_size',
                                                 rc_label.size),
                                  edge_color='#5f5f5f',
                                  edge_width=rc_label.edge_width,
                                  edge_alpha=rc_label.edge_alpha,
                                  fill_color='#5f5f5f',
                                  fill_alpha=rc_label.fill_alpha,
                                  font=rc_label.font,
                                  font_color=rc_label.font_color,
                                  font_size=rc_label.font_size,
                                  font_style=rc_label.font_style,
                                  font_weight=rc_label.font_weight,
                                  text=kwargs.get('wrap_title', None),
                                  )

        if type(self.title_wrap.size) is not list:
            self.title_wrap.size = [self.axes.size[0], self.title_wrap.size]
        if self.title_wrap.on and not self.title_wrap.text:
            self.title_wrap.text = ' | '.join(self.label_wrap.values)

        # Disable x axis grids for boxplot
        if 'box' in self.plot_func:
            self.grid_major_x.on = False
            self.grid_minor_x.on = False
            self.ticks_major_x.on = False
            self.ticks_minor_x.on = False
            self.tick_labels_major_x.on = False
            self.tick_labels_minor_x.on = False
            self.label_x.on = False

        # Confidence interval
        self.conf_int = Element('conf_int', self.fcpp, kwargs,
                                on=True if kwargs.get('conf_int', False) else False,
                                edge_color=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_edge_color',
                                                     copy.copy(line_colors)),
                                edge_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_edge_alpha',
                                                     0.25),
                                fill_color=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_color',
                                                     copy.copy(line_colors)),
                                fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_alpha',
                                                     0.2),
                                )

        # Extras
        self.inline = utl.kwget(kwargs, self.fcpp, 'inline', None)
        self.separate_labels = utl.kwget(kwargs, self.fcpp,
                                         'separate_labels', False)
        self.separate_ticks = utl.kwget(kwargs, self.fcpp,
                                        'separate_ticks', self.separate_labels)
        if not self.axes.share_x or not self.axes.share_y:
            self.separate_ticks = True
        self.tick_cleanup = utl.kwget(kwargs, self.fcpp, 'tick_cleanup', True)

    def init_white_space(self, **kwargs):
        """
        Set the default spacing parameters (plot engine specific)
        Args:
            kwargs: input args from user
        """

        # cbar
        if self.cbar.on:
            self.ws_ax_cbar = utl.kwget(kwargs, self.fcpp, 'ws_ax_cbar', 10)
        else:
            self.ws_ax_cbar = 0

        # rc labels
        ws_rc_label = utl.kwget(kwargs, self.fcpp, 'ws_rc_label', 10)
        self.ws_col_label = utl.kwget(kwargs, self.fcpp,
                                  'ws_col_label', ws_rc_label)
        self.ws_row_label = utl.kwget(kwargs, self.fcpp,
                                  'ws_row_label', ws_rc_label)
        self.ws_col = utl.kwget(kwargs, self.fcpp, 'ws_col', 30)
        self.ws_row = utl.kwget(kwargs, self.fcpp, 'ws_row', 30)

        # figure
        self.ws_fig_label = utl.kwget(kwargs, self.fcpp, 'ws_fig_label', 10)
        self.ws_leg_fig = utl.kwget(kwargs, self.fcpp, 'ws_leg_fig', 10)
        self.ws_fig_ax = utl.kwget(kwargs, self.fcpp, 'ws_fig_ax', 20)
        self.ws_fig_title = utl.kwget(kwargs, self.fcpp, 'ws_fig_title', 10)

        # axes
        self.ws_label_tick = utl.kwget(kwargs, self.fcpp, 'ws_label_tick', 10)
        self.ws_leg_ax = utl.kwget(kwargs, self.fcpp, 'ws_leg_ax', 20)
        self.ws_ticks_ax = utl.kwget(kwargs, self.fcpp, 'ws_ticks_ax', 3)
        self.ws_title_ax = utl.kwget(kwargs, self.fcpp, 'ws_title_ax', 10)
        self.ws_ax_fig = utl.kwget(kwargs, self.fcpp, 'ws_ax_fig', 30)

        # ticks
        self.ws_tick_tick_minimum = utl.kwget(kwargs, self.fcpp,
                                              'ws_tick_tick_minimum', 10)

        # box
        self.ws_ax_box_title = utl.kwget(kwargs, self.fcpp, 'ws_ax_box_title', 10)

    def format_legend_values(self):
        """
        Reformat legend values
        """

        if not self.legend.on:
            return

        df = pd.DataFrame.from_dict(self.legend.values).T.reset_index()
        df['names'] = ''

        # level_0 = legend
        if not (df.level_0==None).all():
            df['names'] = df.level_0

        # level_2 = y
        if len(df.level_2.unique()) > 1:
            df['names'] = df.level_0.map(str) + ': ' + df.level_2.map(str)

        # level_1 = x
        if len(df.level_1.unique()) > 1:
            df['names'] = df['names'].map(str) + ' / ' + df.level_1.map(str)

        for irow, row in df.iterrows():
            key = (row['level_0'], row['level_1'], row['level_2'])
            self.legend.values[row['names']] = self.legend.values[key]
            del self.legend.values[key]

    def from_list(self, base, attrs, name, kwargs):
        """
        Supports definition of object attributes for multiple axes using
        a list

        Index 0 is always 'x'
        Index 1 is always 'y'
        Index 2 is for twinned axes

        Args:
            base (Element): object class
            attrs (list): attributes to check from the base class
            name (str): name of the base class
            kwargs (dict): keyword dict

        Returns:
            updated kwargs
        """

        for attr in attrs:
            if type(getattr(base, attr)) is list:
                setattr(base, attr, getattr(base, attr) +
                        [None] * (len(getattr(base, attr)) - 3))
                kwargs['%s_%s_x' % (name, attr)] = getattr(base, attr)[0]
                kwargs['%s_%s_y' % (name, attr)] = getattr(base, attr)[1]
                if 'twin_x' in kwargs.keys() and kwargs['twin_x']:
                    kwargs['%s_%s_y2' % (name, attr)] = getattr(base, attr)[2]
                if 'twin_y' in kwargs.keys() and kwargs['twin_y']:
                    kwargs['%s_%s_x2' % (name, attr)] = getattr(base, attr)[2]

        return kwargs

    def make_figure(self, data, **kwargs):
        pass

    def make_kwargs(self, element, pop=[]):
        kwargs = {}
        kwargs['position'] = element.position
        kwargs['size'] = element.size
        kwargs['rotation'] = element.rotation
        kwargs['fill_color'] = element.fill_color
        kwargs['edge_color'] = element.edge_color
        kwargs['edge_width'] = element.edge_width
        kwargs['font'] = element.font
        kwargs['font_weight'] = element.font_weight
        kwargs['font_style'] = element.font_style
        kwargs['font_color'] = element.font_color
        kwargs['font_size'] = element.font_size
        kwargs['color'] = element.color
        kwargs['width'] = element.width
        kwargs['style'] = element.style
        for pp in pop:
            if pp in kwargs.keys():
                kwargs.pop(pp)

        return kwargs

    def set_label_text(self, data, **kwargs):
        """
        Set the default label text

        Args:
            data (Data object): current Data object

        """

        labels = ['x', 'y', 'z', 'col', 'row', 'wrap']
        for ilab, lab in enumerate(labels):
            dd = getattr(data, lab)
            if not dd:
                continue

            # Get label override name if in kwargs
            if '%slabel' % lab in kwargs.keys():
                lab_text = kwargs.get('%slabel' % lab)
                lab_text2 = kwargs.get('%s2label' % lab)
            elif 'label_%s' % lab in kwargs.keys():
                lab_text = kwargs.get('label_%s' % lab)
                lab_text2 = kwargs.get('label_%s2' % lab)
            else:
                lab_text = None
                lab_text2 = None

            if lab == 'x' and self.axes.twin_y:
                getattr(self, 'label_x').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_x2').text = \
                    lab_text if lab_text is not None else dd[1]
            elif lab == 'y' and self.axes.twin_x:
                getattr(self, 'label_y').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_y2').text = \
                    lab_text if lab_text is not None else dd[1]
            else:
                if lab == 'wrap':
                    # special case
                    val = 'title_wrap'
                else:
                    val = 'label_%s' % lab
                if type(dd) is list:
                    getattr(self, val).text = \
                        lab_text if lab_text is not None else ' + '.join(dd)
                else:
                    getattr(self, val).text = dd
                if lab != 'z' and hasattr(self, 'label_%s2' % lab):
                    getattr(self, 'label_%s2' % lab).text = \
                        lab_text2 if lab_text2 is not None else ' + '.join(dd)

            if hasattr(data, '%s_vals' % lab):
                getattr(self, 'label_%s' % lab).values = \
                    getattr(data, '%s_vals' % lab)


class Element:
    def __init__(self, label='None', fcpp={}, others={}, **kwargs):
        """
        Element style container
        """

        # Update kwargs
        for k, v in others.items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self.column = None
        self._on = kwargs.get('on', True) # visbile or not
        self.dpi = utl.kwget(kwargs, fcpp, 'dpi', 100)
        self.obj = None  # plot object reference
        self.position = [0, 0, 0, 0]  # left, right, top, bottom
        self._size = kwargs.get('size', [0, 0])  # width, height
        self._size_orig = kwargs.get('size')
        self._text = kwargs.get('text', True)  # text label
        self._text_orig = kwargs.get('text')
        self.rotation = utl.kwget(kwargs, fcpp, '%s_rotation' % label,
                                  kwargs.get('rotation', 0))

        # fill an edge colors
        self.fill_alpha = utl.kwget(kwargs, fcpp, '%s_fill_alpha' % label,
                                    kwargs.get('fill_alpha', 1))
        self.fill_color = utl.kwget(kwargs, fcpp, '%s_fill_color' % label,
                                    kwargs.get('fill_color', '#ffffff'))
        if type(self.fill_color) is RepeatedList:
            self.fill_color.values = [f[0:7] + str(hex(int(self.fill_alpha*255)))[-2:] \
                                      for f in self.fill_color.values]
        else:
            self.fill_color = self.fill_color[0:7] + \
                              str(hex(int(self.fill_alpha*255)))[-2:]

        self.edge_width = utl.kwget(kwargs, fcpp, '%s_edge_width' % label,
                                    kwargs.get('edge_width', 1))
        self.edge_alpha = utl.kwget(kwargs, fcpp, '%s_edge_alpha' % label,
                                    kwargs.get('edge_alpha', 1))
        self.edge_color = utl.kwget(kwargs, fcpp, '%s_edge_color' % label,
                                    kwargs.get('edge_color', '#ffffff'))
        self.edge_width = utl.kwget(kwargs, fcpp, '%s_edge_width' % label,
                                    kwargs.get('edge_width', 1))
        if type(self.edge_color) is RepeatedList:
            self.edge_color.values = [f[0:7] + str(hex(int(self.edge_alpha*255)))[-2:] \
                                      for f in self.edge_color.values]
        else:
            self.edge_color = self.edge_color[0:7] + \
                              str(hex(int(self.edge_alpha*255)))[-2:]

        # fonts
        self.font = utl.kwget(kwargs, fcpp, '%s_font' % label,
                              kwargs.get('font', 'sans-serif'))
        self.font_color = utl.kwget(kwargs, fcpp, '%s_font_color' % label,
                                    kwargs.get('font_color', '#000000'))
        self.font_size = utl.kwget(kwargs, fcpp, '%s_font_size' % label,
                                   kwargs.get('font_size', 14))
        self.font_style = utl.kwget(kwargs, fcpp, '%s_font_style' % label,
                                   kwargs.get('font_style', 'normal'))
        self.font_weight = utl.kwget(kwargs, fcpp, '%s_font_weight' % label,
                                   kwargs.get('font_weight', 'normal'))

        # lines
        self.alpha = utl.kwget(kwargs, fcpp, '%s_alpha' % label,
                               kwargs.get('alpha', 1))
        self.color = utl.kwget(kwargs, fcpp, '%s_color' % label,
                               kwargs.get('color', '#000000'))
        if type(self.color) is list and type(self.alpha) is list:
            self.color = [f[0:7] + str(hex(int(self.alpha[i]*255)))[-2:] \
                          for i, f in enumerate(self.color)]
        elif type(self.color) is RepeatedList:
            self.color.values = [f[0:7] + str(hex(int(self.alpha*255)))[-2:] \
                                 for f in self.color.values]
        elif type(self.color) is list:
            self.color = [f[0:7] + str(hex(int(self.alpha*255)))[-2:] \
                          for f in self.color]
        else:
            self.color = self.color[0:7] + str(hex(int(self.alpha*255)))[-2:]
        self.width = utl.kwget(kwargs, fcpp, '%s_width' % label,
                               kwargs.get('width', 1))
        self.style = utl.kwget(kwargs, fcpp, '%s_style' % label,
                               kwargs.get('style', '-'))

        skip_keys = ['df', 'x', 'y', 'z']
        for k, v in kwargs.items():
            try:
                if not hasattr(self, k) and k not in skip_keys:
                    setattr(self, k, v)
            except:
                pass

    @property
    def kwargs(self):

        temp = self.__dict__
        if 'df' in temp.keys():
            temp.pop('df')
        return temp

    @property
    def on(self):

        return self._on

    @on.setter
    def on(self, state):

        self._on = state

        if not self.on:
            self._size = [0, 0]
            self._text = None

        else:
            self._size = self._size_orig
            self._text = self._text_orig

    @property
    def size(self):

        if self.on:
            return self._size
        else:
            return [0, 0]

    @size.setter
    def size(self, value):

        if self._size_orig is None and value is not None:
            self._size_orig = value

        self._size = value

    @property
    def size_inches(self):

        if self.on:
            return [self._size[0]/self.dpi, self._size[1]/self.dpi]
        else:
            return [0, 0]

    @property
    def text(self):

        return self._text

    @text.setter
    def text(self, value):

        if self._text_orig is None and value is not None:
            self._text_orig = value

        self._text = value

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})

        if '_on' in df.columns:
            del df['_on']
            df['on'] = self.on

        if '_text' in df.columns:
            del df['_text']
            df['text'] = self.text

        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df


class DF_Element(Element):
    def __init__(self, label='None', fcpp={}, others={}, **kwargs):
        Element.__init__(self, label=label, fcpp=fcpp, others=others, **kwargs)

        if not hasattr(self, 'column'):
            self.column = None
        if not hasattr(self, 'values'):
            self.values = []

    @property
    def on(self):

        return True if self._on and len(self.values) > 0 else False

    @on.setter
    def on(self, state):

        self._on = state

        if not self.on:
            self._size = [0, 0]
            self._text = None

        else:
            self._size = self._size_orig
            self._text = self._text_orig


class RepeatedList:
    def __init__(self, values, name):
        """
        Set a default list of items and loop through it beyond the maximum
        index value

        Args:
            values (list): user-defined list of values
            name (str): label to describe contents of class
        """

        self.values = values
        self.shift = 0

        if type(self.values) is not list and len(self.values) < 1:
            raise(ValueError, 'RepeatedList for "%s" must contain an actual '
                              'list with more at least one element')

    def get(self, idx):

        # can we make this a next??

        if type(self.values) is tuple: st()
        return self.values[(idx + self.shift) % len(self.values)]


class LayoutBokeh(BaseLayout):
    def __init__(self, **kwargs):
        BaseLayout.__init__(self, **kwargs)


class LayoutMPL(BaseLayout):

    def __init__(self, plot_func='plot', **kwargs):
        """
        Layout attributes and methods for matplotlib Figure

        Args:
            **kwargs: input args from user
        """

        mplp.close('all')

        # Inherit the base layout properties
        BaseLayout.__init__(self, plot_func, **kwargs)

        # Define white space parameters
        self.init_white_space(**kwargs)

        # Initialize other class variables
        self.col_label_height  = 0
        self.row_label_left    = 0
        self.row_label_width   = 0
        self.wrap_title_bottom = 0

        # Weird spacing defaults out of our control
        self.fig_right_border = 6  # extra border on right side that shows up by default
        self.legend_top_offset = 8 # this is differnt for inline; do we need to toggle on/off with show?
        self.legend_border = 3

    def add_box_labels(self, ir, ic, data):

        num_cols = len(data.changes.columns)
        bottom = 0
        for i in range(0, num_cols):
            if i > 0:
                bottom -= height
            k = num_cols-1-i
            sub = data.changes[num_cols-1-i][data.changes[num_cols-1-i]==1]
            if len(sub) == 0:
                sub = data.changes[num_cols-1-i]

            # Group labels
            if self.box_group_label.on:
                for j in range(0, len(sub)):
                    if j == len(sub) - 1:
                        width = len(data.changes) - sub.index[j]
                    else:
                        width = sub.index[j+1] - sub.index[j]
                    width = width * self.axes.size[0] / len(data.changes)
                    label = data.indices.loc[sub.index[j], num_cols-1-i]
                    height = self.box_group_label.size[i][1] * \
                                (1 + 2 * self.box_group_label.padding / 100)
                    self.add_label(self.axes.obj[ir, ic], label,
                                    (sub.index[j]/len(data.changes),
                                    0, 0,
                                    (bottom - height) / self.axes.size[1]),
                                    rotation=self.box_group_label.rotation[i],
                                    size=[width, height], offset=True,
                                    **self.make_kwargs(self.box_group_label,
                                                        ['size', 'rotation', 'position']))

            # Group titles
            if self.box_group_title.on and ic == data.ncol - 1:
                self.add_label(self.axes.obj[ir, ic], data.groups[k],
                                (1 + self.ws_ax_box_title / self.axes.size[0],
                                0, 0,
                                (bottom - height) / self.axes.size[1]),
                                size=self.box_group_title.size[k],
                                **self.make_kwargs(self.box_group_title,
                                ['position', 'size']))

    def add_box_points(self, ir, ic, x, y):
        """
        Plot x y points with or without jitter
        """

        if self.box_points.jitter:
            x = np.random.normal(x+1, 0.04, size=len(y))
        else:
            x = np.array([x+1]*len(y))
        if len(x) > 0 and len(y) > 0:
            pts = self.axes[ir, ic].plot(
                          x, y,
                          color=self.box_marker.fill_color,
                          markersize=self.box_marker.size,
                          marker=self.box_marker.type,
                          markeredgecolor=self.box_points.edge_color,
                          markerfacecolor='none',
                          markeredgewidth=self.box_marker.edge_width,
                          linestyle='none',
                          zorder=2)
            return pts

    def add_cbar(self, ax, contour):
        """
        Add a color bar
        """

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        size = '%s%%' % (100*self.cbar.size[0]/self.axes.size[0])
        pad = self.ws_ax_cbar/100
        cax = divider.append_axes("right", size=size, pad=pad)

        # Add the colorbar
        cbar = mplp.colorbar(contour, cax=cax)
        cbar.outline.set_edgecolor(self.cbar.edge_color)
        cbar.outline.set_linewidth(self.cbar.edge_width)
        #cbar.dividers.set_color('white')  # could enable
        #cbar.dividers.set_linewidth(2)

        return cbar

    def add_hvlines(self, ir, ic):
        """
        Add axhlines and axvlines

        Args:
            ir (int): row index
            ic (int): col index
        """

        # Set default line attributes
        for axline in ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']:
            ll = getattr(self, axline)
            func = self.axes.obj[ir, ic].axhline if 'hline' in axline \
                   else self.axes.obj[ir, ic].axvline
            if ll.on:
                for ival, val in enumerate(ll.values):
                    func(val, color=ll.color[ival],
                         linestyle=ll.style[ival],
                         linewidth=ll.width[ival],
                         zorder=ll.zorder)

    def add_label(self, axis, text='', position=None, rotation=0, size=None,
                  fill_color='#ffffff', edge_color='#aaaaaa', edge_width=1,
                  font='sans-serif', font_weight='normal', font_style='normal',
                  font_color='#666666', font_size=14, offset=False,**kwargs):
        """ Add a label to the plot

        This function can be used for title labels or for group labels applied
        to rows and columns when plotting facet grid style plots.

        Args:
            label (str):  label text
            pos (tuple): label position tuple of form (left, right, top, bottom)
            old is (left, bottom, width, height)
            axis (matplotlib.axes):  mpl axes object
            rotation (int):  degrees of rotation
            fillcolor (str):  hex color code for label fill (default='#ffffff')
            edgecolor (str):  hex color code for label edge (default='#aaaaaa')
            color (str):  hex color code for label text (default='#666666')
            weight (str):  label font weight (use standard mpl weights like 'bold')
            fontsize (int):  label font size (default=14)
        """

        # Define the label background
        rect = patches.Rectangle((position[0], position[3]),
                                 size[0]/self.axes.size[0],
                                 size[1]/self.axes.size[1],
                                 fill=True, transform=axis.transAxes,
                                 facecolor=fill_color,
                                 edgecolor=edge_color,
                                 clip_on=False, zorder=-1)

        axis.add_patch(rect)

        # Set slight text offset
        if rotation == 270 and offset:
            offsetx = -2/self.axes.size[0]#-font_size/self.axes.size[0]/4
        else:
            offsetx = 0
        if rotation == 0 and offset:
            offsety = -2/self.axes.size[1]#-font_size/self.axes.size[1]/4
        else:
            offsety = 0

        # Add the label text
        text = axis.text(position[0]+size[0]/self.axes.size[0]/2+offsetx,
                         position[3]+size[1]/self.axes.size[1]/2+offsety, text,
                         transform=axis.transAxes, horizontalalignment='center',
                         verticalalignment='center', rotation=rotation,
                         color=font_color, fontname=font, style=font_style,
                         weight=font_weight, size=font_size)

        return text

    def add_legend(self):
        """
        Add a figure legend
            TODO: add separate_label support?

        """

        if self.legend.on and len(self.legend.values) > 0:

            # Format the legend keys
            #self.format_legend_values()

            # Sort the legend keys
            if 'NaN' in self.legend.values.keys():
                del self.legend.values['NaN']
            keys = natsorted(list(self.legend.values.keys()))
            lines = [self.legend.values[f][0] for f in keys]

            # Set the font properties
            fontp = {}
            fontp['family'] = self.legend.font
            fontp['size'] = self.legend.font_size
            fontp['style'] = self.legend.font_style
            fontp['weight'] = self.legend.font_weight

            if self.legend.location == 0:
                self.legend.obj = \
                    self.fig.obj.legend(lines, keys, loc='upper right',
                                        title=self.legend.text if self.legend is not True else '',
                                        bbox_to_anchor=(self.legend.position[1],
                                                        self.legend.position[2]),
                                        numpoints=self.legend.points,
                                        prop=fontp)

            else:
                self.legend.obj = \
                    self.fig.obj.legend(lines, keys, loc=self.legend.position,
                                        title = self.legend.text if self.legend is not True else '',
                                        numpoints=self.legend.points,
                                        prop=fontp)

            for text in self.legend.obj.get_texts():
                text.set_color(self.legend.font_color)

            self.legend.obj.get_frame().set_facecolor(self.legend.fill_color)
            self.legend.obj.get_frame().set_edgecolor(self.legend.edge_color)

    def add_text(self, ir, ic, text, element, offsetx=0, offsety=0):
        """
        Add a text box
        """

        obj = getattr(self, element)
        kwargs = self.make_kwargs(obj, pop=['size'])

        ax = self.axes.obj[ir, ic]
        ax.text(obj.position[0] + offsetx, obj.position[3] + offsety, text,
                transform=ax.transAxes,
                rotation=kwargs['rotation'], color=kwargs['font_color'],
                fontname=kwargs['font'], style=kwargs['font_style'],
                weight=kwargs['font_weight'], size=kwargs['font_size'])

    def fill_between_lines(self, ir, ic, iline, x, lcl, ucl, obj):
        """
        Shade a region between two curves

        Args:

        """

        ax = self.axes.obj[ir, ic]
        obj = getattr(self, obj)
        fc = obj.fill_color
        ec = obj.edge_color
        ax.fill_between(x, lcl, ucl,
                        facecolor=fc.get(iline) if type(fc) is RepeatedList else fc,
                        edgecolor=ec.get(iline) if type(ec) is RepeatedList else ec)

    def format_axes(self):
        """
        Format the axes colors and gridlines
        """

        axes = self.get_axes()

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                for ax in axes:
                    ax = self.set_axes_colors(ax, ir, ic)
                    ax = self.set_axes_grid_lines(ax, ir, ic)
                    #NOT SURE IF THIS SHOULD BE DONE HERE OR IN THE MAIN LOOP

                    # Set axis ticks
                    st()

    def get_axes(self):
        """
        Return list of active axes
        """

        axes = [f for f in [self.axes, self.axes2] if f.on]
        if self.axes2.on:
            axes += [self.axes2]

        return axes

    def get_axes_label_position(self):
        """
        Get the position of the axes labels
            self.label_@.position --> [left, right, top, bottom]
        """

        self.label_x.position[0] = (self.axes.size[0] - self.label_x.size[0]) \
                                    / (2 * self.axes.size[0])
        self.label_x.position[3] = -np.floor(self.labtick_x) / self.axes.size[1]

        self.label_x2.position[0] = (self.axes.size[0] - self.label_x2.size[0]) \
                                    / (2 * self.axes.size[0])
        self.label_x2.position[3] = 1 + (np.floor(self.labtick_x2) - self.label_x2.size[1]) \
                                    / self.axes.size[1]

        self.label_y.position[0] = -np.floor(self.labtick_y) / self.axes.size[0]
        self.label_y.position[3] = (self.axes.size[1] - self.label_y.size[1]) \
                                    / (2 * self.axes.size[1])

        self.label_y2.position[0] = 1 + (np.floor(self.labtick_y2) - self.label_y2.size[0]) \
                                    / self.axes.size[0]
        self.label_y2.position[3] = (self.axes.size[1] - self.label_y2.size[1]) \
                                    / (2 * self.axes.size[1])

        self.label_z.position[0] = 1 + (self.ws_ax_cbar + self.cbar.size[0] + \
                                   self.tick_labels_major_z.size[0] + self.ws_label_tick) \
                                   / self.axes.size[0]
        self.label_z.position[3] = (self.axes.size[1] - self.label_z.size[1]) \
                                    / (2 * self.axes.size[1])

    def get_element_sizes(self, data):
        """
        Calculate the actual rendered size of select elements by pre-plotting
        them.  This is needed to correctly adjust the figure dimensions

        Args:
            data (obj): data class object
        """

        #
        # 2) need to include y2 dimensions and calc
        # 3) can we move the leg dimension calc here
        # 4) rename layout to layout?

        start = datetime.datetime.now()
        now = start.strftime('%Y-%m-%d-%H-%M-%S')

        # Make a dummy figure
        data = copy.deepcopy(data)
        if 'box' in self.plot_func:
            data.x = 'x'
            data.df_fig['x'] = 1

        mplp.ioff()
        fig = mpl.pyplot.figure(dpi=self.fig.dpi)
        ax = fig.add_subplot(111)
        ax2, ax3 = None, None
        if self.axes.twin_x or data.z is not None:
            ax2 = ax.twinx()
        if self.axes.twin_y:
            ax3 = ax.twiny()
        if self.axes.scale in ['logy', 'semilogy', 'loglog', 'log']:
            ax.set_yscale('log')
        elif self.axes.scale in ['logx', 'semilogx', 'loglog', 'log']:
            ax.set_xscale('log')
        elif self.axes.scale in ['symlog']:
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
        elif self.axes.scale in ['logit']:
            ax.set_xscale('logit')
            ax.set_yscale('logit')
        if self.axes.twin_x:
            if self.axes2.scale in ['logy', 'semilogy', 'loglog', 'log']:
                ax2.set_yscale('log')
            elif self.axes2.scale in ['logx', 'semilogx', 'loglog', 'log']:
                ax2.set_xscale('log')
            elif self.axes2.scale in ['symlog']:
                ax2.set_xscale('symlog')
                ax2.set_yscale('symlog')
            elif self.axes2.scale in ['logit']:
                ax2.set_xscale('logit')
                ax2.set_yscale('logit')
        if self.axes.twin_y:
            if self.axes2.scale in ['logy', 'semilogy', 'loglog', 'log']:
                ax3.set_yscale('log')
            elif self.axes2.scale in ['logx', 'semilogx', 'loglog', 'log']:
                ax3.set_xscale('log')
            elif self.axes2.scale in ['symlog']:
                ax3.set_xscale('symlog')
                ax3.set_yscale('symlog')
            elif self.axes2.scale in ['logit']:
                ax3.set_xscale('logit')
                ax3.set_yscale('logit')

        # Set tick and scale properties
        axes = [ax, ax2, ax3]
        for ia, aa in enumerate(axes):
            if aa is None:
                continue
            axes[ia] = self.set_scientific(aa)
            axes[ia].minorticks_on()
            if ia == 0:
                axes[ia].tick_params(axis='both',
                                     which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color,
                                     labelcolor=self.tick_labels_major.font_color,
                                     labelsize=self.tick_labels_major.font_size,
                                     top=False,
                                     bottom=self.ticks_major_x.on,
                                     right=self.ticks_major_y2.on \
                                           if self.axes.twin_x
                                           else self.ticks_major_y.on,
                                     left=self.ticks_major_y.on,
                                     length=self.ticks_major.size[0],
                                     width=self.ticks_major.size[1],
                                     )
                axes[ia].tick_params(axis='both',
                                     which='minor',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_minor.color,
                                     labelcolor=self.tick_labels_minor.font_color,
                                     labelsize=self.tick_labels_minor.font_size,
                                     top=self.ticks_minor_x2.on \
                                         if self.axes.twin_y
                                         else self.ticks_minor_x.on,
                                     bottom=self.ticks_minor_x.on,
                                     right=self.ticks_minor_y2.on \
                                           if self.axes.twin_x
                                           else self.ticks_minor_y.on,
                                     left=self.ticks_minor_y.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )

        # Define label variables
        xticksmaj, x2ticksmaj, yticksmaj, y2ticksmaj, zticksmaj = [], [], [], [], []
        xticksmin, x2ticksmin, yticksmin, y2ticksmin = [], [], [], []
        xticklabelsmaj, x2ticklabelsmaj, yticklabelsmaj, y2ticklabelsmaj = \
            [], [], [], []
        xticklabelsmin, x2ticklabelsmin, yticklabelsmin, y2ticklabelsmin, zticklabelsmaj = \
            [], [], [], [], []
        wrap_labels = np.array([[None]*self.ncol]*self.nrow)
        row_labels = np.array([[None]*self.ncol]*self.nrow)
        col_labels = np.array([[None]*self.ncol]*self.nrow)

        for ir, ic, df in data.get_rc_subset(data.df_fig):
            if len(df) == 0:
                continue
            # twin_x
            if self.axes.twin_x:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax2.plot(df[data.x[0]], df[data.y[1]], 'o-')
            # twin_y
            if self.axes.twin_y:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax3.plot(df[data.x[1]], df[data.y[0]], 'o-')
            # Z axis
            if data.z is not None:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax2.plot(df[data.x[0]], df[data.z[0]], 'o-')
            # Regular
            else:
                for xy in zip(data.x, data.y):
                    pp = ax.plot(df[xy[0]], df[xy[1]], 'o-')
            for xy in zip(data.x, data.y):
                pp = ax.plot(df[xy[0]], df[xy[1]], 'o-')
            if data.ranges[ir, ic]['xmin'] is not None:
                axes[0].set_xlim(left=data.ranges[ir, ic]['xmin'])
            if data.ranges[ir, ic]['xmax'] is not None:
                axes[0].set_xlim(right=data.ranges[ir, ic]['xmax'])
            if data.ranges[ir, ic]['ymin'] is not None:
                axes[0].set_ylim(bottom=data.ranges[ir, ic]['ymin'])
            if data.ranges[ir, ic]['ymax'] is not None:
                axes[0].set_ylim(top=data.ranges[ir, ic]['ymax'])

            # Major ticks
            xticks = axes[0].get_xticks()
            yticks = axes[0].get_yticks()
            xiter_ticks = [f for f in axes[0].xaxis.iter_ticks()] # fails for symlog in 1.5.1
            yiter_ticks = [f for f in axes[0].yaxis.iter_ticks()]
            xticksmaj += [f[2] for f in xiter_ticks[0:len(xticks)]]
            yticksmaj += [f[2] for f in yiter_ticks[0:len(yticks)]]
            if data.twin_x:
                y2ticks = [f[2] for f in axes[1].yaxis.iter_ticks()
                          if f[2] != '']
                y2iter_ticks = [f for f in axes[1].yaxis.iter_ticks()]
                y2ticksmaj += [f[2] for f in y2iter_ticks[0:len(y2ticks)]]
            elif data.twin_y:
                x2ticks = [f[2] for f in axes[2].xaxis.iter_ticks()
                           if f[2] != '']
                x2iter_ticks = [f for f in axes[2].xaxis.iter_ticks()]
                x2ticksmaj += [f[2] for f in x2iter_ticks[0:len(x2ticks)]]
            if data.z is not None:
                zticks = axes[1].get_xticks()
                ziter_ticks = [f for f in axes[1].yaxis.iter_ticks()]
                zticksmaj += [f[2] for f in ziter_ticks[0:len(zticks)]]

            # Minor ticks
            if self.axes.scale in ['logx', 'semilogx', 'loglog', 'log'] and \
                    self.tick_labels_minor_x.on:
                axes[0].xaxis.set_minor_locator(LogLocator())
            elif self.tick_labels_minor_x.on:
                axes[0].xaxis.set_minor_locator(AutoMinorLocator(self.ticks_minor_x.number+1))
                tp = mpl_get_ticks(axes[0])
                incx = (xticks[-1] - xticks[0])/(len(xticks)-1)
                minor_ticks_x = [f[1] for f in tp['x']['labels']][len(tp['x']['ticks']):]
                number_x = len([f for f in minor_ticks_x if f < incx]) + 1
                decimalx = utl.get_decimals(incx/number_x)
                axes[0].xaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimalx+1)))

            if self.axes.scale in ['logy', 'semilogy', 'loglog', 'log'] and \
                    self.tick_labels_minor_y.on:
                axes[0].xaxis.set_minor_locator(LogLocator())
            elif self.tick_labels_minor_y.on:
                axes[0].yaxis.set_minor_locator(AutoMinorLocator(self.ticks_minor_y.number+1))
                tp = mpl_get_ticks(axes[0])
                incy = (yticks[-1] - yticks[0])/(len(yticks)-1)
                minor_ticks_y = [f[1] for f in tp['y']['labels']][len(tp['y']['ticks']):]
                number_y = len([f for f in minor_ticks_y if f < incy]) + 1
                decimaly = utl.get_decimals(incy/number_y)
                axes[0].yaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimaly+1)))

            xticksmin += [f[2] for f in axes[0].xaxis.iter_ticks()][len(xticks):]
            yticksmin += [f[2] for f in axes[0].yaxis.iter_ticks()][len(xticks):]

            ww, rr, cc = self.set_axes_rc_labels(ir, ic, axes[0])
            wrap_labels[ir, ic] = ww
            row_labels[ir, ic] = rr
            col_labels[ir, ic] = cc

        # Make a dummy legend --> move to add_legend???
        if data.legend_vals is not None and len(data.legend_vals) > 0:
            lines = []
            leg_vals = []
            if type(data.legend_vals) == pd.DataFrame:
                for irow, row in data.legend_vals.iterrows():
                    lines += ax.plot([1, 2, 3])
                    leg_vals += [row['names']]
            else:
                for val in data.legend_vals:
                    lines += ax.plot([1, 2, 3])
                    leg_vals += [val]
            if self.yline.on:
                lines += axes[0].plot([1, 2, 3])
                leg_vals += [self.yline.text]
            leg = mpl.pyplot.legend(lines, leg_vals,
                                    title=self.legend.text,
                                    numpoints=self.legend.points,
                                    fontsize=self.legend.font_size)
            if self.legend.marker_size:
                for i in data.legend_vals:
                    leg.legendHandles[0]._legmarker\
                        .set_markersize(self.legend.marker_size)
        else:
            leg = None

        # Write out major tick labels
        for ix, xtick in enumerate(xticksmaj):
            xticklabelsmaj += [fig.text(ix*20, 20, xtick,
                                        fontsize=self.tick_labels_major_x.font_size,
                                        rotation=self.tick_labels_major_x.rotation)]
        for ix, x2tick in enumerate(x2ticksmaj):
            x2ticklabelsmaj += [fig.text(ix*20, 20, x2tick,
                                        fontsize=self.tick_labels_major_x2.font_size,
                                        rotation=self.tick_labels_major_x2.rotation)]
        for iy, ytick in enumerate(yticksmaj):
            yticklabelsmaj += [fig.text(20, iy*20, ytick,
                                        fontsize=self.tick_labels_major_y.font_size,
                                        rotation=self.tick_labels_major_y.rotation)]
        for iy, y2tick in enumerate(y2ticksmaj):
            y2ticklabelsmaj += [fig.text(20, iy*20, y2tick,
                                         fontsize=self.tick_labels_major_y2.font_size,
                                         rotation=self.tick_labels_major_y2.rotation)]
        if data.z is not None:
            for iz, ztick in enumerate(zticksmaj):
                zticklabelsmaj += [fig.text(20, iz*20, ztick,
                                            fontsize=self.tick_labels_major_z.font_size,
                                            rotation=self.tick_labels_major_z.rotation)]

        # Write out minor tick labels
        for ix, xtick in enumerate(xticksmin):
            xticklabelsmin += [fig.text(ix*20, 20, xtick,
                                        fontsize=self.tick_labels_minor_x.font_size,
                                        rotation=self.tick_labels_minor_x.rotation)]
        for ix, x2tick in enumerate(x2ticksmin):
            x2ticklabelsmin += [fig.text(ix*20, 20, x2tick,
                                        fontsize=self.tick_labels_minor_x2.font_size,
                                        rotation=self.tick_labels_minor_x2.rotation)]
        for iy, ytick in enumerate(yticksmin):
            yticklabelsmin += [fig.text(20, iy*20, ytick,
                                        fontsize=self.tick_labels_minor_y.font_size,
                                        rotation=self.tick_labels_minor_y.rotation)]
        for iy, y2tick in enumerate(y2ticksmin):
            y2ticklabelsmin += [fig.text(20, iy*20, y2tick,
                                         fontsize=self.tick_labels_minor_y2.font_size,
                                        rotation=self.tick_labels_minor_y2.rotation)]

        # Write out axes labels
        if type(self.label_x.text) is str:
            label_x = fig.text(0, 0, r'%s' % self.label_x.text,
                               fontsize=self.label_x.font_size,
                               weight=self.label_x.font_weight,
                               style=self.label_x.font_style,
                               color=self.label_x.font_color,
                               rotation=self.label_x.rotation)

        if type(self.label_x2.text) is str:
            label_x2 = fig.text(0, 0, r'%s' % self.label_x2.text,
                               fontsize=self.label_x2.font_size,
                               weight=self.label_x2.font_weight,
                               style=self.label_x2.font_style,
                               color=self.label_x2.font_color,
                               rotation=self.label_x2.rotation)

        if type(self.label_y.text) is str:
            label_y = fig.text(0, 0, r'%s' % self.label_y.text,
                               fontsize=self.label_y.font_size,
                               weight=self.label_y.font_weight,
                               style=self.label_y.font_style,
                               color=self.label_y.font_color,
                               rotation=self.label_y.rotation)

        if type(self.label_y2.text) is str:
            label_y2 = fig.text(0, 0, r'%s' % self.label_y2.text,
                               fontsize=self.label_y2.font_size,
                               weight=self.label_y2.font_weight,
                               style=self.label_y2.font_style,
                               color=self.label_y2.font_color,
                               rotation=self.label_y2.rotation)

        if type(self.label_z.text) is str:
            label_z = fig.text(0, 0, r'%s' % self.label_z.text,
                               fontsize=self.label_z.font_size,
                               weight=self.label_z.font_weight,
                               style=self.label_z.font_style,
                               color=self.label_z.font_color,
                               rotation=self.label_z.rotation)

        # Write out title
        if type(self.title.text) is str:
            title = fig.text(0, 0, r'%s' % self.title.text,
                             fontsize=self.title.font_size,
                             weight=self.title.font_weight,
                             style=self.title.font_style,
                             color=self.title.font_color,
                             rotation=self.title.rotation)

        # Write out boxplot group labels
        box_group_label = []
        box_group_title = []
        if data.groups is None:
            self.box_group_label.on = False
            self.box_group_title.on = False
        if 'box' in self.plot_func and self.box_group_label.on:
            for ii, cc in enumerate(data.indices.columns):
                vals = [str(f) for f in data.indices[cc].unique()]
                box_group_label_row = []
                for val in vals:
                    box_group_label_row += \
                        [fig.text(0, 0, r'%s' % val,
                                  fontsize=self.box_group_label.font_size,
                                  weight=self.box_group_label.font_weight,
                                  style=self.box_group_label.font_style,
                                  color=self.box_group_label.font_color,
                                  rotation=self.box_group_label.rotation,
                                  )]
                box_group_label += [box_group_label_row]
        if 'box' in self.plot_func and self.box_group_title.on:
            for group in data.groups:
                box_group_title += \
                    [fig.text(0, 0, r'%s' % group,
                            fontsize=self.box_group_title.font_size,
                            weight=self.box_group_title.font_weight,
                            style=self.box_group_title.font_style,
                            color=self.box_group_title.font_color,
                            rotation=self.box_group_title.rotation,
                            )]

        # Render dummy figure
        mpl.pyplot.draw()
        # mpl.pyplot.savefig(r'test.png')  # turn on for debugging

        # Get actual sizes
        if self.tick_labels_major_x.on and len(xticklabelsmaj) > 0:
            self.tick_labels_major_x.size = \
                [np.nanmax([t.get_window_extent().width for t in xticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in xticklabelsmaj])]
        if self.tick_labels_major_x2.on and len(x2ticklabelsmaj) > 0:
            self.tick_labels_major_x2.size = \
                [np.nanmax([t.get_window_extent().width for t in x2ticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in x2ticklabelsmaj])]
        if self.tick_labels_major_y.on and len(yticklabelsmaj) > 0:
            self.tick_labels_major_y.size = \
                [np.nanmax([t.get_window_extent().width for t in yticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in yticklabelsmaj])]
        if self.tick_labels_major_y2.on and len(y2ticklabelsmaj) > 0:
            self.tick_labels_major_y2.size = \
                [np.nanmax([t.get_window_extent().width for t in y2ticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in y2ticklabelsmaj])]
        if self.tick_labels_major_z.on and len(zticklabelsmaj) > 0:
            self.tick_labels_major_z.size = \
                [np.nanmax([t.get_window_extent().width for t in zticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in zticklabelsmaj])]

        if self.tick_labels_minor_x.on and len(xticklabelsmin) > 0:
            self.tick_labels_minor_x.size = \
                [np.nanmax([t.get_window_extent().width for t in xticklabelsmin]),
                 np.nanmax([t.get_window_extent().height for t in xticklabelsmin])]
        if self.tick_labels_minor_x2.on and len(x2ticklabelsmin) > 0:
            self.tick_labels_minor_x2.size = \
                [np.nanmax([t.get_window_extent().width for t in x2ticklabelsmin]),
                 np.nanmax([t.get_window_extent().height for t in x2ticklabelsmin])]
        if self.tick_labels_minor_y.on and len(yticklabelsmin) > 0:
            self.tick_labels_minor_y.size = \
                [np.nanmax([t.get_window_extent().width for t in yticklabelsmin]),
                 np.nanmax([t.get_window_extent().height for t in yticklabelsmin])]
        if self.tick_labels_minor_y2.on and len(y2ticklabelsmin) > 0:
            self.tick_labels_minor_y2.size = \
                [np.nanmax([t.get_window_extent().width for t in y2ticklabelsmin]),
                 np.nanmax([t.get_window_extent().height for t in y2ticklabelsmin])]

        if self.axes.twin_x and self.tick_labels_major.on:
            self.ticks_major_y2.size = \
                [np.nanmax([t.get_window_extent().width for t in y2ticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in y2ticklabelsmaj])]
        elif self.axes.twin_x and not self.tick_labels_major.on:
            self.ticks_major_y2.size = \
                [np.nanmax([0 for t in y2ticklabelsmaj]),
                 np.nanmax([0 for t in y2ticklabelsmaj])]

        if self.axes.twin_y and self.tick_labels_major.on:
            self.ticks_major_x2.size = \
                [np.nanmax([t.get_window_extent().width for t in x2ticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in x2ticklabelsmaj])]
        elif self.axes.twin_y and not self.tick_labels_major.on:
            self.ticks_major_x2.size = \
                [np.nanmax([0 for t in x2ticklabelsmaj]),
                 np.nanmax([0 for t in x2ticklabelsmaj])]

        if self.label_x.on:
            self.label_x.size = (label_x.get_window_extent().width,
                                 label_x.get_window_extent().height)
        if self.axes.twin_y:
            self.label_x2.size = (label_x2.get_window_extent().width,
                                  label_x2.get_window_extent().height)
        self.label_y.size = (label_y.get_window_extent().width,
                             label_y.get_window_extent().height)
        if self.axes.twin_x:
            self.label_y2.size = (label_y2.get_window_extent().width,
                                  label_y2.get_window_extent().height)
        if self.label_z.on:
            self.label_z.size = (label_z.get_window_extent().width,
                                 label_z.get_window_extent().height)
        if self.title.on:
            self.title.size[0] = self.axes.size[0]
            self.title.size[1] = title.get_window_extent().height

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                if wrap_labels[ir, ic] is not None:
                    wrap_labels[ir, ic] = \
                        (wrap_labels[ir, ic].get_window_extent().width,
                         wrap_labels[ir, ic].get_window_extent().height)
                if row_labels[ir, ic] is not None:
                    row_labels[ir, ic] = \
                        (row_labels[ir, ic].get_window_extent().width,
                         row_labels[ir, ic].get_window_extent().height)
                if col_labels[ir, ic] is not None:
                    col_labels[ir, ic] = \
                        (col_labels[ir, ic].get_window_extent().width,
                         col_labels[ir, ic].get_window_extent().height)

        self.label_wrap.text_size = wrap_labels
        self.label_row.text_size = row_labels
        self.label_col.text_size = col_labels

        if leg:
            self.legend.size = \
                [leg.get_window_extent().width + self.legend_border,
                 leg.get_window_extent().height + self.legend_border]
        else:
            self.legend.size = [0, 0]

        # box labels
        if 'box' in self.plot_func and self.box_group_label.on \
                and data.groups is not None:
            # Get the size of group labels and adjust the rotation if needed
            rotations = []
            sizes = []
            for irow, row in enumerate(box_group_label):
                # Find the smallest group label box in the row
                labidx = list(data.changes[irow][data.changes[irow]>0].index) + \
                              [len(data.changes)]
                smallest = min(np.diff(labidx))
                max_label_width = self.axes.size[0]/len(data.changes) * smallest
                widest = max([f.get_window_extent().width for f in row])
                tallest = max([f.get_window_extent().height for f in row])
                if widest > max_label_width:
                    rotations += [90]
                    sizes += [(tallest, widest)]
                else:
                    rotations += [0]
                    sizes += [(widest, tallest)]
            sizes.reverse()
            rotations.reverse()
            self.box_group_label._size = sizes
            self.box_group_label.rotation = rotations

        if 'box' in self.plot_func and self.box_group_title.on \
                and data.groups is not None:
            self.box_group_title._size = [(f.get_window_extent().width,
                                           f.get_window_extent().height)
                                           for f in box_group_title]

        # Destroy the dummy figure
        mpl.pyplot.close(fig)

    def get_figure_size(self, **kwargs):
        """
        Determine the size of the mpl figure canvas in pixels and inches
        """

        debug = kwargs.get('debug_size', False)

        # Set some values for convenience
        self.labtick_x = self.label_x.size[1] + \
                         self.ws_label_tick * self.label_x.on + \
                         max(self.tick_labels_major_x.size[1],
                             self.tick_labels_minor_x.size[1]) + \
                         self.ws_ticks_ax * self.tick_labels_major_x.on
        self.labtick_x2 = (self.label_x2.size[1] + self.ws_label_tick + \
                           self.ws_ticks_ax + \
                           max(self.tick_labels_major_x2.size[1],
                               self.tick_labels_minor_x2.size[1])) * self.axes.twin_y
        self.labtick_y = self.label_y.size[0] + self.ws_label_tick + \
                         max(self.tick_labels_major_y.size[0],
                             self.tick_labels_minor_y.size[0]) + self.ws_ticks_ax
        self.labtick_y2 = (self.label_y2.size[0] + self.ws_label_tick + self.ws_ticks_ax + \
                           max(self.tick_labels_major_y2.size[0],
                               self.tick_labels_minor_y2.size[0])) * self.axes.twin_x
        self.labtick_z = (self.ws_ticks_ax + self.ws_label_tick) * self.label_z.on + \
                         self.label_z.size[0] + self.tick_labels_major_z.size[0]
        self.ws_leg_ax = max(0, self.ws_leg_ax - self.labtick_y2) if self.legend.text is not None else 0
        self.ws_leg_fig = self.ws_leg_fig if self.legend.text is not None else self.ws_ax_fig
        self.box_title = max(self.box_group_title.size)[0] if self.box_group_title.on else 0
        if self.box_group_label.on:
            self.box_labels = sum(f[1] * (1 + 2 * self.box_group_label.padding / 100) \
                                  for f in self.box_group_label.size)
        else:
            self.box_labels = 0

        # Adjust the column and row whitespace
        if self.box_group_label.on and self.label_wrap.on and 'ws_row' not in kwargs.keys():
            self.ws_row = self.box_labels + self.title_wrap.size[1]
        else:
            self.ws_row += self.box_labels

        if self.title.on:
            self.ws_title = self.ws_fig_title + self.title.size[1] + self.ws_title_ax
        else:
            self.ws_title = self.ws_fig_ax

        if self.cbar.on:
            self.ws_col = self.labtick_z #- self.label_z.size[0]

        if self.separate_labels:  # may need to move this down
            self.ws_col += self.label_y.size[0]
            self.ws_row += self.label_x.size[1]

        if self.separate_ticks:
            self.ws_col += max(self.tick_labels_major_y.size[0],
                               self.tick_labels_minor_y.size[0])
            self.ws_row += max(self.tick_labels_major_x.size[1],
                               self.tick_labels_minor_x.size[1])

        # Figure width
        self.fig.size[0] = \
            self.ws_fig_label + \
            self.labtick_y + \
            (self.axes.size[0] + self.cbar.size[0] + self.ws_ax_cbar) * self.ncol + \
            self.ws_col * (self.ncol - 1) +  \
            self.ws_leg_ax + self.legend.size[0] + self.ws_leg_fig + \
            self.labtick_y2 + \
            self.label_row.size[0] + self.ws_row_label * self.label_row.on + \
            self.labtick_z * (self.ncol - 1) + \
            self.box_title

        # Figure height
        self.fig.size[1] = \
            self.ws_title + \
            (self.label_col.size[1] + self.ws_col_label) * self.label_col.on + \
            self.title_wrap.size[1] + self.label_wrap.size[1] + \
            self.labtick_x2 + \
            self.axes.size[1]*self.nrow + \
            self.labtick_x + \
            self.ws_fig_label + \
            self.ws_row * (self.nrow - 1) + \
            self.box_labels

        # Debug output
        if debug:
            print('self.fig.size[0] = %s' % self.fig.size[0])
            vals = ['ws_fig_label', 'label_y', 'ws_label_tick', 'tick_labels_major_y',
                    'tick_labels_minor_y', 'ws_ticks_ax', 'axes', 'cbar', 'ws_ax_cbar',
                    'ws_col', 'ws_leg_ax', 'legend', 'ws_leg_fig', 'label_y2',
                    'ws_label_tick', 'ws_ticks_ax', 'tick_labels_major_y2', 'label_row',
                    'ws_row_label', 'label_z', 'tick_labels_major_z', 'box_title',
                    'ncol', 'labtick_y', 'labtick_y2', 'labtick_z']
            for val in vals:
                if isinstance(getattr(self, val), Element):
                    print('   %s.size[0] = %s' % (val, getattr(self, val).size[0]))
                else:
                    print('   %s = %s' % (val, getattr(self, val)))
            print('self.fig.size[1] = %s' % self.fig.size[1])
            vals = ['ws_fig_title', 'title', 'ws_title_ax', 'ws_fig_ax',
                    'label_col', 'ws_col_label', 'title_wrap',
                    'label_wrap', 'label_x2', 'ws_ticks_ax', 'tick_labels_major_x2',
                    'axes', 'label_x', 'ws_label_tick', 'tick_labels_major_x', 'ws_ticks_ax',
                    'ws_fig_label', 'ws_row', 'box_labels',
                    'nrow', 'labtick_x', 'labtick_x2', 'ws_title']
            for val in vals:
                if isinstance(getattr(self, val), Element):
                    print('   %s.size[1] = %s' % (val, getattr(self, val).size[1]))
                else:
                    print('   %s = %s' % (val, getattr(self, val)))


        # Account for legends longer than the figure
        fig_only = self.axes.size[1]*self.nrow + (self.ws_ticks_ax +
                   self.label_x.size[1] + self.ws_fig_label +
                   max(self.tick_labels_major_x.size[1],
                       self.tick_labels_minor_x.size[1])) * \
                   (1 + int(self.separate_labels)*self.nrow)
        self.legend.overflow = max(self.legend.size[1]-fig_only, 0)
        self.fig.size[1] += self.legend.overflow

    def get_legend_position(self):
        """
        Get legend position
        """

        offset_x = 0
        if self.box_group_title.on and self.legend.size[1] < self.axes.size[1]:
            offset_x = max(self.box_group_title.size)[0]

        self.legend.position[1] = \
            1 - (self.ws_leg_fig - self.fig_right_border + \
            offset_x) / self.fig.size[0]

        self.legend.position[2] = \
            self.axes.position[2] + self.legend_top_offset/self.fig.size[1]

    def get_rc_label_position(self):
        """
        Get option group label positions
            self.label.position --> [left, right, top, bottom]
        """

        self.label_row.position[0] = \
            (self.axes.size[0] + self.labtick_y2 + self.ws_row_label +
             (self.ws_ax_cbar if self.cbar.on else 0) + self.cbar.size[0] +
             self.labtick_z)/self.axes.size[0]

        self.label_col.position[3] = (self.axes.size[1] + self.ws_col_label +
                                      self.labtick_x2)/self.axes.size[1]

        self.label_wrap.position[3] = 1
        self.title_wrap.size[0] = self.ncol * self.title_wrap.size[0]
        self.title_wrap.position[3] = 1 + self.label_wrap.size[1] / self.axes.size[1]

    def get_subplots_adjust(self):
        """
        Calculate the subplots_adjust parameters for the axes
            self.axes.position --> [left, right, top, bottom]
        """

        self.axes.position[0] = \
            (self.ws_fig_label + self.labtick_y) / self.fig.size[0]

        self.axes.position[1] = \
            self.axes.position[0] + \
            (self.axes.size[0] * self.ncol + \
            self.ws_col * (self.ncol - 1) + \
            (self.cbar.size[0] + self.ws_ax_cbar) * (self.ncol - 1) + \
            self.labtick_z - self.label_z.size[0]) \
            / self.fig.size[0]

        self.axes.position[2] = \
            1 - (self.ws_title + self.title_wrap.size[1] + \
            (self.label_col.size[1] + self.ws_col_label) * self.label_col.on + \
            self.label_wrap.size[1] + self.labtick_x2) / self.fig.size[1]

        self.axes.position[3] = \
            (self.labtick_x + self.ws_fig_label + self.box_labels) / self.fig.size[1]

    def get_title_position(self):
        """
        Calculate the title position
            self.title.position --> [left, right, top, bottom]
        """

        col_label = (self.label_col.size[1] + \
                     self.ws_col_label * self.label_col.on)
        self.title.position[0] = (self.axes.size[0] - self.title.size[0])/2/self.axes.size[0]
        self.title.position[3] = 1+(self.ws_title_ax + col_label) \
                                 /self.axes.size[1]
        self.title.position[2] = self.title.position[3] + (self.ws_title_ax +
                                 self.title.size[1])/self.axes.size[1]

    def make_figure(self, data,**kwargs):
        """
        Make the figure and axes objects
        """

        self.groups = data.groups
        self.ncol = data.ncol
        self.ngroups = data.ngroups
        self.nrow = data.nrow
        self.nwrap = data.nwrap

        if data.wrap:
            self.separate_labels = kwargs.get('separate_labels', False)
            self.ws_row = kwargs.get('ws_row', self.label_wrap._size[1])
            self.ws_col = kwargs.get('ws_col', 0)

        self.set_colormap(data)
        self.set_label_text(data, **kwargs)
        self.get_element_sizes(data)
        self.get_figure_size(**kwargs)
        self.get_subplots_adjust()
        self.get_rc_label_position()
        self.get_legend_position()

        # Define the subplots
        fig, axes = \
            mplp.subplots(data.nrow, data.ncol,
                          figsize=[self.fig.size_inches[0], self.fig.size_inches[1]],
                          sharex=self.axes.share_x,
                          sharey=self.axes.share_y,
                          dpi=self.fig.dpi,
                          facecolor=self.fig.fill_color,
                          edgecolor=self.fig.edge_color,
                          linewidth=self.fig.edge_width,
                          )
        self.fig.obj = fig
        self.axes.obj = axes

        # Adjust the subplots size
        self.fig.obj.subplots_adjust(left=self.axes.position[0],
                                     right=self.axes.position[1],
                                     top=self.axes.position[2],
                                     bottom=self.axes.position[3],
                                     hspace=1.0*self.ws_row/self.axes.size[1],
                                     wspace=1.0*self.ws_col/self.axes.size[0],
                                     )

        # Reformat the axes variable if it is only one plot
        if not type(self.axes.obj) is np.ndarray:
            self.axes.obj = np.array([self.axes.obj])
        if len(self.axes.obj.shape) == 1:
            if data.nrow == 1:
                self.axes.obj = np.reshape(self.axes.obj, (1, -1))
            else:
                self.axes.obj = np.reshape(self.axes.obj, (-1, 1))

        # Twinning
        self.axes2.obj = np.array([[None]*self.ncol]*self.nrow)
        if self.axes.twin_x:
            for ir in range(0, self.nrow):
                for ic in range(0, self.ncol):
                    self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twinx()
        elif self.axes.twin_y:
            for ir in range(0, self.nrow):
                for ic in range(0, self.ncol):
                    self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twiny()

    def plot_box(self, ir, ic, data,**kwargs):
        """ Plot boxplot data

        Args:
            ir (int): subplot row index
            ic (int): subplot column index
            data (pd.DataFrame): data to plot

        Keyword Args:
            any kwargs allowed by the plotter function selected

        Returns:
            return the box plot object
        """

        if not self.box.on:
            return None

        bp = self.axes.obj[ir, ic].boxplot(data,
                                           labels=[''] * len(data),
                                           showfliers=False,
                                           boxprops={'color': self.box.edge_color,
                                                     'facecolor': self.box.fill_color},
                                           medianprops={'color': self.box.median_color},
                                           notch=self.box.notch,
                                           whiskerprops={'color': self.box.edge_color},
                                           capprops={'color': self.box.edge_color},
                                           patch_artist=True,
                                           zorder=5)

        return bp

    def plot_contour(self, ax, df, x, y, z):
        """
        Plot a contour plot
        """

        # Convert data type
        xx = np.array(df[x])
        yy = np.array(df[y])
        zz = np.array(df[z])

        # Make the grid
        xi = np.linspace(min(xx), max(xx))
        yi = np.linspace(min(yy), max(yy))
        zi = mlab.griddata(xx, yy, zz, xi, yi, interp='linear')

        if self.contour.filled:
            contour = ax.contourf
        else:
            contour = ax.contour
        cc = contour(xi, yi, zi, self.contour.levels, line_width=self.contour.width,
                     cmap=self.contour.cmap, zorder=2)

        if self.cbar.on:
            cbar = self.add_cbar(ax, cc)
        else:
            cbar = None

        return cc, cbar

    def plot_line(self, ir, ic, x0, y0, x1=None, y1=None,**kwargs):
        """
        Plot a simple line

        Args:
            ir (int): subplot row index
            ic (int): subplot column index
            x0 (float): min x coordinate of line
            x1 (float): max x coordinate of line
            y0 (float): min y coordinate of line
            y1 (float): max y coordinate of line
            kwargs: keyword args
        """

        if x1:
            x0 = [x0, x1]
        if y1:
            y0 = [y0, y1]
        self.axes.obj[ir, ic].plot(x0, y0,
                                   linestyle=kwargs.get('style', '-'),
                                   linewidth=kwargs.get('width', 1),
                                   color=kwargs.get('color', '#000000'),
                                   zorder=kwargs.get('zorder', 0))

    def plot_xy(self, ir, ic, iline, df, x, y, leg_name, twin, zorder=1,
                line_type=None, marker_disable=False):
        """ Plot xy data

        Args:
            x (np.array):  x data to plot
            y (np.array):  y data to plot
            color (str):  hex color code for line color (default='#000000')
            marker (str):  marker char string (default='o')
            points (bool):  toggle points on|off (default=False)
            line (bool):  toggle plot lines on|off (default=True)

        Keyword Args:
            any kwargs allowed by the plotter function selected

        Returns:
            return the line plot object
        """

        def format_marker(marker):
            """
            Format the marker string to mathtext
            """

            if marker in ['o', '+', 's', 'x', 'd', '^']:
                return marker
            else:
                return r'$%s$' % marker

        df = df.copy()

        if not line_type:
            line_type = self.lines
        else:
            line_type = getattr(self, line_type)

        # Select the axes
        if twin:
            ax = self.axes2.obj[ir, ic]
        else:
            ax = self.axes.obj[ir, ic]

        # Make the points
        points = None
        if self.markers.on and not marker_disable:
            if self.markers.jitter:
                df[x] = np.random.normal(df[x], 0.03, size=len(df[y]))
            points = ax.plot(df[x], df[y],
                             marker=format_marker(self.markers.type.get(iline)),
                             markerfacecolor=self.markers.fill_color.get(iline) \
                                             if self.markers.filled else 'none',
                             markeredgecolor=self.markers.edge_color.get(iline),
                             markeredgewidth=self.markers.edge_width,
                             linewidth=0,
                             markersize=self.markers.size,
                             zorder=40)

        # Make the line
        lines = None
        if line_type.on:
            lines = ax.plot(df[x], df[y],
                            color=line_type.color.get(iline),
                            linestyle='-' if line_type.style is None \
                                      else line_type.style,
                            linewidth=line_type.width)

        # Add a reference to the line to self.lines
        if leg_name:
            self.legend.values[leg_name] = points if points is not None else lines

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df

    def set_axes_colors(self, ir, ic):
        """
        Set axes colors (fill, alpha, edge)

        Args:
            ir (int): row index
            ic (int): col index

        """

        axes = self.get_axes()

        #for ax in axes:
        try:
            axes[0].obj[ir, ic].set_facecolor(axes[0].fill_color)
        except:
            axes[0].obj[ir, ic].set_axis_bgcolor(axes[0].fill_color)
        try:
            axes[-1].obj[ir, ic].set_edgecolor(axes[-1].edge_color)
        except:
            for f in ['bottom', 'top', 'right', 'left']:
                axes[-1].obj[ir, ic].spines[f].set_color(axes[-1].edge_color)

    def set_axes_grid_lines(self, ir, ic):
        """
        Style the grid lines and toggle visibility

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        """

        axes = self.get_axes()

        for ax in axes:
            # Turn off secondary gridlines
            if not ax.primary:
                ax.obj[ir, ic].grid(False, which='major')
                ax.obj[ir, ic].grid(False, which='minor')
                continue

            # Set major grid
            if self.grid_major_x.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='major', zorder=0,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.style,
                                          linewidth=self.grid_major.width)
            else:
                ax.obj[ir, ic].xaxis.grid(b=False, which='major')
            if self.grid_major_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='major', zorder=0,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.style,
                                          linewidth=self.grid_major.width)
            else:
                ax.obj[ir, ic].yaxis.grid(b=False, which='major')

            # Set minor grid
            if self.grid_minor_x.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='minor', zorder=0,
                                          color=self.grid_minor_x.color,
                                          linestyle=self.grid_minor_x.style,
                                          linewidth=self.grid_minor_x.width)
            if self.grid_minor_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='minor', zorder=0,
                                          color=self.grid_minor_y.color,
                                          linestyle=self.grid_minor_y.style,
                                          linewidth=self.grid_minor_y.width)

    def set_axes_labels(self, ir, ic):
        """
        Set the axes labels

        Args:
            ir (int): current row index
            ic (int): current column index
            kw (dict): kwargs dict

        """

        self.get_axes_label_position()

        axis = ['x', 'x2', 'y', 'y2', 'z']
        for ax in axis:
            label = getattr(self, 'label_%s' % ax)
            if not label.on:
                continue

            if '2' in ax:
                axes = self.axes2.obj[ir, ic]
                pad = self.ws_label_tick*2
            else:
                axes = self.axes.obj[ir, ic]
                pad = self.ws_label_tick

            # Toggle label visibility
            if not self.separate_labels:
                if ax == 'x' and ir != self.nrow - 1 and self.nwrap == 0: continue
                if ax == 'x2' and ir != 0: continue
                if ax == 'y' and ic != 0: continue
                if ax == 'y2' and ic != self.ncol - 1 and (2*ir + ic + 1) != self.nwrap: continue

            # Add the label
            # getattr(axes, 'set_%slabel' % ax[0])(label.text,
            #                                      fontsize=label.font_size,
            #                                      weight=label.font_weight,
            #                                      style=label.font_style,
            #                                      color=label.font_color,
            #                                      rotation=label.rotation,
            #                                      fontname=label.font,
            #                                      labelpad=pad,
            #                                      )
            self.add_label(axes, label.text, **self.make_kwargs(label))

            # if ax == 'y':
            #     # self.label_y.position[0] = (-self.tick_labels_major_y.size[0] - self.label_y.size[1]) / self.axes.size[0]
            #     # self.label_y.position[3] = (self.axes.size[1] - self.label_y.size[1]) / (2 * self.axes.size[1])
            #     # self.add_label(axes, self.label_y.text, **self.make_kwargs(self.label_y))

            #     self.label_y.position[0] = -np.floor(self.labtick_y) / self.axes.size[0]
            #     self.label_y.position[3] = (self.axes.size[1] - self.label_y.size[1]) / (2 * self.axes.size[1])
            #     self.add_label(axes, 'Y1', **self.make_kwargs(self.label_y))
            #     st()

    def set_axes_scale(self, ir, ic):
        """
        Set the scale type of the axes

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        Returns:
            axes scale type
        """

        axes = self.get_axes()

        for ax in axes:
            if ax.scale is None:
                continue
            else:
                if str(ax.scale).lower() in ['loglog', 'semilogx', 'logx', 'log']:
                    ax.obj[ir, ic].set_xscale('log')
                elif str(ax.scale).lower() in ['symlog']:
                    ax.obj[ir, ic].set_xscale('symlog')
                elif str(ax.scale).lower() in ['logit']:
                    ax.obj[ir, ic].set_xscale('logit')
                if str(ax.scale).lower() in ['loglog', 'semilogy', 'logy', 'log']:
                    ax.obj[ir, ic].set_yscale('log')
                elif str(ax.scale).lower() in ['symlog']:
                    ax.obj[ir, ic].set_yscale('symlog')
                elif str(ax.scale).lower() in ['logit']:
                    ax.obj[ir, ic].set_yscale('logit')

    def set_axes_ranges(self, ir, ic, ranges):
        """
        Set the axes ranges

        Args:
            ir (int): subplot row index
            ic (int): subplot col index
            limits (dict): min/max axes limits for each axis

        """

        for k, v in ranges.items():
            if k == 'xmin' and v is not None:
                self.axes.obj[ir, ic].set_xlim(left=v)
            elif k == 'x2min' and v is not None:
                self.axes2.obj[ir, ic].set_xlim(left=v)
            elif k == 'xmax' and v is not None:
                self.axes.obj[ir, ic].set_xlim(right=v)
            elif k == 'x2max' and v is not None:
                self.axes2.obj[ir, ic].set_xlim(right=v)
            elif k == 'ymin' and v is not None:
                self.axes.obj[ir, ic].set_ylim(bottom=v)
            elif k == 'y2min' and v is not None:
                self.axes2.obj[ir, ic].set_ylim(bottom=v)
            elif k == 'ymax' and v is not None:
                self.axes.obj[ir, ic].set_ylim(top=v)
            elif k == 'y2max' and v is not None:
                self.axes2.obj[ir, ic].set_ylim(top=v)
            # elif k == 'zmin' and v is not None and self.cbar.on:
            #     self.cbar.obj[ir, ic].set_clim(bottom=v)
            # elif k == 'zmax' and v is not None and self.cbar.on:
            #     self.cbar.obj[ir, ic].set_clim(top=v)

    def set_axes_rc_labels(self, ir, ic, ax=None):
        """
        Add the row/column label boxes and wrap titles

        Args:
            ir (int): current row index
            ic (int): current column index

        """

        wrap, row, col = None, None, None
        if not ax:
            ax = self.axes.obj[ir, ic]

        # Wrap title  --> this guy's text size is not defined in get_elements_size
        if ir == 0 and ic == 0 and self.title_wrap.on:
            title = self.add_label(ax, self.title_wrap.text,
                                   **self.make_kwargs(self.title_wrap))

        # Row labels
        if ic == self.ncol-1 and self.label_row.on and not self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_row.text_size[ir, ic]
            else:
                text_size = None
            row = self.add_label(ax, '%s=%s' %
                                (self.label_row.text, self.label_row.values[ir]),
                                offset=True, **self.make_kwargs(self.label_row))

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_col.text_size[ir, ic]
            else:
                text_size = None
            if self.label_wrap.on:
                text = ' | '.join([str(f) for f in utl.validate_list(
                    self.label_wrap.values[ir*self.ncol + ic])])
                scol = self.add_label(ax, text,
                                     **self.make_kwargs(self.label_wrap))
            else:
                text = '%s=%s' % (self.label_col.text, self.label_col.values[ic])
                col = self.add_label(ax, text,
                                     **self.make_kwargs(self.label_col))

        return wrap, row, col

    def set_axes_ticks(self, ir, ic):
        """
        Configure the axes tick marks

        Args:
            ax (mpl axes): current axes to scale
            kw (dict): kwargs dict
            y_only (bool): flag to access on the y-axis ticks

        """

        axes = [f.obj[ir, ic] for f in [self.axes, self.axes2] if f.on]

        for ia, aa in enumerate(axes):

            if ia == 0:
                lab = ''
            else:
                lab = '2'

            # Turn off scientific (how do we force it?)
            if not self.tick_labels_major_x.sci \
                    and self.plot_func != 'plot_box' \
                    and self.axes.scale not in ['logx', 'semilogx', 'loglog', 'log'] \
                    and (not self.axes.share_x or ir==0 and ic==0):
                if 'box' not in self.plot_func and (ia == 0 or self.axes.twin_y):
                    axes[ia].get_xaxis().get_major_formatter().set_scientific(False)

            if not self.tick_labels_major_y.sci \
                    and self.axes.scale not in ['logy', 'semilogy', 'loglog', 'log'] \
                    and (not self.axes.share_y or ir==0 and ic==0):
                try:
                    axes[ia].get_yaxis().get_major_formatter().set_scientific(False)
                except:
                    print('bah2: axes[ia].get_xaxis().get_major_formatter().set_scientific(False)')

            # General tick params
            if ia == 0:
                axes[ia].minorticks_on()
                axes[ia].tick_params(axis='both',
                                     which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color,
                                     labelcolor=self.tick_labels_major.font_color,
                                     labelsize=self.tick_labels_major.font_size,
                                     top=False,
                                     bottom=self.ticks_major_x.on,
                                     right=False if self.axes.twin_x
                                           else self.ticks_major_y.on,
                                     left=self.ticks_major_y.on,
                                     length=self.ticks_major.size[0],
                                     width=self.ticks_major.size[1],
                                     )
                axes[ia].tick_params(axis='both',
                                     which='minor',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_minor.color,
                                     labelcolor=self.tick_labels_minor.font_color,
                                     labelsize=self.tick_labels_minor.font_size,
                                     top=False,
                                     bottom=self.ticks_minor_x.on,
                                     right=False if self.axes.twin_x
                                           else self.ticks_minor_y.on,
                                     left=self.ticks_minor_y.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )
            elif self.axes.twin_x:
                axes[ia].minorticks_on()
                axes[ia].tick_params(which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color,
                                     labelcolor=self.tick_labels_major.font_color,
                                     labelsize=self.tick_labels_major.font_size,
                                     right=self.ticks_major_y2.on,
                                     length=self.ticks_major.size[0],
                                     width=self.ticks_major.size[1],
                                     )
                axes[ia].tick_params(which='minor',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_minor.color,
                                     labelcolor=self.tick_labels_minor.font_color,
                                     labelsize=self.tick_labels_minor.font_size,
                                     right=self.ticks_minor_y2.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )
            elif self.axes.twin_y:
                axes[ia].minorticks_on()
                axes[ia].tick_params(which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color,
                                     labelcolor=self.tick_labels_major.font_color,
                                     labelsize=self.tick_labels_major.font_size,
                                     top=self.ticks_major_x2.on,
                                     length=self.ticks_major.size[0],
                                     width=self.ticks_major.size[1],
                                     )
                axes[ia].tick_params(which='minor',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_minor.color,
                                     labelcolor=self.tick_labels_minor.font_color,
                                     labelsize=self.tick_labels_minor.font_size,
                                     top=self.ticks_minor_x2.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )

            tp = mpl_get_ticks(axes[ia],
                               getattr(self, 'ticks_major_x%s' % lab).on,
                               getattr(self, 'ticks_major_y%s' % lab).on)

            # Set custom tick increment
            redo = True
            xinc = getattr(self, 'ticks_major_x%s' % lab).increment
            if xinc is not None:
                axes[ia].set_yticks(np.arange(tp['x']['min'], tp['x']['max'], xinc))
                redo = True
            yinc = getattr(self, 'ticks_major_y%s' % lab).increment
            if yinc is not None:
                axes[ia].set_yticks(np.arange(tp['y']['min'], tp['y']['max'], yinc))
                redo = True
            if redo:
                tp = mpl_get_ticks(axes[ia],
                                   getattr(self, 'ticks_major_x%s' % lab).on,
                                   getattr(self, 'ticks_major_y%s' % lab).on)

            # Force ticks
            if self.separate_ticks or self.nwrap > 0:
                mplp.setp(axes[ia].get_xticklabels(), visible=True)
            if self.separate_ticks:
                mplp.setp(axes[ia].get_yticklabels(), visible=True)
            if not self.separate_ticks and (ic != self.ncol - 1 and (2*ir + ic + 1) != self.nwrap) and self.axes.twin_x and ia == 1:
                mplp.setp(axes[ia].get_yticklabels(), visible=False)
            if not self.separate_ticks and ir != 0 and self.axes.twin_y and ia == 1:
                mplp.setp(axes[ia].get_xticklabels(), visible=False)

            # Major rotation
            if self.tick_labels_major_x.on \
                    and self.tick_labels_major_x.rotation != 0:
                for text in axes[ia].get_xticklabels():
                    text.set_rotation(self.tick_labels_major_x.rotation)
            if self.tick_labels_major_y.on \
                    and self.tick_labels_major_y.rotation != 0:
                for text in axes[ia].get_yticklabels():
                    text.set_rotation(self.tick_labels_major_y.rotation)

            # Tick label shorthand
            tlmajx = getattr(self, 'tick_labels_major_x%s' % lab)
            tlmajy = getattr(self, 'tick_labels_major_y%s' % lab)

            # Check for overlapping major tick labels
            if self.tick_cleanup and tlmajx.on and tlmajy.on:
                xc = [0, -tlmajx.size[1]/2-self.ws_ticks_ax + ia*self.axes.size[1]]
                yc = [-tlmajy.size[0]/2-self.ws_ticks_ax, 0]
                yf = [-tlmajy.size[0]/2-self.ws_ticks_ax, self.axes.size[1]]
                buf = 3
                if len(tp['x']['ticks']) > 2:
                    delx = self.axes.size[0]/(len(tp['x']['ticks'])-2)
                else:
                    delx = self.axes.size[0] - tlmajx.size[0]
                if len(tp['y']['ticks']) > 2:
                    dely = self.axes.size[1]/(len(tp['y']['ticks'])-2)
                else:
                    dely = self.axes.size[1] - tlmajy.size[1]
                x2x, y2y = [], []
                xw, xh = tlmajx.size
                yw, yh = tlmajy.size

                # Calculate overlaps
                x0y0 = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yc])
                x0yf = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yf])
                for ix in range(0, len(tp['x']['ticks']) - 1):
                    x2x += [utl.rectangle_overlap([xw+2*buf, xh+2*buf, [delx*ix,0]],
                                                  [xw+2*buf, xh+2*buf, [delx*(ix+1), 0]])]
                for iy in range(0, len(tp['y']['ticks']) - 1):
                    y2y += [utl.rectangle_overlap([yw+2*buf, yh+2*buf, [0,dely*iy]],
                                                  [yw+2*buf, yh+2*buf, [0,dely*(iy+1)]])]
                # x and y at the origin
                if x0y0 and tp['y']['first']==0:
                    tp['y']['label_text'][tp['y']['first']] = ''
                if x0yf and self.axes.twin_y:
                    tp['y']['label_text'][tp['y']['last']] = ''

                # x overlapping x (this will fail if plot is so small that odd elements overlap)
                if any(x2x) and ((self.axes.share_x and ir==0 and ic==0) \
                        or not self.axes.share_x) \
                        and tp['x']['first'] != -999 and tp['x']['last'] != -999:
                    for i in range(tp['x']['first'] + 1, tp['x']['last'] + 1, 2):
                        tp['x']['label_text'][i] = ''

                # y overlapping y
                if any(y2y) and ((self.axes.share_y and ir==0 and ic==0) \
                        or not self.axes.share_y) \
                        and tp['y']['first'] != -999 and tp['y']['last'] != -999:
                    for i in range(tp['y']['first'], tp['y']['last'] + 1, 2):
                        tp['y']['label_text'][i] = ''

                # overlapping labels between row, col, and wrap plots
                if tp['x']['last'] != -999:
                    last_x = tp['x']['labels'][tp['x']['last']][1]
                    last_x_pos = last_x/(tp['x']['max']-tp['x']['min'])
                    last_x_px = (1-last_x_pos)*self.axes.size[0]
                    if self.ncol > 1 and \
                            xw > last_x_px + self.ws_col - self.ws_tick_tick_minimum and \
                            ic < self.ncol - 1:
                        tp['x']['label_text'][tp['x']['last']] = ''

                if tp['y']['last'] != -999:
                    last_y = tp['y']['labels'][tp['y']['last']][1]
                    last_y_pos = last_y/(tp['y']['max']-tp['y']['min'])
                    last_y_px = (1-last_y_pos)*self.axes.size[1]
                    if self.nrow > 1 and \
                            yh > last_y_px + self.ws_col - self.ws_tick_tick_minimum and \
                            ir < self.nrow - 1 and self.nwrap == 0:
                        tp['y']['label_text'][tp['y']['last']] = ''

                # overlapping last y and first x between row, col, and wraps
                if self.nrow > 1 and ir < self.nrow-1:
                    x2y = utl.rectangle_overlap([xw, xh, xc],
                                                [yw, yh, [yc[0], yc[1]-self.ws_row]])
                    if x2y and \
                            tp['y']['min'] == tp['y']['ticks'][tp['y']['first']]:
                        tp['x']['label_text'][0] = ''

                axes[ia].set_xticklabels(tp['x']['label_text'])
                axes[ia].set_yticklabels(tp['y']['label_text'])

            # Disable major tick labels
            elif not self.tick_labels_major.on:
                axes[ia].tick_params(which='major',
                                     labelbottom='off', labelleft='off',
                                     labelright='off', labeltop='off')

            # Turn on minor tick labels
            ax = ['x', 'y']
            sides = {}
            sides['x'] = {'labelbottom': 'off'}
            sides['x2'] = {'labeltop': 'off'}
            sides['y'] = {'labelleft': 'off'}
            sides['y2'] = {'labelright': 'off'}

            tlminon = False  # "tick label min"

            for axx in ax:
                axl = '%s%s' % (axx, lab)
                tlmin = getattr(self, 'tick_labels_minor_%s' % axl)

                # this is failing in 2.0.2 and I'm not sure what it is solving in 1.5.1
                # if getattr(self, 'ticks_minor_%s' % axl).number is not None:
                #     num_minor = getattr(self, 'ticks_minor_%s' % axl).number
                #     if axx=='x' and self.axes.scale in ['logx', 'semilogx', 'loglog', 'log']:
                #         loc = LogLocator()
                #     elif axx=='y' and self.axes.scale in ['logy', 'semilogy', 'loglog', 'log']:
                #         loc = LogLocator()
                #     else:
                #         loc = AutoMinorLocator(num_minor+1)
                #     getattr(axes[ia], '%saxis' % axx).set_minor_locator(loc)
                if not self.separate_labels and axl == 'x' and ir != self.nrow - 1 and self.nwrap == 0 or \
                        not self.separate_labels and axl == 'y2' and ic != self.ncol - 1 and self.nwrap == 0 or \
                        not self.separate_labels and axl == 'x2' and ir != 0 or \
                        not self.separate_labels and axl == 'y' and ic != 0 or \
                        not self.separate_labels and axl == 'y2' and ic != self.ncol - 1 and (2*ir + ic + 1) != self.nwrap:
                    axes[ia].tick_params(which='minor', **sides[axl])

                elif tlmin.on:
                    # THE DECIMAL THING WONT WORK FOR LOG!

                    tp = mpl_get_ticks(axes[ia],
                                       getattr(self, 'ticks_major_x%s' % lab).on,
                                       getattr(self, 'ticks_major_y%s' % lab).on)
                    inc = tp[axx]['labels'][1][1] - tp[axx]['labels'][0][1]
                    minor_ticks = [f[1] for f in
                                   tp[axx]['labels']][len(tp[axx]['ticks']):]
                    number = len([f for f in minor_ticks if f < inc]) + 1
                    decimals = utl.get_decimals(inc/number)
                    getattr(axes[ia], '%saxis' % axx).set_minor_formatter(
                        ticker.FormatStrFormatter('%%.%sf' % (decimals)))
                    tlminon = True

                    # Rotation
                    if tlmin.rotation != 0:
                        for text in getattr(axes[ia], 'get_%sminorticklabels' % axx)():
                            text.set_rotation(tlmin.rotation)

                    # Minor tick overlap cleanup
                    if self.tick_cleanup and tlminon:
                        tp = mpl_get_ticks(axes[ia],
                                           getattr(self, 'ticks_major_x%s' % lab).on,
                                           getattr(self, 'ticks_major_y%s' % lab).on)  # need to update
                        buf = 1.99
                        if axx == 'x':
                            wh = 0
                        else:
                            wh = 1
                        delmaj = self.axes.size[wh]/(len(tp[axx]['ticks'])-2)
                        labels = tp[axx]['label_text'][len(tp[axx]['ticks']):]
                        delmin = delmaj/number

                        # Check overlap with first and last major label
                        wipe = []
                        m0 = len(tp[axx]['ticks'])
                        majw = getattr(self, 'tick_labels_major_%s' % axl).size[wh]/2
                        if delmaj - 2*majw < tlmin.size[wh] + buf:
                            # No room for any ticks
                            warnings.warn('Insufficient space between %s major tick labels for minor tick labels. Skipping...' % axx)
                            wipe = list(range(0, number-1))
                        elif majw > delmin - tlmin.size[wh]/2 - buf:
                            wipe += [0, number - 2]

                        # There is a weird bug where a tick can be both major and minor; need to remove
                        dups = [i+m0 for i, f in enumerate(minor_ticks) if f in tp[axx]['ticks']]
                        if len(dups) == 0:
                            dups = [-1, len(tp[axx]['label_text'])]
                        if dups[0] != -1:
                            dups = [-1] + dups
                        if dups[len(dups)-1] != len(dups):
                            dups = dups + [len(tp[axx]['label_text'])]
                        temp = []
                        for j in range(1, len(dups)):
                            temp += tp[axx]['label_text'][dups[j-1]+1:dups[j]]

                        # Disable ticks
                        for i, text in enumerate(getattr(axes[ia], 'get_%sminorticklabels' % axx)()):
                            if i in wipe:
                                vals = temp[m0+i::number-1]
                                temp[m0+i::number-1] = ['']*len(vals)

                        # Put back in duplicates
                        tp[axx]['label_text'] = []
                        dups[0] = 0
                        for j in range(1, len(dups)):
                            tp[axx]['label_text'] += temp[dups[j-1]:dups[j]]
                            if j < len(dups) - 1:
                                tp[axx]['label_text'] += ['']

                        # Check minor to minor overlap
                        if tlmin.size[wh] + buf > delmin:
                            for itick, tick in enumerate(tp[axx]['label_text'][m0+1:]):
                                if tp[axx]['label_text'][m0+itick] != '':
                                    tp[axx]['label_text'][m0+itick+1] = ''

                        # Set the labels
                        getattr(axes[ia], 'set_%sticklabels' % axx) \
                            (tp[axx]['label_text'][m0:], minor=True)

    def set_colormap(self, data, **kwargs):
        """
        Replace the color list with discrete values from a colormap

        Args:
            data (Data object)
        """

        if not self.cmap:
            return

        try:
            # Conver the color map into discrete colors
            cmap = mplp.get_cmap(self.cmap)
            color_list = []
            for i in range(0, len(data.legend_vals)):
                color_list += \
                    [mplc_to_hex(cmap((i+1)/(len(data.legend_vals)+1)), False)]

            # Reset colors
            if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
                self.label_y.font_color = color_list[0]
            if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
                self.label_y2.font_color = color_list[1]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x.font_color = color_list[0]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x2.font_color = color_list[1]

            line_colors = RepeatedList(color_list, 'line_colors')
            self.lines.color = copy.copy(line_colors)

            marker_edge_color = RepeatedList(color_list, 'marker_edge_color')
            marker_edge_color.shift = 0
            marker_fill_color = RepeatedList(color_list, 'marker_fill_color')
            marker_fill_color.shift = 0
            self.markers.edge_color = copy.copy(marker_edge_color)
            self.markers.fill_color = copy.copy(marker_fill_color)

        except:
            print('Could not find a colormap called "%s". '
                  'Using default colors...' % self.cmap)

    def set_figure_title(self):
        """
        Add a figure title
        """

        if self.title.on:
            self.get_title_position()
            self.add_label(self.axes.obj[0, 0], self.title.text, offset=True,
                           **self.make_kwargs(self.title))

    def set_scientific(self, ax):
        """
        Turn off scientific notation

        Args:
            ax: axis to adjust

        Returns:
            updated axise
        """

        if not self.tick_labels_major_x.sci and self.plot_func != 'boxplot' \
                and self.axes.scale not in ['logx', 'semilogx', 'loglog', 'symlog', 'logit', 'log']:
            ax.get_xaxis().get_major_formatter().set_scientific(False)
        if not self.tick_labels_major_y.sci \
                and self.axes.scale not in ['logy', 'semilogy', 'loglog', 'symlog', 'logit', 'log']:
            ax.get_yaxis().get_major_formatter().set_scientific(False)

        return ax



