from ..import fcp
import importlib
import os, sys
import pandas as pd
import pdb
import datetime
import time
import numpy as np
import copy
import decimal
import math
from .. colors import *
from .. utilities import RepeatedList
from .. import utilities as utl
from distutils.version import LooseVersion
from random import randint
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

db = pdb.set_trace

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
                   'center': 10, 10: 10,
                   'below': 11, 11:11},
                   )

LOGX = ['logx', 'semilogx', 'log', 'loglog']
LOGY = ['logy', 'semilogy', 'log', 'loglog']
LOGZ = []
SYMLOGX = ['symlogx', 'symlog']
SYMLOGY = ['symlogy', 'symlog']
LOGITX = ['logitx', 'logit']
LOGITY = ['logity', 'logit']
LOG_ALLX = LOGX + SYMLOGX + LOGITX
LOG_ALLY = LOGY + SYMLOGY + LOGITY
ENGINE = ''


class BaseLayout:
    def __init__(self, plot_func, data, **kwargs):
        """
        Generic layout properties class

        Args:
            plot_func (str): name of plot function to use
            data (Data class): data values
            **kwargs: styling, spacing kwargs

        """

        self.plot_func = plot_func

        # Reload default file
        self.fcpp, color_list, marker_list = utl.reload_defaults(kwargs.get('theme', None))

        # Figure
        self.fig = Element('fig', self.fcpp, kwargs,
                           edge_width=3)

        # Color list
        if 'line_color' in kwargs.keys():
            color_list = kwargs['line_color']
        elif kwargs.get('colors'):
            colors = utl.validate_list(kwargs.get('colors'))
            for icolor, color in enumerate(colors):
                if type(color) is int:
                    colors[icolor] = DEFAULT_COLORS[color]
            color_list = colors
        elif 'colors' in self.fcpp.keys():
            colors = utl.validate_list(self.fcpp['colors'])
            for icolor, color in enumerate(colors):
                if type(color) is int:
                    colors[icolor] = DEFAULT_COLORS[color]
            color_list = colors
        elif not color_list:
            color_list = copy.copy(DEFAULT_COLORS)
        self.cmap = kwargs.get('cmap', None)
        if self.plot_func in ['plot_contour', 'plot_heatmap']:
            self.cmap = utl.kwget(kwargs, self.fcpp, 'cmap', None)

        # Axis
        self.ax = ['x', 'y', 'x2', 'y2']
        spines = utl.kwget(kwargs, self.fcpp, 'spines', True)
        self.axes = Element('ax', self.fcpp, kwargs,
                            size=utl.kwget(kwargs, self.fcpp,
                                       'ax_size', [400, 400]),
                            edge_color='#aaaaaa',
                            fill_color='#eaeaea',
                            primary=True,
                            scale=kwargs.get('ax_scale', None),
                            share_x=kwargs.get('share_x', None),
                            share_y=kwargs.get('share_y', None),
                            share_z=kwargs.get('share_z', None),
                            share_x2=kwargs.get('share_x2', None),
                            share_y2=kwargs.get('share_y2', None),
                            share_col = kwargs.get('share_col', None),
                            share_row = kwargs.get('share_row', None),
                            spine_bottom = utl.kwget(kwargs, self.fcpp,
                                                     'spine_bottom', spines),
                            spine_left = utl.kwget(kwargs, self.fcpp,
                                                   'spine_left', spines),
                            spine_right = utl.kwget(kwargs, self.fcpp,
                                                    'spine_right', spines),
                            spine_top = utl.kwget(kwargs, self.fcpp,
                                                  'spine_top', spines),
                            twin_x=kwargs.get('twin_x', False),
                            twin_y=kwargs.get('twin_y', False),
                            )
        for isize, size in enumerate(self.axes.size):
            if 'group' in str(size) and self.plot_func == 'plot_box':
                self.axes.size[isize] = \
                    int(size.split('*')[0].replace(' ', '')) * len(data.indices)
        twinned = kwargs.get('twin_x', False) or kwargs.get('twin_y', False)
        self.axes2 = Element('ax', self.fcpp, kwargs,
                             on=True if twinned else False,
                             edge_color=self.axes.edge_color,
                             fill_color=self.axes.fill_color,
                             primary=False,
                             scale=kwargs.get('ax2_scale', None),
                             xmin=kwargs.get('x2min', None),
                             xmax=kwargs.get('x2max', None),
                             ymin=kwargs.get('y2min', None),
                             ymax=kwargs.get('y2max', None),
                             )

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
                v = kwargs[k]
                if k == 'label_%s' % lab:
                    k = 'label_%s_text' % lab
                setattr(getattr(self, 'label_%s' % lab),
                        k.replace('label_%s_' % lab, ''), v)

            # Update alphas
            getattr(self, 'label_%s' % lab).color_alpha('fill_color', 'fill_alpha')
            getattr(self, 'label_%s' % lab).color_alpha('edge_color', 'edge_alpha')

        # Turn off secondary labels
        if not self.axes.twin_y:
            self.label_x2.on = False
        if not self.axes.twin_x:
            self.label_y2.on = False

        # Twinned label colors
        if 'legend' not in kwargs.keys():
            color_list_unique = pd.Series(color_list).unique()
            if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
                self.label_y.font_color = color_list_unique[0]
            if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
                self.label_y2.font_color = color_list_unique[1]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x.font_color = color_list_unique[0]
            if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                self.label_x2.font_color = color_list_unique[1]

        # Figure title
        title = utl.kwget(kwargs, self.fcpp, 'title', None)
        self.title = Element('title', self.fcpp, kwargs,
                             on=True if title is not None else False,
                             text=title if title is not None else None,
                             font_color='#333333',
                             font_size=18,
                             font_weight='bold',
                             align='center',
                             )
        if type(self.title.size) is not list:
            self.title.size = [self.axes.size[0], self.title.size]

        # Ticks
        if 'ticks' in kwargs.keys() and 'ticks_major' not in kwargs.keys():
            kwargs['ticks_major'] = kwargs['ticks']
        ticks_length = utl.kwget(kwargs, self.fcpp, 'ticks_length', 6.2)
        ticks_width = utl.kwget(kwargs, self.fcpp, 'ticks_width', 2.2)
        self.ticks_major = Element('ticks_major', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major', True),
                                   color='#ffffff',
                                   direction=utl.kwget(kwargs, self.fcpp,
                                                       'ticks_major_direction',
                                                       'in'),
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
                            color=copy.copy(self.ticks_major.color),
                            increment=utl.kwget(kwargs, self.fcpp,
                                                'ticks_major_%s_increment' % ax,
                                                self.ticks_major.increment),
                            padding=utl.kwget(kwargs, self.fcpp,
                                              'ticks_major_%s_padding' % ax,
                                              self.ticks_major.padding),
                            size=self.ticks_major.size,
                            ))

        if 'tick_labels' in kwargs.keys() \
                and 'tick_labels_major' not in kwargs.keys():
            kwargs['tick_labels_major'] = kwargs['tick_labels']
        for k, v in kwargs.copy().items():
            if 'tick_labels' in k and 'major' not in k and 'minor' not in k:
                kwargs['tick_labels_major%s' % k.split('tick_labels')[1]] = v
        self.tick_labels_major = \
            Element('tick_labels_major', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_major',
                                 kwargs.get('tick_labels', True)),
                    edge_alpha=0 if not kwargs.get('tick_labels_edge_alpha', None) and \
                                    not kwargs.get('tick_labels_major_edge_alpha', None) and \
                                    not kwargs.get('tick_labels_major_edge_color', None) \
                                    else 1,
                    fill_alpha=0 if not kwargs.get('tick_labels_fill_alpha', None) and \
                                    not kwargs.get('tick_labels_major_fill_alpha', None) and \
                                    not kwargs.get('tick_labels_major_fill_color', None) \
                                    else 1,
                    font_size=13,
                    offset=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_major_offset', False),
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_major_padding', 4),
                    )
        kwargs = self.from_list(self.tick_labels_major,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_major', kwargs)

        for ax in self.ax + ['z']:
            fill_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_fill_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_fill_alpha',
                                             None))
            fill_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_fill_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_fill_color',
                                             None))
            if not fill_alpha and fill_color:
                fill_alpha = 1
            elif not fill_alpha and not fill_color:
                fill_alpha = 0
            if not fill_color:
                fill_color = copy.copy(self.tick_labels_major.fill_color)

            edge_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_edge_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_edge_alpha',
                                             None))
            edge_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_%s_edge_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_edge_color',
                                             None))
            if not edge_alpha and edge_color:
                edge_alpha = 1
            elif not edge_alpha and not edge_color:
                edge_alpha = 0
            if not edge_color:
                edge_color = copy.copy(self.tick_labels_major.edge_color)

            if '2' in ax:
                axl = '2'
            else:
                axl = ''
            if getattr(self, 'axes%s' % axl).scale in globals()['LOG%s' % ax[0].upper()] and \
                    not utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, False) and \
                    'sci_%s' % ax not in kwargs.keys():
                kwargs['sci_%s' % ax] = 'best'

            setattr(self, 'tick_labels_major_%s' % ax,
                    Element('tick_labels_major_%s' % ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_major_%s' % ax,
                                     self.tick_labels_major.on),
                        edge_color=edge_color,
                        edge_alpha=edge_alpha,
                        edge_width=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_edge_width',
                                       self.tick_labels_major.edge_width),
                        fill_color=fill_color,
                        fill_alpha=fill_alpha,
                        font=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_font',
                                       self.tick_labels_major.font),
                        font_color=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_font_color',
                                       self.tick_labels_major.font_color),
                        font_size=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_font_size',
                                       self.tick_labels_major.font_size),
                        font_style=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_font_style',
                                       self.tick_labels_major.font_style),
                        font_weight=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_font_style',
                                       self.tick_labels_major.font_style),
                        offset=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_offset',
                                       self.tick_labels_major.offset),
                        padding=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_padding',
                                       self.tick_labels_major.padding),
                        rotation=utl.kwget(kwargs, self.fcpp, 'tick_labels_major_rotation',
                                       self.tick_labels_major.rotation),
                        size=[0, 0],
                        sci=utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, 'best'),
                        ))
        self.auto_tick_threshold = utl.kwget(kwargs, self.fcpp,
                                             'auto_tick_threshold', [1e-6, 1e6])

        self.ticks_minor = Element('ticks_minor', self.fcpp, kwargs,
                                   on=utl.kwget(kwargs, self.fcpp,
                                                'ticks_minor', False),
                                   color='#ffffff',
                                   direction=utl.kwget(kwargs, self.fcpp,
                                                       'ticks_minor_direction',
                                                       'in'),
                                   number=utl.kwget(kwargs, self.fcpp,
                                                    'ticks_minor_number',
                                                    3),
                                   padding=utl.kwget(kwargs, self.fcpp,
                                                     'ticks_minor_padding',
                                                     4),
                                   size=[utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_length',
                                                   ticks_length*0.67),
                                         utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_width',
                                                   ticks_width*0.6)],
                                   )
        kwargs = self.from_list(self.ticks_minor,
                                ['color', 'number', 'padding'],
                                'ticks_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'ticks_minor_%s' % ax,
                    Element('ticks_minor_%s' % ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'ticks_minor_%s' % ax, self.ticks_minor.on),
                        color=copy.copy(self.ticks_minor.color),
                        number=utl.kwget(kwargs, self.fcpp,
                                         'ticks_minor_%s_number' % ax,
                                         self.ticks_minor.number),
                        padding=utl.kwget(kwargs, self.fcpp,
                                          'ticks_minor_%s_padding' % ax,
                                          self.ticks_minor.padding),
                        size=self.ticks_minor._size,
                        ))
            if 'ticks_minor_%s_number' % ax in kwargs.keys():
                getattr(self, 'ticks_minor_%s' % ax).on = True

        self.tick_labels_minor = \
            Element('tick_labels_minor', self.fcpp, kwargs,
                    on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_minor',
                                 False),
                    edge_alpha=0 if not kwargs.get('tick_labels_edge_alpha', None) and \
                                    not kwargs.get('tick_labels_minor_edge_alpha', None) and \
                                    not kwargs.get('tick_labels_minor_edge_color', None) \
                                    else 1,
                    fill_alpha=0 if not kwargs.get('tick_labels_fill_alpha', None) and \
                                    not kwargs.get('tick_labels_minor_fill_alpha', None) and \
                                    not kwargs.get('tick_labels_minor_fill_color', None) \
                                    else 1,
                    font_size=10,
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_minor_padding', 3),
                    )
        kwargs = self.from_list(self.tick_labels_minor,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_minor', kwargs)
        for ax in self.ax:
            fill_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_minor_%s_fill_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_fill_alpha',
                                             None))
            fill_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_minor_%s_fill_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_fill_color',
                                             None))
            if not fill_alpha and fill_color:
                fill_alpha = 1
            elif not fill_alpha and not fill_color:
                fill_alpha = 0
            if not fill_color:
                fill_color = copy.copy(self.tick_labels_minor.fill_color)

            edge_alpha = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_minor_%s_edge_alpha' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_edge_alpha',
                                             None))
            edge_color = utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_minor_%s_edge_color' % ax,
                                   utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_edge_color',
                                             None))
            if not edge_alpha and edge_color:
                edge_alpha = 1
            elif not edge_alpha and not edge_color:
                edge_alpha = 0
            if not edge_color:
                edge_color = copy.copy(self.tick_labels_minor.edge_color)

            setattr(self, 'tick_labels_minor_%s' %ax,
                    Element('tick_labels_minor_%s' %ax, self.fcpp, kwargs,
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_minor_%s' % ax,
                                     self.tick_labels_minor.on),
                        edge_color=kwargs.get('tick_labels_minor.edge_color',
                                              self.tick_labels_minor.edge_color),
                        edge_alpha=kwargs.get('tick_labels_minor_edge_alpha',
                                       self.tick_labels_minor.edge_alpha),
                        edge_width=kwargs.get('tick_labels_minor_edge_width',
                                       self.tick_labels_minor.edge_width),
                        fill_color=kwargs.get('tick_labels_minor_fill_color',
                                       self.tick_labels_minor.fill_color),
                        fill_alpha=kwargs.get('tick_labels_minor_fill_alpha',
                                       self.tick_labels_minor.fill_alpha),
                        font=kwargs.get('tick_labels_minor_font',
                                       self.tick_labels_minor.font),
                        font_color=kwargs.get('tick_labels_minor_font_color',
                                       self.tick_labels_minor.font_color),
                        font_size=kwargs.get('tick_labels_minor_font_size',
                                       self.tick_labels_minor.font_size),
                        font_style=kwargs.get('tick_labels_minor_font_style',
                                       self.tick_labels_minor.font_style),
                        font_weight=kwargs.get('tick_labels_minor_font_style',
                                       self.tick_labels_minor.font_style),
                        padding=kwargs.get('tick_labels_minor_padding',
                                       self.tick_labels_minor.padding),
                        rotation=kwargs.get('tick_labels_minor_rotation',
                                       self.tick_labels_minor.rotation),
                        size=[0, 0],
                        sci=utl.kwget(kwargs, self.fcpp, 'sci_%s' % ax, False),
                        ))
            if getattr(self, 'tick_labels_minor_%s' % ax).on:
                getattr(self, 'ticks_minor_%s' % ax).on = True

        # Markers/points
        if 'marker_type' in kwargs.keys():
            marker_list = kwargs['marker_type']
        elif kwargs.get('markers') not in [None, True]:
            marker_list = utl.validate_list(kwargs.get('markers'))
        else:
            marker_list = utl.validate_list(DEFAULT_MARKERS)
        markers = RepeatedList(marker_list, 'markers')
        marker_edge_color = utl.kwget(kwargs, self.fcpp, 'marker_edge_color', color_list)
        marker_fill_color = utl.kwget(kwargs, self.fcpp, 'marker_fill_color', color_list)
        if kwargs.get('marker_fill_color'):
            kwargs['marker_fill'] = True
        self.markers = Element('marker', self.fcpp, kwargs,
                               on=utl.kwget(kwargs, self.fcpp,
                                            'markers', True),
                               filled=utl.kwget(kwargs, self.fcpp,
                                                'marker_fill', False),
                               edge_color=copy.copy(marker_edge_color),
                               edge_width=utl.kwget(kwargs, self.fcpp,
                                                    'marker_edge_width',
                                                     1.5),
                               fill_color=copy.copy(marker_fill_color),
                               jitter=utl.kwget(kwargs, self.fcpp,
                                                'marker_jitter',
                                                kwargs.get('jitter', False)),
                               size=utl.kwget(kwargs, self.fcpp,
                                              'marker_size', 7),
                               type=markers,
                               zorder=utl.kwget(kwargs, self.fcpp,
                                                'zorder', 2),
                               )
        if type(self.markers.size) is not RepeatedList:
            self.markers.size = RepeatedList(self.markers.size, 'marker_size')
        if type(self.markers.edge_width) is not RepeatedList:
            self.markers.edge_width = RepeatedList(self.markers.edge_width,
                                                   'marker_edge_width')

        # Lines
        for k in list(kwargs.keys()):
            if 'line_' in k and '%ss_%s' % (k.split('_')[0], k.split('_')[1]) \
                    not in kwargs.keys():
                kwargs['%ss_%s' % (k.split('_')[0], k.split('_')[1])] = kwargs[k]
        self.lines = Element('lines', self.fcpp, kwargs,
                             on=kwargs.get('lines', True),
                             color=copy.copy(color_list),
                             values=[],
                             )

        # Line fit
        self.fit = Element('fit', self.fcpp, kwargs,
                           on=True if kwargs.get('fit', False) else False,
                           color='#000000',
                           edge_color='none',
                           eqn=utl.kwget(kwargs, self.fcpp, 'fit_eqn', False),
                           fill_color='none',
                           font_size=utl.kwget(kwargs, self.fcpp, 'fit_font_size', 12),
                           padding=utl.kwget(kwargs, self.fcpp, 'fit_padding', 10),
                           rsq=utl.kwget(kwargs, self.fcpp, 'fit_rsq', False),
                           size=[0,0],
                           )
        self.fit.legend_text = utl.kwget(kwargs, self.fcpp, 'fit_legend_text', None)
        self.fit.position[0] = self.fit.padding
        self.fit.position[1] = self.axes.size[1] - \
                               (self.fit.padding + self.fit.font_size)

        # Reference line
        ref_line = kwargs.get('ref_line', False)
        if type(ref_line) is pd.Series:
            ref_col = 'Ref Line'
        elif type(ref_line) is list:
            ref_col = [f for f in ref_line if f in kwargs['df'].columns]
            missing = [f for f in ref_line if f not in ref_col]
            if len(missing) > 0:
                print('Could not find one or more columns for ref line: "%s"' %
                      ', '.join(missing))
            if not kwargs.get('ref_line_legend_text'):
                kwargs['ref_line_legend_text'] = ref_col
        elif type(kwargs.get('ref_line', False)) is str and \
                kwargs.get('ref_line', False) in kwargs['df'].columns:
            ref_col = kwargs.get('ref_line')
        else:
            ref_col = None

        self.ref_line = Element('ref_line', self.fcpp, kwargs,
                                on=False if not ref_col else True,
                                column=RepeatedList(ref_col, 'ref_col') if ref_col else None,
                                color='#000000',
                                legend_text=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                  'ref_line_legend_text', 'Ref Line'),
                                                  'ref_line_legend_text'),
                                )

        # Legend
        kwargs['legend'] = kwargs.get('legend', None)
        if type(kwargs['legend']) is list:
            kwargs['legend'] = ' | '.join(utl.validate_list(kwargs['legend']))
        legend_none = pd.DataFrame({'Key': ['NaN'], 'Line': None}, index=[0])
        self.legend = Legend_Element('legend', self.fcpp, kwargs,
                                 on=True if (kwargs.get('legend') and
                                    kwargs.get('legend_on', True)) else False,
                                 column=kwargs['legend'],
                                 font_size=12,
                                 location=LEGEND_LOCATION[utl.kwget(kwargs,
                                          self.fcpp, 'legend_location', 0)],
                                 marker_alpha=utl.kwget(kwargs, self.fcpp,
                                                        'legend_marker_alpha',
                                                        1),
                                 marker_size=utl.kwget(kwargs, self.fcpp,
                                                       'legend_marker_size',
                                                       7),
                                 nleg=utl.kwget(kwargs, self.fcpp, 'nleg', -1),
                                 points=utl.kwget(kwargs, self.fcpp,
                                                  'legend_points', 1),
                                 ordered_curves=[],
                                 ordered_fits=[],
                                 ordered_ref_lines=[],
                                 overflow=0,
                                 text=kwargs.get('legend_title',
                                                 kwargs.get('legend') if kwargs.get('legend') != True else ''),
                                 #values={} if not kwargs.get('legend') else {'NaN': None},
                                 )
        if not self.legend._on and self.ref_line.on:
            for ref_line_legend_text in self.ref_line.legend_text.values:
                self.legend.values[ref_line_legend_text] = []
            self.legend.on = True
            self.legend.text = ''
        if not self.legend._on and self.fit.on \
                and not (('legend' in kwargs.keys() and kwargs['legend'] == False) or \
                         ('legend_on' in kwargs.keys() and kwargs['legend_on'] == False)):
            #self.legend.values['fit_line'] = []
            self.legend.on = True
            self.legend.text = ''
        if self.legend._on and self.fit.on and 'fit_color' not in kwargs.keys():
            self.fit.color = copy.copy(self.lines.color)
        y = utl.validate_list(kwargs.get('y'))
        if not self.axes.twin_x and y is not None and len(y) > 1 and \
                self.plot_func != 'plot_box' and \
                (kwargs.get('wrap') != 'y' and \
                kwargs.get('row') != 'y' and \
                kwargs.get('col') != 'y') and \
                kwargs.get('legend') != False:
            self.legend.values = self.legend.set_default()
            self.legend.on = True

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
                                                'contour_filled',
                                                kwargs.get('filled', True)),
                               levels=utl.kwget(kwargs, self.fcpp,
                                                'contour_levels',
                                                kwargs.get('levels', 20)),
                              )

        # Heatmaps
        if 'cell_size' in kwargs.keys():
            kwargs['heatmap_cell_size'] = kwargs['cell_size']
        self.heatmap = Element('heatmap', self.fcpp, kwargs,
                               on=True if self.plot_func=='plot_heatmap'
                                  else False,
                               cell_size=utl.kwget(kwargs, self.fcpp,
                                                   'heatmap_cell_size',
                                                   60 if 'ax_size' not in
                                                   kwargs else None),
                               cmap=utl.kwget(kwargs, self.fcpp,
                                              'cmap', 'inferno'),
                               edge_width=0,
                               font_color='#ffffff',
                               font_size=12,
                               interpolation=utl.kwget(kwargs, self.fcpp,
                                              'heatmap_interpolation',
                                              kwargs.get('interpolation', 'none')),
                               text=utl.kwget(kwargs, self.fcpp,
                                              'data_labels', False),
                               )
        if self.heatmap.on:
            grids = [f for f in kwargs.keys() if f in
                     ['grid_major', 'grid_major_x', 'grid_major_y',
                      'grid_minor', 'grid_minor_x', 'grid_minor_y']]
            if len(grids) == 0:
                kwargs['grid_major'] = False
                kwargs['grid_minor'] = False
                kwargs['ticks_major'] = True
            if 'ax_edge_width' not in kwargs.keys():
                self.axes.edge_width = 0
            self.tick_labels_major_x.rotation = \
                utl.kwget(kwargs, self.fcpp, 'tick_labels_major_x', 90)
            if 'x' in kwargs.keys():
                kwargs['tick_cleanup'] = False

        # Bar
        self.bar = Element('bar', self.fcpp, kwargs,
                           on=True if 'bar' in self.plot_func else False,
                           width=utl.kwget(kwargs, self.fcpp, 'bar_width', kwargs.get('width', 0.8)),
                           align=utl.kwget(kwargs, self.fcpp, 'bar_align', kwargs.get('align', 'center')),
                           edge_color=utl.kwget(kwargs, self.fcpp, 'bar_edge_color', copy.copy(color_list)),
                           edge_width=utl.kwget(kwargs, self.fcpp, 'bar_edge_width', 0),
                           fill_alpha=utl.kwget(kwargs, self.fcpp, 'bar_fill_alpha', 0.75),
                           fill_color=utl.kwget(kwargs, self.fcpp, 'bar_fill_color', copy.copy(color_list)),
                           line=utl.kwget(kwargs, self.fcpp, 'bar_line', kwargs.get('line', False) | kwargs.get('lines', False)),
                           horizontal=utl.kwget(kwargs, self.fcpp, 'bar_horizontal', kwargs.get('horizontal', False)),
                           stacked=utl.kwget(kwargs, self.fcpp, 'bar_stacked', kwargs.get('stacked', False)),
                           error_bars=utl.kwget(kwargs, self.fcpp, 'bar_error_bars', kwargs.get('error_bars', None)),
                           error_color=utl.kwget(kwargs, self.fcpp, 'bar_error_color', kwargs.get('error_color', '#555555')),
                           color_by_bar=utl.kwget(kwargs, self.fcpp, 'bar_color_by_bar', kwargs.get('color_by_bar', False)),
                           )
        self.bar.width = self.bar.width.get(0)
        if 'colors' in kwargs.keys():
            self.bar.color_by_bar = True

        # Histogram
        self.hist = Element('hist', self.fcpp, kwargs,
                            on=True if 'hist' in self.plot_func and kwargs.get('hist_on', True) else False,
                            align=utl.kwget(kwargs, self.fcpp, 'hist_align', 'mid'),
                            bins=utl.kwget(kwargs, self.fcpp, 'hist_bins', kwargs.get('bins', 20)),
                            edge_color=copy.copy(color_list),
                            edge_width=0,
                            fill_alpha=0.5,
                            fill_color=copy.copy(color_list),
                            cumulative=utl.kwget(kwargs, self.fcpp, 'hist_cumulative', kwargs.get('cumulative', False)),
                            kde=utl.kwget(kwargs, self.fcpp, 'hist_kde', kwargs.get('kde', False)),
                            normalize=utl.kwget(kwargs, self.fcpp, 'hist_normalize', kwargs.get('normalize', False)),
                            rwidth=utl.kwget(kwargs, self.fcpp, 'hist_width', None),
                            stacked=utl.kwget(kwargs, self.fcpp, 'hist_stacked', kwargs.get('stacked', False)),
                            type=utl.kwget(kwargs, self.fcpp, 'hist_type', 'bar'),
                            horizontal=utl.kwget(kwargs, self.fcpp, 'hist_horizontal', kwargs.get('horizontal', False)),
                            )
        self.kde = Element('kde', self.fcpp, kwargs,
                           on=utl.kwget(kwargs, self.fcpp, 'hist_kde', kwargs.get('kde', False)),
                           color=copy.copy(color_list),
                           width=1.5,
                           zorder=5,
                           )
        if self.kde.on:
            self.hist.normalize = True

        # Boxplot labels
        self.box_group_title = Element('box_group_title', self.fcpp, kwargs,
                                      on=True if 'box' in self.plot_func and kwargs.get('box_labels_on', True) else False,
                                      font_color='#666666',
                                      font_size=12,
                                      padding=15,  # percent
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
        self.violin = Element('violin', self.fcpp, kwargs,
                              on=utl.kwget(kwargs, self.fcpp, 'box_violin',
                                           kwargs.get('violin', False)),
                              box_color=utl.kwget(kwargs, self.fcpp,
                                                  'violin_box_color', '#555555'),
                              box_on=utl.kwget(kwargs, self.fcpp,
                                               'violin_box_on', True),
                              edge_color=utl.kwget(kwargs, self.fcpp,
                                                   'violin_edge_color', '#aaaaaa'),
                              fill_alpha=0.5,
                              fill_color=kwargs.get('color', utl.kwget(kwargs, self.fcpp,
                                                    'violin_fill_color', DEFAULT_COLORS[0])),
                              markers=kwargs.get('markers', utl.kwget(kwargs, self.fcpp,
                                                'violin_markers', False)),
                              median_color=utl.kwget(kwargs, self.fcpp,
                                                     'violin_median_color', '#ffffff'),
                              median_marker=utl.kwget(kwargs, self.fcpp,
                                                     'violin_median_marker', 'o'),
                              median_size=utl.kwget(kwargs, self.fcpp,
                                                     'violin_median_size', 2),
                              whisker_color=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_color', '#555555'),
                              whisker_style=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_style', '-'),
                              whisker_width=utl.kwget(kwargs, self.fcpp,
                                                      'violin_whisker_width', 1.5),
                             )
        box_edge_color = utl.kwget(kwargs, self.fcpp, 'box_edge_color', '#aaaaaa') #['#4b72b0'])
        box_fill_color = utl.kwget(kwargs, self.fcpp, 'box_fill_color', '#ffffff')
        self.box = Element('box', self.fcpp, kwargs,
                           on=True if 'box' in self.plot_func and kwargs.get('box_on', True) else False,
                           edge_color=box_edge_color,
                           edge_width=0.5,
                           fill_color=box_fill_color,
                           median_color=utl.kwget(kwargs, self.fcpp,
                                                 'box_median_line_color',
                                                 '#ff7f0e'),
                           notch=utl.kwget(kwargs, self.fcpp, 'box_notch',
                                           kwargs.get('notch', False)),
                           width=utl.kwget(kwargs, self.fcpp, 'box_width',
                                           kwargs.get('width', 0.5 if not self.violin.on
                                                               else 0.15)),
                           )
        self.box_grand_mean = Element('box_grand_mean', self.fcpp, kwargs,
                                      on=utl.kwget(kwargs, self.fcpp, 'box_grand_mean', kwargs.get('grand_mean', False)),
                                      color=utl.kwget(kwargs, self.fcpp, 'box_grand_mean_color', kwargs.get('grand_mean_color', '#555555')),
                                      style=utl.kwget(kwargs, self.fcpp, 'box_grand_mean_style', kwargs.get('grand_mean_style', '--')),
                                      width=utl.kwget(kwargs, self.fcpp, 'box_grand_mean_width', kwargs.get('grand_mean_width', 1)),
                                      zorder=30)

        self.box_grand_median = Element('box_grand_median', self.fcpp, kwargs,
                                        on=utl.kwget(kwargs, self.fcpp, 'box_grand_median',
                                                     kwargs.get('grand_median', False)),
                                        color=utl.kwget(kwargs, self.fcpp, 'box_grand_median_color',
                                                        kwargs.get('grand_median_color', '#0000ff')),
                                        style=utl.kwget(kwargs, self.fcpp, 'box_grand_median_style',
                                                        kwargs.get('grand_median_style', '--')),
                                        width=utl.kwget(kwargs, self.fcpp, 'box_grand_median_width',
                                                        kwargs.get('grand_median_width', 1)),
                                        zorder=30)

        self.box_group_means = Element('box_group_means', self.fcpp, kwargs,
                                       on=utl.kwget(kwargs, self.fcpp, 'box_group_means',
                                                    kwargs.get('group_means', False)),
                                       color=utl.kwget(kwargs, self.fcpp, 'box_group_means_color',
                                                       kwargs.get('group_means_color', '#FF00FF')),
                                       style=utl.kwget(kwargs, self.fcpp, 'box_group_means_style',
                                                       kwargs.get('group_means_style', '--')),
                                       width=utl.kwget(kwargs, self.fcpp, 'box_group_means_width',
                                                       kwargs.get('group_means_width', 1)),
                                       zorder=30)

        self.box_mean_diamonds = Element('box_mean_diamonds', self.fcpp, kwargs,
                                         on=utl.kwget(kwargs, self.fcpp, 'box_mean_diamonds',
                                                      kwargs.get('mean_diamonds', False)),
                                         alpha=utl.kwget(kwargs, self.fcpp, 'box_mean_diamonds_alpha',
                                                         kwargs.get('mean_diamonds_alpha', 1)),
                                         conf_coeff=utl.kwget(kwargs, self.fcpp, 'conf_coeff', 0.95),
                                         edge_color=utl.kwget(kwargs, self.fcpp,
                                                              'box_mean_diamonds_edge_color',
                                                              kwargs.get('mean_diamonds_edge_color', '#00FF00')),
                                         edge_style=utl.kwget(kwargs, self.fcpp,
                                                         'box_mean_diamonds_edge_style',
                                                         kwargs.get('mean_diamonds_edge_style', '-')),
                                         edge_width=utl.kwget(kwargs, self.fcpp,
                                                              'box_mean_diamonds_edge_width',
                                                              kwargs.get('mean_diamonds_edge_width', 0.7)),
                                         fill_color=utl.kwget(kwargs, self.fcpp,
                                                              'box_mean_diamonds_fill_color',
                                                              kwargs.get('mean_diamonds_fill_color', None)),
                                         width=utl.kwget(kwargs, self.fcpp,
                                                         'box_mean_diamonds_width',
                                                         kwargs.get('mean_diamonds_width', 0.8)),
                                         zorder=30)

        self.box_whisker = Element('box_whisker', self.fcpp, kwargs,
                                   on=self.box.on,
                                   color=self.box.edge_color,
                                   style=self.box.style,
                                   width=self.box.edge_width)

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
                                   color='#bbbbbb', text=None,
                                   zorder=2,
                                   )

        self.box_range_lines = Element('box_range_lines', self.fcpp, kwargs,
                                       on=kwargs.get('box_range_lines',
                                                     not kwargs.get('violin', False)),
                                       color='#cccccc',
                                       style='-',
                                       style2=RepeatedList('--', 'style2'),
                                       zorder=utl.kwget(kwargs, self.fcpp,
                                                        'box_range_lines',
                                                        3),
                                       )
        if 'box' in self.plot_func:
            self.lines.on = False
        if self.violin.on:
            self.markers.on = self.violin.markers
        if 'box' in self.plot_func:
            # edge color
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_edge_color') \
                    and not self.legend._on:
                self.markers.edge_color = DEFAULT_COLORS[1]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            elif not kwargs.get('colors') and not kwargs.get('marker_edge_color'):
                self.markers.edge_color = color_list[1:] + [color_list[0]]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_fill_color') \
                    and not self.legend._on:
                self.markers.fill_color = DEFAULT_COLORS[1]
                self.markers.color_alpha('fill_color', 'fill_alpha')
            elif not kwargs.get('colors'):
                self.markers.fill_color = color_list[1:] + [color_list[0]]
                self.markers.color_alpha('fill_color', 'fill_alpha')
            if 'box_marker_edge_alpha' in self.fcpp.keys():
                self.markers.edge_alpha = self.fcpp['box_marker_edge_alpha']
            if 'box_marker_edge_color' in self.fcpp.keys():
                self.markers.edge_color = self.fcpp['box_marker_edge_color']
                self.markers.color_alpha('edge_color', 'edge_alpha')
            if 'box_marker_fill_alpha' in self.fcpp.keys():
                self.markers.fill_alpha = self.fcpp['box_marker_fill_alpha']
            if 'box_marker_fill_color' in self.fcpp.keys():
                self.markers.fill_color = self.fcpp['box_marker_fill_color']
                self.markers.color_alpha('fill_color', 'fill_alpha')
            self.markers.filled = self.fcpp.get('box_marker_fill', self.markers.filled)
            self.markers.edge_width = self.fcpp.get('box_marker_edge_width', self.markers.edge_width)
            self.markers.jitter = utl.kwget(kwargs, self.fcpp, 'jitter', True)
            if 'box_marker_jitter' in self.fcpp.keys():
                self.markers.jitter = self.fcpp['box_marker_jitter']
            if 'box_marker_size' in self.fcpp.keys():
                self.markers.size = self.fcpp['box_marker_size']
            else:
                self.markers.size = kwargs.get('marker_size', 4)
            if 'marker_type' in kwargs.keys():
                self.markers.type = RepeatedList(kwargs['marker_type'], 'marker_type')
            elif 'box_marker_type' in self.fcpp.keys():
                self.markers.type = RepeatedList(self.fcpp['box_marker_type'], 'marker_type')
            elif not self.legend._on:
                self.markers.type = RepeatedList('o', 'marker_type')
            if 'box_marker_zorder' in self.fcpp.keys():
                self.markers.zorder = self.fcpp['box_marker_zorder']
            if type(self.markers.size) is not RepeatedList:
                self.markers.size = RepeatedList(self.markers.size, 'marker_size')
            if type(self.markers.edge_width) is not RepeatedList:
                self.markers.edge_width = RepeatedList(self.markers.edge_width,
                                                    'marker_edge_width')

        # Axhlines/axvlines
        axlines = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']
        # Todo: list
        for axline in axlines:
            val = kwargs.get(axline, False)
            if type(val) is not tuple:
                vals = utl.validate_list(val)
            else:
                vals = [val]
            values = []
            colors = []
            styles = []
            widths = []
            alphas = []
            labels = []
            for ival, val in enumerate(vals):
                if (type(val) is list or type(val) is tuple) and len(val) > 1:
                    values += [val[0]]
                else:
                    values += [val]
                if (type(val) is list or type(val) is tuple) and len(val) > 1:
                    colors += [val[1]]
                else:
                    colors += [utl.kwget(kwargs, self.fcpp, '%s_color' % axline, '#000000')]
                if (type(val) is list or type(val) is tuple) and len(val) > 2:
                    styles += [val[2]]
                else:
                    styles += [utl.kwget(kwargs, self.fcpp, '%s_style' % axline, '-')]
                if (type(val) is list or type(val) is tuple) and len(val) > 3:
                    widths += [val[3]]
                else:
                    widths += [utl.kwget(kwargs, self.fcpp, '%s_width' % axline, 1)]
                if (type(val) is list or type(val) is tuple) and len(val) > 4:
                    alphas += [val[4]]
                else:
                    alphas += [utl.kwget(kwargs, self.fcpp, '%s_alpha' % axline, 1)]
                if (type(val) is list or type(val) is tuple) and len(val) > 5:
                    labels += [val[5]]
                elif (type(val) is list or type(val) is tuple) and type(val[0]) is str:
                    labels += [val[0]]
                else:
                    labels += [utl.kwget(kwargs, self.fcpp, '%s_label' % axline, None)]
            setattr(self, axline,
                    Element(axline, self.fcpp, kwargs,
                            on=True if axline in kwargs.keys() else False,
                            values=values, color=colors, style=styles,
                            width=widths, alpha=alphas, text=labels,
                            zorder=utl.kwget(kwargs, self.fcpp, '%s_zorder' % axline, 1),
                            ))
            # for label in labels:
            #     if label:
            #         self.legend.values[label] = []

        # Gridlines
        self.grid_major = Element('grid_major', self.fcpp, kwargs,
                                  on=kwargs.get('grid_major', True),
                                  color=utl.kwget(kwargs, self.fcpp,
                                                  'grid_major_color',
                                                  '#ffffff'),
                                  width=1.3,
                                  )
        secondary = ['y2'] if kwargs.get('grid_major_y2') is True else [] + \
                    ['x2'] if kwargs.get('grid_major_x2') is True else []
        for ax in ['x', 'y'] + secondary:
            # secondary axes cannot get the grid
            setattr(self, 'grid_major_%s' %ax,
                    Element('grid_major_%s' %ax, self.fcpp, kwargs,
                            on=kwargs.get('grid_major_%s' % ax,
                                          self.grid_major.on),
                            color=self.grid_major.color,
                            style=self.grid_major.style,
                            width=self.grid_major.width,
                            zorder=self.grid_major.zorder,
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
        secondary = ['y2'] if kwargs.get('grid_major_y2') is True else [] + \
                    ['x2'] if kwargs.get('grid_major_x2') is True else []
        for ax in ['x', 'y'] + secondary:
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
                            zorder=self.grid_minor.zorder,
                            ))
            if getattr(self, 'grid_minor_%s' % ax).on and \
                    ('ticks' not in kwargs.keys() or kwargs['ticks'] != False) and \
                    ('ticks_minor' not in kwargs.keys() or kwargs['ticks_minor'] != False) and \
                    ('ticks_minor_%s' % ax not in kwargs.keys() or kwargs['ticks_minor_%s' % ax] != False):
                getattr(self, 'ticks_minor_%s' % ax).on = True

        # Row column label
        label_rc = DF_Element('label_rc', self.fcpp, kwargs,
                              on=True,
                              size=utl.kwget(kwargs, self.fcpp,
                                             'label_rc_size', 30),
                              edge_color='#8c8c8c',
                              fill_color='#8c8c8c',
                              font_color='#ffffff',
                              font_size=16,
                              font_style='normal',
                              font_weight='bold',
                              align='center',
                              )
        self.label_row = copy.deepcopy(label_rc)
        self.label_row.on = \
            utl.kwget(kwargs, self.fcpp, 'label_row_on', True) \
                if kwargs.get('row') not in [None, 'y'] else False
        self.label_row.column = kwargs.get('row')
        self.label_row.size = [utl.kwget(kwargs, self.fcpp,
                                         'label_row_size', label_rc._size),
                               self.axes.size[1]]
        self.label_row.text_size = None
        self.label_row.edge_color = utl.kwget(kwargs, self.fcpp,
                                              'label_row_edge_color',
                                              label_rc.edge_color)
        self.label_row.edge_alpha = utl.kwget(kwargs, self.fcpp,
                                              'label_row_edge_alpha',
                                              label_rc.edge_alpha)
        self.label_row.edge_width = utl.kwget(kwargs, self.fcpp,
                                              'label_row_edge_width',
                                              label_rc.edge_width)
        self.label_row.fill_color = utl.kwget(kwargs, self.fcpp,
                                              'label_row_fill_color',
                                              label_rc.fill_color)
        self.label_row.font_color = utl.kwget(kwargs, self.fcpp,
                                              'label_row_font_color',
                                              label_rc.font_color)
        self.label_row.rotation = 270

        self.label_col = copy.deepcopy(label_rc)
        self.label_col.on = \
            utl.kwget(kwargs, self.fcpp, 'label_col_on', True) \
                if kwargs.get('col') not in [None, 'x'] else False
        self.label_row.column = kwargs.get('col')
        self.label_col.size = [self.axes.size[0],
                               utl.kwget(kwargs, self.fcpp,
                                         'label_col_size', label_rc._size)]
        self.label_col.text_size = None
        self.label_col.edge_color = utl.kwget(kwargs, self.fcpp,
                                              'label_col_edge_color',
                                              label_rc.edge_color)
        self.label_col.edge_width = utl.kwget(kwargs, self.fcpp,
                                              'label_col_edge_width',
                                              label_rc.edge_width)
        self.label_col.edge_alpha = utl.kwget(kwargs, self.fcpp,
                                              'label_col_edge_alpha',
                                              label_rc.edge_alpha)
        self.label_col.fill_color = utl.kwget(kwargs, self.fcpp,
                                              'label_col_fill_color',
                                              label_rc.fill_color)
        self.label_col.font_color = utl.kwget(kwargs, self.fcpp,
                                              'label_col_font_color',
                                              label_rc.font_color)
        # Wrap label
        self.label_wrap = DF_Element('label_wrap', self.fcpp, kwargs,
                                     on=utl.kwget(kwargs, self.fcpp,
                                                  'label_wrap_on', True)
                                                   if kwargs.get('wrap') else False,
                                     column=kwargs.get('wrap'),
                                     size=[self.axes.size[0],
                                           utl.kwget(kwargs, self.fcpp,
                                           'label_wrap_size', 30)],
                                     edge_color=label_rc.edge_color,
                                     edge_width=label_rc.edge_width,
                                     edge_alpha=label_rc.edge_alpha,
                                     fill_color=label_rc.fill_color,
                                     fill_alpha=label_rc.fill_alpha,
                                     font=label_rc.font,
                                     font_color=label_rc.font_color,
                                     font_size=label_rc.font_size,
                                     font_style=label_rc.font_style,
                                     font_weight=label_rc.font_weight,
                                     text_size=None,
                                     )

        if type(self.label_wrap.size) is not list:
            self.label_wrap.size = [self.label_wrap.size, self.axes.size[1]]

        self.title_wrap = Element('title_wrap', self.fcpp, kwargs,
                                  on=utl.kwget(kwargs, self.fcpp,
                                               'title_wrap_on', True)
                                               if kwargs.get('wrap') else False,
                                  size=utl.kwget(kwargs, self.fcpp,
                                                 'title_wrap_size',
                                                 label_rc.size),
                                  edge_color='#5f5f5f',
                                  edge_width=label_rc.edge_width,
                                  edge_alpha=label_rc.edge_alpha,
                                  fill_color='#5f5f5f',
                                  fill_alpha=label_rc.fill_alpha,
                                  font=label_rc.font,
                                  font_color=label_rc.font_color,
                                  font_size=label_rc.font_size,
                                  font_style=label_rc.font_style,
                                  font_weight=label_rc.font_weight,
                                  text=kwargs.get('title_wrap', None),
                                  )

        if type(self.title_wrap.size) is not list:
            self.title_wrap.size = [self.axes.size[0], self.title_wrap.size]
        # if self.title_wrap.on and not self.title_wrap.text:
        #     self.title_wrap.text = ' | '.join(self.label_wrap.values)

        # Confidence interval
        self.conf_int = Element('conf_int', self.fcpp, kwargs,
                                on=True if kwargs.get('conf_int', False) else False,
                                edge_color=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_edge_color',
                                                     copy.copy(color_list)),
                                edge_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_edge_alpha',
                                                     0.25),
                                fill_color=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_color',
                                                     copy.copy(color_list)),
                                fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_alpha',
                                                     0.2),
                                )

        # Arbitrart text
        position = utl.kwget(kwargs, self.fcpp, 'text_position', [0,0])
        if type(position[0]) is not list:
            position = [position]
        self.text = Element('text', self.fcpp, {},
                            on=True if utl.kwget(kwargs, self.fcpp, 'text', None) \
                               is not None else False,
                            edge_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_edge_color', 'none'),
                                                    'text_edge_color'),
                            fill_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_fill_color', 'none'),
                                                    'text_fill_color'),
                            font=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                              'text_font', 'sans-serif'), 'text_font'),
                            font_color=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_font_color', '#000000'),
                                                    'text_font_color'),
                            font_size=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                   'text_font_size', 14),
                                                   'text_font_size'),
                            font_style=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_font_style', 'normal'),
                                                    'text_font_style'),
                            font_weight=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                    'text_font_weight', 'normal'),
                                                    'text_font_weight'),
                            position=RepeatedList(position, 'text_position'),
                            coordinate=utl.kwget(kwargs, self.fcpp, 'text_coordinate', 'axis'),
                            rotation=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                  'text_rotation', 0), 'text_rotation'),
                            units=utl.kwget(kwargs, self.fcpp, 'text_units', 'pixel'),
                            text=RepeatedList(utl.kwget(kwargs, self.fcpp,
                                                  'text', ''), 'text'),
                            )

        # Extras
        self.inline = utl.kwget(kwargs, self.fcpp, 'inline', None)
        self.separate_labels = utl.kwget(kwargs, self.fcpp,
                                         'separate_labels', False)
        self.separate_ticks = utl.kwget(kwargs, self.fcpp,
                                        'separate_ticks', self.separate_labels)
        if self.separate_labels:
            self.separate_ticks = True
        self.tick_cleanup = utl.kwget(kwargs, self.fcpp, 'tick_cleanup', True)

        # Plot overrides
        if 'bar' in self.plot_func:
            self.grid_major_x.on = True
            self.grid_minor_x.on = False
            self.ticks_major_x.on = False
            self.ticks_minor_x.on = False
        if 'box' in self.plot_func:
            self.grid_major_x.on = False
            self.grid_minor_x.on = False
            self.ticks_major_x.on = False
            self.ticks_minor_x.on = False
            self.tick_labels_major_x.on = False
            self.tick_labels_minor_x.on = False
            self.label_x.on = False
        if 'heatmap' in self.plot_func:
            self.grid_major_x.on = False
            self.grid_major_y.on = False
            self.grid_minor_x.on = False
            self.grid_minor_y.on = False
            self.ticks_major_x.on = False
            self.ticks_major_y.on = False
            self.ticks_minor_x.on = False
            self.ticks_minor_y.on = False
            self.tick_labels_minor_x.on = False
            self.tick_labels_minor_y.on = False

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
        ws_label_rc = utl.kwget(kwargs, self.fcpp, 'ws_label_rc', 10)
        self.ws_label_col = utl.kwget(kwargs, self.fcpp,
                                  'ws_label_col', ws_label_rc)
        self.ws_label_row = utl.kwget(kwargs, self.fcpp,
                                  'ws_label_row', ws_label_rc)
        self.ws_col = utl.kwget(kwargs, self.fcpp, 'ws_col', 30)
        self.ws_row = utl.kwget(kwargs, self.fcpp, 'ws_row', 30)

        # figure
        self.ws_fig_label = utl.kwget(kwargs, self.fcpp, 'ws_fig_label', 10)
        self.ws_leg_fig = utl.kwget(kwargs, self.fcpp, 'ws_leg_fig', 10)
        self.ws_fig_ax = utl.kwget(kwargs, self.fcpp, 'ws_fig_ax', 20)
        self.ws_fig_title = utl.kwget(kwargs, self.fcpp, 'ws_fig_title', 10)

        # axes
        self.ws_label_tick = utl.kwget(kwargs, self.fcpp, 'ws_label_tick', 10)
        self.ws_ax_leg = utl.kwget(kwargs, self.fcpp, 'ws_ax_leg', 5)
        self.ws_ticks_ax = utl.kwget(kwargs, self.fcpp, 'ws_ticks_ax', 5)
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
        kwargs['zorder'] = element.zorder
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
                lab_text = str(kwargs.get('%slabel' % lab))
                lab_text2 = str(kwargs.get('%s2label' % lab))
            elif 'label_%s' % lab in kwargs.keys():
                lab_text = str(kwargs.get('label_%s' % lab))
                lab_text2 = str(kwargs.get('label_%s2' % lab))
            else:
                lab_text = None
                lab_text2 = None

            if lab == 'x' and self.axes.twin_y:
                getattr(self, 'label_x').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_x2').text = \
                    lab_text2 if lab_text2 is not None else getattr(data, '%s2' % lab)[0]
            elif lab == 'y' and self.axes.twin_x:
                getattr(self, 'label_y').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_y2').text = \
                    lab_text2 if lab_text2 is not None else getattr(data, '%s2' % lab)[0]
            else:
                if lab == 'wrap':
                    # special case
                    val = 'title_wrap'
                else:
                    val = 'label_%s' % lab
                if type(dd) is list:
                    if data.wrap == 'y' and lab == 'y' \
                            or data.wrap == 'x' and lab == 'x':
                        getattr(self, val).text = data.wrap_vals
                    elif lab == 'x' and data.col == 'x':
                        getattr(self, val).text = data.x_vals * self.nrow
                    elif lab == 'y' and data.row == 'y':
                        yvals = []
                        for yval in data.y_vals:
                            yvals += [yval] * self.ncol
                        getattr(self, val).text = yvals
                    else:
                        getattr(self, val).text = \
                            lab_text if lab_text is not None \
                            else ' & '.join([str(f) for f in dd])
                else:
                    getattr(self, val).text = dd
                if lab != 'z' and hasattr(self, 'label_%s2' % lab):
                    getattr(self, 'label_%s2' % lab).text = \
                        lab_text2 if lab_text2 is not None \
                        else ' & '.join([str(f) for f in dd])

            if hasattr(data, '%s_vals' % lab):
                getattr(self, 'label_%s' % lab).values = \
                    getattr(data, '%s_vals' % lab)

        if 'hist' in self.plot_func:
            if self.hist.normalize:
                self.label_y.text = kwargs.get('label_y_text', 'Normalized Counts')
            else:
                self.label_y.text = kwargs.get('label_y_text', 'Counts')

    def update_from_data(self, data):
        """
        Make properties updates from the Data class
        """

        self.groups = data.groups
        self.ncol = data.ncol
        self.ngroups = data.ngroups
        self.nrow = data.nrow
        self.nwrap = data.nwrap
        self.axes.share_x = data.share_x
        self.axes2.share_x = data.share_x2
        self.axes.share_y = data.share_y
        self.axes2.share_y = data.share_y2
        self.axes.share_col = data.share_col
        self.axes.share_row = data.share_row
        self.axes.scale = data.ax_scale
        self.axes2.scale =data.ax2_scale
        # if self.plot_func in ['plot_box'] and not self.axes.share_y:
        #     self.separate_ticks = True
        # elif self.plot_func not in ['plot_box'] and \
        #         (not self.axes.share_x or not self.axes.share_y):
        #     self.separate_ticks = True

    def update_wrap(self, data, kwargs):
        """
        Update figure props based on wrap selections
        """

        if data.wrap == 'y' or data.wrap == 'x':
            self.title_wrap.on = False
            self.label_wrap.on = False
            self.separate_labels = kwargs.get('separate_labels', True)
            self.separate_ticks = kwargs.get('separate_ticks', True) \
                                  if not self.separate_labels else True
        elif data.wrap:
            self.separate_labels = kwargs.get('separate_labels', False)
            self.separate_ticks = kwargs.get('separate_ticks', False) \
                                  if not self.separate_labels else True
            self.ws_row = kwargs.get('ws_row', self.label_wrap._size[1])
            self.ws_col = kwargs.get('ws_col', 0)
            self.cbar.on = False  # may want to address this someday

    def add_box_labels(self, ir, ic, dd):
        pass

    def add_hvlines(self, ir, ic):
        pass

    def close(self):
        pass

    def add_legend(self):
        pass

    def make_figure(self):
        pass

    def get_axes(self):
        """
        Return list of active axes
        """

        return [f for f in [self.axes, self.axes2] if f.on]

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

        pass

    def plot_contour(self, ax, df, x, y, z, ranges):
        """
        Plot a contour plot
        """

        pass

    def plot_heatmap(self, ax, df, x, y, z, ranges):
        """
        Plot a heatmap

        Args:
            ax (mpl.axes): current axes obj
            df (pd.DataFrame):  data to plot
            x (str): x-column name
            y (str): y-column name
            z (str): z-column name
            range (dict):  ax limits

        """

        pass

    def plot_hist(self, ir, ic, iline, df, x, y, leg_name, data, zorder=1,
                  line_type=None, marker_disable=False):

        pass

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

        pass

    def save(self, filename):
        pass

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df

    def set_axes_colors(self, ir, ic):
        pass

    def set_axes_grid_lines(self, ir, ic):
        pass

    def set_axes_labels(self, ir, ic):
        pass

    def set_axes_ranges(self, ir, ic):
        pass

    def set_axes_rc_labels(self, ir, ic):
        pass

    def set_axes_scale(self, ir, ic):
        pass

    def set_axes_ticks(self, ir, ic):
        pass

    def set_figure_title(self):
        pass

    def show(self, inline=True):
        pass


class Element:
    def __init__(self, label='None', fcpp={}, others={}, **kwargs):
        """
        Element style container
        """

        # Update kwargs
        for k, v in others.items():
            if k not in kwargs.keys():
                kwargs[k] = v

        self._on = kwargs.get('on', True) # visbile or not
        self.dpi = utl.kwget(kwargs, fcpp, 'dpi', 100)
        self.obj = None  # plot object reference
        self.position = kwargs.get('position', [0, 0, 0, 0])  # left, right, top, bottom
        self._size = kwargs.get('size', [0, 0])  # width, height
        self._size_orig = kwargs.get('size')
        self._text = kwargs.get('text', True)  # text label
        self._text_orig = kwargs.get('text')
        self.rotation = utl.kwget(kwargs, fcpp, '%s_rotation' % label,
                                  kwargs.get('rotation', 0))
        self.zorder = utl.kwget(kwargs, fcpp, '%s_zorder' % label,
                                kwargs.get('zorder', 0))

        # fill and edge colors
        self.fill_alpha = utl.kwget(kwargs, fcpp, '%s_fill_alpha' % label,
                                    kwargs.get('fill_alpha', 1))
        self.fill_color = utl.kwget(kwargs, fcpp, '%s_fill_color' % label,
                                    kwargs.get('fill_color', '#ffffff'))
        if type(self.fill_color) is not RepeatedList \
                and self.fill_color is not None:
            self.color_alpha('fill_color', 'fill_alpha')

        self.edge_width = utl.kwget(kwargs, fcpp, '%s_edge_width' % label,
                                    kwargs.get('edge_width', 1))
        self.edge_alpha = utl.kwget(kwargs, fcpp, '%s_edge_alpha' % label,
                                    kwargs.get('edge_alpha', 1))
        self.edge_color = utl.kwget(kwargs, fcpp, '%s_edge_color' % label,
                                    kwargs.get('edge_color', '#ffffff'))
        if type(self.edge_color) is not RepeatedList:
            self.color_alpha('edge_color', 'edge_alpha')

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
        if type(self.color) is not RepeatedList:
            self.color_alpha('color', 'alpha')

        self.width = utl.kwget(kwargs, fcpp, '%s_width' % label,
                               kwargs.get('width', 1))
        if type(self.width) is not RepeatedList:
            self.width = RepeatedList(self.width, 'width')
        self.style = utl.kwget(kwargs, fcpp, '%s_style' % label,
                               kwargs.get('style', '-'))
        if type(self.style) is not RepeatedList:
            self.style = RepeatedList(self.style, 'style')

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

    def color_alpha(self, attr, alpha):
        """
        Add alpha to each color in the color list and make it a RepeatedList
        """

        # MPL < v2 does not support alpha in hex color code
        skip_alpha = False
        if ENGINE == 'mpl' and LooseVersion(mpl.__version__) < LooseVersion('2'):
            skip_alpha = True

        alpha = RepeatedList(getattr(self, alpha), 'temp')

        if type(getattr(self, attr)) is not RepeatedList:
            color_list = utl.validate_list(getattr(self, attr))

            for ic, color in enumerate(color_list):
                if type(color) is int:
                    color = DEFAULT_COLORS[color]
                if color[0] != '#' and color != 'none':
                    color = '#' + color
                if skip_alpha or color == 'none':
                    astr = ''
                else:
                    astr = str(hex(int(alpha.get(ic) * 255)))[-2:].replace('x', '0')
                color_list[ic] = color[0:7].lower() + astr

            setattr(self, attr, RepeatedList(color_list, attr))

        else:
            # Update existing RepeatedList alphas
            setattr(self, attr, copy.copy(getattr(self, attr)))
            new_vals = []
            for ival, val in enumerate(getattr(self, attr).values):
                if skip_alpha:
                    astr = ''
                else:
                    astr = str(hex(int(alpha.get(ival) * 255)))[-2:].replace('x', '0')
                if len(val) > 7:
                    new_vals += [val[0:-2] + astr]
                else:
                    new_vals += [val + astr]

            getattr(self, attr).values = new_vals

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

        return True if self._on and self.values is not None \
               and len(self.values) > 0 else False

    @on.setter
    def on(self, state):

        self._on = state

        if not self.on:
            self._size = [0, 0]
            self._text = None

        else:
            self._size = self._size_orig
            self._text = self._text_orig


class Legend_Element(DF_Element):
    def __init__(self, label='None', fcpp={}, others={}, **kwargs):
        self.cols = ['Key', 'Curve', 'LineType']
        self.default = pd.DataFrame(columns=self.cols, data=[['NaN', None, None]], index=[0])

        if not kwargs.get('legend'):
            self._values = pd.DataFrame(columns=self.cols)
        else:
            self._values = self.set_default()
        if kwargs.get('sort') == True:
            self.sort = True
        else:
            self.sort = False

        DF_Element.__init__(self, label=label, fcpp=fcpp, others=others, **kwargs)

    @property
    def size(self):

        if self._on:
            return self._size
        else:
            return [0, 0]

    @size.setter
    def size(self, value):

        if self._size_orig is None and value is not None:
            self._size_orig = value

        self._size = value

    @property
    def values(self):
        if len(self._values) <= 1:
            return self._values

        # Re-order single fit lines
        if 'Fit' in self._values.Key.values \
                or 'ref_line' in self._values.LineType.values:
            df = self._values[self._values.LineType=='lines']
            fit = self._values[self._values.LineType=='fit']
            ref = self._values[self._values.LineType=='ref_line']
            return pd.concat([df, fit, ref]).reset_index(drop=True)
        else:
            return self._values.sort_index()

    @values.setter
    def values(self, value):
        self._values = value

    def add_value(self, key, curve, line_type_name):
        """
        Add a new curve to the values dataframe

        Args:
            key (str): string name for legend label
            curve (obj): reference to curve obj
            line_type_name (str): line type description
        """

        temp = pd.DataFrame({'Key': key, 'Curve': curve, 'LineType': line_type_name},
                            index=[len(self._values)])
        self._values = pd.concat([self.values, temp], sort=True)


    def del_value(self, key):
        df = self.values.copy()
        self._values = df[df.Key!=key].copy()

    def set_default(self):
        return self.default.copy()

