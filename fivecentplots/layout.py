from . import fcp
import matplotlib as mpl
import matplotlib.pyplot as mplp
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
import matplotlib.mlab as mlab
import importlib
import os, sys
import pandas as pd
import pdb
import scipy.stats
import datetime
import time
import numpy as np
import copy
import decimal
import math
from .colors import *
import fivecentplots.utilities as utl
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

st = pdb.set_trace

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
        tp[vv]['min'] = min(getattr(ax, 'get_%slim' % vv)())
        tp[vv]['max'] = max(getattr(ax, 'get_%slim' % vv)())
        tp[vv]['ticks'] = getattr(ax, 'get_%sticks' % vv)()
        tp[vv]['labels'] = [f for f in getattr(ax, '%saxis' % vv).iter_ticks()]
        tp[vv]['label_vals'] = [f[1] for f in tp[vv]['labels']]
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

    missing = [f for f in ['x', 'y'] if f not in tp.keys()]
    for mm in missing:
        tp[mm] = {}
        tp[mm]['ticks'] = []
        tp[mm]['labels'] = []
        tp[mm]['label_text'] = []
        tp[mm]['first'] = -999
        tp[mm]['last'] = -999

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
                            share_x=kwargs.get('share_x', True),
                            share_y=kwargs.get('share_y', True),
                            share_z=kwargs.get('share_z', True),
                            share_x2=kwargs.get('share_x2', True),
                            share_y2=kwargs.get('share_y2', True),
                            share_col = kwargs.get('share_col', False),
                            share_row = kwargs.get('share_row', False),
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
                        edge_width=kwargs.get('tick_labels_major_edge_width',
                                       self.tick_labels_major.edge_width),
                        fill_color=fill_color,
                        fill_alpha=fill_alpha,
                        font=kwargs.get('tick_labels_major_font',
                                       self.tick_labels_major.font),
                        font_color=kwargs.get('tick_labels_major_font_color',
                                       self.tick_labels_major.font_color),
                        font_size=kwargs.get('tick_labels_major_font_size',
                                       self.tick_labels_major.font_size),
                        font_style=kwargs.get('tick_labels_major_font_style',
                                       self.tick_labels_major.font_style),
                        font_weight=kwargs.get('tick_labels_major_font_style',
                                       self.tick_labels_major.font_style),
                        offset=kwargs.get('tick_labels_major_offset',
                                       self.tick_labels_major.offset),
                        padding=kwargs.get('tick_labels_major_padding',
                                       self.tick_labels_major.padding),
                        rotation=kwargs.get('tick_labels_major_rotation',
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
                           eqn=utl.kwget(kwargs, self.fcpp, 'fit_eqn', False),
                           font_size=utl.kwget(kwargs, self.fcpp, 'fit_font_size', 12),
                           padding=utl.kwget(kwargs, self.fcpp, 'fit_padding', 10),
                           rsq=utl.kwget(kwargs, self.fcpp, 'fit_rsq', False),
                           size=[0,0],
                           )
        self.fit.position[0] = self.fit.padding / self.axes.size[0]
        self.fit.position[3] = 1 - (self.fit.padding + \
                                    self.fit.font_size)/ self.axes.size[1]

        # Reference line
        self.ref_line = Element('ref_line', self.fcpp, kwargs,
                                on=True if type(kwargs.get('ref_line', False)) is pd.Series else False,
                                color='#000000',
                                size=[0,0],
                                text=utl.kwget(kwargs, self.fcpp, 'ref_line_text', 'Ref Line'),
                                )

        # Legend
        kwargs['legend'] = kwargs.get('legend', None)
        if type(kwargs['legend']) is list:
            kwargs['legend'] = ' | '.join(utl.validate_list(kwargs['legend']))

        self.legend = DF_Element('legend', self.fcpp, kwargs,
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
                                 points=utl.kwget(kwargs, self.fcpp,
                                                  'legend_points', 1),
                                 overflow=0,
                                 text=kwargs.get('legend_title',
                                                 kwargs.get('legend') if kwargs.get('legend') != True else ''),
                                 values={} if not kwargs.get('legend') else {'NaN': None},
                                 )

        if not self.legend.on and self.ref_line.on:
            self.legend.values['ref_line'] = []
            self.legend.on = True
        if not self.legend.on and self.fit.on \
                and not (('legend' in kwargs.keys() and kwargs['legend'] == False) or \
                         ('legend_on' in kwargs.keys() and kwargs['legend_on'] == False)):
            self.legend.values['fit_line'] = []
            self.legend.on = True
        if self.legend.on and self.fit.on and 'fit_color' not in kwargs.keys():
            self.fit.color = copy.copy(self.lines.color)
        y = utl.validate_list(kwargs.get('y'))
        if not self.axes.twin_x and y is not None and len(y) > 1 and \
                self.plot_func != 'plot_box':
            self.legend.values = {'NaN': None}
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
                              fill_color=utl.kwget(kwargs, self.fcpp,
                                                   'violin_fill_color', DEFAULT_COLORS[0]),
                              markers=utl.kwget(kwargs, self.fcpp,
                                                'violin_markers', False),
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
                                   color='#bbbbbb',
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
        if 'box' in self.plot_func:
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_edge_color') \
                    and not self.legend.on:
                self.markers.edge_color = DEFAULT_COLORS[1]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            else:
                self.markers.edge_color = color_list[1:] + [color_list[0]]
                self.markers.color_alpha('edge_color', 'edge_alpha')
            if not kwargs.get('colors') \
                    and not kwargs.get('marker_fill_color') \
                    and not self.legend.on:
                self.markers.fill_color = DEFAULT_COLORS[1]
                self.markers.color_alpha('fill_color', 'fill_alpha')
            else:
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
            elif not self.legend.on:
                self.markers.type = RepeatedList('o', 'marker_type')
            if 'box_marker_zorder' in self.fcpp.keys():
                self.markers.zorder = self.fcpp['box_marker_zorder']
        if self.violin.on and not self.violin.markers:
            self.markers.on = False

        # Axhlines/axvlines
        axlines = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']
        # Todo: list
        for axline in axlines:
            val = kwargs.get(axline, False)
            vals = utl.validate_list(val)
            values = []
            colors = []
            styles = []
            widths = []
            alphas = []
            for ival, val in enumerate(vals):
                if type(val) is list or type(val) is tuple and len(val) > 1:
                    values += [val[0]]
                else:
                    values += [val]
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
                            values=values, color=colors, style=styles,
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

        # Extras
        self.inline = utl.kwget(kwargs, self.fcpp, 'inline', None)
        self.separate_labels = utl.kwget(kwargs, self.fcpp,
                                         'separate_labels', False)
        self.separate_ticks = utl.kwget(kwargs, self.fcpp,
                                        'separate_ticks', self.separate_labels)
        if self.separate_labels:
            self.separate_ticks = True
        if not self.axes.share_x or not self.axes.share_y:
            self.separate_ticks = True
        self.tick_cleanup = utl.kwget(kwargs, self.fcpp, 'tick_cleanup', True)

        # Plot overrides
        if 'box' in self.plot_func:
            self.grid_major_x.on = False
            self.grid_minor_x.on = False
            self.ticks_major_x.on = False
            self.ticks_minor_x.on = False
            self.tick_labels_major_x.on = False
            self.tick_labels_minor_x.on = False
            self.label_x.on = False
            self.axes.share_x = False
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
        self.ws_leg_ax = utl.kwget(kwargs, self.fcpp, 'ws_leg_ax', 20)
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
                            else ' + '.join([str(f) for f in dd])
                else:
                    getattr(self, val).text = dd
                if lab != 'z' and hasattr(self, 'label_%s2' % lab):
                    getattr(self, 'label_%s2' % lab).text = \
                        lab_text2 if lab_text2 is not None \
                        else ' + '.join([str(f) for f in dd])

            if hasattr(data, '%s_vals' % lab):
                getattr(self, 'label_%s' % lab).values = \
                    getattr(data, '%s_vals' % lab)

        if 'hist' in self.plot_func:
            if self.hist.normalize:
                self.label_y.text = kwargs.get('label_y_text', 'Normalized Counts')
            else:
                self.label_y.text = kwargs.get('label_y_text', 'Counts')


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
        self.position = [0, 0, 0, 0]  # left, right, top, bottom
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
                if color[0] != '#':
                    color = '#' + color
                if skip_alpha:
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


class RepeatedList:
    def __init__(self, values, name):
        """
        Set a default list of items and loop through it beyond the maximum
        index value

        Args:
            values (list): user-defined list of values
            name (str): label to describe contents of class
        """

        self.values = utl.validate_list(values)
        self.shift = 0

        if type(self.values) is not list and len(self.values) < 1:
            raise(ValueError, 'RepeatedList for "%s" must contain an actual '
                              'list with more at least one element')

    def get(self, idx):

        # can we make this a next??

        return self.values[(idx + self.shift) % len(self.values)]


class LayoutBokeh(BaseLayout):
    def __init__(self, **kwargs):

        global ENGINE
        ENGINE = 'bokeh'

        BaseLayout.__init__(self, **kwargs)


class LayoutMPL(BaseLayout):

    def __init__(self, plot_func='plot', **kwargs):
        """
        Layout attributes and methods for matplotlib Figure

        Args:
            **kwargs: input args from user
        """

        global ENGINE
        ENGINE = 'mpl'

        if kwargs.get('tick_cleanup', True):
            mplp.style.use('classic')
        else:
            mplp.style.use('default')
        mplp.close('all')

        # Inherit the base layout properties
        BaseLayout.__init__(self, plot_func, **kwargs)

        # Define white space parameters
        self.init_white_space(**kwargs)

        # Initialize other class variables
        self.label_col_height  = 0
        self.label_row_left    = 0
        self.label_row_width   = 0
        self.title_wrap_bottom = 0

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
                    # if self.box_group_title.on:
                    #     height2 = self.box_group_title.size[i][1] * \
                    #               (1 + 2 * self.box_group_title.padding / 100)
                    #     height = max(height, height2)
                    self.add_label(ir, ic, label,
                                   (sub.index[j]/len(data.changes),
                                   0, 0,
                                   (bottom - height) / self.axes.size[1]),
                                   rotation=self.box_group_label.rotation[i],
                                   size=[width, height], offset=True,
                                   **self.make_kwargs(self.box_group_label,
                                                      ['size', 'rotation', 'position']))

            # Group titles
            if self.box_group_title.on and ic == data.ncol - 1:
                self.add_label(ir, ic, data.groups[k],
                               (1 + self.ws_ax_box_title / self.axes.size[0],
                               0, 0,
                               (bottom - height/2 - 1 - \
                                self.box_group_title.size[k][1]/2) / \
                                self.axes.size[1]),
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
        cbar.outline.set_edgecolor(self.cbar.edge_color.get(0))
        cbar.outline.set_linewidth(self.cbar.edge_width)

        # Style tick labels
        ticks_font = \
            font_manager.FontProperties(family=getattr(self, 'tick_labels_major_z').font,
                                        size=getattr(self, 'tick_labels_major_z').font_size,
                                        style=getattr(self, 'tick_labels_major_z').font_style,
                                        weight=getattr(self, 'tick_labels_major_z').font_weight)

        for text in cax.get_yticklabels():
            if getattr(self, 'tick_labels_major_z').rotation != 0:
                text.set_rotation(getattr(self, 'tick_labels_major_z').rotation)
            text.set_fontproperties(ticks_font)
            text.set_bbox(dict(edgecolor=getattr(self, 'tick_labels_major_z').edge_color.get(0),
                                facecolor=getattr(self, 'tick_labels_major_z').fill_color.get(0),
                                linewidth=getattr(self, 'tick_labels_major_z').edge_width))

        #cbar.dividers.set_color('white')  # could enable
        #cbar.dividers.set_linewidth(2)

        return cbar

    def add_hvlines(self, ir, ic):
        """
        Add axhlines and axvlines

        Args:
            ir (int): subplot row index
            ic (int): subplot col index
        """

        # Set default line attributes
        for axline in ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']:
            ll = getattr(self, axline)
            func = self.axes.obj[ir, ic].axhline if 'hline' in axline \
                   else self.axes.obj[ir, ic].axvline
            if ll.on:
                for ival, val in enumerate(ll.values):
                    func(val, color=ll.color.get(ival),
                         linestyle=ll.style.get(ival),
                         linewidth=ll.width.get(ival),
                         zorder=ll.zorder)

    def add_label(self, ir, ic, text='', position=None, rotation=0, size=None,
                  fill_color='#ffffff', edge_color='#aaaaaa', edge_width=1,
                  font='sans-serif', font_weight='normal', font_style='normal',
                  font_color='#666666', font_size=14, offset=False, **kwargs):
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
                                fill=True,
                                transform=self.axes.obj[ir, ic].transAxes,
                                facecolor=fill_color if type(fill_color) is str \
                                        else fill_color.get(ic + ir * self.ncol + 1),
                                edgecolor=edge_color if type(edge_color) is str \
                                        else edge_color.get(ic + ir * self.ncol + 1),
                                clip_on=False, zorder=1)

        self.axes.obj[ir, ic].add_patch(rect)

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
        text = self.axes.obj[ir, ic].text(
            position[0]+size[0]/self.axes.size[0]/2+offsetx,
            position[3]+size[1]/self.axes.size[1]/2+offsety, text,
            transform=self.axes.obj[ir, ic].transAxes,
            horizontalalignment='center',
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
            if 'ref_line' in self.legend.values.keys():
                ref_line = self.legend.values['ref_line']
                del self.legend.values['ref_line']
            else:
                ref_line = None
            if 'fit_line' in self.legend.values.keys():
                fit_line = self.legend.values['fit_line']
                del self.legend.values['fit_line']
            else:
                fit_line = None

            if self.axes.twin_x or self.axes.twin_y:
                keys = self.legend.values.keys()
            else:
                keys = natsorted(list(self.legend.values.keys()))
            lines = [self.legend.values[f][0] for f in keys]
            if ref_line is not None:
                keys = [self.ref_line.text] + keys
                lines = ref_line + lines

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

            for itext, text in enumerate(self.legend.obj.get_texts()):
                text.set_color(self.legend.font_color)
                if self.plot_func != 'plot_hist':
                    self.legend.obj.legendHandles[itext]. \
                        _legmarker.set_markersize(self.legend.marker_size)
                    if self.legend.marker_alpha is not None:
                        self.legend.obj.legendHandles[itext]. \
                            _legmarker.set_alpha(self.legend.marker_alpha)

            self.legend.obj.get_title().set_fontsize(self.legend.font_size)
            self.legend.obj.get_frame().set_facecolor(self.legend.fill_color.get(0))
            self.legend.obj.get_frame().set_alpha(self.legend.fill_alpha)
            self.legend.obj.get_frame().set_edgecolor(self.legend.edge_color.get(0))

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

    def fill_between_lines(self, ir, ic, iline, x, lcl, ucl, obj, twin=False):
        """
        Shade a region between two curves

        Args:

        """

        if twin:
            ax = self.axes2.obj[ir, ic]
        else:
            ax = self.axes.obj[ir, ic]
        obj = getattr(self, obj)
        fc = obj.fill_color
        ec = obj.edge_color
        ax.fill_between(x, lcl, ucl,
                        facecolor=fc.get(iline) if type(fc) is RepeatedList else fc,
                        edgecolor=ec.get(iline) if type(ec) is RepeatedList else ec)

    def get_axes(self):
        """
        Return list of active axes
        """

        axes = [f for f in [self.axes, self.axes2] if f.on]
        #if self.axes2.on:
        #    axes += [self.axes2]

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

        start = datetime.datetime.now()
        now = start.strftime('%Y-%m-%d-%H-%M-%S')

        # Make a dummy figure
        data = copy.deepcopy(data)
        if 'box' in self.plot_func:
            data.x = ['x']
            data.df_fig['x'] = 1

        mplp.ioff()
        fig = mpl.pyplot.figure(dpi=self.fig.dpi)
        ax = fig.add_subplot(111)
        ax2, ax3 = None, None
        if self.axes.twin_x or data.z is not None \
                or self.plot_func == 'plot_heatmap':
            ax2 = ax.twinx()
        if self.axes.twin_y:
            ax3 = ax.twiny()

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

        box_group_label = np.array([[None]*self.ncol]*self.nrow)
        box_group_title = []
        changes = np.array([[None]*self.ncol]*self.nrow)

        # Plot data
        for ir, ic, df in data.get_rc_subset(data.df_fig):
            if len(df) == 0:
                continue
            # twin_x
            if self.axes.twin_x:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax2.plot(df[data.x[0]], df[data.y2[0]], 'o-')
            # twin_y
            elif self.axes.twin_y:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax3.plot(df[data.x2[0]], df[data.y[0]], 'o-')
            # Z axis
            elif self.plot_func == 'plot_heatmap':
                pp = ax.imshow(df, vmin=data.zmin, vmax=data.zmax)
                if self.cbar.on:
                    cbar = self.add_cbar(ax, pp)
                    ax2 = cbar.ax

                # Set ticks
                dtypes = [int, np.int32, np.int64]
                if df.index.dtype not in dtypes:
                    ax.set_yticks(np.arange(len(df)))
                    ax.set_yticklabels(df.index)
                if df.columns.dtype not in dtypes:
                    ax.set_xticks(np.arange(len(df.columns)))
                    ax.set_xticklabels(df.columns)
                if len(df) > len(df.columns) and self.axes.size[0] == self.axes.size[1]:
                    self.axes.size[0] = self.axes.size[0] * len(df.columns) / len(df)
                    self.label_col.size[0] = self.axes.size[0]
                elif len(df) < len(df.columns) and self.axes.size[0] == self.axes.size[1]:
                    self.axes.size[1] = self.axes.size[1] * len(df) / len(df.columns)
                    self.label_row.size[1] = self.axes.size[1]
            elif self.plot_func == 'plot_contour':
                pp, cbar = self.plot_contour(ax, df, data.x[0], data.y[0], data.z[0],
                                             data.ranges[ir, ic])
                if cbar is not None:
                    ax2 = cbar.ax
            elif data.z is not None:
                pp = ax.plot(df[data.x[0]], df[data.y[0]], 'o-')
                pp2 = ax2.plot(df[data.x[0]], df[data.z[0]], 'o-')
                if data.ranges[ir, ic]['zmin'] is not None:
                    ax2.set_ylim(bottom=data.ranges[ir, ic]['zmin'])
                if data.ranges[ir, ic]['y2max'] is not None and data.twin_x:
                    ax2.set_ylim(top=data.ranges[ir, ic]['zmax'])
            # hist
            elif self.plot_func == 'plot_hist':
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    pp = ax.hist(df[data.x[0]], bins=self.hist.bins, normed=self.hist.normalize)
                else:
                    pp = ax.hist(df[data.x[0]], bins=self.hist.bins, density=self.hist.normalize)
            # Regular
            else:
                for xy in zip(data.x, data.y):
                    pp = ax.plot(df[xy[0]], df[xy[1]], 'o-')

        # Set tick and scale properties
        if self.axes.scale in ['loglog', 'log']:
            ax.set_xscale('log')
            ax.set_yscale('log')
        elif self.axes.scale in LOGY:
            ax.set_yscale('log')
        elif self.axes.scale in LOGX:
            ax.set_xscale('log')
        elif self.axes.scale in ['symlog']:
            ax.set_xscale('symlog')
            ax.set_yscale('symlog')
        elif self.axes.scale in SYMLOGY:
            ax.set_yscale('symlog')
        elif self.axes.scale in SYMLOGX:
            ax.set_xscale('symlog')
        elif self.axes.scale in ['logit']:
            ax.set_xscale('logit')
            ax.set_yscale('logit')
        elif self.axes.scale in LOGITY:
            ax.set_yscale('logit')
        elif self.axes.scale in LOGITX:
            ax.set_xscale('logit')
        if self.axes.twin_x:
            if self.axes2.scale in LOGY:
                ax2.set_yscale('log')
            elif self.axes2.scale in SYMLOGY:
                ax2.set_yscale('symlog')
            elif self.axes2.scale in LOGITY:
                ax2.set_yscale('logit')
        if self.axes.twin_y:
            if self.axes2.scale in LOGX:
                ax3.set_xscale('log')
            elif self.axes2.scale in SYMLOGX:
                ax3.set_xscale('symlog')
            elif self.axes2.scale in LOGITX:
                ax3.set_xscale('logit')

        for ir, ic, df in data.get_rc_subset(data.df_fig):
            if len(df) == 0:
                continue
            if data.ranges[ir, ic]['xmin'] is not None:
                ax.set_xlim(left=data.ranges[ir, ic]['xmin'])
            if data.ranges[ir, ic]['xmax'] is not None:
                ax.set_xlim(right=data.ranges[ir, ic]['xmax'])
            if data.ranges[ir, ic]['ymin'] is not None:
                ax.set_ylim(bottom=data.ranges[ir, ic]['ymin'])
            if data.ranges[ir, ic]['ymax'] is not None:
                ax.set_ylim(top=data.ranges[ir, ic]['ymax'])
            if data.ranges[ir, ic]['x2min'] is not None and data.twin_y:
                ax3.set_xlim(left=data.ranges[ir, ic]['x2min'])
            if data.ranges[ir, ic]['x2max'] is not None and data.twin_y:
                ax3.set_xlim(right=data.ranges[ir, ic]['x2max'])
            if data.ranges[ir, ic]['y2min'] is not None and data.twin_x:
                ax2.set_ylim(bottom=data.ranges[ir, ic]['y2min'])
            if data.ranges[ir, ic]['y2max'] is not None and data.twin_x:
                ax2.set_ylim(top=data.ranges[ir, ic]['y2max'])

        axes = [ax, ax2, ax3]
        for ia, aa in enumerate(axes):
            if aa is None:
                continue
            if not (ia == 1 and self.plot_func == 'plot_heatmap'):
                axes[ia] = self.set_scientific(aa, ia)
            if not self.tick_labels_major.offset:
                try:
                    axes[ia].get_xaxis().get_major_formatter().set_useOffset(False)
                except:
                    pass
                try:
                    axes[ia].get_yaxis().get_major_formatter().set_useOffset(False)
                except:
                    pass
            axes[ia].minorticks_on()
            if ia == 0:
                axes[ia].tick_params(axis='both',
                                     which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color.get(0),
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
                                     colors=self.ticks_minor.color.get(0),
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

        for ia, aa in enumerate(axes):
            if aa is None:
                continue
            if not (ia == 1 and self.plot_func == 'plot_heatmap'):
                axes[ia] = self.set_scientific(aa, ia)

        # Ticks
        for ir, ic, df in data.get_rc_subset(data.df_fig):
            # have to do this a second time... may be a better way
            if len(df) == 0:
                continue
            if data.ranges[ir, ic]['xmin'] is not None:
                ax.set_xlim(left=data.ranges[ir, ic]['xmin'])
            if data.ranges[ir, ic]['xmax'] is not None:
                ax.set_xlim(right=data.ranges[ir, ic]['xmax'])
            if data.ranges[ir, ic]['ymin'] is not None:
                ax.set_ylim(bottom=data.ranges[ir, ic]['ymin'])
            if data.ranges[ir, ic]['ymax'] is not None:
                ax.set_ylim(top=data.ranges[ir, ic]['ymax'])
            if data.ranges[ir, ic]['x2min'] is not None and data.twin_y:
                ax3.set_xlim(left=data.ranges[ir, ic]['x2min'])
            if data.ranges[ir, ic]['x2max'] is not None and data.twin_y:
                ax3.set_xlim(right=data.ranges[ir, ic]['x2max'])
            if data.ranges[ir, ic]['y2min'] is not None and data.twin_x:
                ax2.set_ylim(bottom=data.ranges[ir, ic]['y2min'])
            if data.ranges[ir, ic]['y2max'] is not None and data.twin_x:
                ax2.set_ylim(top=data.ranges[ir, ic]['y2max'])

            # Set custom tick increment
            xinc = self.ticks_major_x.increment
            if xinc is not None:
                xlim = ax.get_xlim()
                ax.set_xticks(
                    np.arange(xlim[0] + xinc - xlim[0] % xinc,
                              xlim[1], xinc))
            yinc = self.ticks_major_y.increment
            if yinc is not None:
                ylim = ax.get_ylim()
                ax.set_yticks(
                    np.arange(ylim[0] + yinc - ylim[0] % yinc,
                              ylim[1], yinc))
            x2inc = self.ticks_major_x2.increment
            if ax3 is not None and x2inc is not None:
                xlim = ax3.get_xlim()
                ax3.set_xticks(
                    np.arange(xlim[0] + x2inc - xlim[0] % x2inc,
                              xlmin[1], x2inc))
            y2inc = self.ticks_major_y2.increment
            if ax2 is not None and y2inc is not None:
                ylim = ax2.get_ylim()
                ax2.set_yticks(
                    np.arange(ylim[0] + y2inc - ylim[0] % y2inc,
                              ylim[1], y2inc))

            # Major ticks
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xiter_ticks = [f for f in ax.xaxis.iter_ticks()] # fails for symlog in 1.5.1
            yiter_ticks = [f for f in ax.yaxis.iter_ticks()]
            xticksmaj += [f[2] for f in xiter_ticks[0:len(xticks)]]
            yticksmaj += [f[2] for f in yiter_ticks[0:len(yticks)]]

            if data.twin_x:
                y2ticks = [f[2] for f in ax2.yaxis.iter_ticks()
                          if f[2] != '']
                y2iter_ticks = [f for f in ax2.yaxis.iter_ticks()]
                y2ticksmaj += [f[2] for f in y2iter_ticks[0:len(y2ticks)]]
            elif data.twin_y:
                x2ticks = [f[2] for f in ax3.xaxis.iter_ticks()
                           if f[2] != '']
                x2iter_ticks = [f for f in ax3.xaxis.iter_ticks()]
                x2ticksmaj += [f[2] for f in x2iter_ticks[0:len(x2ticks)]]
            if data.z is not None:
                zticks = ax2.get_yticks()
                ziter_ticks = [f for f in ax2.yaxis.iter_ticks()]
                zticksmaj += [f[2] for f in ziter_ticks[0:len(zticks)]]

            # Minor ticks
            if self.tick_labels_minor_x.on:
                if self.ticks_minor_x.number is not None:
                    if self.axes.scale not in LOG_ALLX:
                        loc = None
                        loc = AutoMinorLocator(self.ticks_minor_x.number + 1)
                        ax.xaxis.set_minor_locator(loc)
                tp = mpl_get_ticks(ax)
                lim = ax.get_xlim()
                vals = [f for f in tp['x']['ticks'] if f > lim[0]]
                label_vals = [f for f in tp['x']['label_vals'] if f > lim[0]]
                inc = label_vals[1] - label_vals[0]
                minor_ticks = [f[1] for f in tp['x']['labels']][len(tp['x']['ticks']):]
                number = len([f for f in minor_ticks if f > vals[0] and f < vals[1]]) + 1
                decimals = utl.get_decimals(inc/number)
                ax.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimals)))
                xiter_ticks = [f for f in ax.xaxis.iter_ticks()]

            if self.tick_labels_minor_y.on:
                if self.ticks_minor_y.number is not None:
                    if self.axes.scale not in LOG_ALLY:
                        loc = None
                        loc = AutoMinorLocator(self.ticks_minor_y.number + 1)
                        ax.yaxis.set_minor_locator(loc)
                tp = mpl_get_ticks(ax)
                lim = ax.get_ylim()
                vals = [f for f in tp['y']['ticks'] if f > lim[0]]
                label_vals = [f for f in tp['y']['label_vals'] if f > lim[0]]
                inc = label_vals[1] - label_vals[0]
                minor_ticks = [f[1] for f in tp['y']['labels']][len(tp['y']['ticks']):]
                number = len([f for f in minor_ticks if f > vals[0] and f < vals[1]]) + 1
                decimals = utl.get_decimals(inc/number)
                ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimals)))
                yiter_ticks = [f for f in ax.yaxis.iter_ticks()]

            xticksmin += [f[2] for f in ax.xaxis.iter_ticks()][len(xticks):]
            yticksmin += [f[2] for f in ax.yaxis.iter_ticks()][len(yticks):]

            if self.tick_labels_minor_x2.on and ax3 is not None:
                if self.ticks_minor_x2.number is not None:
                    if self.axes2.scale not in LOG_ALLX:
                        loc = None
                        loc = AutoMinorLocator(self.ticks_minor_x2.number + 1)
                        ax3.xaxis.set_minor_locator(loc)
                tp = mpl_get_ticks(ax3)
                lim = ax3.get_xlim()
                vals = [f for f in tp['x']['ticks'] if f > lim[0]]
                label_vals = [f for f in tp['x']['label_vals'] if f > lim[0]]
                inc = label_vals[1] - label_vals[0]
                minor_ticks = [f[1] for f in tp['x']['labels']][len(tp['x']['ticks']):]
                number = len([f for f in minor_ticks if f > vals[0] and f < vals[1]]) + 1
                decimals = utl.get_decimals(inc/number)
                ax3.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimals)))
                xiter_ticks = [f for f in ax3.xaxis.iter_ticks()]

            if self.tick_labels_minor_y2.on and ax2 is not None:
                if self.ticks_minor_y2.number is not None:
                    if self.axes.scale not in LOG_ALLY:
                        loc = None
                        loc = AutoMinorLocator(self.ticks_minor_y2.number + 1)
                        ax2.yaxis.set_minor_locator(loc)
                tp = mpl_get_ticks(ax2)
                lim = ax2.get_ylim()
                vals = [f for f in tp['y']['ticks'] if f > lim[0]]
                label_vals = [f for f in tp['y']['label_vals'] if f > lim[0]]
                inc = label_vals[1] - label_vals[0]
                minor_ticks = [f[1] for f in tp['y']['labels']][len(tp['y']['ticks']):]
                number = len([f for f in minor_ticks if f > vals[0] and f < vals[1]]) + 1
                decimals = utl.get_decimals(inc/number)
                ax2.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%%.%sf' % (decimals)))
                yiter_ticks = [f for f in ax2.yaxis.iter_ticks()]

            if ax3 is not None:
                x2ticksmin += [f[2] for f in ax3.xaxis.iter_ticks()][len(xticks):]
            if ax2 is not None:
                y2ticksmin += [f[2] for f in ax2.yaxis.iter_ticks()][len(yticks):]

            self.axes.obj = np.array([[None]*self.ncol]*self.nrow)
            self.axes.obj[ir, ic] = ax
            ww, rr, cc = self.set_axes_rc_labels(ir, ic)
            wrap_labels[ir, ic] = ww
            row_labels[ir, ic] = rr
            col_labels[ir, ic] = cc
            self.axes.obj = None

            # Write out boxplot group labels
            if data.groups is None:
                self.box_group_label.on = False
                self.box_group_title.on = False
            if 'box' in self.plot_func and self.box_group_label.on:
                changes[ir, ic] = data.changes
                box_group_label[ir, ic] = []
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
                    box_group_label[ir, ic] += [box_group_label_row]
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

        # Make a dummy legend --> move to add_legend???
        if data.legend_vals is not None and len(data.legend_vals) > 0 \
                or self.ref_line.on or self.fit.on:
            lines = []
            leg_vals = []
            if type(data.legend_vals) == pd.DataFrame:
                for irow, row in data.legend_vals.iterrows():
                    lines += ax.plot([1, 2, 3])
                    leg_vals += [row['names']]
            elif data.legend_vals:
                for val in data.legend_vals:
                    lines += ax.plot([1, 2, 3])
                    leg_vals += [val]
            if self.ref_line.on:
                lines += ax.plot([1, 2, 3])
                leg_vals += [self.ref_line.text]
            if self.fit.on:
                lines += ax.plot([1, 2, 3])
                if data.legend_vals is not None and \
                        len(data.legend_vals) > 0:
                    leg_vals += [row['names'] + '[Fit]']
                else:
                    leg_vals += ['Fit']
            leg = mpl.pyplot.legend(lines, leg_vals,
                                    title=self.legend.text,
                                    numpoints=self.legend.points,
                                    fontsize=self.legend.font_size)
            leg.get_title().set_fontsize(self.legend.font_size)
            if self.legend.marker_size:
                if type(data.legend_vals) == pd.DataFrame:
                    for irow, row in data.legend_vals.iterrows():
                        leg.legendHandles[irow]._legmarker\
                            .set_markersize(self.legend.marker_size)
                elif data.legend_vals:
                    for irow, row in enumerate(data.legend_vals):
                        leg.legendHandles[irow]._legmarker\
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
            xticklabelsmin += [fig.text(ix*40, 40, xtick,
                                        fontsize=self.tick_labels_minor_x.font_size,
                                        rotation=self.tick_labels_minor_x.rotation)]
        for ix, x2tick in enumerate(x2ticksmin):
            x2ticklabelsmin += [fig.text(ix*40, 40, x2tick,
                                         fontsize=self.tick_labels_minor_x2.font_size,
                                         rotation=self.tick_labels_minor_x2.rotation)]
        for iy, ytick in enumerate(yticksmin):
            yticklabelsmin += [fig.text(40, iy*40, ytick,
                                        fontsize=self.tick_labels_minor_y.font_size,
                                        rotation=self.tick_labels_minor_y.rotation)]
        for iy, y2tick in enumerate(y2ticksmin):
            y2ticklabelsmin += [fig.text(40, iy*40, y2tick,
                                         fontsize=self.tick_labels_minor_y2.font_size,
                                         rotation=self.tick_labels_minor_y2.rotation)]

        # Write out axes labels
        label_x = []
        label_x2 = []
        label_y = []
        label_y2 = []
        label_z = []
        if type(self.label_x.text) is str:
            label_x = [fig.text(0, 0, r'%s' % self.label_x.text,
                                fontsize=self.label_x.font_size,
                                weight=self.label_x.font_weight,
                                style=self.label_x.font_style,
                                color=self.label_x.font_color,
                                rotation=self.label_x.rotation)]

        if type(self.label_x.text) is list:
            label_x = []
            for text in self.label_x.text:
                label_x += [fig.text(0, 0, r'%s' % text,
                                     fontsize=self.label_x.font_size,
                                     weight=self.label_x.font_weight,
                                     style=self.label_x.font_style,
                                     color=self.label_x.font_color,
                                     rotation=self.label_x.rotation)]

        if type(self.label_x2.text) is str:
            label_x2 = [fig.text(0, 0, r'%s' % self.label_x2.text,
                               fontsize=self.label_x2.font_size,
                               weight=self.label_x2.font_weight,
                               style=self.label_x2.font_style,
                               color=self.label_x2.font_color,
                               rotation=self.label_x2.rotation)]

        if type(self.label_y.text) is str:
            label_y = [fig.text(0, 0, r'%s' % self.label_y.text,
                               fontsize=self.label_y.font_size,
                               weight=self.label_y.font_weight,
                               style=self.label_y.font_style,
                               color=self.label_y.font_color,
                               rotation=self.label_y.rotation)]

        if type(self.label_y.text) is list:
            label_y = []
            for text in self.label_y.text:
                label_y += [fig.text(0, 0, r'%s' % text,
                                     fontsize=self.label_y.font_size,
                                     weight=self.label_y.font_weight,
                                     style=self.label_y.font_style,
                                     color=self.label_y.font_color,
                                     rotation=self.label_y.rotation)]

        if type(self.label_y2.text) is str:
            label_y2 = [fig.text(0, 0, r'%s' % self.label_y2.text,
                               fontsize=self.label_y2.font_size,
                               weight=self.label_y2.font_weight,
                               style=self.label_y2.font_style,
                               color=self.label_y2.font_color,
                               rotation=self.label_y2.rotation)]

        if type(self.label_z.text) is str:
            label_z = [fig.text(0, 0, r'%s' % self.label_z.text,
                               fontsize=self.label_z.font_size,
                               weight=self.label_z.font_weight,
                               style=self.label_z.font_style,
                               color=self.label_z.font_color,
                               rotation=self.label_z.rotation)]

        # Write out title
        if type(self.title.text) is str:
            title = fig.text(0, 0, r'%s' % self.title.text,
                             fontsize=self.title.font_size,
                             weight=self.title.font_weight,
                             style=self.title.font_style,
                             color=self.title.font_color,
                             rotation=self.title.rotation)

        # Render dummy figure
        saved = False
        mpl.pyplot.draw()
        try:
            [t.get_window_extent().width for t in xticklabelsmaj]
        except:
            saved = True
            filename = '%s%s' % (int(round(time.time() * 1000)), randint(0, 99))
            mpl.pyplot.savefig(filename + '.png')

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

        if len(label_x) > 0:
            self.label_x.size = (max([f.get_window_extent().width for f in label_x]),
                                 max([f.get_window_extent().height for f in label_x]))
        if len(label_x2) > 0:
            self.label_x2.size = (max([f.get_window_extent().width for f in label_x2]),
                                 max([f.get_window_extent().height for f in label_x2]))
        if len(label_y) > 0:
            self.label_y.size = (max([f.get_window_extent().width for f in label_y]),
                                 max([f.get_window_extent().height for f in label_y]))
        if len(label_y2) > 0:
            self.label_y2.size = (max([f.get_window_extent().width for f in label_y2]),
                                 max([f.get_window_extent().height for f in label_y2]))
        if len(label_z) > 0:
            self.label_z.size = (max([f.get_window_extent().width for f in label_z]),
                                 max([f.get_window_extent().height for f in label_z]))
        if self.title.on and type(self.title.text) is str:
            self.title.size[0] = max(self.axes.size[0],
                                     title.get_window_extent().width)
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
            rotations = np.array([0] * len(data.groups))
            sizes = np.array([[0,0]] * len(data.groups))
            for ir in range(0, self.nrow):
                for ic in range(0, self.ncol):
                    if box_group_label[ir, ic] is None:
                        continue
                    for irow, row in enumerate(box_group_label[ir, ic]):
                        # Find the smallest group label box in the row
                        labidx = list(changes[ir, ic][changes[ir, ic][irow]>0].index) + \
                                    [len(changes[ir, ic])]
                        smallest = min(np.diff(labidx))
                        max_label_width = self.axes.size[0]/len(changes[ir, ic]) * smallest
                        widest = max([f.get_window_extent().width for f in row])
                        tallest = max([f.get_window_extent().height for f in row])
                        if widest > max_label_width:
                            rotations[irow] = 90
                            sizes[irow] = [tallest, widest]
                        elif rotations[irow] != 90:
                            sizes[irow] = [widest, tallest]

            sizes = sizes.tolist()
            sizes.reverse()
            rotations = rotations.tolist()
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
        if saved:
            os.remove(filename + '.png')

    def get_figure_size(self, data, **kwargs):
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
        self.labtick_x2 = (self.label_x2.size[1] + self.ws_label_tick + 2*self.ws_ticks_ax + \
                           max(self.tick_labels_major_x2.size[1],
                               self.tick_labels_minor_x2.size[1])) * self.axes.twin_y
        self.labtick_y = self.label_y.size[0] + self.ws_label_tick + \
                         max(self.tick_labels_major_y.size[0],
                             self.tick_labels_minor_y.size[0]) + self.ws_ticks_ax
        self.labtick_y2 = (self.label_y2.size[0] + self.ws_label_tick + 2*self.ws_ticks_ax + \
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
            self.ws_col += self.label_y.size[0] + self.ws_label_tick + self.ws_fig_label
            self.ws_row += self.label_x.size[1] + self.ws_label_tick + self.ws_fig_label

        if self.separate_ticks:
            self.ws_col += max(self.tick_labels_major_y.size[0],
                               self.tick_labels_minor_y.size[0]) + self.ws_ticks_ax
            self.ws_row += max(self.tick_labels_major_x.size[1],
                               self.tick_labels_minor_x.size[1]) + self.ws_ticks_ax
        if self.plot_func == 'plot_heatmap' and \
                self.heatmap.cell_size is not None and \
                data.num_x is not None:
            self.axes.size = [self.heatmap.cell_size * data.num_x,
                              self.heatmap.cell_size * data.num_y]

        # Figure width
        self.fig.size[0] = \
            self.ws_fig_label + \
            self.labtick_y + \
            (self.axes.size[0] + self.cbar.size[0] + self.ws_ax_cbar) * self.ncol + \
            self.ws_col * (self.ncol - 1) +  \
            self.ws_leg_ax + self.legend.size[0] + self.ws_leg_fig + \
            self.legend.edge_width + self.labtick_y2 + \
            self.label_row.size[0] + self.ws_label_row * self.label_row.on + \
            self.labtick_z * (self.ncol - 1 if self.ncol > 1 else 1) + \
            self.box_title
        self.fig.size[0] = max(self.fig.size[0], self.title.size[0])

        # Figure height
        self.fig.size[1] = \
            self.ws_title + \
            (self.label_col.size[1] + self.ws_label_col) * self.label_col.on + \
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
                    'ws_label_row', 'label_z', 'tick_labels_major_z', 'box_title',
                    'ncol', 'labtick_y', 'labtick_y2', 'labtick_z']
            for val in vals:
                if isinstance(getattr(self, val), Element):
                    print('   %s.size[0] = %s' % (val, getattr(self, val).size[0]))
                else:
                    print('   %s = %s' % (val, getattr(self, val)))
            print('self.fig.size[1] = %s' % self.fig.size[1])
            vals = ['ws_fig_title', 'title', 'ws_title_ax', 'ws_fig_ax',
                    'label_col', 'ws_label_col', 'title_wrap',
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
            (self.axes.size[0] + self.labtick_y2 + self.ws_label_row +
             (self.ws_ax_cbar if self.cbar.on else 0) + self.cbar.size[0] +
             self.labtick_z)/self.axes.size[0]

        self.label_col.position[3] = (self.axes.size[1] + self.ws_label_col +
                                      self.labtick_x2)/self.axes.size[1]

        self.label_wrap.position[3] = 1
        self.title_wrap.size[0] = self.ncol * self.title_wrap.size[0] + (self.ncol - 1) * self.ws_col
        self.title_wrap.position[3] = 1 + (self.label_wrap.size[1] + 1)/ self.axes.size[1]

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
            (self.label_col.size[1] + self.ws_label_col) * self.label_col.on + \
            self.label_wrap.size[1] + self.labtick_x2) / self.fig.size[1]

        self.axes.position[3] = \
            (self.labtick_x + self.ws_fig_label + self.box_labels + \
             self.legend.overflow) / self.fig.size[1]

    def get_title_position(self):
        """
        Calculate the title position
            self.title.position --> [left, right, top, bottom]
        """

        col_label = (self.label_col.size[1] + \
                     self.ws_label_col * self.label_col.on)
        self.title.position[0] = (self.axes.size[0] * self.ncol + \
                                  self.ws_col * (self.ncol - 1) - \
                                  self.title.size[0])/2/self.axes.size[0]
        self.title.position[3] = 1+(self.ws_title_ax + col_label + self.label_wrap.size[1] + \
                                    self.title_wrap.size[1]) / self.axes.size[1]
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

        self.set_colormap(data)
        self.set_label_text(data, **kwargs)
        self.get_element_sizes(data)
        self.get_figure_size(data, **kwargs)
        self.get_subplots_adjust()
        self.get_rc_label_position()
        self.get_legend_position()

        # # Update subplot spacing
        # if self.label_y.size[1] > self.axes.size[1]:
        #     st()
        #     self.ws_row += self.label_y.size[1] - self.axes.size[1] + 10

        # Define the subplots
        fig, axes = \
            mplp.subplots(data.nrow, data.ncol,
                          figsize=[self.fig.size_inches[0], self.fig.size_inches[1]],
                          sharex=self.axes.share_x,
                          sharey=self.axes.share_y,
                          dpi=self.fig.dpi,
                          facecolor=self.fig.fill_color.get(0),
                          edgecolor=self.fig.edge_color.get(0),
                          linewidth=self.fig.edge_width,
                          )
        self.fig.obj = fig
        self.axes.obj = axes
        self.axes.visible = np.array([[True]*self.ncol]*self.nrow)

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

        bp = None

        if self.violin.on:
            bp = self.axes.obj[ir, ic].violinplot(data,
                                                  showmeans=False,
                                                  showextrema=False,
                                                  showmedians=False,
                                                 )
            for ipatch, patch in enumerate(bp['bodies']):
                patch.set_facecolor(self.violin.fill_color.get(ipatch))
                patch.set_edgecolor(self.violin.edge_color.get(ipatch))
                patch.set_alpha(self.violin.fill_alpha)
                patch.set_zorder(2)
                patch.set_lw(self.violin.edge_width)
                if self.violin.box_on:
                    q25 = np.percentile(data[ipatch], 25)
                    med = np.percentile(data[ipatch], 50)
                    q75 = np.percentile(data[ipatch], 75)
                    iqr = q75 - q25
                    offset = 0.05 * len(data) / 7
                    bb = mtransforms.Bbox([[1 + ipatch - offset, q25],
                                           [1 + ipatch + offset, q75]])
                    p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                            abs(bb.width), abs(bb.height),
                            boxstyle="round,pad=0, rounding_size=0.05",
                            ec="none", fc=self.violin.box_color, zorder=12)
                    self.axes.obj[ir, ic].add_patch(p_bbox)
                    whisker_max = min(max(data[ipatch]), q75 + 1.5 * iqr)
                    whisker_min = max(min(data[ipatch]), q25 - 1.5 * iqr)
                    self.axes.obj[ir, ic].plot([ipatch + 1, ipatch + 1],
                                               [whisker_min, whisker_max],
                                               linestyle=self.violin.whisker_style,
                                               color=self.violin.whisker_color,
                                               linewidth=self.violin.whisker_width)
                    self.axes.obj[ir, ic].plot([ipatch + 1], [med],
                                               marker=self.violin.median_marker,
                                               color=self.violin.median_color,
                                               markersize=self.violin.median_size,
                                               markeredgecolor=self.violin.median_color,
                                               zorder=13)

        elif self.box.on and not self.violin.on:
            bp = self.axes.obj[ir, ic].boxplot(data,
                                               labels=[''] * len(data),
                                               showfliers=False,
                                               medianprops={'color': self.box.median_color},
                                               notch=self.box.notch,
                                               patch_artist=True,
                                               zorder=3)#,
                                               #widths=self.box.width.values[0]
                                                #      if len(self.box.width.values) == 1
                                                 #     else self.box.width.values)
            for ipatch, patch in enumerate(bp['boxes']):
                patch.set_edgecolor(self.box.edge_color.get(ipatch))
                patch.set_facecolor(self.box.fill_color.get(ipatch))
                patch.set_alpha(self.box.fill_alpha)
                patch.set_lw(self.box.edge_width)
                patch.set_ls(self.box.style.get(ipatch))
            for ipatch, patch in enumerate(bp['whiskers']):
                patch.set_color(self.box_whisker.color.get(int(ipatch/2)))
                patch.set_lw(1)#self.box_whisker.width.get(ipatch))
                patch.set_ls(self.box_whisker.style.get(ipatch))
            for ipatch, patch in enumerate(bp['caps']):
                patch.set_color(self.box_whisker.color.get(int(ipatch/2)))
                patch.set_lw(self.box_whisker.width.get(ipatch))
                patch.set_ls(self.box_whisker.style.get(ipatch))

        self.axes.obj[ir, ic].set_xticklabels([''])
        self.axes.obj[ir, ic].set_xlim(0.5, len(data) + 0.5)

        return bp

    def plot_contour(self, ax, df, x, y, z, ranges):
        """
        Plot a contour plot
        """

        # Convert data type
        xx = np.array(df[x])
        yy = np.array(df[y])
        zz = np.array(df[z])

        # Make the grid
        if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
            xi = np.linspace(min(xx), max(xx))
            yi = np.linspace(min(yy), max(yy))
            zi = mlab.griddata(xx, yy, zz, xi, yi, interp='linear')
        else:
            xi, yi = np.mgrid[min(xx):max(xx):100j, min(yy):max(yy):200j]
            zi = scipy.interpolate.griddata((xx, yy), zz, (xi, yi), method='linear')

        if self.contour.filled:
            cc = ax.contourf(xi, yi, zi, self.contour.levels, cmap=self.contour.cmap, zorder=2,
                             vmin=ranges['zmin'], vmax=ranges['zmax'])
        else:
            cc = ax.contour(xi, yi, zi, self.contour.levels, linewidths=self.contour.width.values,
                            cmap=self.contour.cmap, zorder=2, vmin=ranges['zmin'], vmax=ranges['zmax'])

        if self.cbar.on:
            cbar = self.add_cbar(ax, cc)
        else:
            cbar = None

        return cc, cbar

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

        # Make the heatmap
        im = ax.imshow(df, self.heatmap.cmap, vmin=ranges['zmin'],
                       vmax=ranges['zmax'],
                       interpolation=self.heatmap.interpolation)

        # Set the axes
        dtypes = [int, np.int32, np.int64]
        if df.index.dtype not in dtypes:
            ax.set_yticks(np.arange(len(df)))
            ax.set_yticklabels(df.index)
            ax.set_yticks(np.arange(df.shape[0]+1)-.5, minor=True)
        if df.columns.dtype not in dtypes:
            ax.set_xticks(np.arange(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xticks(np.arange(df.shape[1]+1)-.5, minor=True)
        if df.index.dtype not in dtypes or df.columns.dtype not in dtypes:
            ax.grid(which="minor", color=self.heatmap.edge_color.get(0),
                    linestyle='-', linewidth=self.heatmap.edge_width)
            ax.tick_params(which="minor", bottom=False, left=False)
        if ranges['xmin'] is not None and ranges['xmin'] > 0:
            xticks = ax.get_xticks()
            ax.set_xticklabels([int(f + ranges['xmin']) for f in xticks])
        if ranges['ymax'] is not None and ranges['ymax'] > 0:
            yticks = ax.get_yticks()
            ax.set_yticklabels([int(f + ranges['ymax']) for f in yticks])

        if self.cbar.on:
            cbar = self.add_cbar(ax, im)
        else:
            cbar = None

        if self.heatmap.text:
            # Loop over data dimensions and create text annotations.
            for iy, yy in enumerate(df.index):
                for ix, xx in enumerate(df.columns):
                    if type(df.loc[yy, xx]) in [float, np.float32, np.float64] and \
                            np.isnan(df.loc[yy, xx]):
                        continue
                    text = ax.text(ix, iy, df.loc[yy, xx],
                                   ha="center", va="center",
                                   color=self.heatmap.font_color)

        return im

    def plot_hist(self, ir, ic, iline, df, x, y, leg_name, data, zorder=1,
                line_type=None, marker_disable=False):

        if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                hist = self.axes.obj[ir, ic].hist(df[x], bins=self.hist.bins,
                                            color=self.hist.fill_color.get(iline),
                                            ec=self.hist.edge_color.get(iline),
                                            lw=self.hist.edge_width,
                                            zorder=3, align=self.hist.align,
                                            cumulative=self.hist.cumulative,
                                            normed=self.hist.normalize,
                                            rwidth=self.hist.rwidth,
                                            stacked=self.hist.stacked,
                                            #type=self.hist.type,
                                            orientation='vertical' if not self.hist.horizontal else 'horizontal',
                                            )
        else:
            hist = self.axes.obj[ir, ic].hist(df[x], bins=self.hist.bins,
                                            color=self.hist.fill_color.get(iline),
                                            ec=self.hist.edge_color.get(iline),
                                            lw=self.hist.edge_width,
                                            zorder=3, align=self.hist.align,
                                            cumulative=self.hist.cumulative,
                                            density=self.hist.normalize,
                                            rwidth=self.hist.rwidth,
                                            stacked=self.hist.stacked,
                                            #type=self.hist.type,
                                            orientation='vertical' if not self.hist.horizontal else 'horizontal',
                                            )

        # Add a reference to the line to self.lines
        if leg_name:
            handle = [patches.Rectangle((0,0),1,1,color=self.hist.fill_color.get(iline))]
            self.legend.values[leg_name] = handle

        # Horizontal adjustments
        if self.hist.horizontal:
            # Swap labels
            if iline == 0 and ir == 0 and ic == 0:
                ylab = self.label_y.text
                self.label_y.text = self.label_x.text
                self.label_x.text = ylab
                self.label_x.size = [self.label_y.size[1], self.label_y.size[0]]
                self.label_y.size = [self.label_x.size[1], self.label_x.size[0]]

            # Rotate ranges
            if iline == 0:
                for irow in range(0, self.nrow):
                    for icol in range(0, self.ncol):
                        ymin = data.ranges[irow, icol]['ymin']
                        ymax = data.ranges[irow, icol]['ymax']
                        data.ranges[irow, icol]['ymin'] = data.ranges[irow, icol]['xmin']
                        data.ranges[irow, icol]['ymax'] = data.ranges[irow, icol]['xmax']
                        data.ranges[irow, icol]['xmin'] = ymin
                        data.ranges[irow, icol]['xmax'] = ymax

        # Add a kde
        if self.kde.on:
            kde = scipy.stats.gaussian_kde(df[x])
            if not self.hist.horizontal:
                x0 = np.linspace(data.ranges[ir, ic]['xmin'],
                                 data.ranges[ir, ic]['xmax'], 1000)
                y0 = kde(x0)
            else:
                y0 = np.linspace(data.ranges[ir, ic]['ymin'],
                                 data.ranges[ir, ic]['ymax'], 1000)
                x0 = kde(y0)
            kwargs = self.make_kwargs(self.kde)
            kwargs['color'] = RepeatedList(kwargs['color'].get(iline), 'color')
            kde = self.plot_line(ir, ic, x0, y0, **kwargs)

        return hist, data

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

        if x1 is not None:
            x0 = [x0, x1]
        if y1 is not None:
            y0 = [y0, y1]

        if 'color' not in kwargs.keys():
            kwargs['color'] = RepeatedList('#000000', 'temp')
        if 'style' not in kwargs.keys():
            kwargs['style'] = RepeatedList('-', 'temp')
        if 'width' not in kwargs.keys():
            kwargs['width'] = RepeatedList('-', 'temp')

        line = self.axes.obj[ir, ic].plot(x0, y0,
                                        linestyle=kwargs['style'].get(0),
                                        linewidth=kwargs['width'].get(0),
                                        color=kwargs['color'].get(0),
                                        zorder=kwargs.get('zorder', 1))
        return line

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
                            #linestyle='-' if line_type.style is None \
                            #          else line_type.style,
                            #linewidth=line_type.width)
                            linestyle=line_type.style.get(iline),
                            linewidth=line_type.width.get(iline),
                            )

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
            ir (int): subplot row index
            ic (int): subplot col index

        """

        axes = self.get_axes()

        #for ax in axes:
        try:
            axes[0].obj[ir, ic].set_facecolor(axes[0].fill_color.get(ic + ir * self.ncol + 1))
        except:
            axes[0].obj[ir, ic].set_axis_bgcolor(axes[0].fill_color.get(ic + ir * self.ncol + 1))

        for f in ['bottom', 'top', 'right', 'left']:
            if len(axes) > 1:
                axes[0].obj[ir, ic].spines[f].set_visible(False)
            if getattr(self.axes, 'spine_%s' % f):
                axes[-1].obj[ir, ic].spines[f].set_color(axes[0].edge_color.get(ic + ir * self.ncol + 1))
            else:
                axes[-1].obj[ir, ic].spines[f].set_color(self.fig.fill_color.get(0))
            axes[-1].obj[ir, ic].spines[f].set_linewidth(self.axes.edge_width)

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
            if not ax.primary and \
                    not (hasattr(self, 'grid_major_x2') or \
                    hasattr(self, 'grid_major_y2')):
                ax.obj[ir, ic].set_axisbelow(True)
                ax.obj[ir, ic].grid(False, which='major')
                ax.obj[ir, ic].grid(False, which='minor')
                continue

            # Set major grid
            ax.obj[ir, ic].set_axisbelow(True)
            if self.grid_major_x.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='major',
                                          #zorder=self.grid_major_x.zorder,
                                          color=self.grid_major_x.color.get(0),
                                          linestyle=self.grid_major_x.style.get(0),
                                          linewidth=self.grid_major_x.width.get(0))
            else:
                ax.obj[ir, ic].xaxis.grid(b=False, which='major')

            if hasattr(self, 'grid_major_x2') and not ax.primary:
                if self.grid_major_x.on:
                    ax.obj[ir, ic].xaxis.grid(b=True, which='major',
                                            #zorder=self.grid_major_x.zorder,
                                            color=self.grid_major_x2.color.get(0),
                                            linestyle=self.grid_major_x2.style.get(0),
                                            linewidth=self.grid_major_x2.width.get(0))
                else:
                    ax.obj[ir, ic].xaxis.grid(b=False, which='major')

            if self.grid_major_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='major',
                                          #zorder=self.grid_major_y.zorder,
                                          color=self.grid_major_y.color.get(0),
                                          linestyle=self.grid_major_y.style.get(0),
                                          linewidth=self.grid_major_y.width.get(0))
            else:
                ax.obj[ir, ic].yaxis.grid(b=False, which='major')

            if hasattr(self, 'grid_major_y2') and not ax.primary:
                if self.grid_major_y2.on:
                    ax.obj[ir, ic].yaxis.grid(b=True, which='major',
                                            #zorder=self.grid_major_y.zorder,
                                            color=self.grid_major_y2.color.get(0),
                                            linestyle=self.grid_major_y2.style.get(0),
                                            linewidth=self.grid_major_y2.width.get(0))
                else:
                    ax.obj[ir, ic].yaxis.grid(b=False, which='major')

            # Set minor grid
            if self.grid_minor_x.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='minor',
                                          #zorder=self.grid_minor_x.zorder,
                                          color=self.grid_minor_x.color.get(0),
                                          linestyle=self.grid_minor_x.style.get(0),
                                          linewidth=self.grid_minor_x.width.get(0))
            if self.grid_minor_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='minor',
                                          #zorder=self.grid_minor_y.zorder,
                                          color=self.grid_minor_y.color.get(0),
                                          linestyle=self.grid_minor_y.style.get(0),
                                          linewidth=self.grid_minor_y.width.get(0))

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
            if type(label.text) not in [str, list]:
                continue
            if type(label.text) is str:
                labeltext = label.text
            if type(label.text) is list:
                labeltext = label.text[ic + ir * self.ncol]

            if '2' in ax:
                axes = self.axes2.obj[ir, ic]
                pad = self.ws_label_tick*2
            else:
                axes = self.axes.obj[ir, ic]
                pad = self.ws_label_tick

            # Toggle label visibility
            if not self.separate_labels:
                if ax == 'x' and ir != self.nrow - 1 and \
                        self.nwrap == 0 and self.axes.visible[ir+1, ic]:
                    continue
                if ax == 'x2' and ir != 0:
                    continue
                if ax == 'y' and ic != 0 and self.axes.visible[ir, ic - 1]:
                    continue
                if ax == 'y2' and ic != self.ncol - 1 and \
                        (ic + ir * self.ncol + 1) != self.nwrap:
                    continue

            # Add the label
            self.add_label(ir, ic, labeltext, **self.make_kwargs(label))

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
                if str(ax.scale).lower() in LOGX:
                    ax.obj[ir, ic].set_xscale('log')
                elif str(ax.scale).lower() in SYMLOGX:
                    ax.obj[ir, ic].set_xscale('symlog')
                elif str(ax.scale).lower() in LOGITX:
                    ax.obj[ir, ic].set_xscale('logit')
                if str(ax.scale).lower() in LOGY:
                    ax.obj[ir, ic].set_yscale('log')
                elif str(ax.scale).lower() in SYMLOGY:
                    ax.obj[ir, ic].set_yscale('symlog')
                elif str(ax.scale).lower() in LOGITY:
                    ax.obj[ir, ic].set_yscale('logit')

    def set_axes_ranges(self, ir, ic, ranges):
        """
        Set the axes ranges

        Args:
            ir (int): subplot row index
            ic (int): subplot col index
            limits (dict): min/max axes limits for each axis

        """

        if self.plot_func == 'plot_heatmap':
            return

        # X-axis
        if self.axes.share_x:
            xvals = ['xmin', 'xmax', 'x2min', 'x2max']
            for xval in xvals:
                xx = None
                for irow in range(0, self.nrow):
                    for icol in range(0, self.ncol):
                        if ranges[irow, icol][xval] is not None:
                            if irow == 0 and icol == 0:
                                xx = ranges[irow, icol][xval]
                            elif 'min' in xval:
                                xx = min(xx, ranges[irow, icol][xval])
                            else:
                                xx = max(xx, ranges[irow, icol][xval])

                if xx is not None and xval == 'xmin':
                    self.axes.obj[ir, ic].set_xlim(left=xx)
                elif xx is not None and xval == 'x2min':
                    self.axes2.obj[ir, ic].set_xlim(left=xx)
                elif xx is not None and xval == 'xmax':
                    self.axes.obj[ir, ic].set_xlim(right=xx)
                elif xx is not None and xval == 'x2max':
                    self.axes2.obj[ir, ic].set_xlim(right=xx)
        else:
            if ranges[ir, ic]['xmin'] is not None:
                self.axes.obj[ir, ic].set_xlim(left=ranges[ir, ic]['xmin'])
            if ranges[ir, ic]['x2min'] is not None:
                self.axes2.obj[ir, ic].set_xlim(left=ranges[ir, ic]['x2min'])
            if ranges[ir, ic]['xmax'] is not None:
                self.axes.obj[ir, ic].set_xlim(right=ranges[ir, ic]['xmax'])
            if ranges[ir, ic]['x2max'] is not None:
                self.axes2.obj[ir, ic].set_xlim(right=ranges[ir, ic]['x2max'])

        # Y-axis
        if self.axes.share_y:
            yvals = ['ymin', 'ymax', 'y2min', 'y2max']
            for yval in yvals:
                yy = None
                for irow in range(0, self.nrow):
                    for icol in range(0, self.ncol):
                        if ranges[irow, icol][yval] is not None:
                            if irow == 0 and icol == 0:
                                yy = ranges[irow, icol][yval]
                            elif 'min' in yval:
                                yy = min(yy, ranges[irow, icol][yval])
                            else:
                                yy = max(yy, ranges[irow, icol][yval])

                if yy is not None and yval == 'ymin':
                    self.axes.obj[ir, ic].set_ylim(bottom=yy)
                elif yy is not None and yval == 'y2min':
                    self.axes2.obj[ir, ic].set_ylim(bottom=yy)
                elif yy is not None and yval == 'ymax':
                    self.axes.obj[ir, ic].set_ylim(top=yy)
                elif yy is not None and yval == 'y2max':
                    self.axes2.obj[ir, ic].set_ylim(top=yy)
        else:
            if ranges[ir, ic]['ymin'] is not None:
                self.axes.obj[ir, ic].set_ylim(bottom=ranges[ir, ic]['ymin'])
            if ranges[ir, ic]['y2min'] is not None:
                self.axes2.obj[ir, ic].set_ylim(bottom=ranges[ir, ic]['y2min'])
            if ranges[ir, ic]['ymax'] is not None:
                self.axes.obj[ir, ic].set_ylim(top=ranges[ir, ic]['ymax'])
            if ranges[ir, ic]['y2max'] is not None:
                self.axes2.obj[ir, ic].set_ylim(top=ranges[ir, ic]['y2max'])

    def set_axes_ranges_hist(self, ir, ic, data, hist, iline):
        """
        Special range overrides for histogram plot
            Needed to deal with the late computation of bin counts

        Args:
            ir (int): subplot row index
            ic (int): subplot col index
            data (Data obj): current data object
            hist (mpl.hist obj): result of hist plot

        """

        if not self.hist.horizontal:
            if data.ymin:
                data.ranges[ir, ic]['ymin'] = data.ymin
            else:
                data.ranges[ir, ic]['ymin'] = None
            if data.ymax:
                data.ranges[ir, ic]['ymax'] = data.ymax
            elif data.ranges[ir, ic]['ymax'] is None:
                data.ranges[ir, ic]['ymax'] = \
                    max(hist[0]) * (1 + data.ax_limit_padding_y_max)
            else:
                data.ranges[ir, ic]['ymax'] = \
                    max(data.ranges[ir, ic]['ymax'], max(hist[0]) * \
                        (1 + data.ax_limit_padding_y_max))
        elif self.hist.horizontal:
            if data.ymin:
                data.ranges[ir, ic]['ymin'] = data.ymin
            elif iline == 0:
                data.ranges[ir, ic]['ymin'] = data.ranges[ir, ic]['xmin']
            if data.ymax:
                data.ranges[ir, ic]['ymax'] = data.ymax
            elif iline == 0:
                data.ranges[ir, ic]['ymax'] = data.ranges[ir, ic]['xmax']
            if data.xmin:
                data.ranges[ir, ic]['xmin'] = data.xmin
            else:
                data.ranges[ir, ic]['xmin'] = None
            if data.xmax:
                data.ranges[ir, ic]['xmax'] = data.xmax
            elif iline == 0:
                data.ranges[ir, ic]['xmax'] = \
                    max(hist[0]) * (1 + data.ax_limit_padding_y_max)
            else:
                data.ranges[ir, ic]['xmax'] = \
                    max(data.ranges[ir, ic]['xmax'], max(hist[0]) * \
                        (1 + data.ax_limit_padding_x_max))

        return data

    def set_axes_rc_labels(self, ir, ic):
        """
        Add the row/column label boxes and wrap titles

        Args:
            ir (int): current row index
            ic (int): current column index

        """

        wrap, row, col = None, None, None

        # Wrap title  --> this guy's text size is not defined in get_elements_size
        if ir == 0 and ic == 0 and self.title_wrap.on:
            title = self.add_label(ir, ic, self.title_wrap.text,
                                   **self.make_kwargs(self.title_wrap))

        # Row labels
        if ic == self.ncol-1 and self.label_row.on and not self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_row.text_size[ir, ic]
            else:
                text_size = None
            row = self.add_label(ir, ic, '%s=%s' %
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
                scol = self.add_label(ir, ic, text,
                                     **self.make_kwargs(self.label_wrap))
            else:
                text = '%s=%s' % (self.label_col.text, self.label_col.values[ic])
                col = self.add_label(ir, ic, text,
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

        def get_tick_position(ax, tp, xy, loc='first', idx=0):
            """
            Find the position of a tick given the actual range

            Args:
                ax (mpl.axes): the axis of interest
                tp (dict): tick location dictionary
                xy (str): which axis to calculate ('x' or 'y')
                loc (str): tick location ('first' or 'last')

            Returns:
                the actual x-position of an xtick or the y-position of a ytick

            """

            if idx == 0:
                lab = ''
            else:
                lab = '2'

            size = 0 if xy == 'x' else 1
            if type(loc) is str:
                tick = tp[xy]['label_vals'][tp[xy][loc]]
            else:
                tick = tp[xy]['label_vals'][loc]
            lim = sorted(getattr(ax, 'get_%slim' % xy)())
            if tick > lim[1]:
                pos = self.axes.size[size]
            elif tick < lim[0]:
                pos = -999  # push it far away from axis
            elif getattr(self, 'axes%s' % lab).scale in (LOGX if xy == 'x' else LOGY):
                pos = (np.log10(tick) - np.log10(lim[0])) / \
                      (np.log10(lim[1]) - np.log10(lim[0])) * \
                       self.axes.size[size]
            else:
                pos = (tick - lim[0]) / (lim[1] - lim[0]) * self.axes.size[size]

            return pos

        axes = [f.obj[ir, ic] for f in [self.axes, self.axes2] if f.on]

        # Format ticks
        for ia, aa in enumerate(axes):

            if ia == 0:
                lab = ''
            else:
                lab = '2'

            # Skip certain calculations if axes are shared and subplots > 1
            skipx, skipy = False, False
            if hasattr(getattr(self, 'axes%s' % lab), 'share_x%s' % lab) and \
                    getattr(getattr(self, 'axes%s' % lab), 'share_x%s' % lab) == True and \
                    (ir != 0 or ic != 0):
                skipx = True
            if hasattr(getattr(self, 'axes%s' % lab), 'share_y%s' %lab) and \
                    getattr(getattr(self, 'axes%s' % lab), 'share_y%s' %lab) and \
                    (ir != 0 or ic != 0):
                skipy = True

            # Turn off scientific
            if ia == 0:
                if not skipx:
                    self.set_scientific(axes[ia])
            elif self.axes.twin_y or self.axes.twin_x:
                if not skipy:
                    self.set_scientific(axes[ia], 2)

            # Turn off offsets
            if not self.tick_labels_major.offset:
                try:
                    if not skipx:
                        aa.get_xaxis().get_major_formatter().set_useOffset(False)
                except:
                    pass
                try:
                    if not skipy:
                        aa.get_yaxis().get_major_formatter().set_useOffset(False)
                except:
                    pass

            # General tick params
            if ia == 0:
                if self.ticks_minor_x.on or self.ticks_minor_y.on:
                    axes[0].minorticks_on()
                axes[0].tick_params(axis='both',
                                    which='major',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_major.color.get(0),
                                    labelcolor=self.tick_labels_major.font_color,
                                    labelsize=self.tick_labels_major.font_size,
                                    top=False,
                                    bottom=self.ticks_major_x.on,
                                    right=False if self.axes.twin_x
                                        else self.ticks_major_y.on,
                                    left=self.ticks_major_y.on,
                                    length=self.ticks_major._size[0],
                                    width=self.ticks_major._size[1],
                                    direction=self.ticks_major.direction,
                                    zorder=100,
                                    )
                axes[0].tick_params(axis='both',
                                    which='minor',
                                    pad=self.ws_ticks_ax,
                                    colors=self.ticks_minor.color.get(0),
                                    labelcolor=self.tick_labels_minor.font_color,
                                    labelsize=self.tick_labels_minor.font_size,
                                    top=False,
                                    bottom=self.ticks_minor_x.on,
                                    right=False if self.axes.twin_x
                                        else self.ticks_minor_y.on,
                                    left=self.ticks_minor_y.on,
                                    length=self.ticks_minor._size[0],
                                    width=self.ticks_minor._size[1],
                                    direction=self.ticks_minor.direction,
                                    )
                if self.axes.twin_x:
                    if self.ticks_minor_y2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=self.ws_ticks_ax*2,
                                        colors=self.ticks_major.color.get(0),
                                        labelcolor=self.tick_labels_major.font_color,
                                        labelsize=self.tick_labels_major.font_size,
                                        right=self.ticks_major_y2.on,
                                        length=self.ticks_major.size[0],
                                        width=self.ticks_major.size[1],
                                        direction=self.ticks_major.direction,
                                        zorder=0,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=self.ws_ticks_ax*2,
                                        colors=self.ticks_minor.color.get(0),
                                        labelcolor=self.tick_labels_minor.font_color,
                                        labelsize=self.tick_labels_minor.font_size,
                                        right=self.ticks_minor_y2.on,
                                        length=self.ticks_minor._size[0],
                                        width=self.ticks_minor._size[1],
                                        direction=self.ticks_minor.direction,
                                        zorder=0,
                                        )
                elif self.axes.twin_y:
                    if self.ticks_minor_x2.on:
                        axes[1].minorticks_on()
                    axes[1].tick_params(which='major',
                                        pad=self.ws_ticks_ax*2,
                                        colors=self.ticks_major.color.get(0),
                                        labelcolor=self.tick_labels_major.font_color,
                                        labelsize=self.tick_labels_major.font_size,
                                        top=self.ticks_major_x2.on,
                                        length=self.ticks_major.size[0],
                                        width=self.ticks_major.size[1],
                                        direction=self.ticks_major.direction,
                                        )
                    axes[1].tick_params(which='minor',
                                        pad=self.ws_ticks_ax*2,
                                        colors=self.ticks_minor.color.get(0),
                                        labelcolor=self.tick_labels_minor.font_color,
                                        labelsize=self.tick_labels_minor.font_size,
                                        top=self.ticks_minor_x2.on,
                                        length=self.ticks_minor._size[0],
                                        width=self.ticks_minor._size[1],
                                        direction=self.ticks_minor.direction,
                                        )

            tp = mpl_get_ticks(axes[ia], True, True)

            # Set custom tick increment
            redo = True
            xinc = getattr(self, 'ticks_major_x%s' % lab).increment
            if not skipx and xinc is not None:
                axes[ia].set_xticks(
                    np.arange(tp['x']['min'] + xinc - tp['x']['min'] % xinc,
                              tp['x']['max'], xinc))
                redo = True
            yinc = getattr(self, 'ticks_major_y%s' % lab).increment
            if not skipy and yinc is not None:
                axes[ia].set_yticks(
                    np.arange(tp['y']['min'] + yinc - tp['y']['min'] % yinc,
                              tp['y']['max'], yinc))
                redo = True
            if redo:
                tp = mpl_get_ticks(axes[ia], True, True)

            # Force ticks
            if self.separate_ticks:
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    mplp.setp(axes[ia].get_xticklabels(), visible=True)
                    mplp.setp(axes[ia].get_yticklabels(), visible=True)
                else:
                    if self.axes.twin_x and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)
                        axes[ia].yaxis.set_tick_params(which='both', labelright=True)
                    elif self.axes.twin_y and ia == 1:
                        axes[ia].xaxis.set_tick_params(which='both', labeltop=True)
                        axes[ia].yaxis.set_tick_params(which='both', labelleft=True)
                    else:
                        axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)
                        axes[ia].yaxis.set_tick_params(which='both', labelleft=True)
            if self.nwrap > 0 and (ic + (ir + 1) * self.ncol + 1) > self.nwrap or \
                    (ir < self.nrow - 1 and not self.axes.visible[ir + 1, ic]):
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    mplp.setp(axes[ia].get_xticklabels()[1:], visible=True)
                elif self.axes.twin_y and ia == 1:
                    axes[ia].yaxis.set_tick_params(which='both', labeltop=True)
                else:
                    axes[ia].xaxis.set_tick_params(which='both', labelbottom=True)
            if not self.separate_ticks and not self.axes.visible[ir, ic - 1]:
                if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                    mplp.setp(axes[ia].get_yticklabels(), visible=True)
                elif self.axes.twin_x and ia == 1:
                    axes[ia].yaxis.set_tick_params(which='both', labelright=True)
                else:
                    axes[ia].yaxis.set_tick_params(which='both', labelleft=True)
            elif not self.separate_ticks and (ic != self.ncol - 1 and \
                    (ic + ir * self.ncol + 1) != self.nwrap) and \
                    self.axes.twin_x and ia == 1:
                mplp.setp(axes[ia].get_yticklabels(), visible=False)
            if not self.separate_ticks and ir != 0 and self.axes.twin_y and ia == 1:
                mplp.setp(axes[ia].get_xticklabels(), visible=False)

            # Major rotation
            if getattr(self, 'tick_labels_major_x%s' % lab).on:
                ticks_font = \
                    font_manager.FontProperties(family=getattr(self, 'tick_labels_major_x%s' % lab).font,
                                                size=getattr(self, 'tick_labels_major_x%s' % lab).font_size,
                                                style=getattr(self, 'tick_labels_major_x%s' % lab).font_style,
                                                weight=getattr(self, 'tick_labels_major_x%s' % lab).font_weight)
                for text in axes[ia].get_xticklabels():
                    if getattr(self, 'tick_labels_major_x%s' % lab).rotation != 0:
                        text.set_rotation(getattr(self, 'tick_labels_major_x%s' % lab).rotation)
                    text.set_fontproperties(ticks_font)
                    text.set_bbox(dict(edgecolor=getattr(self, 'tick_labels_major_x%s' % lab).edge_color.get(0),
                                       facecolor=getattr(self, 'tick_labels_major_x%s' % lab).fill_color.get(0),
                                       linewidth=getattr(self, 'tick_labels_major_x%s' % lab).edge_width))
            if getattr(self, 'tick_labels_major_y%s' % lab).on:
                ticks_font = \
                    font_manager.FontProperties(family=getattr(self, 'tick_labels_major_y%s' % lab).font,
                                                size=getattr(self, 'tick_labels_major_y%s' % lab).font_size,
                                                style=getattr(self, 'tick_labels_major_y%s' % lab).font_style,
                                                weight=getattr(self, 'tick_labels_major_y%s' % lab).font_weight)
                for text in axes[ia].get_yticklabels():
                    if getattr(self, 'tick_labels_major_y%s' % lab).rotation != 0:
                        text.set_rotation(getattr(self, 'tick_labels_major_y%s' % lab).rotation)
                    text.set_fontproperties(ticks_font)
                    text.set_bbox(dict(edgecolor=getattr(self, 'tick_labels_major_y%s' % lab).edge_color.get(0),
                                       facecolor=getattr(self, 'tick_labels_major_y%s' % lab).fill_color.get(0),
                                       linewidth=getattr(self, 'tick_labels_major_y%s' % lab).edge_width))

            # Tick label shorthand
            tlmajx = getattr(self, 'tick_labels_major_x%s' % lab)
            tlmajy = getattr(self, 'tick_labels_major_y%s' % lab)

            # Turn off major tick labels
            if not tlmajx.on:
                axes[ia].set_xticklabels([''])

            if not tlmajy.on:
                axes[ia].set_yticklabels([''])

            # Check for overlapping major tick labels
            lims = {}
            valid_maj = {}
            buf = 3
            if self.tick_cleanup and tlmajx.on:
                # Get rid of out of range labels
                for idx in range(0, tp['x']['first']):
                    tp['x']['label_text'][idx] = ''
                for idx in range(tp['x']['last'] + 1, len(tp['x']['label_text'])):
                    tp['x']['label_text'][idx] = ''
                for idx in range(0, tp['y']['first']):
                    tp['y']['label_text'][idx] = ''
                for idx in range(tp['y']['last'] + 1, len(tp['y']['label_text'])):
                    tp['y']['label_text'][idx] = ''

                # Get the position of the first major x tick
                xcx = get_tick_position(axes[ia], tp, 'x', 'first', ia)
                xfx = get_tick_position(axes[ia], tp, 'x', 'last', ia)
                xc = [xcx, -tlmajx.size[1]/2-self.ws_ticks_ax + ia*self.axes.size[1]]
                lim = axes[ia].get_xlim()
                valid_x = [f for f in tp['x']['ticks']
                           if f >= lim[0] and f <= lim[1]]

                # Get spacings
                if len(tp['x']['ticks']) > 2:
                    delx = self.axes.size[0]/(len(tp['x']['ticks'])-2)
                else:
                    delx = self.axes.size[0] - tlmajx.size[0]
                x2x = []
                xw, xh = tlmajx.size

                # Calculate x-only overlaps
                if not skipx:
                    for ix in range(0, len(tp['x']['ticks']) - 1):
                        x2x += [utl.rectangle_overlap([xw+2*buf, xh+2*buf, [delx*ix,0]],
                                                        [xw+2*buf, xh+2*buf, [delx*(ix+1), 0]])]
                    if any(x2x) and ((self.axes.share_x and ir==0 and ic==0) \
                            or not self.axes.share_x) \
                            and tp['x']['first'] != -999 and tp['x']['last'] != -999:
                        for i in range(tp['x']['first'] + 1, tp['x']['last'] + 1, 2):
                            tp['x']['label_text'][i] = ''

                    # overlapping labels between row, col, and wrap plots
                    if tp['x']['last'] != -999:
                        if self.nwrap > 0 and self.nwrap < self.nrow * self.ncol:
                            if xcx - xw/2 + 2 < 0:
                                tp['x']['label_text'][tp['x']['first']] = ''

                        last_x = tp['x']['labels'][tp['x']['last']][1]
                        if getattr(self, 'axes%s' % lab).scale not in LOG_ALLX:
                            last_x_pos = (last_x - tp['x']['min']) / \
                                            (tp['x']['max'] - tp['x']['min'])
                        else:
                            last_x_pos = (np.log10(last_x) - np.log10(tp['x']['min'])) / \
                                            (np.log10(tp['x']['max']) - np.log10(tp['x']['min']))
                        last_x_px = (1-last_x_pos)*self.axes.size[0]
                        if self.ncol > 1 and \
                                xw / 2 - xcx > last_x_px + self.ws_col - \
                                                self.ws_tick_tick_minimum and \
                                ic < self.ncol - 1 and \
                                tp['x']['label_text'][tp['x']['first']] != '':
                            tp['x']['label_text'][tp['x']['last']] = ''

            if self.tick_cleanup and tlmajy.on:
                # Get the position of the first and last major y tick
                ycy = get_tick_position(axes[ia], tp, 'y', 'first', ia)
                yfy = get_tick_position(axes[ia], tp, 'y', 'last', ia)
                yc = [-tlmajy.size[0]/2-self.ws_ticks_ax, ycy]
                yf = [-tlmajy.size[0]/2-self.ws_ticks_ax, yfy]
                xlim = axes[ia].get_xlim()
                if xlim[0] > xlim[1]:
                    yyc = yc
                    yc = yf
                    yf = yyc
                lim = axes[ia].get_ylim()
                valid_y = [f for f in tp['y']['ticks']
                           if f >= lim[0] and f <= lim[1]]

                # Get spacings
                if len(tp['y']['ticks']) > 2:
                    dely = self.axes.size[1]/(len(tp['y']['ticks'])-2)
                else:
                    dely = self.axes.size[1] - tlmajy.size[1]
                y2y = []
                yw, yh = tlmajy.size

                # Calculate y-only overlaps
                if not skipy:
                    for iy in range(0, len(tp['y']['ticks']) - 1):
                        y2y += [utl.rectangle_overlap([yw+2*buf, yh+2*buf, [0,dely*iy]],
                                                    [yw+2*buf, yh+2*buf, [0,dely*(iy+1)]])]
                    if any(y2y) and ((self.axes.share_y and ir==0 and ic==0) \
                            or not self.axes.share_y) \
                            and tp['y']['first'] != -999 and tp['y']['last'] != -999:
                        for i in range(tp['y']['first'], tp['y']['last'] + 1, 2):
                            tp['y']['label_text'][i] = ''

                    # overlapping labels between row, col, and wrap plots
                    if tp['y']['last'] != -999:
                        last_y = tp['y']['labels'][tp['y']['last']][1]
                        if getattr(self, 'axes%s' % lab).scale not in LOG_ALLY:
                            last_y_pos = (last_y - tp['y']['min']) / \
                                        (tp['y']['max']-tp['y']['min'])
                        else:
                            last_y_pos = (np.log10(last_y) - np.log10(tp['y']['min'])) / \
                                        (np.log10(tp['y']['max'])-np.log10(tp['y']['min']))
                        last_y_px = (1-last_y_pos)*self.axes.size[1]
                        # there is a discrepancy here compared with x (yh?)
                        if self.nrow > 1 and \
                                yh > last_y_px + self.ws_col - self.ws_tick_tick_minimum and \
                                ir < self.nrow - 1 and self.nwrap == 0:
                            tp['y']['label_text'][tp['y']['last']] = ''

            if self.tick_cleanup and tlmajx.on and tlmajy.on:
                # Calculate overlaps
                x0y0 = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yc])
                x0yf = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yf])

                # x and y at the origin
                if x0y0 and lim[0] < lim[1]:  # and tp['y']['first']==0:  not sure about this
                    tp['y']['label_text'][tp['y']['first']] = ''
                elif x0y0:
                    tp['y']['label_text'][tp['y']['last']] = ''
                if x0yf and self.axes.twin_y:
                    tp['y']['label_text'][tp['y']['last']] = ''

                # overlapping last y and first x between row, col, and wraps
                if self.nrow > 1 and ir < self.nrow-1:
                    x2y = utl.rectangle_overlap([xw, xh, xc],
                                                [yw, yh, [yc[0], yc[1]-self.ws_row]])

            if self.tick_cleanup and tlmajx.on and not skipx:
                axes[ia].set_xticklabels(tp['x']['label_text'])

            if self.tick_cleanup and tlmajy.on and not skipy:
                axes[ia].set_yticklabels(tp['y']['label_text'])

            # Turn on minor tick labels
            ax = ['x', 'y']
            sides = {}
            if LooseVersion(mpl.__version__) < LooseVersion('2.2'):
                sides['x'] = {'labelbottom': 'off'}
                sides['x2'] = {'labeltop': 'off'}
                sides['y'] = {'labelleft': 'off'}
                sides['y2'] = {'labelright': 'off'}
            else:
                sides['x'] = {'labelbottom': False}
                sides['x2'] = {'labeltop': False}
                sides['y'] = {'labelleft': False}
                sides['y2'] = {'labelright': False}

            tlminon = False  # "tick label min"
            for axx in ax:
                axl = '%s%s' % (axx, lab)
                tlmin = getattr(self, 'ticks_minor_%s' % axl)

                if ia == 1 and axx == 'x' and self.axes.twin_x:
                    continue

                if ia == 1 and axx == 'y' and self.axes.twin_y:
                    continue

                if getattr(self, 'ticks_minor_%s' % axl).number is not None:
                    num_minor = getattr(self, 'ticks_minor_%s' % axl).number
                    if getattr(self, 'axes%s' % lab).scale not in (LOG_ALLX if axx == 'x' else LOG_ALLY):
                        loc = None
                        loc = AutoMinorLocator(num_minor+1)
                        getattr(axes[ia], '%saxis' % axx).set_minor_locator(loc)

                if not self.separate_ticks and axl == 'x' and ir != self.nrow - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'y2' and ic != self.ncol - 1 and self.nwrap == 0 or \
                        not self.separate_ticks and axl == 'x2' and ir != 0 or \
                        not self.separate_ticks and axl == 'y' and ic != 0 or \
                        not self.separate_ticks and axl == 'y2' and ic != self.ncol - 1 and (ic + ir * self.ncol + 1) != self.nwrap:
                    axes[ia].tick_params(which='minor', **sides[axl])

                elif tlmin.on:
                    if not getattr(self, 'tick_labels_minor_%s' % axl).on:
                        continue
                    elif 'x' in axx and skipx:
                        continue
                    elif 'y' in axx and skipy:
                        continue
                    else:
                        tlminon = True

                    # Determine the minimum number of decimals needed to display the minor ticks
                    tp = mpl_get_ticks(axes[ia],
                                      getattr(self, 'ticks_major_x%s' % lab).on,
                                      getattr(self, 'ticks_major_y%s' % lab).on)
                    m0 = len(tp[axx]['ticks'])
                    lim = getattr(axes[ia], 'get_%slim' % axx)()
                    vals = [f for f in tp[axx]['ticks'] if f > lim[0]]
                    label_vals = [f for f in tp[axx]['label_vals'] if f > lim[0]]
                    inc = label_vals[1] - label_vals[0]
                    minor_ticks = [f[1] for f in tp[axx]['labels']][m0:]

                    # Remove any major tick labels from the list of minor ticks
                    dups = []
                    dups_idx = []
                    for imt, mt in enumerate(minor_ticks):
                        for majt in tp[axx]['ticks']:
                            if math.isclose(mt, majt):
                                dups += [mt]
                                dups_idx += [m0 + imt]
                    minor_ticks2 = [f for f in minor_ticks if f not in dups]
                    number = len([f for f in minor_ticks2 if f > vals[0] and f < vals[1]]) + 1
                    decimals = utl.get_decimals(inc/number)

                    # Check for minor ticks below the first major tick for log axes
                    if getattr(self, 'axes%s' % lab).scale in (LOGX if axx == 'x' else LOGY):
                        extra_minor = [f for f in minor_ticks
                                       if f < label_vals[0] and f > lim[0]]
                        if len(extra_minor) > 0:
                            decimals += 1

                    # Set the tick decimal format
                    getattr(axes[ia], '%saxis' % axx).set_minor_formatter(
                        ticker.FormatStrFormatter('%%.%sf' % (decimals)))

                    # Text formatting
                    tlminlab = getattr(self, 'tick_labels_minor_%s' % axl)
                    ticks_font_minor = \
                        font_manager.FontProperties(family=tlminlab.font,
                                                    size=tlminlab.font_size,
                                                    style=tlminlab.font_style,
                                                    weight=tlminlab.font_weight)
                    for text in getattr(axes[ia], 'get_%sminorticklabels' % axx)():
                        if tlminlab.rotation != 0:
                            text.set_rotation(getattr(self, 'tick_labels_major_%s' % axx).rotation)
                        text.set_fontproperties(ticks_font_minor)
                        text.set_bbox(dict(edgecolor=tlminlab.edge_color.get(0),
                                        facecolor=tlminlab.fill_color.get(0),
                                        linewidth=tlminlab.edge_width))

                    if tlminlab.rotation != 0:
                        for text in getattr(axes[ia], 'get_%sminorticklabels' % axx)():
                            text.set_rotation(tlminlab.rotation)

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

                        # Check overlap with major tick labels
                        majw = getattr(self, 'tick_labels_major_%s' % axl).size[wh]/2
                        delmaj = self.axes.size[wh]
                        lim = getattr(axes[ia], 'get_%slim' % axx)()
                        ticks = [f for f in tp[axx]['ticks'] if f >= lim[0] and f <= lim[1]]
                        if len(ticks) > 0:
                            if axx == 'x':
                                if xfx != xcx:
                                    delmaj = (xfx-xcx)/(len(valid_x)-1)
                                else:
                                    delmaj = xfx
                            else:
                                if yfy != ycy:
                                    delmaj = (yfy-ycy)/(len(valid_y)-1)
                                else:
                                    delmaj = yfy

                            labels = tp[axx]['label_text'][len(tp[axx]['ticks']):]
                            delmin = delmaj/number
                            wipe = []

                            for i in range(0, number - 1):
                                if getattr(self, 'axes%s' % lab).scale in \
                                        (LOGX if axx == 'x' else LOGY):
                                    pos = np.log10(i + 2) * delmaj
                                else:
                                    pos = delmaj / number * (i + 1)
                                if majw > pos - tlminlab.size[wh]/2 - buf or \
                                        delmaj - majw < pos + tlminlab.size[wh]/2 + buf:
                                    wipe += [i]
                            offset = len([f for f in minor_ticks if f < ticks[0]])

                            # There is a weird bug where a tick can be both major and minor; need to remove
                            # dups = [i+m0 for i, f in enumerate(minor_ticks)
                            #         if f in tp[axx]['ticks']]
                            if len(dups_idx) == 0:
                                dups_idx = [-1, len(tp[axx]['label_text'])]
                            if dups_idx[0] != -1:
                                dups_idx = [-1] + dups_idx
                            if dups_idx[len(dups_idx)-1] != len(dups_idx):
                                dups_idx = dups_idx + [len(tp[axx]['label_text'])]
                            temp = []
                            for j in range(1, len(dups_idx)):
                                temp += tp[axx]['label_text'][dups_idx[j-1]+1:dups_idx[j]]

                            # Disable ticks above first major tick
                            for i, text in enumerate(tp[axx]['label_text']):
                                # above first major tick
                                if i in wipe:
                                    vals = temp[m0+i+offset::number-1]
                                    temp[m0+i+offset::number-1] = ['']*len(vals)
                            # Disable ticks below first major tick
                            rwipe = [number - 1 - f for f in wipe]
                            rticks = list(reversed(temp[m0:m0+offset]))
                            for i, tick in enumerate(rticks):
                                if i + 1 in rwipe:
                                    rticks[i] = ''
                            temp[m0:m0+offset] = list(reversed(rticks))

                            # Put back in duplicates
                            for jj in dups_idx[1:]:
                                temp.insert(jj, '')
                            tp[axx]['label_text'] = temp

                        # Check for overlap of first minor with opposite major axis (need to account for twin_x?)
                        if len(tp['x']['ticks']) > 0:
                            minor = tp['x']['label_vals'][m0:]
                            lim = getattr(axes[ia], 'get_%slim' % axx)()
                            first = [i for i, f in enumerate(minor) if f > lim[0]][0]
                            xwmin, xhmin = tlmajx.size
                            ywmin, yhmin = tlmajy.size
                            xcxmin = get_tick_position(axes[ia], tp, axx, first+m0, ia)
                            xcmin = [xcxmin, -tlmin.size[1]/2-self.ws_ticks_ax + ia*self.axes.size[1]]
                            x0miny0 = utl.rectangle_overlap([xwmin+2*buf, xhmin+2*buf, xcmin],
                                                            [yw+2*buf, yh+2*buf, yc])
                            if x0miny0:
                                tp[axx]['label_text'][m0+first] = ''

                        # Check minor to minor overlap
                        if axx == 'x' and self.axes.scale in LOGX or \
                                axx == 'y' and self.axes.scale in LOGY:
                            for itick, t0 in enumerate(tp[axx]['label_text'][m0+1:-1]):
                                if t0 == '':
                                    continue
                                t0 = float(t0)
                                if t0 == 0:
                                    continue
                                t0 = np.log10(t0) % 1
                                t1 = tp[axx]['label_text'][m0+itick+2]
                                if t1 == '':
                                    continue
                                t1 = np.log10(float(t1)) % 1
                                delmin = (t1-t0) * delmaj
                                if tlminlab.size[wh] + buf > delmin:
                                    if tp[axx]['label_text'][m0+itick+1] != '':
                                        tp[axx]['label_text'][m0+itick+2] = ''

                        elif tlminlab.size[wh] + buf > delmin:
                            for itick, tick in enumerate(tp[axx]['label_text'][m0+1:]):
                                if tp[axx]['label_text'][m0+itick] != '':
                                    tp[axx]['label_text'][m0+itick+1] = ''

                        # Set the labels
                        for itick, tick in enumerate(tp[axx]['label_text'][m0:]):
                            if tick == '':
                                continue
                            tp[axx]['label_text'][m0 + itick] = \
                                str(decimal.Decimal(tick).normalize())
                        getattr(axes[ia], 'set_%sticklabels' % axx) \
                            (tp[axx]['label_text'][m0:], minor=True)

    def set_colormap(self, data, **kwargs):
        """
        Replace the color list with discrete values from a colormap

        Args:
            data (Data object)
        """

        if not self.cmap or self.plot_func in ['plot_contour', 'plot_heatmap']:
            return

        try:
            # Conver the color map into discrete colors
            cmap = mplp.get_cmap(self.cmap)
            color_list = []
            if data.legend_vals is None or len(data.legend_vals) == 0:
                if self.axes.twin_x or self.axes.twin_y:
                    maxx = 2
                else:
                    maxx = 1
            else:
                maxx = len(data.legend_vals)
            for i in range(0, maxx):
                color_list += \
                    [mplc_to_hex(cmap((i+1)/(maxx+1)), False)]

            # Reset colors
            if self.legend.column is None:
                if self.axes.twin_x and 'label_y_font_color' not in kwargs.keys():
                    self.label_y.font_color = color_list[0]
                if self.axes.twin_x and 'label_y2_font_color' not in kwargs.keys():
                    self.label_y2.font_color = color_list[1]
                if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                    self.label_x.font_color = color_list[0]
                if self.axes.twin_y and 'label_x_font_color' not in kwargs.keys():
                    self.label_x2.font_color = color_list[1]

            self.lines.color = copy.copy(color_list)
            self.lines.color_alpha('color', 'alpha')
            self.markers.edge_color = copy.copy(color_list)
            self.markers.color_alpha('edge_color', 'edge_alpha')
            self.markers.fill_color = copy.copy(color_list)
            self.markers.color_alpha('fill_color', 'fill_alpha')

        except:
            print('Could not find a colormap called "%s". '
                  'Using default colors...' % self.cmap)

    def set_figure_title(self):
        """
        Add a figure title
        """

        if self.title.on:
            self.get_title_position()
            self.add_label(0, 0, self.title.text, offset=True,
                           **self.make_kwargs(self.title))

    def set_scientific(self, ax, idx=0):
        """
        Turn off scientific notation

        Args:
            ax: axis to adjust

        Returns:
            updated axise
        """

        if idx == 0:
            lab = ''
        else:
            lab = '2'

        # Select scientific notation unless specified
        tp = mpl_get_ticks(ax)
        bestx, besty = False, False

        if self.tick_labels_major_x.sci == 'best' and len(tp['x']['ticks']) > 0:
            xrange = tp['x']['ticks'][-1] - tp['x']['ticks'][0]
            nonzero = tp['x']['ticks'][tp['x']['ticks'] != 0]
            xthresh = np.any(np.abs(nonzero) <= self.auto_tick_threshold[0]) or \
                    np.any(np.abs(nonzero) >= self.auto_tick_threshold[1])
            if xrange <= self.auto_tick_threshold[0] or \
            xrange >= self.auto_tick_threshold[1] or xthresh:
                tick_labels_major_x_sci = True
            else:
                tick_labels_major_x_sci = False
            bestx = True
        elif self.tick_labels_major_x.sci == 'best':
            tick_labels_major_x_sci = False
        else:
            tick_labels_major_x_sci = self.tick_labels_major_x.sci

        if self.tick_labels_major_y.sci == 'best' and len(tp['y']['ticks']) > 0:
            yrange = tp['y']['ticks'][-1] - tp['y']['ticks'][0]
            nonzero = tp['y']['ticks'][tp['y']['ticks'] != 0]
            ythresh = np.any(np.abs(nonzero) <= self.auto_tick_threshold[0]) or \
                    np.any(np.abs(nonzero) >= self.auto_tick_threshold[1])
            if yrange <= self.auto_tick_threshold[0] or \
            yrange >= self.auto_tick_threshold[1] or ythresh:
                tick_labels_major_y_sci = True
            else:
                tick_labels_major_y_sci = False
            besty = True
        elif self.tick_labels_major_y.sci == 'best':
            tick_labels_major_y_sci = False
        else:
            tick_labels_major_y_sci = self.tick_labels_major_y.sci

        # Set labels
        logx = getattr(self, 'axes%s' % lab).scale in LOGX + SYMLOGX + LOGITX
        if self.plot_func in ['plot_hist'] and self.hist.horizontal == True and \
                self.hist.kde == False:
            ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        elif not tick_labels_major_x_sci \
                and self.plot_func not in ['plot_box', 'plot_heatmap'] \
                and not logx:
            try:
                ax.get_xaxis().get_major_formatter().set_scientific(False)
            except:
                pass

        elif not tick_labels_major_x_sci \
                and self.plot_func not in ['plot_box', 'plot_heatmap']:
            try:
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
                tp = mpl_get_ticks(ax)
                for itick, tick in enumerate(tp['x']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['x']['label_text'][itick] = \
                            '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_xticklabels(tp['x']['label_text'])
            except:
                pass
        elif (bestx and not logx \
                or not bestx and tick_labels_major_x_sci and logx) \
                and self.plot_func not in ['plot_box', 'plot_heatmap']:
            xlim = ax.get_xlim()
            max_dec = 0
            for itick, tick in enumerate(tp['x']['ticks']):
                if tick != 0:
                    power = np.ceil(-np.log10(tick))
                    if np.isnan(power) or tick < xlim[0] or tick > xlim[1]:
                        continue
                    dec = utl.get_decimals(tick*10**power)
                    max_dec = max(max_dec, dec)
            dec = '%%.%se' % max_dec
            ax.get_xaxis().set_major_formatter(ticker.FormatStrFormatter(dec))

        logy = getattr(self, 'axes%s' % lab).scale in LOGY + SYMLOGY + LOGITY
        if self.plot_func in ['plot_hist'] and self.hist.horizontal == False and \
                self.hist.kde == False:
            ax.get_yaxis().set_major_locator(MaxNLocator(integer=True))
        elif not tick_labels_major_y_sci \
                and self.plot_func not in ['plot_heatmap'] \
                and not logy:
            try:
                ax.get_yaxis().get_major_formatter().set_scientific(False)
            except:
                pass

        elif not tick_labels_major_y_sci \
                and self.plot_func not in ['plot_heatmap']:
            try:
                ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
                tp = mpl_get_ticks(ax)
                for itick, tick in enumerate(tp['y']['ticks']):
                    if tick < 1 and tick > 0:
                        digits = -np.log10(tick)
                    else:
                        digits = 0
                    tp['y']['label_text'][itick] = \
                            '{0:.{digits}f}'.format(tick, digits=int(digits))
                ax.set_yticklabels(tp['y']['label_text'])
            except:
                pass
        elif (besty and not logy \
                or not besty and tick_labels_major_y_sci and logy) \
                and self.plot_func not in ['plot_heatmap']:
            ylim = ax.get_ylim()
            max_dec = 0
            for itick, tick in enumerate(tp['y']['ticks']):
                if tick != 0:
                    power = np.ceil(-np.log10(tick))
                    if np.isnan(power) or tick < ylim[0] or tick > ylim[1]:
                        continue
                    dec = utl.get_decimals(tick*10**power)
                    max_dec = max(max_dec, dec)
            dec = '%%.%se' % max_dec
            ax.get_yaxis().set_major_formatter(ticker.FormatStrFormatter(dec))

        return ax
