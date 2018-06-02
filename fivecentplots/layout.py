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
#warnings.filterwarnings("ignore")  # do it with kwargs?
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


def mpl_get_ticks(ax):
    """
    Divine a bunch of tick and label parameters for mpl layouts

    Args:
        ax (mpl.axes)

    Returns:
        dict of x and y ax tick parameters

    """

    tp = {}
    xy = ['x', 'y']

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
    def __init__(self, **kwargs):
        """
        Generic layout properties class

        Args:
            **kwargs: styling, spacing kwargs

        """

        # Reload default file
        self.fcpp, color_list, marker_list = utl.reload_defaults()

        # Figure
        self.fig = Element(dpi=utl.kwget(kwargs, self.fcpp, 'dpi', 100),
                           size=[0, 0],
                           size_px=[0, 0])

        # Color list
        if 'line_color' in kwargs.keys():
            color_list = kwargs['line_color']
        elif not color_list:
            color_list = DEFAULT_COLORS

        # Axis
        self.ax = ['x', 'y', 'x2', 'y2']
        self.axes = Element(size=utl.kwget(kwargs, self.fcpp,
                                       'ax_size', [400, 400]),
                            edge_color=utl.kwget(kwargs, self.fcpp,
                                             'ax_edge_color', '#aaaaaa'),
                            fill_color=utl.kwget(kwargs, self.fcpp,
                                             'ax_fill_color', '#eaeaea'),
                            fill_alpha=utl.kwget(kwargs, self.fcpp,
                                             'ax_fill_alpha', 1),
                            primary=True,
                            scale=kwargs.get('ax_scale', None),
                            sci_x=utl.kwget(kwargs, self.fcpp, 'sci_x', False),
                            sci_y=utl.kwget(kwargs, self.fcpp, 'sci_y', False),
                            share_x=kwargs.get('share_x', True),
                            share_y=kwargs.get('share_y', True),
                            share_z=kwargs.get('share_z', True),
                            share_x2=kwargs.get('share_x2', True),  #placeholders
                            share_y2=kwargs.get('share_y2', True),
                            share_col = kwargs.get('share_col', False),
                            share_row = kwargs.get('share_row', False),
                            twinx=kwargs.get('twinx', False),
                            twiny=kwargs.get('twiny', False),
                            xmin=kwargs.get('xmin', None),
                            xmax=kwargs.get('xmax', None),
                            ymin=kwargs.get('ymin', None),
                            ymax=kwargs.get('ymax', None),
                            )
        if self.axes.scale:
            self.axes.scale = self.axes.scale.lower()
        if self.axes.share_row or self.axes.share_col:
            self.axes.share_x = False
            self.axes.share_y = False

        twinned = kwargs.get('twinx', False) or kwargs.get('twiny', False)
        self.axes2 = Element(on=True if twinned else False,
                             edge_color=utl.kwget(kwargs, self.fcpp,
                                             'ax_edge_color', '#aaaaaa'),
                             fill_color=utl.kwget(kwargs, self.fcpp,
                                             'ax_fill_color', '#eaeaea'),
                             primary=False,
                             scale=kwargs.get('ax2_scale', None),
                             xmin=kwargs.get('x2min', None),
                             xmax=kwargs.get('x2max', None),
                             ymin=kwargs.get('y2min', None),
                             ymax=kwargs.get('y2max', None),
                             )
        if self.axes2.scale:
            self.axes2.scale = self.axes2.scale.lower()

        # Axes labels
        label = Element(edge_color=utl.kwget(kwargs, self.fcpp,
                                             'label_edge_color', '#ffffff'),
                        fill_color=utl.kwget(kwargs, self.fcpp,
                                             'label_fill_color', '#ffffff'),
                        fill_alpha=utl.kwget(kwargs, self.fcpp,
                                             'label_fill_alpha', 1),
                        font=utl.kwget(kwargs, self.fcpp, 'label_font',
                                       'sans-serif'),
                        font_color=utl.kwget(kwargs, self.fcpp,
                                             'label_font_color', '#000000'),
                        font_size=utl.kwget(kwargs, self.fcpp,
                                        'label_font_size', 14),
                        font_style=utl.kwget(kwargs, self.fcpp,
                                             'label_font_style', 'italic'),
                        font_weight=utl.kwget(kwargs, self.fcpp,
                                              'label_font_weight','bold'),
                        )
        labels = ['x', 'x2', 'y', 'y2', 'z']
        rotations = [0, 0, 90, 270, 0]
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
        if not self.axes.twiny:
            self.label_x2.on = False
        if not self.axes.twinx:
            self.label_y2.on = False

        # Twinned label colors
        if self.axes.twinx and 'label_y_font_color' not in kwargs.keys():
            self.label_y.font_color = color_list[0]
        if self.axes.twinx and 'label_y2_font_color' not in kwargs.keys():
            self.label_y2.font_color = color_list[1]
        if self.axes.twiny and 'label_x_font_color' not in kwargs.keys():
            self.label_x.font_color = color_list[0]
        if self.axes.twiny and 'label_x_font_color' not in kwargs.keys():
            self.label_x2.font_color = color_list[1]

        # Figure title
        title = kwargs.get('title', None)
        self.title = Element(on=True if title else False,
                             text=title if title is not None else None,
                             size=utl.kwget(kwargs, self.fcpp, 'title_h', 40),
                             edge_color=utl.kwget(kwargs, self.fcpp,
                                              'title_edge_color', '#ffffff'),
                             fill_color=utl.kwget(kwargs, self.fcpp,
                                              'title_fill_color', '#ffffff'),
                             fill_alpha=utl.kwget(kwargs, self.fcpp,
                                              'title_fill_alpha', 1),
                             font=utl.kwget(kwargs, self.fcpp, 'title_font',
                                            'sans-serif'),
                             font_color=utl.kwget(kwargs, self.fcpp,
                                              'title_font_color', '#333333'),
                             font_size=utl.kwget(kwargs, self.fcpp,
                                             'title_font_size', 18),
                             font_style=utl.kwget(kwargs, self.fcpp,
                                              'title_font_style', 'normal'),
                             font_weight=utl.kwget(kwargs, self.fcpp,
                                               'title_font_weight', 'bold'),
                             rotation=0,
                             )
        if type(self.title.size) is not list:
            self.title.size = [self.axes.size[0], self.title.size]

        # Ticks
        if 'ticks' in kwargs.keys() and 'ticks_major' not in kwargs.keys():
            kwargs['ticks_major'] = kwargs['ticks']
        tick_labels = kwargs.get('tick_labels', True)
        ticks_length = utl.kwget(kwargs, self.fcpp, 'ticks_length', 6)
        ticks_width = utl.kwget(kwargs, self.fcpp, 'ticks_width', 2.5)
        self.ticks_major = Element(on=utl.kwget(kwargs, self.fcpp,
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
                                                   ticks_width)]
                                   )
        kwargs = self.from_list(self.ticks_major,
                                ['color', 'increment', 'padding'],
                                'ticks_major', kwargs)
        for ia, ax in enumerate(self.ax):
            setattr(self, 'ticks_major_%s' %ax,
                    Element(on=utl.kwget(kwargs, self.fcpp,
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
            Element(on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_major', tick_labels),
                    font=utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_major_font', 'sans-serif'),
                    font_color=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_major_font_color',
                                         '#000000'),
                    font_size=utl.kwget(kwargs, self.fcpp,
                                        'tick_labels_major_font_size', 13),
                    font_style=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_major_font_style',
                                         'normal'),
                    font_weight=utl.kwget(kwargs, self.fcpp,
                                          'tick_labels_major_font_weight',
                                          'normal'),
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_major_padding', 4),
                    rotation=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_major_rotation', 0),
                   )
        kwargs = self.from_list(self.tick_labels_major,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_major', kwargs)
        for ax in self.ax:
            setattr(self, 'tick_labels_major_%s' %ax,
                    Element(
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_major_%s' % ax,
                                     self.tick_labels_major.on),
                        font=utl.kwget(kwargs, self.fcpp,
                                       'tick_labels_major_font_%s' % ax,
                                       self.tick_labels_major.font),
                        font_color=utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_font_color',
                                             self.tick_labels_major.font_color),
                        font_size=utl.kwget(kwargs, self.fcpp,
                                            'tick_labels_major_font_size_%s' % ax,
                                            self.tick_labels_major.font_size),
                        font_style=utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_major_font_style_%s' % ax,
                                             self.tick_labels_major.font_style),
                        font_weight=utl.kwget(kwargs, self.fcpp,
                                              'tick_labels_major_font_weight_%s' % ax,
                                              self.tick_labels_major.font_weight),
                        padding=utl.kwget(kwargs, self.fcpp,
                                          'tick_labels_major_padding_%s' % ax,
                                          self.tick_labels_major.padding),
                        rotation=utl.kwget(kwargs, self.fcpp,
                                           'tick_labels_major_rotation_%s' % ax,
                                           self.tick_labels_major.rotation),
                        size=[0, 0],
                        ))

        if 'ticks' in kwargs.keys() and 'ticks_minor' not in kwargs.keys():
            kwargs['ticks_minor'] = kwargs['ticks']
        self.ticks_minor = Element(on=utl.kwget(kwargs, self.fcpp,
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
                                                   ticks_width*0.7)]
                                   )
        kwargs = self.from_list(self.ticks_minor,
                                ['color', 'number', 'padding'],
                                'ticks_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'ticks_minor_%s' % ax,
                    Element(
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
                        size=self.ticks_minor.size,
                        ))

        if 'tick_labels' in kwargs.keys() and 'tick_labels_minor' not in kwargs.keys():
            kwargs['tick_labels_minor'] = kwargs['tick_labels']
        self.tick_labels_minor = \
            Element(on=utl.kwget(kwargs, self.fcpp,
                                 'tick_labels_minor', False),
                    font=utl.kwget(kwargs, self.fcpp,
                                   'tick_labels_minor_font', 'sans-serif'),
                    font_color=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_minor_font_color',
                                         '#000000'),
                    font_size=utl.kwget(kwargs, self.fcpp,
                                        'tick_labels_minor_font_size', 10),
                    font_style=utl.kwget(kwargs, self.fcpp,
                                         'tick_labels_minor_font_style',
                                         'normal'),
                    font_weight=utl.kwget(kwargs, self.fcpp,
                                          'tick_labels_minor_font_weight',
                                          'normal'),
                    padding=utl.kwget(kwargs, self.fcpp,
                                      'tick_labels_minor_padding', 3),
                    rotation=utl.kwget(kwargs, self.fcpp,
                                       'tick_labels_minor_rotation', 0)
                                   )
        kwargs = self.from_list(self.tick_labels_minor,
                                ['font', 'font_color', 'font_size',
                                 'font_style', 'font_weight', 'padding',
                                 'rotation'], 'tick_labels_minor', kwargs)
        for ax in self.ax:
            setattr(self, 'tick_labels_minor_%s' %ax,
                    Element(
                        on=utl.kwget(kwargs, self.fcpp,
                                     'tick_labels_minor_%s' % ax,
                                     self.tick_labels_minor.on),
                        font=utl.kwget(kwargs, self.fcpp,
                                       'tick_labels_minor_%s_font' % ax,
                                       self.tick_labels_minor.font),
                        font_color=utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_%s_font_color' % ax,
                                             self.tick_labels_minor.font_color),
                        font_size=utl.kwget(kwargs, self.fcpp,
                                            'tick_labels_minor_%s_font_size' % ax,
                                            self.tick_labels_minor.font_size),
                        font_style=utl.kwget(kwargs, self.fcpp,
                                             'tick_labels_minor_%s_font_style' % ax,
                                             self.tick_labels_minor.font_style),
                        font_weight=utl.kwget(kwargs, self.fcpp,
                                              'tick_labels_minor_%s_font_weight' % ax,
                                              self.tick_labels_minor.font_weight),
                        padding=utl.kwget(kwargs, self.fcpp,
                                          'tick_labels_minor_%s_padding' % ax,
                                          self.tick_labels_minor.padding),
                        rotation=utl.kwget(kwargs, self.fcpp,
                                           'tick_labels_minor_%s_rotation' % ax,
                                           self.tick_labels_minor.rotation),
                        size=[0, 0],
                        ))

        # Boxplot labels
        self.box_item_label = Element(on=kwargs.get('box_labels_on', True),
                                     size=utl.kwget(kwargs, self.fcpp,
                                                    'box_label_size', 25),
                                     edge_color=utl.kwget(kwargs, self.fcpp,
                                                          'box_item_label_edge_color',
                                                          '#aaaaaa'),
                                     fill_color=utl.kwget(kwargs, self.fcpp,
                                                          'box_item_label_fill_color',
                                                          '#ffffff'),
                                     fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                          'box_item_label_fill_alpha',
                                                          1),
                                     font=utl.kwget(kwargs, self.fcpp,
                                                    'box_item_label_font',
                                                    'sans-serif'),
                                     font_color=utl.kwget(kwargs, self.fcpp,
                                                          'box_item_label_font_color',
                                                          '#666666'),
                                     font_size=utl.kwget(kwargs, self.fcpp,
                                                         'box_item_label_font_size',
                                                         13),
                                     font_style=utl.kwget(kwargs, self.fcpp,
                                                          'box_item_label_font_style',
                                                          'normal'),
                                     font_weight=utl.kwget(kwargs, self.fcpp,
                                                           'box_item_label_font_weight',
                                                           'normal'),
                                     )
        self.box_group_label = Element(on=kwargs.get('box_labels_on', True),
                                      edge_color=utl.kwget(kwargs, self.fcpp,
                                                           'box_group_label_edge_color',
                                                           '#ffffff'),
                                      fill_color=utl.kwget(kwargs, self.fcpp,
                                                           'box_group_label_fill_color',
                                                           '#ffffff'),
                                      fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                          'box_group_label_fill_alpha',
                                                          1),
                                      font=utl.kwget(kwargs, self.fcpp,
                                                     'box_group_label_font',
                                                     'sans-serif'),
                                      font_color=utl.kwget(kwargs, self.fcpp,
                                                           'box_group_label_font_color',
                                                           '#666666'),
                                      font_size=utl.kwget(kwargs, self.fcpp,
                                                          'box_group_label_font_size',
                                                          12),
                                      font_style=utl.kwget(kwargs, self.fcpp,
                                                           'box_group_label_font_style',
                                                           'normal'),
                                      font_weight=utl.kwget(kwargs, self.fcpp,
                                                            'box_group_label_font_weight',
                                                            'normal'),
                                      )

        # Boxplot boxes '#4b72b0', '#c34e52'
        self.box_boxes = Element(edge_color=utl.kwget(kwargs, self.fcpp,
                                                     'box_boxes_edge_color',
                                                     '#4b72b0'),
                                fill_color=utl.kwget(kwargs, self.fcpp,
                                                     'box_boxes_fill_color',
                                                     '#ffffff'),
                                fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'box_boxes_fill_alpha', 1),
                                line_color=utl.kwget(kwargs, self.fcpp,
                                                     'box_boxes_median_line_color',
                                                     '#c34e52'),
                                range_line_color=utl.kwget(kwargs, self.fcpp,
                                                           'box_range_line_color',
                                                           '#dddddd'),
                                range_line_style=utl.kwget(kwargs, self.fcpp,
                                                           'box_range_line_style',
                                                           '--'),
                                range_line_width=utl.kwget(kwargs, self.fcpp,
                                                           'box_range_line_width',
                                                           1),
                                dividers=utl.kwget(kwargs, self.fcpp,
                                                   'box_dividers', True),
                                divider_line_color=utl.kwget(kwargs, self.fcpp,
                                                             'box_dividers',
                                                             True),
                                divider_line_style=utl.kwget(kwargs, self.fcpp,
                                    'box_divider_line_style', '#bbbbbb'),
                                divider_line_weight=utl.kwget(kwargs, self.fcpp,
                                    'box_divider_line_weight', 1.5),
                                connect_means=utl.kwget(kwargs, self.fcpp,
                                                        'box_connect_means',
                                                        False),
                                jitter=utl.kwget(kwargs, self.fcpp,
                                                 'box_jitter', False)
                                )

        # Legend
        self.legend = DF_Element(on=True if (kwargs.get('legend') and
                                 kwargs.get('legend_on', True)) else False,
                                 column=kwargs.get('legend', None),
                                 edge_color=utl.kwget(kwargs, self.fcpp,
                                                      'legend_edge_color',
                                                      '#ffffff'),
                                 fill_color=utl.kwget(kwargs, self.fcpp,
                                                      'legend_fill_color',
                                                      '#ffffff'),
                                 fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                      'legend_fill_alpha', 1),
                                 font=utl.kwget(kwargs, self.fcpp,
                                                'legend_font', 'sans-serif'),
                                 font_color=utl.kwget(kwargs, self.fcpp,
                                                      'legend_font_color',
                                                      '#000000'),
                                 font_size=utl.kwget(kwargs, self.fcpp,
                                                     'legend_font_size', 12),
                                 font_style=utl.kwget(kwargs, self.fcpp,
                                                      'legend_font_style', 'normal'),
                                 font_weight=utl.kwget(kwargs, self.fcpp,
                                                       'legend_font_weight',
                                                       'normal'),
                                 location=LEGEND_LOCATION[utl.kwget(kwargs,
                                          self.fcpp, 'legend_location', 0)],
                                 marker_size=utl.kwget(kwargs, self.fcpp,
                                                       'legend_marker_size',
                                                       None),
                                 num_points=utl.kwget(kwargs, self.fcpp,
                                                      'legend_points', 1),
                                 overflow=0,
                                 text=kwargs.get('legend_title', kwargs.get('legend', '')),
                                 values={} if not kwargs.get('legend') else {'NaN': None},
                                 )
        # Color bar
        self.cbar = Element(on=kwargs.get('cbar', False),
                            size=[utl.kwget(kwargs, self.fcpp,
                                           'cbar_width', 30),
                                  self.axes.size[1]],
                            )

        # Line fit
        self.line_fit = Element(on=kwargs.get('line_fit', False),
                                line_color=utl.kwget(kwargs, self.fcpp,
                                                     'line_fit_color',
                                                     '#000000'),
                                line_sytle=utl.kwget(kwargs, self.fcpp,
                                                     'line_fit_style', 1),
                                line_width=utl.kwget(kwargs, self.fcpp,
                                                     'line_fit_width', 1),
                                font=utl.kwget(kwargs, self.fcpp,
                                               'line_fit_font', 'sans-serif'),
                                font_color=utl.kwget(kwargs, self.fcpp,
                                                     'line_fit_font_color',
                                                     '#000000'),
                                font_size=utl.kwget(kwargs, self.fcpp,
                                                    'line_fit_font_size',
                                                    12),
                                font_style=utl.kwget(kwargs, self.fcpp,
                                                     'line_fit_font_style',
                                                     'normal'),
                                font_weight=utl.kwget(kwargs, self.fcpp,
                                                      'line_fit_font_weight',
                                                      'normal'),
                                )
        # Lines
        line_colors = RepeatedList(color_list, 'line_colors')
        self.lines = Element(on=kwargs.get('lines', True),
                             color=line_colors,
                             line_sytle=utl.kwget(kwargs, self.fcpp,
                                                  'line_style', None),
                             line_width=utl.kwget(kwargs, self.fcpp,
                                                  'line_width', 1),
                             values=[],
                             )

        # Markers/points
        if 'marker_type' in kwargs.keys():
            marker_list = kwargs['marker_type']
        elif not marker_list:
            marker_list = DEFAULT_MARKERS
        markers = RepeatedList(marker_list, 'markers')
        marker_edge_color = utl.kwget(kwargs, self.fcpp, 'marker_edge_color', None)
        marker_edge_color = DEFAULT_COLORS if marker_edge_color is None else marker_edge_color
        marker_edge_color = RepeatedList(marker_edge_color, 'marker_edge_color')
        marker_fill_color = utl.kwget(kwargs, self.fcpp, 'marker_fill_color', None)
        marker_fill_color = DEFAULT_COLORS if marker_fill_color is None else marker_fill_color
        marker_fill_color = RepeatedList(marker_fill_color, 'marker_fill_color')

        self.markers = Element(on=utl.kwget(kwargs, self.fcpp,
                                            'markers', True),
                               size=utl.kwget(kwargs, self.fcpp,
                                              'marker_size', 7),
                               type=markers,
                               filled=utl.kwget(kwargs, self.fcpp,
                                                'marker_fill', False),
                               edge_color=marker_edge_color,
                               edge_width=utl.kwget(kwargs, self.fcpp,
                                                    'marker_edge_width',
                                                     1.5),
                               fill_color=marker_fill_color,
                               fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                    'marker_fill_alpha', 1),

                               )

        # Axhlines/axvlines
        axlines = ['ax_hlines', 'ax_vlines', 'yline',
                   'ax2_hlines', 'ax2_vlines', 'y2line']
        # Todo: list
        for axline in axlines:
            val = kwargs.get(axline, False)
            vals = utl.validate_list(val)
            setattr(self, axline,
                    Element(on=True if val in kwargs.keys() else False,
                            values=utl.validate_list(vals[0]),
                            line_color=vals[1] if len(vals) > 1 else None,
                            line_style=vals[2] if len(vals) > 2 else None,
                            line_width=vals[3] if len(vals) > 3 else None,
                           ))

        # Gridlines
        self.grid_major = Element(on=kwargs.get('grid_major', True),
                                  color=utl.kwget(kwargs, self.fcpp,
                                                  'grid_major_color',
                                                  '#ffffff'),
                                  line_style=utl.kwget(kwargs, self.fcpp,
                                                       'grid_major_line_style',
                                                       '-'),
                                  line_width=utl.kwget(kwargs, self.fcpp,
                                                       'grid_major_line_width',
                                                       1.3)
                                  )
        for ax in ['x', 'y']:
            # secondary axes cannot get the grid
            setattr(self, 'grid_major_%s' %ax,
                    Element(on=kwargs.get('grid_major_%s' % ax,
                                          self.grid_major.on),
                            color=utl.kwget(kwargs, self.fcpp,
                                            'grid_major_color_%s' % ax,
                                            self.grid_major.color),
                            line_style=utl.kwget(kwargs, self.fcpp,
                                                 'grid_major_line_style_%s' % ax,
                                                 self.grid_major.line_style),
                            line_width=utl.kwget(kwargs, self.fcpp,
                                                 'grid_major_line_width_%s' % ax,
                                                 self.grid_major.line_width)
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

        self.grid_minor = Element(on=kwargs.get('grid_minor', False),
                                  color=utl.kwget(kwargs, self.fcpp,
                                                  'grid_minor_color',
                                                  '#ffffff'),
                                  line_style=utl.kwget(kwargs, self.fcpp,
                                                       'grid_minor_line_style',
                                                       '-'),
                                  line_width=utl.kwget(kwargs, self.fcpp,
                                                       'grid_minor_line_width',
                                                       0.5)
                                  )
        if self.grid_minor.on and \
                ('ticks' not in kwargs.keys() or kwargs['ticks'] != False) and \
                ('ticks_minor' not in kwargs.keys() or kwargs['ticks_minor'] != False):
            self.ticks_minor.on = True
        for ax in ['x', 'y']:
            # secondary axes cannot get the grid
            setattr(self, 'grid_minor_%s' %ax,
                    Element(on=kwargs.get('grid_minor_%s' % ax,
                                          self.grid_minor.on),
                            color=utl.kwget(kwargs, self.fcpp,
                                            'grid_minor_color_%s' % ax,
                                            self.grid_minor.color),
                            line_style=utl.kwget(kwargs, self.fcpp,
                                                 'grid_minor_line_style_%s' % ax,
                                                 self.grid_minor.line_style),
                            line_width=utl.kwget(kwargs, self.fcpp,
                                                 'grid_minor_line_width_%s' % ax,
                                                 self.grid_minor.line_width)
                            ))

        # Row column label
        rc_label = DF_Element(on=True,
                              size=utl.kwget(kwargs, self.fcpp,
                                             'rc_label_size', 30),
                              edge_color=utl.kwget(kwargs, self.fcpp,
                                                   'rc_label_edge_color',
                                                   '#8c8c8c'),
                              fill_color=utl.kwget(kwargs, self.fcpp,
                                                   'rc_label_fill_color',
                                                   '#8c8c8c'),
                              fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                   'rc_label_fill_alpha', 1),
                              font=utl.kwget(kwargs, self.fcpp, 'rc_label_font',
                                             'sans-serif'),
                              font_color=utl.kwget(kwargs, self.fcpp,
                                                   'rc_label_font_color',
                                                   '#ffffff'),
                              font_size=utl.kwget(kwargs, self.fcpp,
                                                  'rc_label_font_size', 16),
                              font_style=utl.kwget(kwargs, self.fcpp,
                                                   'rc_label_font_style',
                                                   'normal'),
                              font_weight=utl.kwget(kwargs, self.fcpp,
                                                    'rc_label_font_weight',
                                                    'bold'),
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
        self.label_col.fill_color = utl.kwget(kwargs, self.fcpp,
                                              'col_label_fill_color',
                                              rc_label.fill_color)
        self.label_col.font_color = utl.kwget(kwargs, self.fcpp,
                                              'col_label_font_color',
                                              rc_label.font_color)
        # Wrap label
        self.label_wrap = DF_Element(on=utl.kwget(kwargs, self.fcpp,
                                                  'wrap_label_on', True)
                                                   if kwargs.get('wrap') else False,
                                     column=kwargs.get('wrap'),
                                     size=[self.axes.size[0],
                                           utl.kwget(kwargs, self.fcpp,
                                           'wrap_label_size', rc_label.size)],
                                     edge_color=utl.kwget(kwargs, self.fcpp,
                                                          'wrap_label_edge_color',
                                                          rc_label.edge_color),
                                     fill_color=utl.kwget(kwargs, self.fcpp,
                                                          'wrap_label_fill_color',
                                                          rc_label.fill_color),
                                     fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                          rc_label.fill_alpha,
                                                          1),
                                     font=utl.kwget(kwargs, self.fcpp,
                                                    'wrap_label_font',
                                                    rc_label.font),
                                     font_color=utl.kwget(kwargs, self.fcpp,
                                                          'wrap_label_font_color',
                                                          rc_label.font_color),
                                     font_size=utl.kwget(kwargs, self.fcpp,
                                                         'wrap_label_font_size',
                                                         rc_label.font_size),
                                     font_style=utl.kwget(kwargs, self.fcpp,
                                                          'wrap_label_font_style',
                                                          rc_label.font_style),
                                     font_weight=utl.kwget(kwargs, self.fcpp,
                                                           'wrap_label_font_weight',
                                                           rc_label.font_weight),
                                     text_size=None,
                                     )
        if type(self.label_wrap.size) is not list:
            self.label_wrap.size = [self.label_wrap.size, self.axes.size[1]]

        self.title_wrap = Element(on=utl.kwget(kwargs, self.fcpp,
                                               'wrap_title_on', True)
                                               if kwargs.get('wrap') else False,
                                  size=utl.kwget(kwargs, self.fcpp,
                                                 'wrap_title_size',
                                                 rc_label.size),
                                  edge_color=utl.kwget(kwargs, self.fcpp,
                                                       'wrap_title_edge_color',
                                                       rc_label.edge_color),
                                  fill_color=utl.kwget(kwargs, self.fcpp,
                                                       'wrap_title_fill_color',
                                                       rc_label.fill_color),
                                  fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                       'wrap_title_fill_alpha',
                                                       rc_label.fill_alpha),
                                  font=utl.kwget(kwargs, self.fcpp,
                                                 'wrap_title_font',
                                                 rc_label.font),
                                  font_color=utl.kwget(kwargs, self.fcpp,
                                                       'wrap_title_font_color',
                                                       rc_label.font_color),
                                  font_size=utl.kwget(kwargs, self.fcpp,
                                                      'wrap_title_font_size',
                                                      rc_label.font_size),
                                  font_style=utl.kwget(kwargs, self.fcpp,
                                                       'wrap_title_font_style',
                                                       rc_label.font_style),
                                  font_weight=utl.kwget(kwargs, self.fcpp,
                                                        'wrap_title_font_weight',
                                                        rc_label.font_weight),
                                  text=kwargs.get('wrap_title', None),
                                  )
        if type(self.title_wrap.size) is not list:
            self.title_wrap.size = [self.axes.size[0], self.title_wrap.size]
        if self.title_wrap.on and not self.title_wrap.text:
            self.title_wrap.text = ' | '.join(self.label_wrap.values)

        # Confidence interval
        self.conf_int = Element(on=kwargs.get('conf_int', None),
                                fill_color=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_color',
                                                     None),
                                fill_alpha=utl.kwget(kwargs, self.fcpp,
                                                     'conf_int_fill_alpha',
                                                     0.2),
                                )

        # Extras
        self.cmap = utl.kwget(kwargs, self.fcpp, 'cmap', None)
        self.inline = utl.kwget(kwargs, self.fcpp, 'inline', None)
        self.separate_labels = utl.kwget(kwargs, self.fcpp,
                                         'separate_labels', False)
        self.separate_ticks = utl.kwget(kwargs, self.fcpp,
                                        'separate_ticks', False)
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
        self.ws_cbar_ax = utl.kwget(kwargs, self.fcpp, 'ws_cbar_ax', 10)

        # rc labels
        ws_rc_label = utl.kwget(kwargs, self.fcpp, 'ws_rc_label', 10)
        self.ws_col_label = utl.kwget(kwargs, self.fcpp,
                                  'ws_col_label', ws_rc_label)
        self.ws_row_label = utl.kwget(kwargs, self.fcpp,
                                  'ws_row_label', ws_rc_label)
        self.ws_col = utl.kwget(kwargs, self.fcpp, 'ws_col', 30)
        self.ws_row = utl.kwget(kwargs, self.fcpp, 'ws_row', 30)

        # figure
        self.ws_label_fig = utl.kwget(kwargs, self.fcpp, 'ws_label_fig', 10)
        self.ws_leg_fig = utl.kwget(kwargs, self.fcpp, 'ws_leg_fig', 10)
        self.ws_fig_title = utl.kwget(kwargs, self.fcpp, 'ws_fig_title', 10)

        # axes
        self.ws_label_tick = utl.kwget(kwargs, self.fcpp, 'ws_label_tick', 10)
        self.ws_leg_ax = utl.kwget(kwargs, self.fcpp, 'ws_leg_ax', 20)
        self.ws_ticks_ax = utl.kwget(kwargs, self.fcpp, 'ws_ticks_ax', 3)
        self.ws_title_ax = utl.kwget(kwargs, self.fcpp, 'ws_title_ax', 20)
        self.ws_wrap_title = utl.kwget(kwargs, self.fcpp, 'wrap_label_ws', 0)

        # ticks
        self.ws_tick_tick_minimum = utl.kwget(kwargs, self.fcpp,
                                              'ws_tick_tick_minimum', 10)

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
                if 'twinx' in kwargs.keys() and kwargs['twinx']:
                    kwargs['%s_%s_y2' % (name, attr)] = getattr(base, attr)[2]
                if 'twiny' in kwargs.keys() and kwargs['twiny']:
                    kwargs['%s_%s_x2' % (name, attr)] = getattr(base, attr)[2]

        return kwargs

    def make_figure(self, data, **kwargs):
        pass

    def make_kwargs(self, element):
        kwargs = {}
        kwargs['position'] = element.position
        kwargs['size'] = element.size
        kwargs['rotation'] = element.rotation
        kwargs['fill_color'] = element.fill_color
        kwargs['edge_color'] = element.edge_color
        kwargs['font'] = element.font
        kwargs['font_weight'] = element.font_weight
        kwargs['font_style'] = element.font_style
        kwargs['font_color'] = element.font_color
        kwargs['font_size'] = element.font_size

        return kwargs

    def set_labels(self, data, **kwargs):
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

            if lab == 'x' and self.axes.twiny:
                getattr(self, 'label_x').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_x2').text = \
                    lab_text if lab_text is not None else dd[1]
            elif lab == 'y' and self.axes.twinx:
                getattr(self, 'label_y').text = \
                    lab_text if lab_text is not None else dd[0]
                getattr(self, 'label_y2').text = \
                    lab_text if lab_text is not None else dd[1]
            else:
                if type(dd) is list:
                    getattr(self, 'label_%s' % lab).text = \
                        lab_text if lab_text is not None else ' + '.join(dd)
                else:
                    getattr(self, 'label_%s' % lab).text = dd
                if lab != 'z' and hasattr(self, 'label_%s2' % lab):
                    getattr(self, 'label_%s2' % lab).text = \
                        lab_text2 if lab_text2 is not None else ' + '.join(dd)

            if hasattr(data, '%s_vals' % lab):
                getattr(self, 'label_%s' % lab).values = \
                    getattr(data, '%s_vals' % lab)


class Element:
    def __init__(self, **kwargs):
        """
        Element style container
        """

        self.column = None
        self._on = kwargs.get('on', True) # visbile or not
        self.obj = None  # plot object reference
        self.position = [0, 0, 0, 0]  # left, right, top, bottom
        self._size = kwargs.get('size', [0, 0])  # width, height
        self.size_orig = None
        self._text = kwargs.get('text', True)  # text label
        self.text_orig = None  # text label
        self.rotation = 0

        # colors
        self.edge_color = None
        self.fill_color = None
        self.fill_alpha = None
        self.line_color = None

        # styles
        self.line_style = None
        self.line_width = None

        # fonts
        self.font = None
        self.font_color = None
        self.font_size = None
        self.font_style = None
        self.font_weight = None

        for k, v in kwargs.items():
            try:
                setattr(self, k, v)
            except:
                pass

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
            self._size = self.size_orig
            self._text = self.text_orig

    @property
    def size(self):

        if self.on:
            return self._size
        else:
            return [0, 0]

    @size.setter
    def size(self, value):

        if self.size_orig is None and value is not None:
            self.size_orig = value

        self._size = value

    @property
    def text(self):

        return self._text

    @text.setter
    def text(self, value):

        if self.text_orig is None and value is not None:
            self.text_orig = value

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
    def __init__(self, **kwargs):
        Element.__init__(self, **kwargs)

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
            self._size = self.size_orig
            self._text = self.text_orig


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

        if type(self.values) is not list and len(self.values) < 1:
            raise(ValueError, 'RepeatedList for "%s" must contain an actual '
                              'list with more at least one element')

    def get(self, idx):

        # can we make this a next??

        return self.values[idx % len(self.values)]


class LayoutBokeh(BaseLayout):
    def __init__(self, **kwargs):
        BaseLayout.__init__(self, **kwargs)


class LayoutMPL(BaseLayout):

    def __init__(self, ptype='plot', **kwargs):
        """
        Layout attributes and methods for matplotlib Figure

        Args:
            **kwargs: input args from user
        """

        mplp.close('all')

        # Inherit the base layout properties
        BaseLayout.__init__(self, **kwargs)

        # Define white space parameters
        self.init_white_space(**kwargs)

        # Initialize other class variables
        self.ptype             = ptype
        self.col_label_height  = 0
        self.row_label_left    = 0
        self.row_label_width   = 0
        self.wrap_title_bottom = 0

        # Weird spacing defaults out of our control
        self.fig_right_border = 6  # extra border on right side that shows up by default
        self.legend_top_offset = 8
        self.legend_border = 3

    def add_cbar(self):
        # Define colorbar position
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(axes[ir, ic])
        size = '%s%%' % (100*kw['cbar_width']/layout.ax_size[0])
        pad = kw['ws_cbar_ax']/100
        cax = divider.append_axes("right", size=size, pad=pad)

        # Add the colorbar and label
        cbar = mplp.colorbar(cc, cax=cax)
        cbar.ax.set_ylabel(r'%s' % kw['cbar_label'], rotation=270,
                           labelpad=kw['label_font_size'],
                           style=kw['label_style'],
                           fontsize=kw['label_font_size'],
                           weight=kw['label_weight'],
                           color=kw['ylabel_color'])

    def add_hvlines(self, ir, ic):
        """
        Add axhlines and axvlines

        Args:
            ir (int): row index
            ic (int): col index
        """

        # Set default line attributes
        for line in ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']:
            line = getattr(self, line)
            if not line.line_color:
                line.line_color = 'k'
            if not line.line_style:
                line.line_style = '-'
            if not line.line_width:
                line.line_width = \
                    1 if not self.lines.line_width else self.lines.line_width

        # Add lines
        if self.ax_hlines.on:
            for val in self.ax_hlines.values:
                self.axes[ir, ic].axhline(val,
                                          color=line.line_color,
                                          linestyle=line.line_style,
                                          linewidth=line.line_width)

        if self.ax_vlines.on:
            for val in self.ax_vlines.values:
                self.axes[ir, ic].axvline(val,
                                          color=line.line_color,
                                          linestyle=line.line_style,
                                          linewidth=line.line_width)

        if self.ax2_hlines.on:
            for val in self.ax2_hlines.values:
                self.axes2[ir, ic].axhline(val,
                                           color=line.line_color,
                                           linestyle=line.line_style,
                                           linewidth=line.line_width)

        if self.ax2_vlines.on:
            for val in self.ax_vlines.values:
                self.axes2[ir, ic].axvline(val,
                                           color=line.line_color,
                                           linestyle=line.line_style,
                                           linewidth=line.line_width)

    def add_label(self, axis, text='', position=None, rotation=0, size=None,
                  fill_color='#ffffff', edge_color='#aaaaaa', font='',
                  font_weight='normal', font_style='normal',
                  font_color='#666666', font_size=14):
        """ Add a label to the plot

        This function can be used for title labels or for group labels applied
        to rows and columns when plotting facet grid style plots.

        Args:
            label (str):  label text
            pos (tuple): label position tuple of form (left, bottom, width, height)
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
                                 facecolor=fill_color, edgecolor=edge_color,
                                 clip_on=False, zorder=-1)

        axis.add_patch(rect)

        # Set slight text offset
        # if rotation == 270:
        #     offsetx = -2/self.axes.size[0]#-font_size/self.axes.size[0]/4
        # else:
        #     offsetx = 0
        # if rotation == 0:
        #     offsety = -2/self.axes.size[1]#-font_size/self.axes.size[1]/4
        # else:
        #     offsety = 0
        # print(offsetx, offsety)
        offsetx, offsety = 0, 0

        # Add the label text
        text = axis.text(position[0]+size[0]/self.axes.size[0]/2+offsetx,
                         position[3]+size[1]/self.axes.size[1]/2+offsety, text,
                         transform=axis.transAxes, horizontalalignment='center',
                         verticalalignment='center', rotation=rotation,
                         color=font_color, fontname=font,
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
                                        title=self.legend.text,
                                        bbox_to_anchor=(self.legend.position[1],
                                                        self.legend.position[2]),
                                        numpoints=self.legend.num_points,
                                        prop=fontp)

            else:
                self.legend.obj = \
                    self.fig.obj.legend(lines, keys, loc=self.legend.position,
                                        title = self.legend.text,
                                        numpoints=self.legend.num_points,
                                        prop=fontp)

            for text in self.legend.obj.get_texts():
                text.set_color(self.legend.font_color)

            self.legend.obj.get_frame().set_facecolor(self.legend.fill_color)
            self.legend.obj.get_frame().set_edgecolor(self.legend.edge_color)

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
        mplp.ioff()
        fig = mpl.pyplot.figure(dpi=self.fig.dpi)
        ax = fig.add_subplot(111)
        ax2, ax3 = None, None
        if self.axes.twinx:
            ax2 = ax.twinx()
        if self.axes.twiny:
            ax3 = ax.twiny()
        if self.axes.scale in ['logy', 'semilogy']:
            plottype = 'semilogy'
        elif self.axes.scale in ['logx', 'semilogx']:
            plottype = 'semilogx'
        elif self.axes.scale in ['loglog']:
            plottype = 'loglog'
        else:
            plottype = 'plot'
        plotter = getattr(ax, plottype)
        if self.axes.twinx:
            plotter2 = getattr(ax2, plottype)
        if self.axes.twiny:
            plotter3 = getattr(ax3, plottype)

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
                                     top=self.ticks_major_x2.on \
                                         if self.axes.twiny
                                         else self.ticks_major_x.on,
                                     bottom=self.ticks_major_x.on,
                                     right=self.ticks_major_y2.on \
                                           if self.axes.twinx
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
                                         if self.axes.twiny
                                         else self.ticks_minor_x.on,
                                     bottom=self.ticks_minor_x.on,
                                     right=self.ticks_minor_y2.on \
                                           if self.axes.twinx
                                           else self.ticks_minor_y.on,
                                     left=self.ticks_minor_y.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )

        # Define label variables
        xticksmaj, x2ticksmaj, yticksmaj, y2ticksmaj = [], [], [], []
        xticksmin, x2ticksmin, yticksmin, y2ticksmin = [], [], [], []
        xticklabelsmaj, x2ticklabelsmaj, yticklabelsmaj, y2ticklabelsmaj = \
            [], [], [], []
        xticklabelsmin, x2ticklabelsmin, yticklabelsmin, y2ticklabelsmin = \
            [], [], [], []
        wrap_labels = np.array([[None]*self.ncol]*self.nrow)
        row_labels = np.array([[None]*self.ncol]*self.nrow)
        col_labels = np.array([[None]*self.ncol]*self.nrow)

        for ir, ic, df in data.get_rc_subset(data.df_fig):
            if data.twinx:
                y2ticks = [f[2] for f in axes[1].yaxis.iter_ticks()
                          if f[2] != '']
                y2iter_ticks = [f for f in axes[1].yaxis.iter_ticks()]
                y2ticksmaj += [f[2] for f in y2iter_ticks[0:len(y2ticks)]]
            elif data.twiny:
                x2ticks = [f[2] for f in axes[2].xaxis.iter_ticks()
                           if f[2] != '']
            for xy in zip(data.x, data.y):
                plotter(df[xy[0]], df[xy[1]], 'o-')
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
            xiter_ticks = [f for f in axes[0].xaxis.iter_ticks()]
            yiter_ticks = [f for f in axes[0].yaxis.iter_ticks()]
            xticksmaj += [f[2] for f in xiter_ticks[0:len(xticks)]]
            yticksmaj += [f[2] for f in yiter_ticks[0:len(yticks)]]

            # Minor ticks
            if self.axes.scale in ['logx', 'semilogx', 'loglog'] and \
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

            if self.axes.scale in ['logy', 'semilogy', 'loglog'] and \
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
        if type(data.legend_vals) == pd.DataFrame and len(data.legend_vals) > 0:
            lines = []
            leg_vals = []
            for irow, row in data.legend_vals.iterrows():
                lines += ax.plot([1, 2, 3])
                leg_vals += [row['names']]
            if self.yline.on:
                lines += axes[0].plot([1, 2, 3])
                leg_vals += [self.yline.text]
            leg = mpl.pyplot.legend(lines, leg_vals,
                                    title=self.legend.text,
                                    numpoints=self.legend.num_points,
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
        if self.label_x.text:
            label_x = fig.text(0, 0, r'%s' % self.label_x.text,
                               fontsize=self.label_x.font_size,
                               weight=self.label_x.font_weight,
                               style=self.label_x.font_style,
                               color=self.label_x.font_color,
                               rotation=self.label_x.rotation)

        if self.label_x2.text:
            label_x2 = fig.text(0, 0, r'%s' % self.label_x2.text,
                               fontsize=self.label_x2.font_size,
                               weight=self.label_x2.font_weight,
                               style=self.label_x2.font_style,
                               color=self.label_x2.font_color,
                               rotation=self.label_x2.rotation)

        if self.label_y.text:
            label_y = fig.text(0, 0, r'%s' % self.label_y.text,
                               fontsize=self.label_y.font_size,
                               weight=self.label_y.font_weight,
                               style=self.label_y.font_style,
                               color=self.label_y.font_color,
                               rotation=self.label_y.rotation)

        if self.label_y2.text:
            label_y2 = fig.text(0, 0, r'%s' % self.label_y2.text,
                               fontsize=self.label_y2.font_size,
                               weight=self.label_y2.font_weight,
                               style=self.label_y2.font_style,
                               color=self.label_y2.font_color,
                               rotation=self.label_y2.rotation)

        # what about z text?  this is cbar?


        # Render dummy figure
        mpl.pyplot.draw()
        #mpl.pyplot.savefig(r'C:\data\test.png')

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

        if self.axes.twinx and self.tick_labels_major.on:
            self.ticks_major_y2.size = \
                [np.nanmax([t.get_window_extent().width for t in y2ticklabelsmaj]),
                 np.nanmax([t.get_window_extent().height for t in y2ticklabelsmaj])]
        elif self.axes.twinx and not self.tick_labels_major.on:
            self.ticks_major_y2.size = \
                [np.nanmax([0 for t in y2ticklabelsmaj]),
                 np.nanmax([0 for t in y2ticklabelsmaj])]

        if self.axes.twiny and self.tick_labels_major.on:
            self.ticks_major_x2.size = \
                [np.nanmax([t.get_window_extent().width for t in x2ticklabels]),
                 np.nanmax([t.get_window_extent().height for t in x2ticklabels])]
        elif self.axes.twiny and not self.tick_labels_major.on:
            self.ticks_major_x2.size = \
                [np.nanmax([0 for t in x2ticklabels]),
                 np.nanmax([0 for t in x2ticklabels])]

        self.label_x.size = (label_x.get_window_extent().width,
                             label_x.get_window_extent().height)
        if self.axes.twiny:
            self.label_x2.size = (label_x2.get_window_extent().width,
                                  label_x2.get_window_extent().height)
        self.label_y.size = (label_y.get_window_extent().width,
                             label_y.get_window_extent().height)
        if self.axes.twinx:
            self.label_y2.size = (label_y2.get_window_extent().width,
                                  label_y2.get_window_extent().height)

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

        # Destroy the dummy figure
        mpl.pyplot.close(fig)

    def get_figure_size(self):
        """
        Determine the size of the mpl figure canvas in pixels and inches
        """
        ### NEED TO RESET DEFAULT WS IF NO LEG  --> ws_col_label; ws_row_label?
        # Calculate the canvas size

        if self.separate_labels:
            self.ws_col += self.label_y.size[0]
            self.ws_row += self.label_x.size[1]

        if self.separate_ticks:
            self.ws_col += max(self.tick_labels_major_y.size[0],
                               self.tick_labels_minor_y.size[0])
            self.ws_row += max(self.tick_labels_major_x.size[1],
                               self.tick_labels_minor_x.size[1])

        y2 = self.label_y2.size[0] + 2*self.ws_label_tick + \
             max(self.tick_labels_major_y2.size[0],
                 self.tick_labels_minor_y2.size[0])
        tick_labels_major_x = max(self.tick_labels_major_x.size[1],
                                  self.tick_labels_minor_x.size[1]) + \
                              max(self.tick_labels_major_x2.size[1],
                                  self.tick_labels_minor_x2.size[1])
        tick_labels_major_y = max(self.tick_labels_major_y.size[0],
                                  self.tick_labels_minor_y.size[0])
        ws_leg_ax = max(0, self.ws_leg_ax - y2) if self.legend.text is not None else 0
        ws_leg_fig = self.ws_leg_fig if self.legend.text is not None else 0
        cbar = self.cbar.size[0] + self.ws_cbar_ax if self.cbar.on else 0

        self.fig.size_px[0] = \
            self.ws_label_fig + self.label_y.size[0] + 2*self.ws_label_tick + \
            tick_labels_major_y + self.ws_ticks_ax + \
            self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1) +  \
            ws_leg_ax + self.legend.size[0] + ws_leg_fig + \
            (y2 + self.ws_ticks_ax) * int(self.axes.twinx) + \
            self.label_row.size[0] + self.ws_row_label * self.label_row.on + \
            cbar * self.ncol

        self.fig.size_px[1] = \
            self.ws_fig_title * self.title.on + self.title.size[1] + \
            self.ws_title_ax + self.label_col.size[1] + \
            self.ws_col_label * self.label_col.on + \
            self.axes.size[1]*self.nrow + 2*self.ws_label_tick + \
            self.ws_ticks_ax + self.label_x.size[1] + self.ws_label_fig + \
            tick_labels_major_x + self.title_wrap.size[1] + \
            self.ws_row * (self.nrow - 1)  # leg_overflow?

        fig_only = self.axes.size[1]*self.nrow + (self.ws_ticks_ax +
                   self.label_x.size[1] + self.ws_label_fig +
                   max(self.tick_labels_major_x.size[1],
                       self.tick_labels_minor_x.size[1])) * \
                   (1 + int(self.separate_labels)*self.nrow)
        self.legend.overflow = max(self.legend.size[1]-fig_only, 0)
        self.fig.size[0] = self.fig.size_px[0]/self.fig.dpi
        self.fig.size[1] = \
            (self.fig.size_px[1] + self.legend.overflow)/self.fig.dpi

    def get_legend_position(self):
        """
        Get legend position
        """

        self.legend.position[1] = \
            1 - (self.ws_leg_fig - self.fig_right_border)/self.fig.size_px[0]
        self.legend.position[2] = \
            self.axes.position[2] + self.legend_top_offset/self.fig.size_px[1]

    def get_rc_label_position(self):
        """
        Get option group label positions
        """

        self.label_row.position[0] = \
            (self.axes.size[0] + self.ws_row_label +
             (self.ws_cbar_ax if self.cbar.on else 0) + self.cbar.size[0] +
             self.label_z.size[0])/self.axes.size[0]

        #self.label_row.siz = self.row_label_size/self.ax_w
        self.label_col.position[3] = \
            (self.axes.size[1] + self.ws_col_label)/self.axes.size[1]
        #self.col_label_height = self.col_label_size/self.ax_h
        #self.wrap_title_bottom = (self.ax_h + self.col_labels +
        #                          self.ws_wrap_title)/self.ax_h

    def get_subplots_adjust(self):
        """
        Calculate the subplots_adjust parameters for the axes
            self.axes.position --> [left, right, top, bottom]
        """

        self.axes.position[0] = \
            (self.ws_label_fig + self.label_y.size[0] + 2*self.ws_label_tick + \
             max(self.tick_labels_major_y.size[0],
                 self.tick_labels_minor_y.size[0]) + \
             self.ws_ticks_ax)/self.fig.size_px[0]

        self.axes.position[1] = \
            (self.ws_label_fig + self.label_y.size[0] + 2*self.ws_label_tick + \
             max(self.tick_labels_major_y.size[0],
                 self.tick_labels_minor_y.size[0]) + \
             self.ws_ticks_ax + self.axes.size[0] * self.ncol + \
             self.ws_col * (self.ncol - 1)) / self.fig.size_px[0]

        self.axes.position[2] = \
            1 - (self.ws_fig_title * self.title.on + self.title.size[1] + \
            self.ws_title_ax + self.title_wrap.size[1] + \
            (self.label_col.size[1] + self.ws_col_label) * self.label_col.on) \
            / self.fig.size_px[1]

        self.axes.position[3] = \
            (self.ws_ticks_ax + self.label_x.size[1] + self.ws_label_fig + \
             2*self.ws_label_tick + \
             max(self.tick_labels_major_x.size[1],
                 self.tick_labels_minor_x.size[1])) / self.fig.size_px[1]

    def get_title_position(self):
        """
        Calculate the title position
            self.title.position --> [left, right, top, bottom]
        """

        col_label = (self.label_col.size[1] + \
                     self.ws_col_label * self.label_col.on)
        self.title.position[0] = 0
        self.title.position[3] = 1+(self.ws_title_ax + col_label) \
                                 /self.axes.size[1]
        self.title.position[2] = self.title.position[3] + (self.ws_title_ax +
                                 self.title.size[1])/self.axes.size[1]

    def make_figure(self, data, **kwargs):
        """
        Make the figure and axes objects
        """

        self.nrow = data.nrow
        self.ncol = data.ncol

        self.set_labels(data)
        self.get_element_sizes(data)
        self.get_figure_size()
        self.get_subplots_adjust()
        self.get_rc_label_position()
        self.get_legend_position()

        # Define the subplots
        fig, axes = \
            mplp.subplots(data.nrow, data.ncol,
                          figsize=[self.fig.size[0], self.fig.size[1]],
                          sharex=self.axes.share_x,
                          sharey=self.axes.share_y,
                          dpi=self.fig.dpi,
                          facecolor=self.fig.fill_color,
                          edgecolor=self.fig.edge_color)
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
        if self.axes.twinx:
            for ir in range(0, self.nrow):
                for ic in range(0, self.ncol):
                    self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twinx()
        elif self.axes.twiny:
            for ir in range(0, self.nrow):
                for ic in range(0, self.ncol):
                    self.axes2.obj[ir, ic] = self.axes.obj[ir, ic].twiny()

        # # Row/col/wrap
        # if data.wrap:
        #     self.label_row.on = False
        #     self.label_col.on = self.label_wrap.on
        #
        # else:
        #     self.label_wrap.on = False

        # Set axes label colors

        #self.format_axes()

    def plot_data(self, ir, ic, iline, df, x, y, leg_name, twin):
        """ Plot data

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

        # Select the axes
        if twin:
            ax = self.axes2.obj[ir, ic]
        else:
            ax = self.axes.obj[ir, ic]

        # Make the points
        points = None
        if self.markers.on:
            points = ax.plot(df[x], df[y],
                             marker=format_marker(self.markers.type.get(iline)),
                             markerfacecolor=self.markers.fill_color.get(iline) \
                                             if self.markers.filled else 'none',
                             markeredgecolor=self.markers.edge_color.get(iline),
                             markeredgewidth=self.markers.edge_width,
                             linewidth=0)

        # Make the line
        lines = None
        if self.lines.on:
            lines = ax.plot(df[x], df[y],
                            color=self.lines.color.get(iline),
                            linestyle='-' if self.lines.line_style is None \
                                      else self.lines.line_style,
                            linewidth=self.lines.line_width)

        # Add a reference to the line to self.lines
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
            axes[-1].obj[ir, ic].set_facecolor(axes[-1].fill_color)
        except:
            axes[-1].obj[ir, ic].set_axis_bgcolor(axes[-1].fill_color)
        axes[-1].obj[ir, ic].set_alpha(axes[-1].fill_alpha)
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
                ax.obj[ir, ic].xaxis.grid(b=True, which='major', zorder=3,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.line_style,
                                          linewidth=self.grid_major.line_width)
            else:
                ax.obj[ir, ic].xaxis.grid(b=False, which='major')
            if self.grid_major_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='major', zorder=3,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.line_style,
                                          linewidth=self.grid_major.line_width)
            else:
                ax.obj[ir, ic].yaxis.grid(b=False, which='major')

            # Set minor grid
            if self.grid_minor_x.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='minor', zorder=4,
                                          color=self.grid_minor_x.color,
                                          linestyle=self.grid_minor_x.line_style,
                                          linewidth=self.grid_minor_x.line_width)
            if self.grid_minor_y.on:
                ax.obj[ir, ic].yaxis.grid(b=True, which='minor', zorder=4,
                                          color=self.grid_minor_y.color,
                                          linestyle=self.grid_minor_y.line_style,
                                          linewidth=self.grid_minor_y.line_width)

    def set_axes_labels(self, ir, ic):
        """
        Set the axes labels

        Args:
            ir (int): current row index
            ic (int): current column index
            kw (dict): kwargs dict

        """

        axis = ['x', 'x2', 'y', 'y2']

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
                if ax == 'x' and ir != self.nrow - 1: continue
                if ax == 'x2' and ir != 0: continue
                if ax == 'y' and ic != 0: continue
                if ax == 'y2' and ic != self.ncol - 1: continue

            # Add the label
            getattr(axes, 'set_%slabel' % ax[0])(label.text,
                                                 fontsize=label.font_size,
                                                 weight=label.font_weight,
                                                 style=label.font_style,
                                                 color=label.font_color,
                                                 rotation=label.rotation,
                                                 fontname=label.font,
                                                 labelpad=pad,
                                                 )

        # Add cbars for contour
        if self.cbar.on: #ir, ic??
            self.cbar.obj.ax.set_ylabel(fontsize=self.label_z.font_size,
                                        weight=self.label_z.font_weight,
                                        style=self.label_z.font_style,
                                        color=self.label_z.font_color,
                                        rotation=self.label_z.rotation,
                                        fontname=self.label_z.font,
                                        )

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
                if str(ax.scale).lower() in ['loglog', 'semilogx', 'logx']:
                    ax.obj[ir, ic].set_xscale('log')
                if str(ax.scale).lower() in ['loglog', 'semilogy', 'logy']:
                    ax.obj[ir, ic].set_yscale('log')

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
            elif k == 'zmin' and v is not None:
                self.cbar.obj[ir, ic].set_clim(bottom=v)
            elif k == 'zmax' and v is not None:
                self.cbar.obj[ir, ic].set_clim(top=v)

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
            wrap = self.add_label(ax, self.title_wrap.text,
                                  **self.make_kwargs(self.title_wrap))

        # Row labels
        if ic == self.ncol-1 and self.label_row.on and not self.label_wrap.on:
            if self.label_row.text_size is not None:
                text_size = self.label_row.text_size[ir, ic]
            else:
                text_size = None
            row = self.add_label(ax, '%s=%s' %
                                (self.label_row.text, self.label_row.values[ir]),
                                **self.make_kwargs(self.label_row))

        # Col/wrap labels
        if (ir == 0 or self.label_wrap.on) and self.label_col.on:
            if self.label_row.text_size is not None:
                text_size = self.label_col.text_size[ir, ic]
            else:
                text_size = None
            if self.label_wrap.on:
                text = ' | '.join([str(f) for f in utl.validate_list(
                    self.label_wrap.values[ir*self.ncol + ic])])
                col = self.add_label(ax, text,
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
            if not self.axes.sci_x \
                    and self.ptype != 'boxplot' \
                    and self.axes.scale not in ['logx', 'semilogx', 'loglog'] \
                    and (not self.axes.share_x or ir==0 and ic==0):
                try:
                    axes[ia].get_xaxis().get_major_formatter().set_scientific(False)
                except:
                    print('bah: axes[ia].get_xaxis().get_major_formatter().set_scientific(False)')

            if not self.axes.sci_y \
                    and self.axes.scale not in ['logy', 'semilogy', 'loglog'] \
                    and (not self.axes.share_y or ir==0 and ic==0):
                axes[ia].get_yaxis().get_major_formatter().set_scientific(False)

            # General tick params
            if ia == 0:
                axes[ia].minorticks_on()
                axes[ia].tick_params(axis='both',
                                     which='major',
                                     pad=self.ws_ticks_ax,
                                     colors=self.ticks_major.color,
                                     labelcolor=self.tick_labels_major.font_color,
                                     labelsize=self.tick_labels_major.font_size,
                                     top=False if self.axes.twiny
                                         else self.ticks_major_x.on,
                                     bottom=self.ticks_major_x.on,
                                     right=False if self.axes.twinx
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
                                     top=False if self.axes.twiny
                                         else self.ticks_minor_x.on,
                                     bottom=self.ticks_minor_x.on,
                                     right=False if self.axes.twinx
                                           else self.ticks_minor_y.on,
                                     left=self.ticks_minor_y.on,
                                     length=self.ticks_minor.size[0],
                                     width=self.ticks_minor.size[1],
                                     )
            elif self.axes.twinx:
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
            elif self.axes.twiny:
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

            tp = mpl_get_ticks(axes[ia])

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
                tp = mpl_get_ticks(axes[ia])

            # Force ticks
            if self.separate_ticks:
                mplp.setp(axes[ia].get_xticklabels(), visible=True)
                mplp.setp(axes[ia].get_yticklabels(), visible=True)

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
                xc = [0, -tlmajx.size[1]/2-self.ws_ticks_ax]
                yc = [-tlmajy.size[0]/2-self.ws_ticks_ax, 0]
                buf = 6
                delx = self.axes.size[0]/(len(tp['x']['ticks'])-2)
                dely = self.axes.size[1]/(len(tp['y']['ticks'])-2)
                x2x, y2y = [], []
                xw, xh = tlmajx.size
                yw, yh = tlmajy.size

                x0y0 = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yc])

                for ix in range(0, len(tp['x']['ticks']) - 1):
                    x2x += [utl.rectangle_overlap([xw+2*buf, xh+2*buf, [delx*ix,0]],
                                                  [xw+2*buf, xh+2*buf, [delx*(ix+1), 0]])]
                for iy in range(0, len(tp['y']['ticks']) - 1):
                    y2y += [utl.rectangle_overlap([yw+2*buf, yh+2*buf, [0,dely*iy]],
                                                  [yw+2*buf, yh+2*buf, [0,dely*(iy+1)]])]

                # x and y at the origin
                if x0y0 and tp['y']['first']==0:
                    tp['y']['label_text'][tp['y']['first']] = ''

                # x overlapping x
                if any(x2x) and (not (self.axes.share_x and ir > 0 or ic > 0)) \
                        and tp['x']['first'] != -999 and tp['x']['last'] != -999:
                    for i in range(tp['x']['first'], tp['x']['last'], 2):
                        tp['x']['label_text'][i] = ''

                # y overlapping y
                if any(y2y) and (not (self.axes.share_y and ir > 0 or ic > 0)) \
                        and tp['y']['first'] != -999 and tp['y']['last'] != -999:
                    for i in range(tp['y']['first'], tp['y']['last'], 2):
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
                            ir < self.nrow - 1:
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

                if getattr(self, 'ticks_minor_%s' % axl).number is not None:
                    num_minor = getattr(self, 'ticks_minor_%s' % axl).number
                    if axx=='x' and self.axes.scale in ['logx', 'semilogx', 'loglog']:
                        loc = LogLocator()
                    elif axx=='y' and self.axes.scale in ['logy', 'semilogy', 'loglog']:
                        loc = LogLocator()
                    else:
                        loc = AutoMinorLocator(num_minor+1)
                    getattr(axes[ia], '%saxis' % axx).set_minor_locator(loc)

                if not self.separate_labels and axl == 'x' and ir != self.nrow - 1 or \
                        not self.separate_labels and axl == 'x2' and ir != 0 or \
                        not self.separate_labels and axl == 'y' and ic != 0 or \
                        not self.separate_labels and axl == 'y2' and ic != self.ncol - 1:
                    axes[ia].tick_params(which='minor', **sides[axl])

                elif tlmin.on:
                    # THE DECIMAL THING WONT WORK FOR LOG!

                    tp = mpl_get_ticks(axes[ia])
                    inc = tp[axl]['labels'][1][1] - tp[axl]['labels'][0][1]
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
                    if axx=='y': st()
                    if self.tick_cleanup and tlminon:
                        tp = mpl_get_ticks(axes[ia])  # need to update
                        buf = 1.99
                        if axx == 'x':
                            wh = 0
                        else:
                            wh = 1
                        delmaj = self.axes.size[wh]/(len(tp[axl]['ticks'])-2)
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

    def set_figure_title(self):
        """
        Add a figure title
        """

        if self.title.on:
            self.get_title_position()
            self.add_label(self.axes.obj[0, 0], self.title.text,
                           **self.make_kwargs(self.title))

    def set_scientific(self, ax):
        """
        Turn off scientific notation

        Args:
            ax: axis to adjust

        Returns:
            updated axise
        """

        if not self.axes.sci_x and self.ptype != 'boxplot' \
                and self.axes.scale not in ['logx', 'semilogx', 'loglog']:
            ax.get_xaxis().get_major_formatter().set_scientific(False)
        if not self.axes.sci_y \
                and self.axes.scale not in ['logy', 'semilogy', 'loglog']:
            ax.get_yaxis().get_major_formatter().set_scientific(False)

        return ax



