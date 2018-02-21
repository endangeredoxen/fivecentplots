from . import fcp
import matplotlib as mpl
import matplotlib.pyplot as mplp
import matplotlib.ticker as ticker
import matplotlib.patches as patches
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

        # Axis
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
                            sharex=kwargs.get('sharex', True),
                            sharey=kwargs.get('sharey', True),
                            twinx=kwargs.get('twinx', False),
                            twiny=kwargs.get('twiny', False),
                            xmin=kwargs.get('xmin', None),
                            xmax=kwargs.get('xmax', None),
                            ymin=kwargs.get('ymin', None),
                            ymax=kwargs.get('ymax', None),
                            )

        twinned = kwargs.get('twinx', False) or kwargs.get('twiny', False)
        self.axes2 = Element(on=True if twinned else False,
                             primary=False,
                             scale=kwargs.get('ax2_scale', None),
                             xmin=kwargs.get('x2min', None),
                             xmax=kwargs.get('x2max', None),
                             ymin=kwargs.get('y2min', None),
                             ymax=kwargs.get('y2max', None),
                             )
        
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
            if kwargs.get('%slabel' % lab):
                print('"%slabel" is deprecated. Please use "label_%s" instead'
                      % (lab, lab))
                if kwargs.get('%slabel_color' % lab):
                    kwargs['label_%s' % lab] = kwargs['%slabel_color' % lab]
            setattr(self, 'label_%s' % lab, copy.deepcopy(label))
            getattr(self, 'label_%s' % lab).rotation = rotations[ilab]
            getattr(self, 'label_%s' % lab).font_color = \
                    kwargs.get('label_%s_color' % lab, '#000000')
        if not self.axes.twiny:
            self.label_x2.on = False
        if not self.axes.twinx:
            self.label_y2.on = False

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
                                   increment_x=kwargs.get('ticks_major_increment_x'),
                                   increment_x2=kwargs.get('ticks_major_increment_x2'),
                                   increment_y=kwargs.get('ticks_major_increment_y'),
                                   increment_y2=kwargs.get('ticks_major_increment_y2'),
                                   padding=utl.kwget(kwargs, self.fcpp,
                                                     'ticks_major_min_padding',
                                                     4),
                                   size=[utl.kwget(kwargs, self.fcpp,
                                                   'ticks_major_length',
                                                   ticks_length),
                                         utl.kwget(kwargs, self.fcpp,
                                                   'ticks_major_width',
                                                   ticks_width)]
                                   )
        if self.ticks_major.increment is not None:
            self.ticks_major.increment = \
                utl.validate_list(self.ticks_major.increment)
            self.ticks_major.increment += \
                [None]*(4-len(self.ticks_major_increment))
            self.ticks_major.increment_x = self.ticks_major.increment[0]
            self.ticks_major.increment_x2 = self.ticks_major.increment[1]
            self.ticks_major.increment_y = self.ticks_major.increment[2]
            self.ticks_major.increment_y2 = self.ticks_major.increment[3]

        if 'tick_labels' in kwargs.keys() \
                and 'tick_labels_major' not in kwargs.keys():
            kwargs['tick_labels_major'] = kwargs['tick_labels']
        self.tick_labels_major = \
            Element(on=utl.kwget(kwargs, self.fcpp, 
                                 'tick_labels_major', tick_labels),
                    color=utl.kwget(kwargs, self.fcpp, 
                                    'tick_labels_major_color', '#000000'),
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
                                      'tick_labels_major_min_padding', 4),
                                   )
        self.tick_labels_major.size_x = [0, 0]
        self.tick_labels_major.size_x2 = [0, 0]
        self.tick_labels_major.size_y = [0, 0]
        self.tick_labels_major.size_y2 = [0, 0]

        if 'ticks' in kwargs.keys() and 'ticks_minor' not in kwargs.keys():
            kwargs['ticks_minor'] = kwargs['ticks']
        self.ticks_minor = Element(on=utl.kwget(kwargs, self.fcpp, 
                                                'ticks_minor', False),
                                   color=utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_color', 
                                                   '#ffffff'),
                                   number=utl.kwget(kwargs, self.fcpp,
                                                    'ticks_minor_number',
                                                    None),
                                   padding=utl.kwget(kwargs, self.fcpp,
                                                     'ticks_minor_min_padding',
                                                     4),
                                   size=[utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_length_minor',
                                                   ticks_length*0.8),
                                         utl.kwget(kwargs, self.fcpp,
                                                   'ticks_minor_width',
                                                   ticks_width*0.7)]
                                   )

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
                                      'tick_labels_minor_min_padding', 3),
                                   )
        self.tick_labels_minor.size_x = [0, 0]
        self.tick_labels_minor.size_x2 = [0, 0]
        self.tick_labels_minor.size_y = [0, 0]
        self.tick_labels_minor.size_y2 = [0, 0]

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
        if kwargs.get('leg_groups'):
            kwargs['legend'] = kwargs['leg_groups']
            print('"leg_groups" is deprecated. Please use "leg" instead')
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
                              text=kwargs.get('legend_title', kwargs['legend']),
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
        if 'line_color' in kwargs.keys():
            color_list = kwargs['line_color']
        elif not color_list:
            color_list = DEFAULT_COLORS
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
        if self.grid_major.on and \
                ('ticks' not in kwargs.keys() or kwargs['ticks'] != False) and \
                ('ticks_major' not in kwargs.keys() or kwargs['ticks_major'] != False):
            self.ticks_major.on = True
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
        if not self.axes.sharex or not self.axes.sharey:
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
        self.fig_right_border = 6
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
            plotter = 'semilogy'
        elif self.axes.scale in ['logx', 'semilogx']:
            plotter = 'semilogx'
        elif self.axes.scale in ['loglog']:
            plotter = 'loglog'
        else:
            plotter = 'plot'
        plotter = getattr(ax, plotter)
        if self.axes.twinx:
            plotter2 = getattr(ax2, plotter)
        if self.axes.twiny:
            plotter3 = getattr(ax3, plotter)

        # Set tick and scale properties
        axes = [ax, ax2, ax3]
        for ia, aa in enumerate(axes):
            if aa is None:
                continue
            axes[ia] = self.set_scientific(aa)
            axes[ia].tick_params(axis='both',
                                 which='major',
                                 pad=self.ws_ticks_ax,
                                 labelsize=self.tick_labels_major.font_size,
                                 colors=self.tick_labels_major.font_color,
                                 top=self.ticks_major.on,
                                 bottom=self.ticks_major.on,
                                 right=self.ticks_major.on,
                                 left=self.ticks_major.on,
                                 )
            axes[ia].tick_params(axis='both',
                                 which='minor',
                                 pad=self.ws_ticks_ax,
                                 labelsize=self.tick_labels_minor.font_size,
                                 colors=self.tick_labels_minor.font_color,
                                 top=self.ticks_minor.on,
                                 bottom=self.ticks_minor.on,
                                 right=self.ticks_minor.on,
                                 left=self.ticks_minor.on,
                                 )

        # Define label variables
        xticks, x2ticks, yticks, y2ticks = [], [], [], []
        xticklabels, x2ticklabels, yticklabels, y2ticklabels = [], [], [], []
        wrap_labels = np.array([[None]*self.ncol]*self.nrow)
        row_labels = np.array([[None]*self.ncol]*self.nrow)
        col_labels = np.array([[None]*self.ncol]*self.nrow)

        for ir, ic, df in data.get_rc_subset(data.df_fig):

            if data.twinx:
                st()
            elif data.twiny:
                st()
            else:
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

                xticks += [f[2] for f in axes[0].xaxis.iter_ticks()
                           if f[2] != '']
                yticks += [f[2] for f in axes[0].yaxis.iter_ticks()
                           if f[2] != '']

            ww, rr, cc = self.set_axes_rc_labels(ir, ic, axes[0])
            wrap_labels[ir, ic] = ww
            row_labels[ir, ic] = rr
            col_labels[ir, ic] = cc

        # Make a dummy legend --> move to add_legend???
        if len(data.legend_vals) > 0:
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

        # Write out tick labels
        for ix, xtick in enumerate(xticks):
            xticklabels += [fig.text(ix*20, 20, xtick,
                                     fontsize=self.ticks_major.font_size)]
        for ix, x2tick in enumerate(x2ticks):
            x2ticklabels += [fig.text(ix*20, 20, x2tick,
                                     fontsize=self.ticks_major.font_size)]
        for iy, ytick in enumerate(yticks):
            yticklabels += [fig.text(20, iy*20, ytick,
                                     fontsize=self.ticks_major.font_size)]
        for iy, y2tick in enumerate(y2ticks):
            y2ticklabels += [fig.text(20, iy*20, y2tick,
                                      fontsize=self.ticks_major.font_size)]

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

        # Get actual sizes
        if self.tick_labels_major.on:
            self.tick_labels_major.size_x = \
                [np.nanmax([t.get_window_extent().width for t in xticklabels]),
                 np.nanmax([t.get_window_extent().height for t in xticklabels])]
            self.tick_labels_major.size_y = \
                [np.nanmax([t.get_window_extent().width for t in yticklabels]),
                 np.nanmax([t.get_window_extent().height for t in yticklabels])]
        else:
            self.tick_labels_major.size_x = \
                [np.nanmax([0 for t in xticklabels]),
                 np.nanmax([0 for t in xticklabels])]
            self.tick_labels_major.size_y = \
                [np.nanmax([0 for t in yticklabels]),
                 np.nanmax([0 for t in yticklabels])]
        
        if self.axes.twinx and self.tick_labels_major.on:
            self.ticks_major.size_y2 = \
                [np.nanmax([t.get_window_extent().width for t in y2ticklabels]),
                 np.nanmax([t.get_window_extent().height for t in y2ticklabels])]
        elif self.axes.twinx and not self.tick_labels_major.on:
            self.ticks_major.size_y2 = \
                [np.nanmax([0 for t in y2ticklabels]),
                 np.nanmax([0 for t in y2ticklabels])]
        
        if self.axes.twiny and self.tick_labels_major.on:
            self.ticks_major.size_x2 = \
                [np.nanmax([t.get_window_extent().width for t in x2ticklabels]),
                 np.nanmax([t.get_window_extent().height for t in x2ticklabels])]
        elif self.axes.twiny and not self.tick_labels_major.on:
            self.ticks_major.size_x2 = \
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
            self.ws_col += max(self.tick_labels_major.size_y[0], 
                               self.tick_labels_minor.size_y[0])
            self.ws_row += max(self.tick_labels_major.size_x[1], 
                               self.tick_labels_minor.size_x[1])

        self.fig.size_px[0] = \
            self.ws_label_fig + self.label_y.size[0] + self.ws_label_tick + \
             max(self.tick_labels_major.size_y[0], 
                 self.tick_labels_minor.size_y[0]) + \
             self.ws_ticks_ax + \
            self.axes.size[0] * self.ncol + self.ws_col * (self.ncol - 1) +  \
            (self.ws_leg_ax if self.legend.text is not None else 0) + \
            self.legend.size[0] + \
            (self.ws_leg_fig if self.legend.text is not None else 0) + \
            (self.label_y2.size[0] + self.ws_label_tick + \
             max(self.tick_labels_major.size_y2[0], 
                 self.tick_labels_minor.size_y2[0]) + \
             self.ws_ticks_ax) * int(self.axes.twinx) + \
            self.label_row.size[0] + self.ws_row_label + \
            (self.cbar.size[0] + self.ws_cbar_ax if self.cbar.on else 0) * \
            self.ncol

        self.fig.size_px[1] = \
            self.ws_fig_title * self.title.on + self.title.size[1] + \
            self.ws_title_ax + self.label_col.size[1] + \
            self.ws_col_label * self.label_col.on + \
            self.axes.size[1]*self.nrow + \
            self.ws_ticks_ax + self.label_x.size[1] + self.ws_label_fig + \
            max(self.tick_labels_major.size_x[1], 
                self.tick_labels_minor.size_x[1]) + \
            max(self.tick_labels_major.size_x2[1], 
                self.tick_labels_minor.size_x2[1]) + \
            self.title_wrap.size[1] + self.ws_row * (self.nrow - 1)  # leg_overflow?

        fig_only = self.axes.size[1]*self.nrow + (self.ws_ticks_ax +
                   self.label_x.size[1] + self.ws_label_fig +
                   max(self.tick_labels_major.size_x[1],
                       self.tick_labels_minor.size_x[1])) * \
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
            (self.ws_label_fig + self.label_y.size[0] + self.ws_label_tick + \
             max(self.tick_labels_major.size_y[0],
                 self.tick_labels_minor.size_y[0]) + \
             self.ws_ticks_ax)/self.fig.size_px[0]
        
        self.axes.position[1] = \
            (self.ws_label_fig + self.label_y.size[0] + self.ws_label_tick + \
             max(self.tick_labels_major.size_y[0],
                 self.tick_labels_minor.size_y[0]) + \
             self.ws_ticks_ax + self.axes.size[0] * self.ncol + \
             self.ws_col * (self.ncol - 1)) / self.fig.size_px[0]

        self.axes.position[2] = \
            1 - (self.ws_fig_title * self.title.on + self.title.size[1] + \
            self.ws_title_ax + self.title_wrap.size[1] + \
            (self.label_col.size[1] + self.ws_col_label) * self.label_col.on) \
            / self.fig.size_px[1]

        self.axes.position[3] = \
            (self.ws_ticks_ax + self.label_x.size[1] + self.ws_label_fig + \
             max(self.tick_labels_major.size_x[1],
                 self.tick_labels_minor.size_x[1])) / self.fig.size_px[1]

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
                          sharex=self.axes.sharex,
                          sharey=self.axes.sharey,
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
        self.axes2.obj = np.empty(self.axes.obj.shape)
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

        for ax in axes:
            try:
                ax.obj[ir, ic].set_facecolor(ax.fill_color)
            except:
                ax.obj[ir, ic].set_axis_bgcolor(ax.fill_color)
            ax.obj[ir, ic].set_alpha(ax.fill_alpha)
            try:
                ax.obj[ir, ic].set_edgecolor(ax.edge_color)
            except:
                for f in ['bottom', 'top', 'right', 'left']:
                    ax.obj[ir, ic].spines[f].set_color(ax.edge_color)

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

            # Set major grid
            if self.grid_major.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='major', zorder=3,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.line_style,
                                          linewidth=self.grid_major.line_width)
                ax.obj[ir, ic].yaxis.grid(b=True, which='major', zorder=3,
                                          color=self.grid_major.color,
                                          linestyle=self.grid_major.line_style,
                                          linewidth=self.grid_major.line_width)

            # Set minor grid
            if self.grid_minor.on:
                ax.obj[ir, ic].xaxis.grid(b=True, which='minor', zorder=4,
                                          color=self.grid_minor.color,
                                          linestyle=self.grid_minor.line_style,
                                          linewidth=self.grid_minor.line_width)
                ax.obj[ir, ic].yaxis.grid(b=True, which='minor', zorder=4,
                                          color=self.grid_minor.color,
                                          linestyle=self.grid_minor.line_style,
                                          linewidth=self.grid_minor.line_width)

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
            else:
                axes = self.axes.obj[ir, ic]

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

            # Turn off scientific (how do we force it?)
            if not self.axes.sci_x \
                    and self.ptype != 'boxplot' \
                    and self.axes.scale not in ['logx', 'semilogx', 'loglog'] \
                    and (not self.axes.sharex or ir==0 and ic==0):
                axes[ia].get_xaxis().get_major_formatter().set_scientific(False)

            if not self.axes.sci_y \
                    and self.axes.scale not in ['logy', 'semilogy', 'loglog'] \
                    and (not self.axes.sharey or ir==0 and ic==0):
                axes[ia].get_yaxis().get_major_formatter().set_scientific(False)

            # Set custom tick increment
            if ia==0:
                lab = ''
            else:
                lab = '2'
            xinc = getattr(getattr(self, 'ticks_major'), 'increment_x%s' % lab)
            if getattr(getattr(self, 'ticks_major'), 'increment_x%s' % lab):
                xmin, xmax = axes[ia].get_xlim()
                axes[ia].set_yticks(np.arange(xmin, xmax, xinc))
            yinc = getattr(getattr(self, 'ticks_major'), 'increment_y%s' % lab)
            if yinc is not None:
                ymin, ymax = axes[ia].get_ylim()
                axes[ia].set_yticks(np.arange(ymin, ymax, yinc))

            # General tick params
            axes[ia].tick_params(axis='both',
                                 which='major',
                                 pad=self.ws_ticks_ax,
                                 colors=self.ticks_major.color,
                                 labelcolor=self.tick_labels_major.font_color,
                                 labelsize=self.tick_labels_major.font_size,
                                 top=self.ticks_major.on,
                                 bottom=self.ticks_major.on,
                                 right=self.ticks_major.on,
                                 left=self.ticks_major.on,
                                 length=self.ticks_major.size[0],
                                 width=self.ticks_major.size[1],
                                 )
            axes[ia].tick_params(axis='both',
                                 which='minor',
                                 pad=self.ws_ticks_ax,
                                 colors=self.ticks_minor.color,
                                 labelcolor=self.tick_labels_minor.font_color,
                                 labelsize=self.tick_labels_minor.font_size,
                                 top=self.ticks_minor.on,
                                 bottom=self.ticks_minor.on,
                                 right=self.ticks_minor.on,
                                 left=self.ticks_minor.on,
                                 length=self.ticks_minor.size[0],
                                 width=self.ticks_minor.size[1],
                                 )
            if self.ticks_minor.on:
                axes[ia].minorticks_on()

            # Force ticks
            if self.separate_ticks:
                mplp.setp(axes[ia].get_xticklabels(), visible=True)
                mplp.setp(axes[ia].get_yticklabels(), visible=True)

            # Check for overlapping major tick labels
            if self.tick_labels_major.on and self.tick_cleanup:
                xmin, xmax = axes[ia].get_xlim()
                xticks = axes[ia].get_xticks()
                xlabels = [f for f in axes[ia].xaxis.iter_ticks()]
                xlabel_text = [f[2] for f in xlabels]
                try:
                    x_first = [i for i, f in enumerate(xlabels)
                               if f[1] >= xmin and f[2] != ''][0]
                except:
                    x_first = -999
                try:
                    x_last = [i for i, f in enumerate(xlabels)
                              if f[1] <= xmax and f[2] != ''][-1]
                except:
                    x_last = -999

                ymin, ymax = axes[ia].get_ylim()
                yticks = axes[ia].get_yticks()
                ylabels = [f for f in axes[ia].yaxis.iter_ticks()]
                ylabel_text = [f[2] for f in ylabels]
                try:
                    y_first = [i for i, f in enumerate(ylabels)
                               if f[1] >= ymin and f[2] != ''][0]
                except:
                    y_first = -999
                try:
                    y_last = [i for i, f in enumerate(ylabels)
                              if f[1] <= ymax and f[2] != ''][-1]
                except:
                    y_last = -999

                xc = [0, -self.tick_labels_major.size_x[1]/2-self.ws_ticks_ax]
                yc = [-self.tick_labels_major.size_y[0]/2-self.ws_ticks_ax, 0]
                buf = 6
                delx = self.axes.size[0]/(len(xticks)-2)
                dely = self.axes.size[1]/(len(yticks)-2)
                x2x, y2y = [], []
                if ia == 0:
                    lab = ''
                else:
                    lab = '2'
                xw, xh = getattr(getattr(self, 'tick_labels_major'),
                                 'size_x%s' % lab)
                yw, yh = getattr(getattr(self, 'tick_labels_major'),
                                 'size_y%s' % lab)

                x0y0 = utl.rectangle_overlap([xw+2*buf, xh+2*buf, xc],
                                             [yw+2*buf, yh+2*buf, yc])

                for ix in range(0, len(xticks) - 1):
                    xwh = self.tick_labels_major.size_x
                    x2x += [utl.rectangle_overlap([xw+2*buf, xh+2*buf, [delx*ix,0]],
                                                  [xw+2*buf, xh+2*buf, [delx*(ix+1), 0]])]
                for iy in range(0, len(yticks) - 1):
                    ywh = self.tick_labels_major.size_y
                    y2y += [utl.rectangle_overlap([yw+2*buf, yh+2*buf, [0,dely*iy]],
                                                  [yw+2*buf, yh+2*buf, [0,dely*(iy+1)]])]

                # x and y at the origin
                if x0y0 and y_first==0:
                    # what if the first tick is missing?  need to add buffer
                    # like in case below xc and yc are not necessarily correct
                    ylabel_text[y_first] = ''

                # x overlapping x
                if any(x2x) and (not (self.axes.sharex and ir > 0 or ic > 0)) \
                        and x_first != -999 and x_last != -999:
                    for i in range(x_first, x_last, 2):
                        xlabel_text[i] = ''

                # y overlapping y
                if any(y2y) and (not (self.axes.sharey and ir > 0 or ic > 0)) \
                        and y_first != -999 and y_last != -999:
                    for i in range(y_first, y_last, 2):
                        ylabel_text[i] = ''

                # overlapping labels between row, col, and wrap plots
                if x_last != -999:
                    last_x = xlabels[x_last][1]
                    last_x_pos = last_x/(xmax-xmin)
                    last_x_px = (1-last_x_pos)*self.axes.size[0]
                    if self.ncol > 1 and \
                            xw > last_x_px + self.ws_col - self.ws_tick_tick_minimum and \
                            ic < self.ncol - 1:
                        xlabel_text[x_last] = ''

                if y_last != -999:
                    last_y = ylabels[y_last][1]
                    last_y_pos = last_y/(ymax-ymin)
                    last_y_px = (1-last_y_pos)*self.axes.size[1]
                    if self.nrow > 1 and \
                            yh > last_y_px + self.ws_col - self.ws_tick_tick_minimum and \
                            ir < self.nrow - 1:
                        ylabel_text[y_last] = ''

                # overlapping last y and first x between row, col, and wraps
                if self.nrow > 1 and ir < self.nrow-1:
                    x2y = utl.rectangle_overlap([xw, xh, xc],
                                                [yw, yh, [yc[0], yc[1]-self.ws_row]])
                    if x2y:
                        xlabel_text[0] = ''

                axes[ia].set_xticklabels(xlabel_text)
                axes[ia].set_yticklabels(ylabel_text)

            # Disable major tick labels
            elif not self.tick_labels_major.on:
                axes[ia].tick_params(which='major',
                                     labelbottom='off', labelleft='off',
                                     labelright='off', labeltop='off')

            # Set minor tick labels
            if self.tick_labels_minor.on:
                # does this need to be axis specific?  probably to deal
                # with log on one axis only OR do you just ignore with log?
                if self.ticks_minor.number is not None:
                    pass

                else:
                    xinc = (xlabels[x_last][1]-xlabels[x_first][1]) \
                           / (x_last-x_first)
                    decimals = utl.get_decimals(xinc)
                    axes[ia].xaxis.set_minor_formatter(
                        ticker.FormatStrFormatter('%%.%sf' % (decimals+1)))
                    # yinc = (ylabels[y_last][1]-ylabels[y_first][1]) \
                    #        / (y_last-y_first)
                    # decimals = utl.get_decimals(yinc)
                    # axes[ia].yaxis.set_minor_formatter(
                    #     ticker.FormatStrFormatter('%%.%sf' % (decimals+1)))


                #and self.tick_cleanup:
                pass

            # Disable minor tick labels
            elif not self.tick_labels_minor.on:
                axes[ia].tick_params(which='minor',
                                     labelbottom='off', labelleft='off',
                                     labelright='off', labeltop='off')


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



