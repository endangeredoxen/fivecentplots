from .. import fcp
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
from .. colors import *
from .. utilities import RepeatedList
from .. import utilities as utl
from distutils.version import LooseVersion
from random import randint
from collections import defaultdict
from . layout import *
import warnings
import bokeh.plotting as bp
import bokeh.layouts as bl
import bokeh.models as bm
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


DASHES = {'-': 'solid',
          'solid': 'solid',
          '--': 'dashed ',
          'dashed': 'dashed',
          '.': 'dotted',
          'dotted': 'dotted',
          '.-': 'dotdash',
          'dotdash': 'dotdash',
          '-.': 'dashdot',
          'dashdot': 'dashdot'}


def fill_alpha(hexstr):
    """
    Break an 8-digit hex string into a hex color and a fractional alpha value
    """

    if len(hexstr) == 6:
        return hexstr, 1

    return hexstr[0:7], int(hexstr[-2:], 16) / 255


def format_marker(fig, marker):
            """
            Format the marker string to mathtext
            """

            markers = {'o': fig.circle,
                       'circle': fig.circle,
                       '+': fig.cross,
                       'cross': fig.cross,
                       's': fig.square,
                       'square': fig.square,
                       'x': fig.x,
                       'd': fig.diamond,
                       'diamond': fig.diamond,
                       't': fig.triangle,
                       'triangle': fig.triangle}

            return markers[marker]


class Layout(BaseLayout):
    def __init__(self, plot_func, data, **kwargs):

        global ENGINE
        ENGINE = 'bokeh'

        BaseLayout.__init__(self, plot_func, data, **kwargs)

        # Update kwargs
        if not kwargs.get('save_ext'):
            kwargs['save_ext'] = '.html'
        self.kwargs = kwargs

    def add_box_labels(self, ir, ic, dd):
        pass

    def add_hvlines(self, ir, ic, df=None):
        pass

    def add_legend(self):
        """
        Add a figure legend
        """

        if not self.legend.on or len(self.legend.values) == 0:
            return

        tt = list(self.legend.values.items())
        tt = [f for f in tt if f[0] is not 'NaN']
        title = self.axes.obj[0, 0].circle(0, 0, size=0.00000001,
                                             color=None)
        tt = [(self.legend.text, [title])] + tt
        legend = bm.Legend(items=tt, location='top_right')
        self.axes.obj[0,0].add_layout(legend, 'right')

    def add_text(self, ir, ic, text=None, element=None, offsetx=0, offsety=0,
                 **kwargs):
        """
        Add a text box
        """

        pass

    def get_element_sizes(self, data):

        if data.legend_vals is None:
            return data

        # Get approx legend size
        name_size = 0
        for name in data.legend_vals['names']:
            name_size = max(name_size,
                            utl.get_text_dimensions(name, self.legend.font_size,
                                                    self.legend.font)[0])

        if self.legend.on:
            self.legend.size = [10 + 20 + 5 + name_size + 10, 0]

        return data

    def get_figure_size(self, data, **kwargs):

        self.axes.size[0] += self.legend.size[0]

    def make_figure(self, data, **kwargs):
        """
        Make the figure and axes objects
        """

        self.update_from_data(data)
        self.update_wrap(data, kwargs)
        self.set_label_text(data, **kwargs)
        data = self.get_element_sizes(data)
        self.get_figure_size(data, **kwargs)

        self.axes.obj = np.array([[None]*self.ncol]*self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.axes.obj[ir, ic] = bp.figure(plot_width=self.axes.size[0],
                                                  plot_height=self.axes.size[1])

        self.axes.visible = np.array([[True]*self.ncol]*self.nrow)

        return data

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

        df = df.copy()

        if not line_type:
            line_type = self.lines
        else:
            line_type = getattr(self, line_type)

        # TWINNING
        # Insert here

        # Make the points
        points = None
        if self.markers.on and not marker_disable:
            if self.markers.jitter:
                df[x] = np.random.normal(df[x], 0.03, size=len(df[y]))
            marker = format_marker(self.axes.obj[ir, ic],
                                   self.markers.type.get(iline))
            ecolor, ealpha = fill_alpha(self.markers.edge_color.get(iline))
            fcolor, falpha = fill_alpha(self.markers.fill_color.get(iline))
            if marker != 'None':
                points = marker(df[x], df[y],
                                fill_color=fcolor if self.markers.filled else None,
                                fill_alpha=falpha,
                                line_color=ecolor,
                                line_alpha=ealpha,
                                size=self.markers.size.get(iline),
                                )
            else:
                #??
                points = marker(df[x], df[y],
                                color=line_type.color.get(iline),
                                linestyle=line_type.style.get(iline),
                                linewidth=line_type.width.get(iline))

        # Make the line
        lines = None
        if line_type.on:
            lines = self.axes.obj[ir, ic].line(df[x], df[y],
                                               color=line_type.color.get(iline)[0:7],
                                               line_dash=DASHES[line_type.style.get(iline)],
                                               line_width=line_type.width.get(iline),
                                               )

        # Add a reference to the line to self.lines
        if leg_name is not None:
            leg_vals = []
            if self.markers.on:
                leg_vals += [points]
            if line_type.on:
                leg_vals += [lines]
            self.legend.values[leg_name] = leg_vals

    def save(self, filename, idx=0):

        #bp.output_file(filename)
        #self.saved = bl.gridplot(self.axes.obj.flatten(), ncols=self.ncol)
        #bp.save(self.saved)
        pass  # conflicts with jupyter

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

        fill, alpha = fill_alpha(axes[0].fill_color.get(utl.plot_num(ir, ic, self.ncol)))
        self.axes.obj[ir, ic].background_fill_color = fill
        self.axes.obj[ir, ic].background_fill_alpha = alpha

    def set_axes_grid_lines(self, ir, ic):
        """
        Style the grid lines and toggle visibility

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        """

        axis = ['x', 'y']

        for ax in axis:
            grid = getattr(self.axes.obj[ir, ic], '%sgrid' % ax)
            grid.grid_line_color = self.grid_major.color.get(0)[0:7] if self.grid_major.on \
                                   else None
            grid.grid_line_width = self.grid_major.width.get(0)
            grid.grid_line_dash = DASHES[self.grid_major.style.get(0)]

            grid.minor_grid_line_color = self.grid_minor.color.get(0)[0:7] if self.grid_minor.on \
                                   else None
            grid.minor_grid_line_width = self.grid_minor.width.get(0)
            grid.minor_grid_line_dash = DASHES[self.grid_minor.style.get(0)]

    def set_axes_labels(self, ir, ic):
        """
        Set the axes labels

        Args:
            ir (int): current row index
            ic (int): current column index
            kw (dict): kwargs dict

        """

        axis = ['x', 'y'] #x2', 'y', 'y2', 'z']
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

            # Twinning?

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
                        utl.plot_num(ir, ic, self.ncol) != self.nwrap:
                    continue

            lkwargs = self.make_kwargs(label)
            laxis = getattr(self.axes.obj[ir, ic], '%saxis' % ax)
            laxis.axis_label = labeltext
            laxis.axis_label_text_font = lkwargs['font']
            laxis.axis_label_text_font_size = '%spt' % lkwargs['font_size']
            laxis.axis_label_text_color = lkwargs['font_color']
            laxis.axis_label_text_font_style = lkwargs['font_style']

    def set_axes_ranges(self, ir, ic, ranges):
        pass

    def set_axes_rc_labels(self, ir, ic):
        pass

    def set_axes_scale(self, ir, ic):
        pass

    def set_axes_ticks(self, ir, ic):

        self.axes.obj[ir, ic].xaxis.major_label_text_font_size = \
            '%spt' % self.tick_labels_major.font_size
        self.axes.obj[ir, ic].yaxis.major_label_text_font_size = \
            '%spt' % self.tick_labels_major.font_size

    def set_figure_title(self):
        """
        Add a figure title
        """

        if self.title.on:
            title = self.axes.obj[0,0].title
            title.text = self.title.text
            title.align = self.title.align
            title.text_color = self.title.font_color
            title.text_font_size = '%spt' % self.title.font_size
            title.background_fill_alpha = self.title.fill_alpha
            title.background_fill_color = self.title.fill_color.get(0)[0:7]

    def show(self, inline=True):
        """
        Handle display of the plot window
        """

        try:
            app = str(get_ipython())
        except:
            app = ''

        # jupyter notebook special
        if 'zmqshell.ZMQInteractiveShell' in app:
            bp.output_notebook()
            bp.show(bl.gridplot(self.axes.obj.flatten(), ncols=self.ncol))

        # other
        else:
            bp.show(self.saved)
