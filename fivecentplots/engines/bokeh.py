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
        pass

    def add_text(self, ir, ic, text=None, element=None, offsetx=0, offsety=0,
                 **kwargs):
        """
        Add a text box
        """

        pass

    def make_figure(self, data, **kwargs):
        """
        Make the figure and axes objects
        """

        self.update_from_data(data)
        self.update_wrap(data, kwargs)
        self.set_label_text(data, **kwargs)


        self.fig.obj = bp.figure(plot_width=self.axes.size[0],
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
            marker = format_marker(self.fig.obj,
                                   self.markers.type.get(iline))
            if marker != 'None':
                marker(df[x], df[y],
                       fill_color=self.markers.fill_color.get(iline) \
                                  if self.markers.filled else None,
                       line_color=self.markers.edge_color.get(iline)[0:7],
                       size=self.markers.size.get(iline),
                       )
            else:
                st()
                #??
                points = ax.plot(df[x], df[y],
                                marker=marker,
                                color=line_type.color.get(iline),
                                linestyle=line_type.style.get(iline),
                                linewidth=line_type.width.get(iline),
                                zorder=40)

        # Make the line
        lines = None
        if line_type.on:
            lines = self.fig.obj.line(df[x], df[y],
                                      color=line_type.color.get(iline)[0:7],
                                      #linestyle=line_type.style.get(iline),
                                      line_width=line_type.width.get(iline),
                                     )

    def save(self, filename, idx=0):

        bp.output_file(filename)

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

        self.fig.obj.xaxis.axis_label = self.label_x.text
        self.fig.obj.yaxis.axis_label = self.label_y.text

    def set_axes_ranges(self, ir, ic, ranges):
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
        """
        Handle display of the plot window
        """

        bp.show(self.fig.obj)