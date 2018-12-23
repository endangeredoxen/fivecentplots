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
import matplotlib as mpl
import matplotlib.pyplot as mplp
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
import matplotlib.mlab as mlab
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

class Layout(BaseLayout):
    def __init__(self, **kwargs):

        global ENGINE
        ENGINE = 'bokeh'

        BaseLayout.__init__(self, **kwargs)

    def add_box_labels(self, ir, ic, dd):
        pass

    def add_hvlines(self, ir, ic):
        pass

    def add_legend(self):
        pass

    def make_figure(self):
        pass

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