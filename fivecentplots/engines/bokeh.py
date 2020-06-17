from .. import fcp
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

db = pdb.set_trace


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

        x = 0
        y = self.ncol - 1
        tt = list(self.legend.values.items())
        tt = [f for f in tt if f[0] is not 'NaN']
        title = self.axes.obj[x, y].circle(0, 0, size=0.00000001,
                                             color=None)
        tt = [(self.legend.text, [title])] + tt
        legend = bm.Legend(items=tt, location='top_right')
        self.axes.obj[x, y].add_layout(legend, 'right')

    def add_text(self, ir, ic, text=None, element=None, offsetx=0, offsety=0,
                 **kwargs):
        """
        Add a text box
        """

        pass

    def get_element_sizes(self, data):

        # Add label size
        self.label_x.size = utl.get_text_dimensions(self.label_x.text,
                                **self.make_kwargs(self.label_x))
        self.label_y.size += utl.get_text_dimensions(self.label_y.text,
                                **self.make_kwargs(self.label_y))

        # Ticks (rough)
        xlab = '%s.0' % int(data.ranges[0, 0]['xmax'])
        ylab = '%s.0' % int(data.ranges[0, 0]['ymax'])
        self.tick_labels_major_x.size = utl.get_text_dimensions(xlab,
            **self.make_kwargs(self.tick_labels_major_x))
        self.tick_labels_major_y.size += utl.get_text_dimensions(ylab,
            **self.make_kwargs(self.tick_labels_major_y))

        # title
        if self.title.text is not None:
            self.title.size = utl.get_text_dimensions(self.title.text,
                                **self.make_kwargs(self.title))

        # rc labels
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                if self.label_row.text is not None:
                    text = '%s=%s' % (self.label_row.text, self.label_row.values[ir])
                    self.label_row.size[1] = max(self.label_row.size[1],  \
                        utl.get_text_dimensions(text,
                                    **self.make_kwargs(self.label_row))[1])
                if self.label_col.text is not None:
                    text = '%s=%s' % (self.label_col.text, self.label_col.values[ir])
                    self.label_col.size[1] = max(self.label_col.size[1],  \
                        utl.get_text_dimensions(text,
                                    **self.make_kwargs(self.label_col))[1])

        # Legend
        if data.legend_vals is None:
            return data

        # Get approx legend size
        name_size = 0
        for name in data.legend_vals['names']:
            name_size = max(name_size,
                            utl.get_text_dimensions(str(name),
                                **self.make_kwargs(self.legend))[0])

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
        #self.get_figure_size(data, **kwargs)

        self.axes.obj = np.array([[None]*self.ncol]*self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                x_scale, y_scale = self.set_axes_scale2(ir, ic)
                x_size, y_size = self.axes.size
                # legend
                if ir == 0 and ic == self.ncol - 1:
                    x_size += self.legend.size[0]
                # rc labels
                y_size += self.label_col.size[1]
                # axes labels
                if ir == 0 and ic == 0:
                    x_size += self.label_x.size[1] + 15
                if ir == self.nrow - 1:
                    y_size += self.label_y.size[1] + 15
                # ticks (separate out to label and ticks later)
                x_size += self.tick_labels_major_x.size[0] + 15
                y_size += self.tick_labels_major_y.size[0] + 15
                # title
                y_size += self.title.size[0]

                self.axes.obj[ir, ic] = bp.figure(plot_width=int(x_size),
                                                  plot_height=int(y_size),
                                                  x_axis_type=x_scale,
                                                  y_axis_type=y_scale)

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
            if self.markers.on and not marker_disable:
                leg_vals += [points]
            if line_type.on:
                leg_vals += [lines]
            self.legend.values[leg_name] = leg_vals

    def save(self, filename, idx=0):

        bp.output_file(filename)
        # by row
        if self.ncol == 1 and self.nrow > 1:
            self.saved = bl.column(self.axes.obj.flatten().tolist())
        # by column
        elif self.ncol > 1 and self.nrow == 1:
            self.saved = bl.row(self.axes.obj.flatten().tolist())
        # grid
        else:
            self.saved = bl.gridplot(self.axes.obj.tolist(),
                                      ncols=self.ncol if self.ncol > 1 else None)
        bp.save(self.saved)

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
                    self.axes.obj[ir, ic].x_range=bm.Range1d(start=xx)
                elif xx is not None and xval == 'x2min':
                    self.axes2.obj[ir, ic].x_range=bm.Range1d(start=xx)
                elif xx is not None and xval == 'xmax':
                    self.axes.obj[ir, ic].x_range=bm.Range1d(end=xx)
                elif xx is not None and xval == 'x2max':
                    self.axes2.obj[ir, ic].x_range=bm.Range1d(end=xx)
        else:
            if ranges[ir, ic]['xmin'] is not None:
                self.axes.obj[ir, ic].x_range=bm.Range1d(start=ranges[ir, ic]['xmin'])
            if ranges[ir, ic]['x2min'] is not None:
                self.axes2.obj[ir, ic].x_range=bm.Range1d(start=ranges[ir, ic]['x2min'])
            if ranges[ir, ic]['xmax'] is not None:
                self.axes.obj[ir, ic].x_range=bm.Range1d(end=ranges[ir, ic]['xmax'])
            if ranges[ir, ic]['x2max'] is not None:
                self.axes2.obj[ir, ic].x_range=bm.Range1d(end=ranges[ir, ic]['x2max'])

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
                    self.axes.obj[ir, ic].y_range=bm.Range1d(start=yy)
                elif yy is not None and yval == 'y2min':
                    self.axes2.obj[ir, ic].y_range=bm.Range1d(start=yy)
                elif yy is not None and yval == 'ymax':
                    self.axes.obj[ir, ic].y_range=bm.Range1d(end=yy)
                elif yy is not None and yval == 'y2max':
                    self.axes2.obj[ir, ic].y_range=bm.Range1d(end=yy)
        else:
            if ranges[ir, ic]['ymin'] is not None:
                self.axes.obj[ir, ic].y_range=bm.Range1d(start=ranges[ir, ic]['ymin'])
            if ranges[ir, ic]['y2min'] is not None:
                self.axes2.obj[ir, ic].y_range=bm.Range1d(start=ranges[ir, ic]['y2min'])
            if ranges[ir, ic]['ymax'] is not None:
                self.axes.obj[ir, ic].y_range=bm.Range1d(end=ranges[ir, ic]['ymax'])
            if ranges[ir, ic]['y2max'] is not None:
                self.axes2.obj[ir, ic].y_range=bm.Range1d(end=ranges[ir, ic]['y2max'])

    def set_axes_rc_labels(self, ir, ic):

        title = self.axes.obj[ir, ic].title

        # Row labels
        if ic == self.ncol-1 and self.label_row.on and not self.label_wrap.on:
            title.text = ' %s=%s ' % (self.label_row.text, self.label_row.values[ir])
            title.align = self.label_row.align
            title.text_color = self.label_row.font_color
            title.text_font_size = '%spt' % self.label_row.font_size
            title.background_fill_alpha = self.label_row.fill_alpha
            title.background_fill_color = self.label_row.fill_color.get(0)[0:7]

        # Col/wrap labels
        if (ir == 0 and self.label_col.on) or self.label_wrap.on:
            # if self.label_wrap.on:
            #     text = ' | '.join([str(f) for f in utl.validate_list(
            #         self.label_wrap.values[ir*self.ncol + ic])])
            #     scol = self.add_label(ir, ic, text,
            #                          **self.make_kwargs(self.label_wrap))
            # else:
            title.text = ' %s=%s ' % (self.label_col.text, self.label_col.values[ic])
            title.align = self.label_col.align
            title.text_color = self.label_col.font_color
            title.text_font_size = '%spt' % self.label_col.font_size
            title.background_fill_alpha = self.label_col.fill_alpha
            title.background_fill_color = self.label_col.fill_color.get(0)[0:7]

    def set_axes_scale(self, ir, ic):
        pass

    def set_axes_scale2(self, ir, ic):
        """
        This appears to need to happen at instantiation of the figure
        """

        if str(self.axes.scale).lower() in LOGX:
            x_scale = 'log'
        # elif str(ax.scale).lower() in SYMLOGX:
        #     ax.obj[ir, ic].set_xscale('symlog')
        # elif str(ax.scale).lower() in LOGITX:
        #     ax.obj[ir, ic].set_xscale('logit')
        else:
            x_scale = 'linear'
        if str(self.axes.scale).lower() in LOGY:
            y_scale = 'log'
        # elif str(ax.scale).lower() in SYMLOGY:
        #     ax.obj[ir, ic].set_yscale('symlog')
        # elif str(ax.scale).lower() in LOGITY:
        #     ax.obj[ir, ic].set_yscale('logit')
        else:
            y_scale = 'linear'

        return x_scale, y_scale

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

    def show(self, filename=None):
        """
        Handle display of the plot window
        """

        try:
            app = str(get_ipython())
        except:
            app = ''

        # jupyter notebook special
        if 'zmqshell.ZMQInteractiveShell' in app:
            #bp.output_notebook()
            #bp.show(bl.gridplot(self.axes.obj.flatten(), ncols=self.ncol))
            from IPython.core.display import HTML
            return HTML(filename)

        # other
        else:
            os.startfile(filename)
