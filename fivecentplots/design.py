from fivecentplots.themes.gray import *
import matplotlib as mpl
import matplotlib.pyplot as mplp
import os
import pandas as pd
import pdb
import datetime

st = pdb.set_trace


class FigDesign:

    def __init__(self, **kwargs):
        """ Compute design spacing rules for a matplotlib Figure

        Overall design is an alternative to the built-in tight_layout()
        function and isloosely modeled on the Overlay Plot in JMP

        Keyword Args:
            ws_leg_ax:
            ax_size:
            cols:
            dpi:
            leg_items:
            ws_fig_leg:
            leg_font_size:
            leg_points:
            leg_title:
            ax_fig_ws:
            fig_ax_ws:
            ws_fig_title:
            ws_row:
            rows:
            ws_title_ax:
            title_h:
            ws_col:
        Returns:
            self
        """
        # Handle kwargs and defaults
        self.ax_size         = kwargs.get('ax_size',
                                          fcp_params['ax_size'])
        self.cbar_width      = kwargs.get('cbar_width',
                                          fcp_params['cbar_width'])
        self.col_labels_on   = kwargs.get('col_labels_on',
                                          False)
        self.col_label_size  = kwargs.get('col_label_size',
                                         fcp_params['rc_label_size'])
        self.dpi             = kwargs.get('dpi',
                                          fcp_params['dpi'])
        self.group_labels    = kwargs.get('group_labels', 0)
        self.leg_font_size   = kwargs.get('leg_font_size',
                                          fcp_params['leg_font_size'])
        self.leg_items       = kwargs.get('leg_items',
                                          fcp_params['leg_items'])
        self.leg_on          = kwargs.get('leg_on', True)
        self.leg_points      = kwargs.get('leg_points',
                                          fcp_params['leg_points'])
        self.leg_title       = kwargs.get('leg_title',
                                           fcp_params['leg_title'])
        self.ncol            = kwargs.get('ncol', 1)
        self.nrow            = kwargs.get('nrow', 1)
        self.row_labels_on   = kwargs.get('row_labels_on',
                                          False)
        self.row_label_size  = kwargs.get('row_label_size',
                                         fcp_params['rc_label_size'])
        self.title_h         = kwargs.get('title_h',
                                          fcp_params['title_h'])
        self.twinx           = kwargs.get('twinx', False)
        self.wrap_title_size = kwargs.get('wrap_title_size',
                                          fcp_params['wrap_title_size'])
        self.ws_cbar_ax      = kwargs.get('ws_cbar_ax',
                                          fcp_params['ws_cbar_ax'])
        self.ws_col          = kwargs.get('ws_col',
                                          fcp_params['ws_col'])
        self.ws_col_label    = kwargs.get('ws_col_label',
                                         fcp_params['ws_rc_label'])
        self.ws_fig_label    = kwargs.get('ws_fig_label',
                                          fcp_params['ws_fig_label'])
        self.ws_fig_leg      = kwargs.get('ws_fig_leg',
                                          fcp_params['ws_fig_leg'])
        self.ws_fig_title    = kwargs.get('ws_fig_title',
                                          fcp_params['ws_fig_title'])
        self.ws_label_tick   = kwargs.get('ws_label_tick',
                                          fcp_params['ws_label_tick'])
        self.ws_leg_ax       = kwargs.get('ws_leg_ax',
                                          fcp_params['ws_leg_ax'])
        self.ws_row_label    = kwargs.get('ws_row_label',
                                         fcp_params['ws_rc_label'])
        self.ws_row          = kwargs.get('ws_row',
                                          fcp_params['ws_row'])
        self.ws_tick_ax      = kwargs.get('ws_tick_ax',
                                          fcp_params['ws_tick_ax'])
        self.ws_title_ax     = kwargs.get('ws_title_ax',
                                          fcp_params['ws_title_ax'])
        self.ws_wrap_title   = kwargs.get('wrap_label_ws',
                                          fcp_params['ws_wrap_title'])
        self.xlabel_size     = kwargs.get('xlabel_size', (0, 0))
        self.xtick_size      = kwargs.get('xtick_size', (0, 0))
        self.ylabel_size     = kwargs.get('ylabel_size', (0, 0))
        self.ytick_size      = kwargs.get('ytick_size', (0, 0))

        # Initialize other variables
        self.ax_h              = self.ax_size[1]
        self.ax_w              = self.ax_size[0]
        self.bottom            = 0
        self.col_label_bottom  = 0
        self.col_label_height  = 0
        self.leg_h             = 0
        self.leg_overflow      = 0
        self.leg_right         = 0
        self.leg_top           = 0
        self.leg_w             = 0
        self.left              = 0
        self.fig_h             = 0
        self.fig_w             = 0
        self.fig_h_px          = 0
        self.fig_w_px          = 0
        self.right             = 0
        self.row_label_left    = 0
        self.row_label_width   = 0
        self.top               = 0
        self.wrap_title_bottom = 0

        # Account for colorbars
        if not kwargs['cbar']:
            self.ws_cbar_ax = 0
            self.cbar_width = 0
            self.cbar_label = 0
        else:
            self.ws_fig_leg = 100 # better solution?
            self.cbar_label = self.ws_fig_leg

        # Set label size
        if self.row_labels_on and not kwargs['wrap']:
            self.row_labels = self.row_label_size + self.ws_row_label
        else:
            self.row_labels = 0
        if self.col_labels_on and not kwargs['wrap']:
            self.col_labels = self.col_label_size + self.ws_col_label
        elif self.col_labels_on:
            self.col_labels = self.col_label_size + self.ws_col_label
            self.ws_row += self.col_label_size + self.ws_col_label
        else:
            self.col_labels = 0

        # Add an optional column label for wrap plots
        if kwargs['wrap_title']:
            self.wrap_title = self.wrap_title_size + self.ws_wrap_title
        else:
            self.wrap_title = 0

        # Weird spacing defaults out of our control
        self.fig_right_border = 6
        self.leg_top_offset = 8
        self.leg_border = 3

        # Calculate dimensions
        self.get_legend_size()
        self.get_figure_size()
        self.get_subplots_adjust()
        self.get_label_position()
        self.get_legend_position()
        self.get_title_position()

    def get_label_position(self):
        """
        Get option group label positions
        """

        self.row_label_left = (self.ax_w + self.ws_row_label +
                               self.ws_cbar_ax + self.cbar_width +
                               self.cbar_label)/self.ax_w
        self.row_label_width = self.row_label_size/self.ax_w
        self.col_label_bottom = (self.ax_h + self.ws_col_label)/self.ax_h
        self.col_label_height = self.col_label_size/self.ax_h
        self.wrap_title_bottom = (self.ax_h + self.col_labels +
                                  self.ws_wrap_title)/self.ax_h

    def get_legend_position(self):
        """
        Get legend position
        """

        self.leg_top = self.top + self.leg_top_offset/self.fig_h_px
        self.leg_right = 1 - (self.ws_fig_leg - self.fig_right_border -
                              0*self.group_labels)/self.fig_w_px

    def get_legend_size(self):
        """
        Determine the size of the legend by building a dummy figure and
        extracting its properties
        """

        if len(self.leg_items) > 0 and self.leg_on:
            now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            mplp.ioff()
            fig = mpl.pyplot.figure(dpi=self.dpi)
            ax = fig.add_subplot(111)
            lines = []
            for i in self.leg_items:
                lines += ax.plot([1,2,3])
            leg = mpl.pyplot.legend(lines,
                                    list(self.leg_items),
                                    title=self.leg_title,
                                    numpoints=self.leg_points,
                                    prop={'size':self.leg_font_size})
            mpl.pyplot.draw()
            mpl.pyplot.savefig('dummy_legend_%s.png' % now)
            self.leg_h = leg.get_window_extent().height + self.leg_border
            self.leg_w = leg.get_window_extent().width + self.leg_border
            mpl.pyplot.close(fig)
            os.remove('dummy_legend_%s.png' % now)
            # self.leg_w = max(self.leg_w, self.ax_ws_fig_leg)
        else:
            self.leg_h = 0
            self.leg_w = 0

    def get_figure_size(self):
        """
        Determine the size of the mpl figure canvas in pixels and inches
        """

        # Account for second y-axis
        if self.twinx:
            y2 = self.fig_ax_ws - self.ws_leg_ax
        else:
            y2 = 0

        # Calculate the canvas size
        self.fig_w_px = self.ws_fig_label + self.ws_label_tick + \
                        self.ylabel_size[0] + self.ytick_size[0] + \
                        self.ax_w*self.ncol + y2 + \
                        self.ws_leg_ax + self.leg_w + self.ws_fig_leg + \
                        self.ws_col*(self.ncol-1) + self.row_labels - \
                        self.fig_right_border + self.ws_tick_ax*self.ncol + \
                        self.group_labels + \
                        self.ws_cbar_ax*self.ncol + self.cbar_width*self.ncol
        self.fig_h_px = self.ws_fig_title + self.title_h + \
                        self.ws_title_ax + self.ax_h*self.nrow + \
                        self.ws_fig_label + self.ws_label_tick + \
                        self.xlabel_size[1] + self.xtick_size[1] + \
                        self.ws_row*(self.nrow-1) + \
                        self.col_labels + self.ws_tick_ax*self.nrow + \
                        self.wrap_title
        self.fig_only = self.ax_h*self.nrow + \
                        self.ws_row*(self.nrow-1) + \
                        self.ws_tick_ax*self.ncol
        self.leg_overflow = max(self.leg_h-self.fig_only, 0)
        self.fig_w = self.fig_w_px/self.dpi
        self.fig_h = (self.fig_h_px+self.leg_overflow)/self.dpi
        self.ax_w += self.ws_cbar_ax + self.cbar_width

    def get_subplots_adjust(self):
        """
        Calculate the subplots_adjust parameters for the axes
        """

        self.left = (self.ws_fig_label + self.ws_label_tick + \
                     self.ylabel_size[0] + self.ytick_size[0] + self.ws_tick_ax)/self.fig_w_px
        self.right = (self.ws_fig_label + self.ws_label_tick + \
                      self.ylabel_size[0] + self.ytick_size[0] + \
                      self.ax_w*self.ncol + self.ws_tick_ax + \
                      self.ws_col*(self.ncol-1))/self.fig_w_px
        self.top = 1 - (self.ws_fig_title + self.title_h + self.wrap_title + \
                   self.ws_title_ax + self.col_labels)/self.fig_h/self.dpi
        self.bottom = (self.leg_overflow + \
                       self.ws_fig_label + self.ws_label_tick + \
                       self.xlabel_size[1] + self.xtick_size[1] + \
                       self.ws_tick_ax)/self.fig_h/self.dpi

    def get_title_position(self):
        """
        Calculate the title position
        """

        self.title_bottom = 1+(self.ws_title_ax+self.col_labels+
                            self.ws_tick_ax*self.ncol)/self.ax_h
        self.title_top = self.title_bottom+(self.ws_title_ax+
                         self.title_h)/self.ax_h

        self.title_h_px = self.title_h
        self.title_w_px = self.fig_w_px
        self.title_h = self.title_h/self.ax_h
        self.title_w = self.fig_w_px/self.ax_w
        self.title_left = 0 #- self.fig_ax_ws/self.ax_w

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df
