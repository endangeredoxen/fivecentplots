from fivecentplots.defaults import *
import matplotlib as mpl
import os
import pandas as pd
import pdb

st = pdb.set_trace


class FigDesign:

    def __init__(self, **kwargs):
        """ Compute design spacing rules for a matplotlib Figure

        Overall design is an alternative to the built-in tight_layout()
        function and isloosely modeled on the Overlay Plot in JMP

        Args:
            **kwargs:
              * ax_leg_ws:
              * ax_size:
              * cols:
              * dpi:
              * leg_items:
              * leg_fig_ws:
              * leg_font_size:
              * leg_points:
              * leg_title:
              * ax_fig_ws:
              * fig_ax_ws:
              * fig_title_ws:
              * row_padding:
              * rows:
              * title_ax_ws:
              * title_h:
              * col_padding:
        Returns:
            self
        """
        # Handle kwargs and defaults
        self.ax_fig_ws       = kwargs.get('ax_fig_ws',
                                          fcp_params['ax_fig_ws'])
        self.ax_leg_fig_ws   = kwargs.get('ax_leg_fig_ws',
                                          fcp_params['ax_leg_fig_ws'])
        self.ax_label_pad    = kwargs.get('ax_label_pad',
                                          fcp_params['ax_label_pad'])
        self.ax_leg_ws       = kwargs.get('ax_leg_ws',
                                          fcp_params['ax_leg_ws'])
        self.ax_size         = kwargs.get('ax_size',
                                          fcp_params['ax_size'])
        self.col_labels_on   = kwargs.get('col_labels_on',
                                          False)
        self.col_label_size  = kwargs.get('col_label_size',
                                         fcp_params['rc_label_size'])
        self.col_label_ws    = kwargs.get('col_label_ws',
                                         fcp_params['rc_label_ws'])
        self.col_padding     = kwargs.get('col_padding',
                                          fcp_params['col_padding'])
        self.dpi             = kwargs.get('dpi',
                                          fcp_params['dpi'])
        self.fig_ax_ws       = kwargs.get('fig_ax_ws',
                                          fcp_params['fig_ax_ws'])
        self.fig_title_ws    = kwargs.get('fig_title_ws',
                                          fcp_params['fig_title_ws'])
        self.leg_fig_ws      = kwargs.get('leg_fig_ws',
                                          fcp_params['leg_fig_ws'])
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
        self.row_label_size = kwargs.get('row_label_size',
                                        fcp_params['rc_label_size'])
        self.row_label_ws   = kwargs.get('row_label_ws',
                                        fcp_params['rc_label_ws'])
        self.row_padding    = kwargs.get('row_padding',
                                         fcp_params['row_padding'])
        self.title_ax_ws    = kwargs.get('title_ax_ws',
                                         fcp_params['title_ax_ws'])
        self.title_h        = kwargs.get('title_h',
                                         fcp_params['title_h'])
        self.twinx          = kwargs.get('twinx', False)

        # Initialize other variables
        self.ax_h             = self.ax_size[1]
        self.ax_w             = self.ax_size[0]
        self.bottom           = 0
        self.col_label_bottom = 0
        self.col_label_height = 0
        self.leg_h            = 0
        self.leg_overflow     = 0
        self.leg_right        = 0
        self.leg_top          = 0
        self.leg_w            = 0
        self.left             = 0
        self.fig_h            = 0
        self.fig_w            = 0
        self.fig_h_px         = 0
        self.fig_w_px         = 0
        self.right            = 0
        self.row_label_left   = 0
        self.row_label_width  = 0
        self.top              = 0

        # Set label size
        if self.row_labels_on:
            self.row_labels = self.row_label_size + self.row_label_ws
        else:
            self.row_labels = 0
        if self.col_labels_on:
            self.col_labels = self.col_label_size + self.col_label_ws
        else:
            self.col_labels = 0
        
        # # Update title position
        # if self.col_labels > 0:
            # self.title_ax_ws += self.col_labels
        
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
        self.row_label_left = (self.ax_w + self.row_label_ws)/self.ax_w
        self.row_label_width = self.row_label_size/self.ax_w
        self.col_label_bottom = (self.ax_h + self.col_label_ws)/self.ax_h
        self.col_label_height = self.col_label_size/self.ax_h

    def get_legend_position(self):
        """
        Get legend position
        """

        self.leg_top = self.top + self.leg_top_offset/self.fig_h_px
        self.leg_right = 1 - \
            (self.leg_fig_ws-self.fig_right_border)/self.fig_w_px

    def get_legend_size(self):
        """
        Determine the size of the legend by building a dummy figure and
        extracting its properties
        """

        if len(self.leg_items) > 0 and self.leg_on:
            mpl.pylab.ioff()
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
            fig.canvas.draw()
            mpl.pyplot.savefig('dummy_legend.png')
            self.leg_h = leg.get_window_extent().height + self.leg_border
            self.leg_w = leg.get_window_extent().width + self.leg_border
            mpl.pyplot.close(fig)
            os.remove('dummy_legend.png')

        self.leg_w = max(self.leg_w, self.ax_leg_fig_ws)

    def get_figure_size(self):
        """
        Determine the size of the mpl figure canvas in pixels and inches
        """

        # Account for second y-axis
        if self.twinx:
            y2 = self.fig_ax_ws - self.ax_leg_ws
        else:
            y2 = 0

        # Calculate the canvas size
        self.fig_w_px = self.fig_ax_ws + self.ax_w*self.ncol + y2 + \
                        self.ax_leg_ws + self.leg_w + self.leg_fig_ws + \
                        self.col_padding*(self.ncol-1) + self.row_labels - \
                        self.fig_right_border + self.ax_label_pad*self.ncol
        self.fig_h_px = self.fig_title_ws + self.title_h + \
                        self.title_ax_ws + self.ax_h*self.nrow + \
                        self.ax_fig_ws + self.row_padding*(self.nrow-1) + \
                        self.col_labels + self.ax_label_pad*self.ncol
        self.fig_only = self.ax_h*self.nrow + \
                        self.ax_fig_ws + self.row_padding*(self.nrow-1) + \
                        self.ax_label_pad*self.ncol
        self.leg_overflow = max(self.leg_h-self.fig_only, 0)
        self.fig_w = self.fig_w_px/self.dpi
        self.fig_h = (self.fig_h_px+self.leg_overflow)/self.dpi

    def get_subplots_adjust(self):
        """
        Calculate the subplots_adjust parameters for the axes
        """
        
        self.left = self.fig_ax_ws/self.fig_w_px
        self.right = (self.fig_ax_ws + self.ax_w*self.ncol + \
                      self.col_padding*(self.ncol-1))/self.fig_w_px
        self.top = 1 - (self.fig_title_ws + self.title_h + \
                   self.title_ax_ws + self.col_labels)/self.fig_h_px
        self.bottom = (self.leg_overflow + self.ax_fig_ws)/self.fig_h_px

    def get_title_position(self):
        """
        Calculate the title position
        """

        self.title_bottom = 1+(self.title_ax_ws+self.col_labels+self.ax_label_pad*self.ncol)/self.ax_h
        self.title_top = self.title_bottom+(self.title_ax_ws+self.title_h)/self.ax_h
        
        self.title_h_px = self.title_h
        self.title_w_px = self.fig_w_px
        self.title_h = self.title_h/self.ax_h
        self.title_w = self.fig_w_px/self.ax_w
        self.title_left = 0 - self.fig_ax_ws/self.ax_w

    def see(self):
        """
        Prints a readable list of class attributes
        """
        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df
