from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
utl = utilities
db = pdb.set_trace


class Histogram(data.Data):
    def __init__(self, **kwargs):

        name = 'hist'
        req = ['x']
        opt = []

        # overrides
        kwargs['ax_limit_padding_ymax'] = kwargs.get('ax_limit_padding', 0.05)
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', 0)
        
        super().__init__(name, req, opt, **kwargs)

        # overrides post
        self.auto_scale = False
        self.ax_limit_padding_ymax = kwargs['ax_limit_padding_ymax']
        self.stacked = utl.kwget(kwargs, self.fcpp, 
                                 ['hist_stacked', 'stacked'],
                                 kwargs.get('stacked', False))
        
    def df_hist(self, df_in):
        """
        Iterate over groups and build a dataframe of counts
        """

        hist = pd.DataFrame()

        for iline, df, x, y, z, leg_name, twin, ngroups in self.get_plot_data(df_in):
            counts = np.histogram(df[self.x[0]].dropna(), bins=self.bins, 
                                  normed=self.norm)[0]
            hist = pd.concat([hist, pd.DataFrame({self.y[0]: counts})])

        return hist
    
    def get_data_ranges(self):

        # Handle all but y-axis which needs histogram binning
        self.axs = [f for f in self.axs if f != 'y']
        self._get_data_ranges()
        self.axs += ['y']
        
        # set ranges by subset
        self.y = ['Counts']
        temp_ranges = self.range_dict()
        max_y = 0
        max_y_row = np.zeros(self.nrow)
        max_y_col = np.zeros(self.ncol)
        min_y = 0
        min_y_row = np.zeros(self.nrow)
        min_y_col = np.zeros(self.ncol)
        
        # iterate through all rc_subsets in order to compute histogram counts
        for ir, ic, plot_num in self.get_subplot_index():
            df_rc = self.subset(ir, ic)
            
            if len(df_rc) == 0:
                temp_ranges[ir, ic]['ymin'] = None
                temp_ranges[ir, ic]['ymin'] = None
                continue

            hist = self.df_hist(df_rc)
            vals = self.get_data_range('y', hist, plot_num)
            temp_ranges[ir, ic]['ymin'] = vals[0]
            temp_ranges[ir, ic]['ymin'] = vals[1]
            min_y = min(min_y, vals[0])
            min_y_row[ir] = min(min_y_row[ir], vals[0])
            min_y_col[ic] = min(min_y_col[ic], vals[0])
            max_y = max(max_y, vals[1])
            max_y_row[ir] = min(max_y_col[ir], vals[1])
            max_y_col[ic] = min(max_y_col[ir], vals[1])

        # compute actual ranges with option y-axis sharing    
        for ir, ic, plot_num in self.get_subplot_index():
            # share y
            if self.share_y:
                self.add_range(ir, ic, 'y', 'min', min_y)
                self.add_range(ir, ic, 'y', 'max', max_y)
            
            # share row
            elif self.share_row:
                self.add_range(ir, ic, 'y', 'min', min_y_row[ir])
                self.add_range(ir, ic, 'y', 'max', max_y_row[ir])

            # share col
            elif self.share_col:
                self.add_range(ir, ic, 'y', 'min', min_y_col[ic])
                self.add_range(ir, ic, 'y', 'max', max_y_col[ic])
            
            # not share y
            else:
                self.add_range(ir, ic, 'y', 'min', temp_ranges[ir][ic]['ymin'])
                self.add_range(ir, ic, 'y', 'max', temp_ranges[ir][ic]['ymax'])

        self.y = None
        
    def subset_modify(self, df, ir, ic):

        return self._subset_modify(df, ir, ic)

    def subset_wrap(self, ir, ic):
        """
        For wrap plots, select the revelant subset from self.df_fig

        Need one additional line of code compared with data.subset_wrap
        """

        if ir * self.ncol + ic > self.nwrap-1:
            return pd.DataFrame()
        elif self.wrap == 'y':
            # can we drop these validate calls for speed
            self.y = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend) if self.legend not in [None, True, False] else [])
            return self.df_fig[cols]
        elif self.wrap == 'x':
            self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend) if self.legend is not None else [])
            # need this extra line for hist
            cols = [f for f in cols if f != 'Counts']
            return self.df_fig[cols]
        else:
            if self.sort:
                self.wrap_vals = \
                    natsorted(list(self.df_fig.groupby(self.wrap).groups.keys()))
            else:
                self.wrap_vals = list(self.df_fig.groupby(self.wrap, sort=False).groups.keys())
            wrap = dict(zip(self.wrap,
                        utl.validate_list(self.wrap_vals[ir*self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

    