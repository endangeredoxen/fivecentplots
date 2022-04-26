from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Histogram(data.Data):
    def __init__(self, **kwargs):

        name = 'hist'
        req = []
        opt = ['x']
        kwargs['df'] = utl.df_from_array2d(kwargs['df'])

        self.fcpp, dummy, dummy2 = utl.reload_defaults(
            kwargs.get('theme', None))
        bars = utl.kwget(kwargs, self.fcpp, 'bars', kwargs.get('bars', True))
        kwargs['2D'] = False

        # 2D image input
        if not kwargs.get('x', None):
            # Color plane splitting
            cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
            if cfa is not None:
                kwargs['df'] = utl.split_color_planes(kwargs['df'], cfa)
            kwargs['2D'] = True
            bars = utl.kwget(kwargs, self.fcpp, 'bars',
                             kwargs.get('bars', False))

        # overrides
        kwargs['ax_limit_padding_ymax'] = kwargs.get('ax_limit_padding', 0.05)
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', 0)

        super().__init__(name, req, opt, **kwargs)

        # cdf/pdf option (if conflict, prefer cdf)
        self.cdf = utl.kwget(kwargs, self.fcpp, ['cdf'], kwargs.get('cdf', False))
        self.pdf = utl.kwget(kwargs, self.fcpp, ['pdf'], kwargs.get('pdf', False))
        if self.cdf and kwargs.get('preset') == 'HIST':
            self.ax_scale = 'lin'

        # Toggle bars vs lines
        if not bars:
            self.switch_type(kwargs)

        # overrides post
        self.auto_scale = False
        self.ax_limit_padding_ymax = kwargs['ax_limit_padding_ymax']
        self.stacked = utl.kwget(kwargs, self.fcpp,
                                 ['hist_stacked', 'stacked'],
                                 kwargs.get('stacked', False))

    def df_hist(self, df_in, brange=None):
        """
        Iterate over groups and build a dataframe of counts
        """

        hist = pd.DataFrame()

        groups = self.groupers

        if len(groups) > 0:
            for nn, df in df_in.groupby(self.groupers):
                if self.kwargs['2D']:
                    dfx = df[utl.df_int_cols(df)].values
                    self.x = ['Value']
                else:
                    dfx = df[self.x[0]]

                if brange:
                    counts, vals = np.histogram(dfx[~np.isnan(dfx)],
                                                bins=self.bins,
                                                normed=self.norm, range=brange)
                else:
                    counts, vals = np.histogram(dfx[~np.isnan(dfx)],
                                                bins=self.bins,
                                                normed=self.norm)

                temp = pd.DataFrame({self.x[0]: vals[:-1], self.y[0]: counts})
                for ig, group in enumerate(self.groupers):
                    if type(nn) is tuple:
                        temp[group] = nn[ig]
                    else:
                        temp[group] = nn
                hist = pd.concat([hist, temp])
        else:
            if self.kwargs['2D']:
                dfx = df_in[utl.df_int_cols(df_in)].values
                self.x = ['Value']
            else:
                dfx = df_in[self.x[0]].dropna()
            if brange:
                counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins,
                                            normed=self.norm, range=brange)
            else:
                counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins,
                                            normed=self.norm)
            if self.cdf:
                pdf = counts / sum(counts)
                counts = np.cumsum(pdf)
                self.y = ['Cumulative Probability']
            elif self.pdf:
                counts = counts / sum(counts)
                self.y = ['Probability Density']
            hist = pd.DataFrame({self.x[0]: vals[:-1], self.y[0]: counts})

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

        # self.y = None

        if hasattr(self, 'horizontal') and self.horizontal:
            self.swap_xy_ranges()

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
                   (utl.validate_list(self.legend)
                    if self.legend not in [None, True, False] else [])
            return self.df_fig[cols]
        elif self.wrap == 'x':
            self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend)
                    if self.legend is not None else [])
            # need this extra line for hist
            cols = [f for f in cols if f != 'Counts']
            return self.df_fig[cols]
        else:
            if self.sort:
                self.wrap_vals = \
                    natsorted(
                        list(self.df_fig.groupby(self.wrap).groups.keys()))
            else:
                self.wrap_vals = list(self.df_fig.groupby(
                    self.wrap, sort=False).groups.keys())
            wrap = dict(zip(self.wrap,
                        utl.validate_list(self.wrap_vals[ir*self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

    def switch_type(self, kwargs):
        """
        If bars are not enabled, switch everything to line plot
        """

        self.name = 'xy'
        self.y = ['Counts']
        self.get_data_ranges = self._get_data_ranges
        self.subset_wrap = self._subset_wrap

        # Update the bins to integer values if not specified and 2D image data
        bins = utl.kwget(kwargs, self.fcpp, 'bins', kwargs.get('bars', None))
        if kwargs['2D']:
            cols = utl.df_int_cols(self.df_all)
            vals = self.df_all[cols].values
            vmin = int(np.nanmin(vals))
            vmax = int(np.nanmax(vals))
            if not bins:
                self.bins = vmax - vmin + 1  # add 1 to get the last bin
        elif not kwargs.get('vmin') and not kwargs.get('vmax'):
            vmin = int(np.nanmin(self.df_all.Value))
            vmax = int(np.nanmax(self.df_all.Value))

        # Convert the image data to a histogram
        temp = self.legend
        self.get_legend_groupings(self.df_all)
        self.df_all = self.df_hist(self.df_all, [vmin, vmax + 1])
        self.legend = temp  # reset the original legend param
