from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Histogram(data.Data):
    def __init__(self, fcpp: dict = {}, **kwargs):
        """Histogram-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            fcpp: theme-file kwargs
            kwargs: user-defined keyword args
        """
        name = 'hist'
        req = []
        opt = ['x']
        kwargs['df'] = utl.df_from_array2d(kwargs['df'])

        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        # Replace certain kwargs
        bars = utl.kwget(kwargs, self.fcpp, 'bars', kwargs.get('bars', True))
        kwargs['2D'] = False

        # 2D image input
        if not kwargs.get('x', None):
            # Color plane splitting
            cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
            if cfa is not None:
                kwargs['df'] = utl.split_color_planes(kwargs['df'], cfa)
            kwargs['2D'] = True
            bars = utl.kwget(kwargs, self.fcpp, 'bars', kwargs.get('bars', False))

        # overrides
        kwargs['ax_limit_padding_ymax'] = kwargs.get('ax_limit_padding', 0.05)
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', 0)

        # invalid options
        if 'wrap' in kwargs and kwargs['wrap'] == 'y':
            raise data.GroupingError('Cannot wrap by "y" for hist plots')

        super().__init__(name, req, opt, self.fcpp, **kwargs)

        self.use_parent_ranges = False

        # cdf/pdf option (if conflict, prefer cdf)
        self.cdf = utl.kwget(kwargs, self.fcpp, ['cdf'], kwargs.get('cdf', False))
        self.pdf = utl.kwget(kwargs, self.fcpp, ['pdf'], kwargs.get('pdf', False))
        if self.cdf and kwargs.get('preset') == 'HIST':
            self.ax_scale = 'lin'
        if self.cdf or self.pdf:
            bars = False

        # Other options
        self.cumulative = utl.kwget(kwargs, self.fcpp, ['hist_cumulative', 'cumulative'],
                                    kwargs.get('cumulative', False))

        # Toggle bars vs lines
        if not bars:
            self.switch_type(kwargs)

        # overrides post
        self.auto_scale = False
        self.ax_limit_padding_ymax = kwargs['ax_limit_padding_ymax']

    def df_hist(self, df_in: pd.DataFrame, brange: [float, None] = None) -> pd.DataFrame:
        """Iterate over groups and build a dataframe of counts.

        Args:
            df_in: input DataFrame
            brange (optional): range for histogram calculation

        Returns:
            new DataFrame with histogram counts and values
        """
        hist = pd.DataFrame()

        groups = self._groupers
        if len(groups) > 0:
            for nn, df in df_in.groupby(self._groupers):
                if self.kwargs['2D']:
                    dfx = df[utl.df_int_cols(df)].values
                    self.x = ['Value']
                else:
                    dfx = df[self.x[0]]

                if brange:
                    counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins, density=self.norm, range=brange)
                else:
                    counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins, density=self.norm)

                # cdf + pdf
                if self.cdf:
                    pdf = counts / sum(counts)
                    counts = np.cumsum(pdf)
                    self.y = ['Cumulative Probability']
                elif self.pdf:
                    counts = counts / sum(counts)
                    self.y = ['Probability Density']

                temp = pd.DataFrame({self.x[0]: vals[:-1], self.y[0]: counts})
                for ig, group in enumerate(self._groupers):
                    if isinstance(nn, tuple):
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
                counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins, density=self.norm, range=brange)
            else:
                counts, vals = np.histogram(dfx[~np.isnan(dfx)], bins=self.bins, density=self.norm)

            # special case of all values being equal
            if len(counts) == 1:
                vals = np.insert(vals, 0, vals[0])
                if self.ax_scale in ['logy', 'log']:
                    counts = np.insert(counts, 0, 1)
                else:
                    counts = np.insert(counts, 0, 0)

            # cdf + pdf
            if self.cdf:
                pdf = counts / sum(counts)
                counts = np.cumsum(pdf)
                self.y = ['Cumulative Probability']
            elif self.pdf:
                counts = counts / sum(counts)
                self.y = ['Probability Density']

            hist = pd.DataFrame({self.x[0]: vals[:-1], self.y[0]: counts})

        return hist

    def _get_data_ranges(self):
        """Histogram-specific data range calculator by subplot."""
        # If switch_type applied, just use the parent range function
        if self.use_parent_ranges:
            data.Data._get_data_ranges(self)
            return

        # Handle all but y-axis which needs histogram binning
        self.axs = [f for f in self.axs if f != 'y']
        data.Data._get_data_ranges(self)  # call original parent function
        self.axs += ['y']

        # set ranges by subset
        self.y = ['Counts']
        temp_ranges = self._range_dict()
        max_y = 0
        max_y_row = np.zeros(self.nrow)
        max_y_col = np.zeros(self.ncol)
        min_y = 0
        min_y_row = np.zeros(self.nrow)
        min_y_col = np.zeros(self.ncol)

        # iterate through all rc_subsets in order to compute histogram counts
        for ir, ic, plot_num in self._get_subplot_index():
            df_rc = self._subset(ir, ic)

            if len(df_rc) == 0:
                temp_ranges[ir, ic]['ymin'] = None
                temp_ranges[ir, ic]['ymin'] = None
                continue

            hist = self.df_hist(df_rc)
            if self.cumulative:
                hist.Counts = hist.Counts.sum()
            vals = self._get_data_range('y', hist, plot_num)
            temp_ranges[ir, ic]['ymin'] = vals[0]
            temp_ranges[ir, ic]['ymax'] = vals[1]
            min_y = min(min_y, vals[0])
            min_y_row[ir] = min(min_y_row[ir], vals[0])
            min_y_col[ic] = min(min_y_col[ic], vals[0])
            max_y = max(max_y, vals[1])
            max_y_row[ir] = max(max_y_row[ir], vals[1])
            max_y_col[ic] = max(max_y_col[ic], vals[1])

        # compute actual ranges with option y-axis sharing
        for ir, ic, plot_num in self._get_subplot_index():
            # share y
            if self.share_y:
                self._add_range(ir, ic, 'y', 'min', min_y)
                self._add_range(ir, ic, 'y', 'max', max_y)

            # share row
            elif self.share_row:
                self._add_range(ir, ic, 'y', 'min', min_y_row[ir])
                self._add_range(ir, ic, 'y', 'max', max_y_row[ir])

            # share col
            elif self.share_col:
                self._add_range(ir, ic, 'y', 'min', min_y_col[ic])
                self._add_range(ir, ic, 'y', 'max', max_y_col[ic])

            # not share y
            else:
                if 'ymin' in temp_ranges[ir, ic]:
                    self._add_range(ir, ic, 'y', 'min', temp_ranges[ir][ic]['ymin'])
                if 'ymax' in temp_ranges[ir, ic]:
                    self._add_range(ir, ic, 'y', 'max', temp_ranges[ir][ic]['ymax'])

        if hasattr(self, 'horizontal') and self.horizontal:
            self.swap_xy_ranges()

    def _subset_wrap(self, ir: int, ic: int) -> pd.DataFrame:
        """Histogram-specific version of subset_wrap.  Select the revelant subset
        from self.df_fig with one additional line of code compared with parent func

        Args:
            ir: subplot row index
            ic: subplot column index

        Returns:
            self.df_fig DataFrame subset based on self.wrap value
        """
        if ir * self.ncol + ic > self.nwrap - 1:
            return pd.DataFrame()
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
                    natsorted(list(self.df_fig.groupby(self.wrap).groups.keys()))
            else:
                self.wrap_vals = list(self.df_fig.groupby(self.wrap, sort=False).groups.keys())
            wrap = dict(zip(self.wrap,
                        utl.validate_list(self.wrap_vals[ir * self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

    def switch_type(self, kwargs):
        """If bars are not enabled, switch everything to line plot.

        Args:
            kwargs: user-defined keyword args
        """
        self.name = 'xy'
        self.y = ['Counts']
        self.use_parent_ranges = True
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
        self._get_legend_groupings(self.df_all)
        self.df_all = self.df_hist(self.df_all, [vmin, vmax + 1])
        self.legend = temp  # reset the original legend param
