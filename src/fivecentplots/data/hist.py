from . import data
import pdb
import pandas as pd
import numpy as np
import numpy.typing as npt
from .. import utilities
from natsort import natsorted
from typing import List, Union
utl = utilities
db = pdb.set_trace


class Histogram(data.Data):
    name = 'hist'
    req = []
    opt = ['x']
    url = 'hist.html'

    def __init__(self, fcpp: dict = {}, **kwargs):
        """Histogram-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            fcpp: theme-file kwargs
            kwargs: user-defined keyword args
        """
        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        # Hist supports image data and regular non-image data
        if 'imgs' in kwargs and isinstance(kwargs['imgs'], dict) and len(kwargs['imgs']) > 0 \
                or isinstance(kwargs['df'], np.ndarray) and len(kwargs['df'].shape) > 1:
            # Format input image data
            #   kwargs['df']:
            #       * Single image only:
            #           - 2D numpy array of pixel data
            #           - OR a 2D pd.DataFrame of pixel data
            #       * Multiple images:
            #           - pd.DataFrame with 1 row per image
            #           - row index value must match a key in the kwargs['imgs'] dict
            #           - other columns in this DataFrame are grouping columns
            #   kwargs['imgs']:
            #       * Single image only:
            #           - Not defined or used
            #       * Multiple images:
            #           - dict of the actual image data; dict key must match a row index value in kwargs['df']
            kwargs['df'], kwargs['imgs'] = utl.img_data_format(kwargs)
            self.hists = {}

            # Other parameters
            self.channels = kwargs['df'].iloc[0]['channels']
            # Image data defaults to a bin size of 1 digital number so the number of bins equals
            # img.max() - img.min() + 1, unless user overrides this parameter in the function call
            self.bins = utl.kwget(kwargs, self.fcpp, 'bins', kwargs.get('bins', 0))

            # Image data defaults to no bars
            self.bars = utl.kwget(kwargs, self.fcpp, 'bars', kwargs.get('bars', False))

            # Optional color plane splitting
            self.cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
            if self.cfa is not None and self.channels == 1:
                kwargs['df'], kwargs['imgs'] = utl.split_color_planes_wrapper(kwargs['df'], kwargs['imgs'], self.cfa)

            # TODO:: Reformat RGBA
            if self.channels > 1:
                # Need to reformat the group and image DataFrames
                groups = self._kwargs_groupers(kwargs)
                if 'Channel' in groups:
                    # Separate by color channel
                    imgs = {}
                    for ii, (k, v) in enumerate(kwargs['imgs'].items()):
                        # Separate the RGB columns into separate images
                        for icol, col in enumerate(['R', 'G', 'B']):
                            imgs[3 * ii + icol] = v[['Row', 'Column', col]]
                            imgs[3 * ii + icol].columns = ['Row', 'Column', 'Value']

                    # Update the grouping table and image DataFrame dict
                    kwargs['df'] = pd.merge(kwargs['df'], pd.DataFrame({'Channel': ['R', 'G', 'B']}), how='cross')
                else:
                    # Stack? need some kind of luma
                    db()

                # Update the grouping table and image DataFrame dict
                kwargs['df'] = pd.merge(kwargs['df'], pd.DataFrame({'Channel': ['R', 'G', 'B']}), how='cross')
                kwargs['imgs'] = imgs

        elif isinstance(kwargs['df'], pd.DataFrame):
            # Non-image data
            kwargs['imgs'] = None
            self.channels = -1
            self.bars = utl.kwget(kwargs, self.fcpp, 'bars', kwargs.get('bars', True))
            self.bins = utl.kwget(kwargs, self.fcpp, ['hist_bins', 'bins'], kwargs.get('bins', 20))

        else:
            raise TypeError('only a DataFrame or a 2D/3D numpy arrays of image data can be passed to fcp.hist')

        # Pre-super overrides
        kwargs['ax_limit_padding_ymax'] = kwargs.get('ax_limit_padding', 0.05)
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', 0)

        # Invalid options
        if kwargs.get('wrap') == 'y':
            raise data.GroupingError('Cannot wrap by "y" for hist plots')
        if kwargs.get('row') == 'y':
            raise data.GroupingError('Cannot subplot by rows="y" for hist plots')

        # Super
        super().__init__(self.name, self.req, self.opt, self.fcpp, **kwargs)

        # Set axes
        self.axs_on = ['x', 'y']
        if self.imgs:
            self.x = ['Value']
            self.y = ['Counts']

        # Other attributes for histogram
        self.auto_scale = False
        normalize = utl.kwget(kwargs, self.fcpp, ['hist_normalize', 'normalize'], kwargs.get('normalize', False))
        kde = utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'], kwargs.get('kde', False))
        if normalize or kde:
            self.norm = True
        else:
            self.norm = False
        self.cumulative = utl.kwget(kwargs, self.fcpp, ['hist_cumulative', 'cumulative'],
                                    kwargs.get('cumulative', False))

        # cdf/pdf option (if conflict, prefer cdf)
        self.cdf = utl.kwget(kwargs, self.fcpp, ['cdf'], kwargs.get('cdf', False))
        if not self.cdf:
            self.pdf = utl.kwget(kwargs, self.fcpp, ['pdf'], kwargs.get('pdf', False))
        if (self.cdf or self.pdf) and kwargs.get('preset') == 'HIST':
            self.ax_scale = 'lin'
        if self.cdf or self.pdf:
            bars = False

        # Toggle bars vs lines
        if not self.bars:
            self.switch_to_xy_plot(kwargs)

    def _calc_distribution(self, counts: npt.NDArray[int]) -> npt.NDArray[int]:
        """
        Compute cdf or pdf calculations

        Args:
            array of bin counts

        Return:
            cumsum of counts
        """
        if self.cdf:
            pdf = counts / sum(counts)
            counts = np.cumsum(pdf)
            self.y = ['Cumulative Probability']
        elif self.pdf:
            counts = counts / sum(counts)
            self.y = ['Probability Density']

        return counts

    def _calc_histograms(self, data: Union[pd.DataFrame, npt.NDArray]) -> List[npt.NDArray]:
        """Calculate the histogram data for one data set.

        Args:
            some input DataFrame

        Returns:
            list of histogram counts and bin values
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Remove nans
        data = data[~np.isnan(data)]

        # Calculate the histogram counts
        if self.bins == 0 and data.dtype not in [float, np.float16, np.float32, np.float64]:
            # If no bins defined, assume a bin size of 1 unit and use np.bincount for better speed
            data = data.astype(int)
            offset = data.min() if data.min() < 0 else 0  # bincount requires positives only
            vals = np.arange(offset, data.max() + 1)
            counts = np.bincount(data - offset)

        else:
            # Case of image data but invalid dtype
            if self.bins == 0:
                self.bins = int(data.max() - data.min() + 1)

            # If bins specified or data contains floats, use np.histogram (slower)
            brange = data.min(), data.max()
            counts, vals = np.histogram(data, bins=self.bins, density=self.norm, range=brange)

        # Clean up
        if len(vals) != len(counts):
            vals = vals[:-1]
        counts = counts[(vals >= data.min()) & (vals <= data.max())]
        vals = vals[(vals >= data.min()) & (vals <= data.max())]

        # Additional manipulations for xy plots
        if self.name == 'xy':
            # Special case of all values being equal
            if len(counts) == 1:
                vals = np.insert(vals, 0, vals[0])
                if self.ax_scale in ['logx', 'log']:
                    counts = np.insert(counts, 0, 1)
                else:
                    counts = np.insert(counts, 0, 0)

            # Eliminate repeating count values of zero to reduce data points and speed up processing/plotting
            mask = (counts[:-2]==0) & (counts[2:]==0) & (counts[1:len(counts) - 1]==0)
            counts = np.concatenate((counts[:1], counts[1:-1][~mask], counts[-1:]))
            vals = np.concatenate((vals[:1], vals[1:-1][~mask], vals[-1:]))

            # Remove leading zero bin
            if counts[0] == 0 and vals[0] == 0 and self.bins == 0:
                counts = counts[1:]
                vals = vals[1:]

            # Optionally sub-sample the data, preserving the last discrete bin because it often contains valuable
            # image bin data
            if self.sample > 1:
                vals_last_bin = vals[-self.sample:]
                counts_last_bin = counts[-self.sample:]
                vals = np.concatenate((vals[:-self.sample:self.sample], vals_last_bin))
                counts = np.concatenate((counts[:-self.sample:self.sample], counts_last_bin))

        # cdf + pdf
        counts = self._calc_distribution(counts)

        return counts, vals

    def _get_data_ranges(self):
        """Histogram-specific data range calculator by subplot."""
        # If switch_to_xy_plot applied, just use the parent range function
        if self.name == 'xy' and self.imgs is None:
            data.Data._get_data_ranges(self)

        elif self.imgs is not None:
            # If xy plot but data are images, use custom range calculations to avoid building a massive and
            # slow DataFrame and just use the histogram vals/counts arrays
            data.Data._get_data_ranges_user_defined(self)

            # Apply shared axes
            for ir, ic, plot_num in self.get_subplot_index():
                for ax in self.axs:
                    if ax == 'x':
                        idx = 1
                    elif ax == 'y':
                        idx = 0

                    # Share axes
                    if getattr(self, 'share_%s' % ax) and (ir > 0 or ic > 0):
                        self._add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                        self._add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])
                    elif getattr(self, 'share_%s' % ax):
                        df_fig = np.concatenate([self.hists[f][idx] for f in self.df_fig.index])
                        vals = self._get_data_range(ax, df_fig, plot_num)
                        self._add_range(ir, ic, ax, 'min', vals[0])
                        self._add_range(ir, ic, ax, 'max', vals[1])

                    # Share row
                    elif self.share_row and self.row is not None and ic > 0 and self.row != 'y':
                        self._add_range(ir, ic, ax, 'min', self.ranges[ir, 0]['%smin' % ax])
                        self._add_range(ir, ic, ax, 'max', self.ranges[ir, 0]['%smax' % ax])

                    elif self.share_row and self.row is not None:
                        df_fig = df_fig[self.df_fig[self.row[0]] == self.row_vals[ir]]
                        df_fig =np.concatenate([self.hists[f][idx] for f in df_fig.index])
                        vals = self._get_data_range(ax, df_fig, plot_num)
                        self._add_range(ir, ic, ax, 'min', vals[0])
                        self._add_range(ir, ic, ax, 'max', vals[1])

                    # Share col
                    elif self.share_col and self.col is not None and ir > 0 and self.col != 'x':
                        self._add_range(ir, ic, ax, 'min', self.ranges[0, ic]['%smin' % ax])
                        self._add_range(ir, ic, ax, 'max', self.ranges[0, ic]['%smax' % ax])
                    elif self.share_col and self.col is not None:
                        df_fig = df_fig[df_fig[self.cols[0] == self.col_vals[ic]]]
                        df_fig = np.concatenate([self.hists[f][idx] for f in df_fig.index])
                        vals = self._get_data_range(ax, df_fig[df_fig[self.col[0]] == self.col_vals[ic]], plot_num)
                        self._add_range(ir, ic, ax, 'min', vals[0])
                        self._add_range(ir, ic, ax, 'max', vals[1])

                    # subplot level when not shared
                    else:
                        df_rc = self._subset(ir, ic)

                        # Empty rc
                        if len(df_rc) == 0:  # this doesn't exist yet!
                            self._add_range(ir, ic, ax, 'min', None)
                            self._add_range(ir, ic, ax, 'max', None)
                            continue

                        # Not shared or wrap by x or y
                        elif not getattr(self, 'share_%s' % ax) or \
                                (self.wrap is not None and self.wrap == 'y' or self.wrap == 'x'):
                            df_rc = np.concatenate([self.hists[f][idx] for f in df_rc.index])
                            vals = self._get_data_range(ax, df_rc, plot_num)
                            self._add_range(ir, ic, ax, 'min', vals[0])
                            self._add_range(ir, ic, ax, 'max', vals[1])

        else:
            # Normal histogram func
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
            for ir, ic, plot_num in self.get_subplot_index():
                df_rc = self._subset(ir, ic)

                if len(df_rc) == 0:
                    temp_ranges[ir, ic]['ymin'] = None
                    temp_ranges[ir, ic]['ymin'] = None
                    continue

                hist = self._iterate_hists(df_rc)
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
            for ir, ic, plot_num in self.get_subplot_index():
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

    def get_plot_data(self, df: pd.DataFrame):
        """Generator to subset into discrete sets of data for each curve.

        Args:
            df: data subset to plot

        Yields:
            iline: legend index
            df: data subset to plot
            row['x'] [x]: x-axis column name
            row['y'] [y]: y-axis column name
            self.z [z]: z-column name
            leg [leg_name]: legend value name if legend enabled
            twin: denotes if twin axis is enabled or not
            len(vals) [ngroups]: total number of groups in the full data
        """
        if not self.imgs:
            yield from data.Data.get_plot_data(self, df)

        else:
            if not isinstance(self.legend_vals, pd.DataFrame):
                # Make the subset
                df_hist = []
                for idx in df.index:
                    df_hist += [pd.DataFrame({self.x[0]: self.hists[idx][1], self.y[0]: self.hists[idx][0]})]
                df_hist = pd.concat(df_hist)

                yield 0, df_hist, self.x[0], self.y[0], None, None, False, 1

            else:
                for iline, row in self.legend_vals.iterrows():
                    # Subset by legend value
                    df2 = df[df[self.legend] == row['Leg']].copy()
                    if len(df2) == 0:
                        continue

                    # Make the subset
                    df_hist = []
                    for idx in df2.index:
                        df_hist += [pd.DataFrame({row['x']: self.hists[idx][1], row['y']: self.hists[idx][0]})]
                    df_hist = pd.concat(df_hist)

                    yield iline, df_hist, row['x'], row['y'], None, row['names'], False, len(self.legend_vals)

    def _get_rc_groupings(self, df: pd.DataFrame):
        """Determine the row, column, or wrap grid groupings.

        Args:
            df: subset
        """
        # Simplify the df for speed by removing non-grouping columns
        non_group_cols = [self.x[0]]
        if self.y:
            non_group_cols += [self.y[0]]
        group_cols = [f for f in df.columns if f not in non_group_cols]
        df = df[group_cols].drop_duplicates()

        data.Data._get_rc_groupings(self, df)

    def _iterate_hists(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Iterate over groups and build a data set of counts and values.

        Args:
            df_in: input DataFrame (for images, this is the grouping table)

        Returns:
            for non-images: new DataFrame with histogram counts and values
            for images: updates to the self.hists dict with counts and values arrays
        """
        if self.imgs:
            # Iterate over df indices and compute a histogram for each corresponding image
            for irow, row in df_in.iterrows():
                self.hists[irow] = self._calc_histograms(self.imgs[irow])
            return df_in

        else:
            # Non-image data
            groups = self._groupers
            if len(groups) > 0:
                # Grouping column(s) present
                hist = []
                for nn, df in df_in.groupby(self._groupers, sort=self.sort):
                    counts, vals = self._calc_histograms(df[self.x[0]])

                    # Add grouping columns
                    temp = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                    for ig, group in enumerate(self._groupers):
                        if isinstance(nn, tuple):
                            temp[group] = nn[ig]
                        else:
                            temp[group] = nn
                    hist += [temp]

                return pd.concat(hist)

            else:
                # No grouping columns
                counts, vals = self._calc_histograms(df_in[self.x[0]])
                return pd.DataFrame({self.x[0]: vals, self.y[0]: counts})

    def _kwargs_groupers(self, kwargs) -> list:
        """Get all grouping values from kwargs"""
        props = ['row', 'cols', 'wrap', 'groups', 'legend', 'fig']
        grouper = []

        for prop in props:
            if kwargs.get(prop, None) not in ['x', 'y', None]:
                grouper += utl.validate_list(kwargs.get(prop))

        return list(set(grouper))

    # def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
    #     """Wrapper to handle image data.

    #     Args:
    #         ir: subplot row index
    #         ic: subplot column index
    #         df: df_groups subset based on self._subset

    #     Returns:
    #         updated DataFrame
    #     """
    #     if len(df.index) == 0:
    #         return df

    #     return data.Data._subset_modify(self, ir, ic, df)

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
            wrap = dict(zip(self.wrap, utl.validate_list(self.wrap_vals[ir * self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

    def switch_to_xy_plot(self, kwargs):
        """If bars are not enabled, switch everything to line plot.

        Args:
            kwargs: user-defined keyword args
        """
        self.name = 'xy'
        if not self.y:
            self.y = ['Counts']

        # Convert the image data to a histogram
        self._get_legend_groupings(self.df_all)
        self.df_all = self._iterate_hists(self.df_all)
