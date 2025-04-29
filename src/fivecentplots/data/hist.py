from . import data
import pdb
import pandas as pd
import numpy as np
import numpy.typing as npt
from .. import utilities
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
        self.timer = kwargs['timer']

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

            # Reformat RGBA
            if self.channels > 1:
                # Need to reformat the group and image DataFrames
                groups = self._kwargs_groupers(kwargs)
                if 'Channel' in groups:
                    imgs = {}
                    for ii, (k, v) in enumerate(kwargs['imgs'].items()):
                        # Separate the RGB columns into separate images
                        for icol, col in enumerate(['R', 'G', 'B']):
                            imgs[3 * ii + icol] = v[icol]

                    # Update the grouping table and image DataFrame dict
                    kwargs['imgs'] = imgs
                    kwargs['df'] = pd.merge(kwargs['df'], pd.DataFrame({'Channel': ['R', 'G', 'B']}), how='cross')
                    kwargs['df']['channels'] = 1
                else:
                    # Use luminosity histogram of grayscale
                    for k, v in kwargs['imgs'].items():
                        kwargs['imgs'][k] = utl.img_grayscale(v, bit_depth=8)

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
        self.auto_scale = True
        normalize = utl.kwget(kwargs, self.fcpp, ['hist_normalize', 'normalize'], kwargs.get('normalize', False))
        self.branges = None
        kde = utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'], kwargs.get('kde', False))
        if normalize or kde:
            self.norm = True
        else:
            self.norm = False
        self.cumulative = utl.kwget(kwargs, self.fcpp, ['hist_cumulative', 'cumulative'],
                                    kwargs.get('cumulative', False))
        self.horizontal = utl.kwget(kwargs, self.fcpp, ['hist_horizontal', 'horizontal'],
                                    kwargs.get('horizontal', False))
        self.warning_speed = False

        # cdf/pdf option (if conflict, prefer cdf)
        self.cdf = utl.kwget(kwargs, self.fcpp, ['cdf'], kwargs.get('cdf', False))
        if not self.cdf:
            self.pdf = utl.kwget(kwargs, self.fcpp, ['pdf'], kwargs.get('pdf', False))
        else:
            self.pdf = False
        if (self.cdf or self.pdf) and kwargs.get('preset') == 'HIST':
            self.ax_scale = 'lin'

        # Toggle bars vs lines
        if not self.bars or self.cdf or self.pdf:
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
        elif self.pdf:
            counts = counts / sum(counts)

        return counts

    def _calc_histograms(self, ir: int, ic: int, data_set: Union[pd.DataFrame, npt.NDArray]) -> List[npt.NDArray]:
        """Calculate the histogram data for one data set.

        Args:
            ir: current axes row index
            ic: current axes column index
            data_set: data subset

        Returns:
            list of histogram counts and bin values
        """
        if self.branges is None:
            self.branges = np.array([[None] * self.ncol] * self.nrow)

        if isinstance(data_set, pd.DataFrame):
            data_set = data_set.values

        # Remove nans
        data_set = data_set[~np.isnan(data_set)]

        # Calculate the histogram counts
        if self.bins == 0 and data_set.dtype not in [float, np.float16, np.float32, np.float64]:
            # If no bins defined, assume a bin size of 1 unit and use np.bincount for better speed
            data_set = data_set.astype(int)
            offset = data_set.min() if data_set.min() < 0 else 0  # bincount requires positives only
            vals = np.arange(offset, data_set.max() + 1)
            counts = np.bincount(data_set - offset)

        else:
            if self.imgs is not None and not self.warning_speed:
                print('Warning: histograms of image data with float values is slow; consider using ints instead')
                self.warning_speed = True  # to not print this again

            # Case of image data but invalid dtype
            if self.bins == 0:
                self.bins = int(data_set.max() - data_set.min() + 1)

            # If bins specified or data contains floats, use np.histogram (slower)
            # Step 1: get the x-axis range in which to apply self.bins number of bins; wherever the x-axis range
            #   is shared, this should be the same range.  Normally, shared ranges are not computed until after
            #   the plot is created (i.e., after counts are calculated), so we have to check shared x-axis cases
            #   here instead of waiting for data.get_data_ranges
            if self.share_x:
                if self.imgs is None:
                    brange = self.df_all[self.x[0]].min(), self.df_all[self.x[0]].max()
                else:
                    brange = min([self.imgs[f].min() for f in self.df_all.index]), \
                             max([self.imgs[f].max() for f in self.df_all.index])
            elif self.share_row and self.row is not None:
                df_row = self.df_all.loc[self.df_all[self.row[0]] == self.row_vals[ir]]
                if self.imgs is None:
                    brange = df_row[self.x[0]].min(), df_row[self.x[0]].max()
                else:
                    brange = min([self.imgs[f].min() for f in df_row.index]), \
                             max([self.imgs[f].max() for f in df_row.index])
            elif self.share_col and self.col is not None:
                df_col = self.df_all.loc[self.df_all[self.col[0]] == self.col_vals[ic]]
                if self.imgs is None:
                    brange = df_col[self.x[0]].min(), df_col[self.x[0]].max()
                else:
                    brange = min([self.imgs[f].min() for f in df_col.index]), \
                             max([self.imgs[f].max() for f in df_col.index])
            else:
                # No sharing, compute for this subset only
                brange = data_set.min(), data_set.max()
            self.branges[ir, ic] = brange

            # Step 2: compute the counts
            counts, vals = np.histogram(data_set, bins=self.bins, density=self.norm, range=brange)

        # Clean up
        if len(vals) != len(counts):
            vals = vals[:-1]

        # Additional manipulations for image histograms
        if self.imgs is not None:
            counts = counts[(vals >= data_set.min()) & (vals <= data_set.max())]
            vals = vals[(vals >= data_set.min()) & (vals <= data_set.max())]

            # Special case of all values being equal
            if len(counts) == 1:
                vals = np.insert(vals, 0, vals[0])
                if self.ax_scale in ['logy', 'log', 'semilogy', 'loglog']:
                    counts = np.insert(counts, 0, 1)
                else:
                    counts = np.insert(counts, 0, 0)

            # Eliminate repeating count values of zero to reduce data points and speed up processing/plotting
            mask = (counts[:-2] == 0) & (counts[2:] == 0) & (counts[1:len(counts) - 1] == 0)
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

        # cdf or pdf
        if self.cdf:
            counts = self._calc_distribution(counts)

            # Fill to max x-range
            if self.imgs is None:
                xmax = self.df_all[self.x[0]].max()
            else:
                xmax = 0
                xmax = max([f.max() for f in self.imgs.values()])
            vals = np.append(vals, xmax)
            counts = np.append(counts, 1)

        return counts, vals

    def _range_overrides(self, ir: int, ic: int, df_rc: pd.DataFrame):
        """
        For non-image histograms (i.e., histograms that are created by a `hist` plotting function),
        need to compute y-column "Counts" for the df_rc
        """
        if self.imgs is not None or self.name == 'xy':
            # Fix y-axis for single-bin case
            if len(df_rc) == 2 and df_rc.Counts[0] in [0, 1] and self.ymin.values == [None]:
                self.ranges['ymin'][ir, ic] = df_rc.Counts[0]
            return

        plot_num = utl.plot_num(ir, ic, self.ncol) - 1

        groups = self._groupers
        if len(groups) > 0:
            for ii, (nn, sub) in enumerate(df_rc.groupby(self._groupers)):
                counts, vals = self._calc_histograms(ir, ic, sub[self.x[0]])
                if len(counts) == 1:
                    # Fix to get a valid range with only one data point
                    counts = np.array([0, counts[0]])
                    vals = np.array([0, vals[0]])
                if self.cumulative:
                    counts = counts.cumsum()
                df_hist = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                if ii == 0:
                    ymin, ymax = self._get_data_range('y', df_hist, plot_num)
                else:
                    ymin_, ymax_ = self._get_data_range('y', df_hist, plot_num)
                    ymin = min(ymin, ymin_)
                    ymax = max(ymax, ymax_)

        else:
            counts, vals = self._calc_histograms(ir, ic, df_rc[self.x[0]])
            if self.cumulative:
                counts = counts.cumsum()
            df_hist = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
            ymin, ymax = self._get_data_range('y', df_hist, plot_num)

        # Pass the min/max values through _get_data_range to get ax
        if self.ymin[plot_num] is None:
            ymin = min(0, ymin)
        self.ranges['ymin'][ir, ic], self.ranges['ymax'][ir, ic] = ymin, ymax

        # Flip horizontal
        if self.horizontal:
            ymin = self.ranges['ymin'][ir, ic]
            ymax = self.ranges['ymax'][ir, ic]
            self.ranges['ymin'][ir, ic] = self.ranges['xmin'][ir, ic]
            self.ranges['ymax'][ir, ic] = self.ranges['xmax'][ir, ic]
            self.ranges['xmin'][ir, ic] = ymin
            self.ranges['xmax'][ir, ic] = ymax

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df

        if self.imgs is None and self.name == 'xy':
            counts, vals = self._calc_histograms(ir, ic, df[self.x[0]])
            df_sub = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
            for group in self._groupers:
                df_sub[group] = df[group].iloc[0]

            return df_sub

        elif self.imgs is None:
            return data.Data._subset_modify(self, ir, ic, df)

        else:
            if self.legend is None:
                subset_dict = {key: value for key, value in self.imgs.items() if key in list(df.index)}
                if len(subset_dict) > 0:
                    counts, vals = self._calc_histograms(ir, ic, np.concatenate(list(subset_dict.values()), 1))
                    df_sub = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                    return df_sub
                else:
                    return pd.DataFrame()  # is this enough?

            else:
                df_sub = []
                for iline, row in self.legend_vals.iterrows():
                    idx = list(df.loc[df[self.legend] == row.Leg].index)
                    subset_dict = {key: value for key, value in self.imgs.items() if key in idx}
                    if len(subset_dict) == 0:
                        print('Warning: could not find image for legend row "{row.Leg}"')
                        continue
                    counts, vals = self._calc_histograms(ir, ic, np.concatenate(list(subset_dict.values()), 1))
                    temp = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                    temp[self.legend] = row.Leg
                    df_sub += [temp]
                return pd.concat(df_sub)

    def _subset_wrap(self, ir: int, ic: int) -> pd.DataFrame:
        """For wrap plots, select the revelant subset from self.df_fig.  Subclassed to deal with missing hist column
        names that don't show up until after hist calcs.

        Args:
            ir: subplot row index
            ic: subplot column index

        Returns:
            self.df_fig DataFrame subset based on self.wrap value
        """
        if ir * self.ncol + ic > self.nwrap - 1:
            return pd.DataFrame()
        elif self.wrap == 'y':
            # NOTE: can we drop these validate calls for speed?
            self.y = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) \
                + (self.y if self.y is not None else []) \
                + (self.groups if self.groups is not None else []) \
                + (utl.validate_list(self.legend)
                   if self.legend not in [None, True, False] else [])
            return self.df_fig[cols]
        elif self.wrap == 'x':
            self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend)
                    if self.legend is not None else [])
            return self.df_fig[cols]
        else:
            wrap = dict(zip(self.wrap, utl.validate_list(self.wrap_vals[ir * self.ncol + ic])))
            mask = pd.concat([self.df_fig[x[0]].eq(x[1]) for x in wrap.items()], axis=1).all(axis=1)
            return self.df_fig[mask]

    def switch_to_xy_plot(self, kwargs):
        """If bars are not enabled, switch everything to line plot.

        Args:
            kwargs: user-defined keyword args
        """
        self.name = 'xy'
        if self.cdf:
            self.y = ['Cumulative Probability']
        if self.pdf:
            self.y = ['Probability Density']
        if not self.y:
            self.y = ['Counts']
