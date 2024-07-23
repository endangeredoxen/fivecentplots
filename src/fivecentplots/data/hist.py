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

    def _post_range_calculations(self, ir: int, ic: int, df_rc: pd.DataFrame):
        """
        For non-image histograms (i.e., hists that are calculated by a `hist` plotting function), we must
        manually recalculate the y-axis ranges because "counts" are not in self.df_rc.
        """
        if self.imgs is not None:
            return

        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        counts, vals = self._calc_histograms(df_rc[self.x])
        self.ranges['ymin'][ir, ic], self.ranges['ymax'][ir, ic] = self._get_data_range('y', counts, plot_num)

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        if self.imgs is None:
            return data.Data._subset_modify(self, ir, ic, df)

        else:
            if self.legend is None:
                subset_dict = {key: value for key, value in self.imgs.items() if key in list(df.index)}
                counts, vals = self._calc_histograms(np.concatenate(list(subset_dict.values()), 1))
                df_sub = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                return df_sub

            else:
                df_sub = []
                for iline, row in self.legend_vals.iterrows():
                    idx = list(df.loc[df[self.legend] == row['Leg']].index)
                    subset_dict = {key: value for key, value in self.imgs.items() if key in idx}
                    counts, vals = self._calc_histograms(np.concatenate(list(subset_dict.values()), 1))
                    temp = pd.DataFrame({self.x[0]: vals, self.y[0]: counts})
                    temp[self.legend] = row['Leg']
                    df_sub += [temp]
                return pd.concat(df_sub)

    def switch_to_xy_plot(self, kwargs):
        """If bars are not enabled, switch everything to line plot.

        Args:
            kwargs: user-defined keyword args
        """
        self.name = 'xy'
        if not self.y:
            self.y = ['Counts']
