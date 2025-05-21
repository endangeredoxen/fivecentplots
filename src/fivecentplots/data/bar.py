from . import data
import pdb
import pandas as pd
from .. import utilities
import numpy.typing as npt
from typing import Union
utl = utilities
db = pdb.set_trace


class Bar(data.Data):
    name = 'bar'
    url = 'barplot.html'

    def __init__(self, **kwargs):
        """Barplot-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        # Check for bad ranges
        horizontal = utl.kwget(kwargs, kwargs.get('fcpp', {}),
                               ['bar_horizontal', 'horizontal'], kwargs.get('horizontal', False))
        if horizontal:
            if 'ymin' in kwargs or 'ymax' in kwargs:
                raise data.RangeError('y-limits not allowed for horizontal bar plot!')
            if 'xmin' in kwargs:
                kwargs['ymin'] = kwargs['xmin']
                kwargs.pop('xmin')
            if 'xmax' in kwargs:
                kwargs['ymax'] = kwargs['xmax']
                kwargs.pop('xmax')
        else:
            if 'xmin' in kwargs or 'xmin' in kwargs:
                raise data.RangeError('x-limits not allowed for bar plot!')

        super().__init__(self.name, **kwargs)

        # overrides
        self.horizontal = horizontal
        self.stacked = utl.kwget(kwargs, self.fcpp, ['bar_stacked', 'stacked'], kwargs.get('stacked', False))
        if utl.kwget(kwargs, self.fcpp, ['bar_error_bars', 'error_bars'], kwargs.get('error_bars', False)):
            self.error_bars = True

    def _filter_data(self, kwargs):
        """Apply an optional filter to the data but allow showing of all groups with kwarg `show_all_groups`

        Args:
            kwargs: user-defined keyword args
        """
        # Get all the groups before filtering
        if kwargs.get('show_all_groups', False):
            groups = self.df_all[self.x[0]].unique()

        # Apply a custom filter
        if self.filter:
            self.df_all = utl.df_filter(self.df_all, self.filter)
            if len(self.df_all) == 0:
                raise data.DataError('DataFrame is empty after applying filter')

        # Handle the show all groups option
        if kwargs.get('show_all_groups', False):
            missing = [f for f in groups if f not in self.df_all[self.x[0]].unique()]
            temp = pd.DataFrame(columns=self.df_all.columns, index=range(0, len(missing)))
            for mm in missing:
                temp[self.x[0]] = mm
                temp[self.y[0]] = 0
            self.df_all = pd.concat([self.df_all, temp]).reset_index(drop=True)

    def _get_auto_scale(self,
                        data_set: Union[pd.DataFrame, npt.NDArray],
                        plot_num: int) -> Union[pd.DataFrame, npt.NDArray]:
        """Auto-scale the plot data.  For Bar, sum up by group first.

        Args:
            df: either self.df_fig or self.df_rc
            plot_num: plot number

        Returns:
            updated df with potentially reduced range of data
        """
        # Sum up by group
        if self.stacked:
            data_sum = data_set.groupby(self.x).sum(numeric_only=True)
        else:
            data_sum = data_set.groupby(self.x + self._groupers).sum(numeric_only=True)

        # Add error bar range
        if self.error_bars:
            if self.stacked:
                yys = data_set.groupby(self.x).std(numeric_only=True)
            else:
                yys = data_set.groupby(self.x + self._groupers).std(numeric_only=True)
            data_plus = data_sum.copy()
            data_plus[self.y] += yys[self.y]
            data_minus = data_sum.copy()
            data_minus[self.y] -= yys[self.y]
            data_sum = pd.concat([data_plus, data_minus])

        # Run the usual range algorithm
        return data.Data._get_auto_scale(self, data_sum, plot_num)

    def _get_data_range(self, ax: str, data_set: Union[pd.DataFrame, npt.NDArray], plot_num: int) -> tuple:
        """Determine the min/max values for a given axis based on user inputs. Modification for Bar

        Args:
            ax: name of the axis ('x', 'y', etc)
            data_set: data to use for range calculation
            plot_num: index number of the current subplot

        Returns:
            min, max tuple
        """
        vmin, vmax = data.Data._get_data_range(self, ax, data_set, plot_num)

        # Set ymin = 0 unless data or user require a negative value
        if ax == 'y' and vmin < 0 and data_set[self.y].values.min() >= 0 and self.ymin[plot_num] is None:
            vmin = 0

        return vmin, vmax
