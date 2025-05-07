from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
from natsort import natsorted
import numpy.typing as npt
from typing import Union
utl = utilities
db = pdb.set_trace


class Heatmap(data.Data):
    name = 'heatmap'
    req = []
    opt = ['x', 'y', 'z']
    url = 'heatmap.html'

    def __init__(self, **kwargs):
        """Heatmap-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Deprecating support for image plotting --> use imshow

        Args:
            kwargs: user-defined keyword args
        """
        kwargs['auto_scale'] = False

        # check for invalid axis options
        vals = ['twin_x', 'twin_y']
        for val in vals:
            if val in kwargs:
                raise data.AxisError(f'{val} is not a valid option for heatmap plots')

        # check for invalid grouping options
        if 'row' in kwargs and kwargs['row'] == 'y':
            raise data.GroupingError('Cannot group row by "y" for heatmap plots')
        if 'col' in kwargs and kwargs['col'] == 'x':
            raise data.GroupingError('Cannot group col by "x" for heatmap plots')
        if 'wrap' in kwargs and kwargs['wrap'] == 'y':
            raise data.GroupingError('Cannot wrap by "y" for heatmap plots')
        if 'legend' in kwargs and kwargs['legend'] is not None:
            raise data.GroupingError('legend not available for heatmap plots')

        super().__init__(self.name, self.req, self.opt, **kwargs)

        # Set values
        if 'x' not in kwargs.keys() and \
                'y' not in kwargs.keys() and \
                'z' not in kwargs.keys():

            self.auto_cols = True
            self.x = ['Column']
            self.y = ['Row']
            self.z = ['Value']

        else:
            self.pivot = True

        self.num_x = None
        self.num_y = None

        self.ax_limit_padding = kwargs.get('ax_limit_padding', None)

        # Update valid axes
        self.axs_on = ['x', 'y', 'z']

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.

        Args:
            xyz: name of variable to check
        """
        if xyz in self.opt and getattr(self, xyz) is None:
            return None

        vals = getattr(self, xyz)

        if vals is None and xyz not in self.opt:
            raise data.AxisError(f'Must provide a column name for "{xyz}"')

        return vals

    def _get_data_range(self, ax: str, data_set: Union[pd.DataFrame, npt.NDArray], plot_num: int) -> tuple:
        """Determine the min/max values for a given axis based on user inputs.

        Args:
            ax: name of the axis ('x', 'y', etc)
            dd: data to use for range calculation
            plot_num: index number of the current subplot

        Returns:
            min, max tuple
        """
        if self.pivot:
            if ax == 'x':
                return -0.5, len(data_set.columns.values) - 0.5
            elif ax == 'y':
                return len(data_set.index.values) - 0.5, -0.5
            else:
                data_set = pd.DataFrame(data_set.stack())
                data_set.columns = [self.z]

        return data.Data._get_data_range(self, ax, data_set, plot_num)

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Extra modifications for Heatmap subsets

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        if len(df) == 0:
            return df

        if self.pivot:
            # Reshape if input DataFrame is stacked
            df = pd.pivot_table(df, values=self.z[0], index=self.y[0], columns=self.x[0])
        if self.sort:
            cols = natsorted(df.columns)
            df = df[cols]
            df.index = natsorted(df.index)

        # Ensure only int columns are present for imshow case and set range
        if self.auto_cols:
            df = df[utl.df_int_cols(df)]

            if 'xmin' in self.ranges and \
                    self.ranges['xmin'][ir, ic] is not None:
                df = df[[f for f in df.columns if f >= self.ranges['xmin'][ir, ic]]]
            if 'xmax' in self.ranges and \
                    self.ranges['xmax'][ir, ic] is not None:
                df = df[[f for f in df.columns if f <= self.ranges['xmax'][ir, ic]]]
            if 'ymin' in self.ranges and \
                    self.ranges['ymin'][ir, ic] is not None:
                df = df.loc[[f for f in df.index if f >= self.ranges['ymin'][ir, ic]]]
            if 'ymax' in self.ranges and \
                    self.ranges['ymax'][ir, ic] is not None:
                df = df.loc[[f for f in df.index if f <= self.ranges['ymax'][ir, ic]]]

        # Check dtypes to properly designate tick labels
        dtypes = [int, np.int32, np.int64]
        if df.index.dtype in dtypes and list(df.index) != \
                [f + df.index[0] for f in range(0, len(df.index))]:
            df.index = df.index.astype('O')
        if df.columns.dtype == 'object':
            ddtypes = list(set([type(f) for f in df.columns]))
            if all(f in dtypes for f in ddtypes):
                df.columns = [np.int64(f) for f in df.columns]
        elif df.columns.dtype in dtypes and list(df.columns) != \
                [f + df.columns[0] for f in range(0, len(df.columns))]:
            df.columns = df.columns.astype('O')

        # set heatmap element size parameters
        if self.x[0] in self.df_fig.columns:
            self.num_x = len(self.df_fig[self.x].drop_duplicates())
        else:
            self.num_x = None
        if self.y[0] in self.df_fig.columns:
            self.num_y = len(self.df_fig[self.y].drop_duplicates())
        else:
            self.num_y = None

        return df
