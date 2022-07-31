from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Heatmap(data.Data):
    def __init__(self, **kwargs):
        """Heatmap-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        name = 'heatmap'
        req = []
        opt = ['x', 'y', 'z']
        kwargs['auto_scale'] = False

        super().__init__(name, req, opt, **kwargs)

        if 'x' not in kwargs.keys() and \
                'y' not in kwargs.keys() and \
                'z' not in kwargs.keys():

            self.auto_cols = True
            self.x = ['Column']
            self.y = ['Row']
            self.z = ['Value']

        else:
            self.pivot = True

        self.ax_limit_padding = kwargs.get('ax_limit_padding', None)

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.

        Args:
            xyz: name of variable to check
        """
        if xyz not in self.req and xyz not in self.opt:
            return

        if xyz in self.opt and getattr(self, xyz) is None:
            return None

        vals = getattr(self, xyz)

        if vals is None and xyz not in self.opt:
            raise data.AxisError('Must provide a column name for "%s"' % xyz)

        # Check for axis errors
        if self.twin_x and len(self.y) != 2:
            raise data.AxisError('twin_x error! %s y values were specified but'
                                 ' two are required' % len(self.y))
        if self.twin_x and len(self.x) > 1:
            raise data.AxisError('twin_x error! only one x value can be specified')
        if self.twin_y and len(self.x) != 2:
            raise data.AxisError('twin_y error! %s x values were specified but'
                                 ' two are required' % len(self.x))
        if self.twin_y and len(self.y) > 1:
            raise data.AxisError('twin_y error! only one y value can be specified')

        return vals

    def _get_data_ranges(self):
        """Heatmap-specific data range calculator by subplot."""
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self._get_auto_scale(df_fig)

        # set ranges by subset
        for ir, ic, plot_num in self._get_subplot_index():
            df_rc = self._subset(ir, ic)

            # auto cols option
            if self.auto_cols:
                df_rc = df_rc[utl.df_int_cols(df_rc)]

                # x
                cols = [f for f in df_rc.columns if isinstance(f, int)]
                self._add_range(ir, ic, 'x', 'min', min(cols))
                self._add_range(ir, ic, 'x', 'max', max(cols))

                # y
                rows = [f for f in df_rc.index if isinstance(f, int)]
                self._add_range(ir, ic, 'y', 'min', min(rows))
                self._add_range(ir, ic, 'y', 'max', max(rows))

                # z
                self._add_range(ir, ic, 'z', 'min', df_rc.min().min())
                self._add_range(ir, ic, 'z', 'max', df_rc.max().max())

            else:
                # x
                self._add_range(ir, ic, 'x', 'min', -0.5)
                self._add_range(ir, ic, 'x', 'max', len(df_rc.columns) + 0.5)

                # y (can update all the get plot nums to range?)
                if self.ymin[plot_num] is not None \
                        and self.ymax[plot_num] is not None \
                        and self.ymin[plot_num] < self.ymax[plot_num]:
                    ymin = self.ymin[plot_num]
                    self._add_range(ir, ic, 'y', 'min', self.ymax[plot_num])
                    self._add_range(ir, ic, 'y', 'max', ymin)
                self._add_range(ir, ic, 'y', 'max', -0.5)
                self._add_range(ir, ic, 'y', 'min', len(df_rc) + 0.5)

                # z
                if self.share_col:
                    pass
                elif self.share_row:
                    pass
                elif self.share_z and ir == 0 and ic == 0:
                    self._add_range(ir, ic, 'z', 'min',
                                    self.df_fig[self.z[0]].min())
                    self._add_range(ir, ic, 'z', 'max',
                                    self.df_fig[self.z[0]].max())
                elif self.share_z:
                    self._add_range(ir, ic, 'z', 'min',
                                    self.ranges[0, 0]['zmin'])
                    self._add_range(ir, ic, 'z', 'max',
                                    self.ranges[0, 0]['zmax'])
                else:
                    self._add_range(ir, ic, 'z', 'min', df_rc.min().min())
                    self._add_range(ir, ic, 'z', 'max', df_rc.max().max())

            # not used
            self._add_range(ir, ic, 'x2', 'min', None)
            self._add_range(ir, ic, 'y2', 'min', None)
            self._add_range(ir, ic, 'x2', 'max', None)
            self._add_range(ir, ic, 'y2', 'max', None)

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
            # Reshape if input dataframe is stacked
            df = pd.pivot_table(df, values=self.z[0],
                                index=self.y[0], columns=self.x[0])
        if self.sort:
            cols = natsorted(df.columns)
            df = df[cols]
            df.index = natsorted(df.index)

        # Ensure only int columns are present for imshow case and set range
        if self.auto_cols:
            df = df[utl.df_int_cols(df)]

            if 'xmin' in self.ranges[ir, ic].keys() and \
                    self.ranges[ir, ic]['xmin'] is not None:
                df = df[[f for f in df.columns if f >= self.ranges[ir, ic]['xmin']]]
            if 'xmax' in self.ranges[ir, ic].keys() and \
                    self.ranges[ir, ic]['xmax'] is not None:
                df = df[[f for f in df.columns if f <= self.ranges[ir, ic]['xmax']]]
            if 'ymin' in self.ranges[ir, ic].keys() and \
                    self.ranges[ir, ic]['ymin'] is not None:
                df = df.loc[[f for f in df.index if f >= self.ranges[ir, ic]['ymin']]]
            if 'ymax' in self.ranges[ir, ic].keys() and \
                    self.ranges[ir, ic]['ymax'] is not None:
                df = df.loc[[f for f in df.index if f <= self.ranges[ir, ic]['ymax']]]

        # check dtypes to properly designated tick labels
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
