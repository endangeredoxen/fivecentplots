from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
try:
    from natsort import natsorted
except:
    natsorted = sorted
utl = utilities
db = pdb.set_trace


class Heatmap(data.Data):
    def __init__(self, **kwargs):

        name = 'heatmap'
        req = []
        opt = ['x', 'y', 'z']
        kwargs['auto_scale'] = False
        auto_cols = False
        pivot = False

        if 'x' not in kwargs.keys() and 'y' not in kwargs.keys() and \
                'z' not in kwargs.keys():
            kwargs['x'] = ['Column']
            kwargs['y'] = ['Row']
            kwargs['z'] = ['Value']
            auto_cols = True
        else:
            pivot = True

        super().__init__(name, req, opt, **kwargs)

        if pivot:
            self.pivot = True

        if auto_cols:
            self.auto_cols = True

        self.ax_limit_padding = kwargs.get('ax_limit_padding', None)

    def get_data_range(self, ax, df, plot_num):
        """
        Determine the min/max values for a given axis based on user inputs

        Args:
            axis (str): x, x2, y, y2, z
            df (pd.DataFrame): data table to use for range calculation

        Returns:
            min, max tuple
        """

        if self.auto_cols:
            df = df[utl.df_int_cols(df)]

            if ax == 'x':
                vmin = min([f for f in df.columns if type(f) is int])
                vmax = max([f for f in df.columns if type(f) is int])
            elif ax == 'y':
                vmin = min([f for f in df.index if type(f) is int])
                vmax = max([f for f in df.index if type(f) is int])
            else:
                vmin = df.min().min()
                vmax = df.max().max()
            return vmin, vmax

        axx = getattr(self, ax)
        if ax not in ['x2', 'y2', 'z']:
            vmin = 0
            vmax = len(df[axx].drop_duplicates())
        elif ax not in ['x2', 'y2']:
            vmin = df[axx].min().iloc[0]
            vmax = df[axx].max().iloc[0]
        else:
            vmin = None
            vmax = None
        # if getattr(self, '%smin' % ax).get(plot_num):
        #     vmin = getattr(self, '%smin' % ax).get(plot_num)
        # if getattr(self, '%smax' % ax).get(plot_num):
        #     vmax = getattr(self, '%smax' % ax).get(plot_num)
        # if type(vmin) is str:
        #     vmin = None
        # if type(vmax) is str:
        #     vmax = None

        return vmin, vmax

    def get_data_ranges(self):

        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self.get_auto_scale(df_fig)

        # set ranges by subset
        for ir, ic, plot_num in self.get_subplot_index():
            df_rc = self.subset(ir, ic)

            # auto cols option
            if self.auto_cols:
                df_rc = df_rc[utl.df_int_cols(df_rc)]

                # x
                cols = [f for f in df_rc.columns if type(f) is int]
                self.add_range(ir, ic, 'x', 'min', min(cols))
                self.add_range(ir, ic, 'x', 'max', max(cols))

                # y
                rows = [f for f in df_rc.index if type(f) is int]
                self.add_range(ir, ic, 'y', 'min', min(cols))
                self.add_range(ir, ic, 'y', 'max', max(cols))

                # z
                self.add_range(ir, ic, 'z', 'min', df_rc.min().min())
                self.add_range(ir, ic, 'z', 'max', df_rc.max().max())

            else:
                # x
                self.add_range(ir, ic, 'x', 'min', -0.5)
                self.add_range(ir, ic, 'x', 'max', len(df_rc.columns) + 0.5)

                # y (can update all the get plot nums to range?)
                if self.ymin.get(plot_num) is not None \
                        and self.ymax.get(plot_num) is not None \
                        and self.ymin.get(plot_num) < self.ymax.get(plot_num):
                    ymin = self.ymin.get(plot_num)
                    self.add_range(ir, ic, 'y', 'min', self.ymax.get(plot_num))
                    self.add_range(ir, ic, 'y', 'max', ymin)
                self.add_range(ir, ic, 'y', 'max', -0.5)
                self.add_range(ir, ic, 'y', 'min', len(df_rc) + 0.5)

                # z
                if self.share_col:
                    pass
                elif self.share_row:
                    pass
                elif self.share_z and ir==0 and ic==0:
                    self.add_range(ir, ic, 'z', 'min', self.df_fig[self.z[0]].min())
                    self.add_range(ir, ic, 'z', 'max', self.df_fig[self.z[0]].max())
                elif self.share_z:
                    self.add_range(ir, ic, 'z', 'min', self.ranges[0, 0]['zmin'])
                    self.add_range(ir, ic, 'z', 'max', self.ranges[0, 0]['zmax'])
                else:
                    self.add_range(ir, ic, 'z', 'min', df_rc.min().min())
                    self.add_range(ir, ic, 'z', 'max', df_rc.max().max())

            # not used
            self.add_range(ir, ic, 'x2', 'min', None)
            self.add_range(ir, ic, 'y2', 'min', None)
            self.add_range(ir, ic, 'x2', 'max', None)
            self.add_range(ir, ic, 'y2', 'max', None)

    def subset_modify(self, df, ir, ic):

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
