from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
utl = utilities
db = pdb.set_trace


class Gantt(data.Data):
    def __init__(self, **kwargs):

        name = 'gantt'
        req = ['x', 'y']
        opt = []
        kwargs['share_y'] = False

        super().__init__(name, req, opt, **kwargs)

        # error checks
        if len(self.x) !=2:
            raise data.DataError('Gantt charts require both a start and a stop column')
        if self.df_all[self.x[0]].dtype != 'datetime64[ns]':
            raise data.DataError('Start column in gantt chart must be of type datetime')
        if self.df_all[self.x[1]].dtype != 'datetime64[ns]':
            raise data.DataError('Stop column in gantt chart must be of type datetime')

    def get_data_ranges(self):

        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self.get_auto_scale(df_fig)

        for ir, ic, plot_num in self.get_subplot_index():
            df_rc = self.subset(ir, ic)

            if len(df_rc) == 0:
                for ax in self.axs:
                    self.add_range(ir, ic, ax, 'min', None)
                    self.add_range(ir, ic, ax, 'max', None)
                continue

            self.add_range(ir, ic, 'y', 'min', -0.5)

            # shared axes
            if self.share_x or (self.nrow == 1 and self.ncol == 1):
                self.add_range(ir, ic, 'x', 'min', df_fig[self.x[0]].min())
                self.add_range(ir, ic, 'x', 'max', df_fig[self.x[1]].max())
            if self.share_y or (self.nrow == 1 and self.ncol == 1):
                if self.legend is not None:
                    self.add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]]) - 0.5)
                else:
                    self.add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]].unique()) - 0.5)

            # non-shared axes
            if not self.share_x:
                self.add_range(ir, ic, 'x', 'min', df_rc[self.x[0]].min())
                self.add_range(ir, ic, 'x', 'max', df_rc[self.x[1]].max())
            if not self.share_y:
                self.add_range(ir, ic, 'y', 'max', len(df_rc[self.y[0]]) - 0.5)
                # ymaxes += [len(df_rc[self.y[0]]) - 0.5]

            # not used
            self.add_range(ir, ic, 'x2', 'min', None)
            self.add_range(ir, ic, 'y2', 'min', None)
            self.add_range(ir, ic, 'x2', 'max', None)
            self.add_range(ir, ic, 'y2', 'max', None)
            self.add_range(ir, ic, 'z', 'min', None)
            self.add_range(ir, ic, 'z', 'max', None)
        
    def subset_modify(self, df, ir, ic):

        # deal with duplicate gantt entries
        if self.legend is None and len(df) > 0:
            idx = []
            [idx.append(x) for x in df.set_index(self.y).index if x not in idx]
            df_start = df.groupby(self.y).min()
            df_stop = df.groupby(self.y).max()
            df_start[self.x[1]] = df_stop.loc[df_start.index, self.x[1]]
            df = df_start.reindex(idx).reset_index()

        return df

