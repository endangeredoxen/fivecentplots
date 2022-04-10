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
            try:
                # check to see if 'O' type is still a valid datetime
                self.df_all[self.x[0]].astype('datetime64[ns]')
            except:
                raise data.DataError('Start column in gantt chart must be of type datetime')
        if self.df_all[self.x[1]].dtype != 'datetime64[ns]':
            try:
                self.df_all[self.x[1]].astype('datetime64[ns]')
            except:
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

    def get_plot_data(self, df):
        """
        Generator to subset into discrete sets of data for each curve

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            subset
        """

        if type(self.legend_vals) != pd.DataFrame:
            xx = [self.x[0]]  # make sure we only get one group for self.x
            yy = [] if not self.y else self.y + self.y2
            lenx = 1 if not self.x else len(xx)
            leny = 1 if not self.y else len(yy)
            vals = pd.DataFrame({'x': self.x if not self.x else xx*leny,
                                 'y': self.y if not self.y else yy*lenx})

            for irow, row in vals.iterrows():
                # Set twin ax status
                twin = False
                if (row['x'] != vals.loc[0, 'x'] and self.twin_y) \
                        or (row['y'] != vals.loc[0, 'y'] and self.twin_x):
                    twin = True
                if self.legend_vals is not None and self.twin_y:
                    leg = row['x']
                elif self.legend_vals is not None:
                    leg = row['y']
                else:
                    leg = None
                if self.wrap == 'y':
                    irow = self.wrap_vals.index(leg)

                yield irow, df, row['x'], row['y'], \
                      None if self.z is None else self.z[0], leg, twin, len(vals)

        else:
            for irow, row in self.legend_vals.iterrows():
                # Fix unique wrap vals
                if self.wrap == 'y' or self.wrap == 'x':
                    wrap_col = list(set(df.columns) & set(getattr(self, self.wrap)))[0]
                    df = df.rename(columns={self.wrap: wrap_col})
                    row[self.wrap] = wrap_col
                if self.row == 'y':
                    row['y'] = self.y[0]
                    self.legend_vals['y'] = self.y[0]
                if self.col == 'x':
                    row['x'] = self.x[0]
                    self.legend_vals['x'] = self.x[0]

                # Subset by legend value
                if row['Leg'] is not None:
                    df2 = df[df[self.legend]==row['Leg']].copy()

                # Filter out all nan data
                if row['x'] and row['x'] in df2.columns and len(df2[row['x']].dropna()) == 0 \
                        or row['y'] and row['y'] in df2.columns and len(df2[row['y']].dropna()) == 0:
                    continue

                # Set twin ax status
                twin = False
                if (row['x'] != self.legend_vals.loc[0, 'x'] and self.twin_y) \
                        or (row['y'] != self.legend_vals.loc[0, 'y'] and self.twin_x):
                    twin = True
                yield irow, df2, row['x'], row['y'], \
                      None if self.z is None else self.z[0], row['names'], \
                      twin, len(self.legend_vals)

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

    def subset_wrap(self, ir, ic):

        return self._subset_wrap(ir, ic)

