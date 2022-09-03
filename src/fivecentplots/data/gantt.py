from . import data
import pdb
import pandas as pd
from .. import utilities
utl = utilities
db = pdb.set_trace


class Gantt(data.Data):
    def __init__(self, **kwargs):
        """Gantt-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        name = 'gantt'
        req = ['x', 'y']
        opt = []
        kwargs['share_y'] = False

        super().__init__(name, req, opt, **kwargs)

        # error checks
        if len(self.x) != 2:
            raise data.DataError('Gantt charts require both a start and a stop column')
        if self.df_all[self.x[0]].dtype != 'datetime64[ns]':
            try:
                # check to see if 'O' type is still a valid datetime
                self.df_all[self.x[0]].astype('datetime64[ns]')
            except:  # noqa
                raise data.DataError('Start column in gantt chart must be of type datetime')
        if self.df_all[self.x[1]].dtype != 'datetime64[ns]':
            try:
                self.df_all[self.x[1]].astype('datetime64[ns]')
            except:  # noqa
                raise data.DataError('Stop column in gantt chart must be of type datetime')

    def _get_data_ranges(self):
        """Gantt-specific data range calculator by subplot."""
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self._get_auto_scale(df_fig)

        for ir, ic, plot_num in self._get_subplot_index():
            df_rc = self._subset(ir, ic)

            if len(df_rc) == 0:
                for ax in self.axs:
                    self._add_range(ir, ic, ax, 'min', None)
                    self._add_range(ir, ic, ax, 'max', None)
                continue

            self._add_range(ir, ic, 'y', 'min', -0.5)

            # shared axes
            if self.share_x or (self.nrow == 1 and self.ncol == 1):
                self._add_range(ir, ic, 'x', 'min', df_fig[self.x[0]].min())
                self._add_range(ir, ic, 'x', 'max', df_fig[self.x[1]].max())
            if self.share_y or (self.nrow == 1 and self.ncol == 1):
                if self.legend is not None:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]]) - 0.5)
                else:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]].unique()) - 0.5)

            # non-shared axes
            if not self.share_x:
                self._add_range(ir, ic, 'x', 'min', df_rc[self.x[0]].min())
                self._add_range(ir, ic, 'x', 'max', df_rc[self.x[1]].max())
            if not self.share_y:
                self._add_range(ir, ic, 'y', 'max', len(df_rc[self.y[0]]) - 0.5)
                # ymaxes += [len(df_rc[self.y[0]]) - 0.5]

            # not used
            self._add_range(ir, ic, 'x2', 'min', None)
            self._add_range(ir, ic, 'y2', 'min', None)
            self._add_range(ir, ic, 'x2', 'max', None)
            self._add_range(ir, ic, 'y2', 'max', None)
            self._add_range(ir, ic, 'z', 'min', None)
            self._add_range(ir, ic, 'z', 'max', None)

    def get_plot_data(self, df):
        """Gantt-specific generator to subset into discrete sets of data for
        each curve.

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
        if not isinstance(self.legend_vals, pd.DataFrame):
            xx = [self.x[0]]  # make sure we only get one group for self.x
            yy = [] if not self.y else self.y + self.y2
            lenx = 1 if not self.x else len(xx)
            leny = 1 if not self.y else len(yy)
            vals = pd.DataFrame({'x': self.x if not self.x else xx * leny,
                                 'y': self.y if not self.y else yy * lenx})

            for irow, row in vals.iterrows():
                yield irow, df, row['x'], row['y'], None if self.z is None else self.z[0], None, False, len(vals)

        else:
            for irow, row in self.legend_vals.iterrows():
                # Subset by legend value
                if row['Leg'] is not None:
                    df2 = df[df[self.legend] == row['Leg']].copy()

                # Filter out all nan data
                if row['x'] and row['x'] in df2.columns and len(df2[row['x']].dropna()) == 0 \
                        or row['y'] and row['y'] in df2.columns and len(df2[row['y']].dropna()) == 0:
                    continue

                # Set twin ax status
                yield irow, df2, row['x'], row['y'], \
                    None if self.z is None else self.z[0], row['names'], \
                    False, len(self.legend_vals)

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Modify subset to deal with duplicate Gantt entries

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        # remove duplicates with legend
        if self.legend is None and len(df) > 0:
            idx = []
            [idx.append(x) for x in df.set_index(self.y).index if x not in idx]
            df_start = df.groupby(self.y).min()
            df_stop = df.groupby(self.y).max()
            df_start[self.x[1]] = df_stop.loc[df_start.index, self.x[1]]
            df = df_start.reindex(idx).reset_index()

        # account for rc plots with shared y-axis
        if (self.wrap is not None or self.col is not None or self.row is not None) \
                and self.share_y:
            # set the top level index
            idx = []
            [idx.append(x) for x in self.df_all.set_index(self.y).index if x not in idx]
            df_start = self.df_all.groupby(self.y).min()
            df_stop = self.df_all.groupby(self.y).max()
            df_start[self.x[1]] = df_stop.loc[df_start.index, self.x[1]]
            df_all = df_start.reindex(idx).reset_index()

            # check for matches in the subset
            df = pd.merge(df_all, df, how='left', indicator='Exist')
            df.loc[df.Exist != 'both', self.x[1]] = df[self.x[0]]  # set start/stop date to the same for sorting
            del df['Exist']

        return df
