from . import data
import pdb
import pandas as pd
from .. import utilities
utl = utilities
db = pdb.set_trace


class Gantt(data.Data):
    name = 'gantt'
    req = ['x', 'y']
    opt = []
    url = 'gantt.html'

    def __init__(self, **kwargs):
        """Gantt-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        super().__init__(self.name, self.req, self.opt, **kwargs)

        # Get some important kwargs
        self.workstreams = utl.kwget(kwargs, self.fcpp, ['gantt_workstreams', 'workstreams'], None)
        if self.workstreams is not None and 'legend' not in kwargs:
            self.legend = self.workstreams
        self.duration = utl.kwget(kwargs, self.fcpp, ['gantt_duration', 'duration'], 'Duration')
        self.dependencies = utl.kwget(kwargs, self.fcpp, ['gantt_dependencies', 'dependencies'], 'Dependency')
        self.milestone=utl.kwget(kwargs, self.fcpp, ['gantt_milestones', 'milestones'], 'Milestone'),

        # error checks
        if self.workstreams is not None and self.workstreams not in self.df_all.columns:
            raise data.DataError('Workstreams column "{self.workstreams}" is not in DataFrame')
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
        for col in utl.validate_list(utl.kwget(kwargs, self.fcpp, ['bar_labels', 'gantt_bar_labels'], [])):
            if col not in self.df_all.columns:
                raise data.DataError(f'Bar label column "{col}" is not in DataFrame')

        self._populate_dates()

        # Update date time formats and replace NaT with fake datetime
        self.df_all[self.x[0]] = pd.to_datetime(self.df_all[self.x[0]])
        self._strip_timestamp(self.x[0])
        self.df_all[self.x[1]] = pd.to_datetime(self.df_all[self.x[1]])
        self._strip_timestamp(self.x[1])

    def get_data_ranges(self):
        """Gantt-specific data range calculator by subplot."""
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        for ir, ic, plot_num in self.get_subplot_index():
            df_rc = self._subset(ir, ic)

            if len(df_rc) == 0:
                for ax in self.axs_on:
                    self._add_range(ir, ic, ax, 'min', None)
                    self._add_range(ir, ic, ax, 'max', None)
                continue

            self._add_range(ir, ic, 'y', 'min', -0.5)

            # shared axes
            if self.share_x or (self.nrow == 1 and self.ncol == 1):
                if self.ranges['xmin'][ir, ic] is not None:
                    self._add_range(ir, ic, 'x', 'min', self.ranges['xmin'][ir, ic])
                else:
                    self._add_range(ir, ic, 'x', 'min', df_fig[self.x[0]].min())
                if self.ranges['xmax'][ir, ic] is not None:
                    self._add_range(ir, ic, 'x', 'max', self.ranges['xmax'][ir, ic])
                else:
                    self._add_range(ir, ic, 'x', 'max', df_fig[self.x[1]].max())
            if self.share_y or (self.nrow == 1 and self.ncol == 1):
                if self.legend is not None and self.workstreams != self.legend:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]]) - 0.5)
                else:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]].unique()) - 0.5)

            # non-shared axes
            if not self.share_x:
                self._add_range(ir, ic, 'x', 'min', df_rc[self.x[0]].min())
                if self.ranges['xmax'][ir, ic] is not None:
                    self._add_range(ir, ic, 'x', 'max', self.ranges['xmax'][ir, ic])
                else:
                    self._add_range(ir, ic, 'x', 'max', df_rc[self.x[1]].max())
            if not self.share_y:
                self._add_range(ir, ic, 'y', 'max', len(df_rc[self.y[0]]) - 0.5)

    def get_plot_data(self, df):
        """Gantt-specific generator to subset into discrete sets of data for each curve.

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
            if self.workstreams is not None and self.legend != self.workstreams:
                # Workstream defined with different legend column
                for iws, ws in enumerate(df[self.workstreams].unique()):
                    # Subset by workstream value
                    df2 = df.loc[df[self.workstreams] == ws].copy()

                    # Set twin ax status
                    yield iws, df2, None, self.y[0], None, ws, False, len(df2)

            else:
                # No workstream defined OR workstream defined with no legend
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

    def _populate_dates(self):
        """Compute dates that are based on a dependency or duration."""
        # Fill in missing start dates based on dependencies
        missing_start = self.df_all.loc[self.df_all[self.x[0]].isna()]
        for irow, row in missing_start.iterrows():
            if self.dependencies not in row or str(row[self.dependencies]) == 'nan':
                raise data.DataError(f'No start time defined for "{self.y[0]}" (specify a date or dependency)')
            deps = [f.lstrip() for f in row[self.dependencies].split(';')]
            start = self.df_all.loc[self.df_all[self.y[0]].isin(deps)][self.x[1]].max()
            self.df_all.loc[irow, self.x[0]] = start

        # Fill in missing end dates using duration or treat as discrete milestones
        missing_end = self.df_all.loc[self.df_all[self.x[1]].isna()]
        for irow, row in missing_end.iterrows():
            if self.duration in row:
                db()
            else:
                # Milestones have same start and end date
                self.df_all.loc[irow, self.x[1]] = row[self.x[0]]


    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Modify subset to deal with duplicate Gantt entries

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        # Deal with missing dates
        earliest_start = df[self.x[0]].min()
        latest_end = df[self.x[1]].max()

        # Missing dates case 1:  no start or end date --> set duration as 1 day at earliest start date
        no_start_end = df.loc[(df[self.x[0]].isnull()) & (df[self.x[1]].isnull())].index
        df.loc[no_start_end, self.x[0]] = earliest_start
        df.loc[no_start_end, self.x[1]] = earliest_start + pd.Timedelta(days=1)

        # Missing dates case 2:  start date but no end data --> set as latest date
        no_end = df.loc[df[self.x[1]].isnull()].index
        df.loc[no_end, self.x[1]] = latest_end

        # Missing dates case 3: end date but no start date --> set start as one before end date
        no_start = df.loc[df[self.x[0]].isnull()].index
        df.loc[no_start, self.x[0]] = latest_end - pd.Timedelta(days=1)

        # Remove duplicates with legend
        if self.legend is None and len(df) > 0:
            idx = []
            [idx.append(x) for x in df.set_index(self.y).index if x not in idx]
            df_start = df.groupby(self.y)[self.x].min()
            df_stop = df.groupby(self.y)[self.x].max()
            df_start[self.x[1]] = df_stop.loc[df_start.index, self.x[1]]
            df = df_start.reindex(idx).reset_index()

        # Account for rc plots with shared y-axis
        if (self.wrap is not None or self.col is not None or self.row is not None) \
                and self.share_y:
            # set the top level index
            idx = []
            [idx.append(x) for x in self.df_all.set_index(self.y).index if x not in idx]
            df_start = self.df_all.groupby(self.y)[self.x].min()
            df_stop = self.df_all.groupby(self.y)[self.x].max()
            df_start[self.x[1]] = df_stop.loc[df_start.index, self.x[1]]
            df_all = df_start.reindex(idx).reset_index()

            # check for matches in the subset
            df = pd.merge(df_all, df, how='left', indicator='Exist')
            df.loc[df.Exist != 'both', self.x[1]] = df[self.x[0]]  # set start/stop date to the same for sorting
            del df['Exist']

        return df
