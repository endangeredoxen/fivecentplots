from . import data
import pdb
import pandas as pd
import numpy as np
try:
    from pandas.tseries.offset import BusinessDay
    from pandas.tseries.offset import DateOffset
    from pandas.tseries.offset import CustomBusinessDay
except ImportError:
    from pandas.tseries.offsets import BusinessDay
    from pandas.tseries.offsets import DateOffset
    from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from .. import utilities
utl = utilities
db = pdb.set_trace
NULLS = [None, np.nan, 'nan', pd.NaT, 'NaT', 'N/A', 'n/a', 'Nan', 'NAN', 'NaN', 'None', '']


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
        self.DATE_TYPES = ['year', 'quarter', 'month', 'week', 'quarter-year', 'month-year']  # not linked to layout.py
        self.business_days = utl.kwget(kwargs, self.fcpp, ['gantt_business_days', 'business_days'], True)
        self.us_holidays = utl.kwget(kwargs, self.fcpp, ['gantt_us_holidays', 'us_holidays'], True)
        self.date_type = utl.validate_list(utl.kwget(kwargs, self.fcpp, ['gantt_date_type', 'date_type'], []))
        self.workstreams = utl.kwget(kwargs, self.fcpp, ['gantt_workstreams', 'workstreams'], None)
        if self.workstreams is not None and 'legend' not in kwargs:
            self.legend = self.workstreams
        inline = utl.kwget(kwargs, self.fcpp, ['gantt_workstreams_location', 'workstreams_location'], 'left')
        if inline == 'inline':
            self.workstreams_inline = True
        else:
            self.workstreams_inline = False
        self.show_all = utl.kwget(kwargs, self.fcpp, ['gantt_show_all', 'show_all'], False)
        self.duration = utl.kwget(kwargs, self.fcpp, ['gantt_duration', 'duration'], 'Duration')
        self.dependencies = utl.kwget(kwargs, self.fcpp, ['gantt_dependencies', 'dependencies'], 'Dependency')
        self.milestone = utl.kwget(kwargs, self.fcpp, ['gantt_milestones', 'milestones'], 'Milestone')

        # error checks
        if self.workstreams not in [None, False] and self.workstreams not in self.df_all.columns:
            raise data.DataError(f'Workstreams column "{self.workstreams}" is not in DataFrame')
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

        # Date type restrictions
        invalid_date_types = [f for f in self.date_type if f not in self.DATE_TYPES]
        if len(invalid_date_types) > 0:
            raise data.DataError(f'Invalid date type[s] {invalid_date_types} specfied for Gantt chart; '
                                 f'allowed values: {self.DATE_TYPES}')
        if 'quarter-year' in self.date_type:
            allowed_combos = ['month', 'week', 'quarter-year']
            if any([f for f in self.date_type if f not in allowed_combos]):
                raise data.DataError('Date type "quarter-year" can only be combined with "month" or "week"')
        if 'month-year' in self.date_type:
            allowed_combos = ['quarter', 'week', 'month-year']
            if any([f for f in self.date_type if f not in allowed_combos]):
                raise data.DataError('Date type "month-year" can only be combined with "quarter" or "week"')

        # Attempt to populate missing dates
        self._populate_dates()

        # Inline workstreams df adjustment
        if self.workstreams_inline:
            df_inline = []
            for nn, gg in self.df_all.groupby(self.workstreams):
                df_inline += [pd.DataFrame({self.workstreams: [nn],
                                            self.x[0]: gg[self.x[0]].min(),
                                            self.x[1]: gg[self.x[1]].max(),
                                            self.y[0]: [nn],
                                            '_is_workstream': [1]},
                                           index=[0])]
            self.df_all['_is_workstream'] = 2
            self.df_all = pd.concat([pd.concat(df_inline), self.df_all])

        # Optionally hide tasks with no dates specified
        if not self.show_all:
            self.df_all = self.df_all.loc[~((self.df_all[self.x[0]].isin(NULLS)) &
                                            (self.df_all[self.x[1]].isin(NULLS)))]

        # Update date times to np.datetime64
        # self.df_all[self.x[0]] = pd.to_datetime(self.df_all[self.x[0]])  # maybe automatic?
        # self.df_all[self.x[1]] = pd.to_datetime(self.df_all[self.x[1]])
        for vv in ['min', 'max']:
            if getattr(self, f'x{vv}').values != [None]:
                getattr(self, f'x{vv}').values = [np.datetime64(f) for f in getattr(self, f'x{vv}').values]

    def _calc_dependent_start_dates(self):
        """Calculate start dates that are based on a dependency"""
        if self.dependencies not in self.df_all.columns:
            return

        # Find the dependent rows ("children" are dependent on "parents")
        children = self.df_all.loc[~self.df_all[self.dependencies].isin(NULLS)].copy()
        dep_vals = [f.lstrip() for f in ';'.join(children[self.dependencies].values).split(';')]

        # Check parent dates for errors
        parents = self.df_all.loc[(self.df_all[self.y[0]].isin(dep_vals)) &
                                  (self.df_all[self.dependencies].isin(NULLS))]
        for irow, row in parents.iterrows():
            if row[self.x[1]] in NULLS:
                raise data.DataError(f'Missing end date for "{row[self.y[0]]}"; cannot calculate dependent dates')

        # Multiple dependencies can be specified for a single row, separated by a semicolon
        children_idx = ~self.df_all[self.dependencies].isin(NULLS)
        self.df_all.loc[children_idx, self.dependencies] = \
            self.df_all.loc[children_idx, self.dependencies].str.split(';')
        self.df_all.loc[children_idx, self.dependencies] = \
            self.df_all.loc[children_idx, self.dependencies].apply(lambda x: [f.strip() for f in x])

        # Fill in child dates
        def resolve_dep(row):
            """Resolve nested dependencies"""
            parent = self.df_all.loc[self.df_all[self.y[0]].isin(row[self.dependencies])]
            if len(parent.loc[parent[self.x[1]].isin(NULLS)]) == 0:
                # If start date already exists, use it
                if row[self.x[0]] not in NULLS:
                    return row[self.x[0]]

                # If not, find all rows in the main dataframe that contain the depedency strings
                dep = self.df_all.loc[self.df_all[self.y[0]].isin(row[self.dependencies])]

                # Nothing found in dependency column but milestone column exists
                if len(dep) == 0 and self.milestone in row:
                    date = self.df_all.loc[self.df_all[self.milestone].isin(row[self.dependencies]), self.x[0]]
                # One or more dependencies found; use latest date as new start date
                elif len(dep) > 0:
                    date = dep[self.x[1]]
                # Nothing found, raise error
                else:
                    raise data.DataError(f'Cannot find dependency date for "{row[self.y[0]]}"')
                return date.max()
            elif len(parent.loc[~parent[self.dependencies].isin(NULLS)]) > 0:
                deps = parent.loc[~parent[self.dependencies].isin(NULLS)]
                for irow, row_ in deps.iterrows():
                    self.df_all.loc[irow, self.x[0]] = resolve_dep(row_)
            else:
                # ERROR
                raise data.DataError(f'Cannot find dependency date for "{row[self.y[0]]}"')

        for irow, row in self.df_all.iterrows():
            if row[self.dependencies] in NULLS or (row[self.x[0]] not in NULLS and row[self.x[1]] not in NULLS):
                continue
            self.df_all.loc[irow, self.x[0]] = resolve_dep(row)

            # Update duration
            if row[self.x[1]] in NULLS and self.duration in row and row[self.duration] not in NULLS:
                self.df_all.loc[irow, self.x[1]] = self._calc_durations(self.df_all.loc[irow])

    def _calc_durations(self, row):
        """Calculate durations for a given row in the DataFrame"""
        if row[self.x[1]] not in NULLS:
            # Don't calculate if there is already an end date
            return row[self.x[1]]
        if isinstance(row[self.duration], int):
            # Default is days
            duration = int(row[self.duration])
        elif isinstance(row[self.duration], str):
            date_type = row[self.duration][-1:]
            try:
                duration = float(row[self.duration][:-1])
                full, partial = divmod(duration, 1)
            except ValueError:
                raise data.DataError(f'Invalid duration "{date_type}" defined for row index {row.name}')
            if date_type.lower() == 'w':
                if partial > 0:
                    partial = DateOffset(days=int(7 * partial))
                else:
                    partial = DateOffset(days=0)
                return row[self.x[0]] + DateOffset(weeks=int(full)) + partial
            elif date_type.lower() == 'm':
                if partial > 0:
                    partial = DateOffset(days=int(30 * partial))
                else:
                    partial = DateOffset(days=0)
                return row[self.x[0]] + DateOffset(months=int(full))
            elif date_type.lower() == 'd':
                if partial > 0:
                    raise data.DataError('Partial days not allowed for duration; use only integers to specify days')
                if self.business_days and self.us_holidays:
                    return row[self.x[0]] + CustomBusinessDay(calendar=USFederalHolidayCalendar()) * int(full)
                elif self.business_days:
                    return row[self.x[0]] + BusinessDay() * int(full)
                else:
                    return row[self.x[0]] + pd.Timedelta(days=int(full))
            else:
                raise data.DataError(f'Unknown duration date type "{date_type}" defined for row index {row.name}')

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
            if self.share_x and (self.nrow != 1 or self.ncol != 1):
                self._add_range(ir, ic, 'x', 'min', self.ranges['xmin'][self.ranges['xmin'] != None].min())  # noqa
                self._add_range(ir, ic, 'x', 'max', self.ranges['xmax'][self.ranges['xmax'] != None].max())  # noqa
            if self.share_y or (self.nrow == 1 and self.ncol == 1):
                if self.legend is not None and self.workstreams != self.legend:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]]) - 0.5)
                else:
                    self._add_range(ir, ic, 'y', 'max', len(df_fig[self.y[0]].unique()) - 0.5)

            # non-shared axes
            if not self.share_x:
                if self.ranges['xmin'][ir, ic] is None:
                    self._add_range(ir, ic, 'x', 'min', df_rc[self.x[0]].min())
                if self.ranges['xmax'][ir, ic] is None:
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
        """Fill in missing dates"""
        # For all rows with a start date and a duration, calculate the end date
        if self.duration in self.df_all.columns:
            sub = self.df_all.loc[~(self.df_all[self.x[0]].isin(NULLS)) &
                                  ~(self.df_all[self.duration].isin(NULLS))]
            for irow, row in sub.iterrows():
                self.df_all.loc[irow, self.x[1]] = self._calc_durations(row)

        # Calculate start dates based on dependencies
        self._calc_dependent_start_dates()

        # # Check duration again since some start dates were filled in by dependencies (DO I NEED THIS??)
        # if self.duration in self.df_all.columns:
        #     sub = self.df_all.loc[~(self.df_all[self.x[0]].isin(NULLS)) & \
        #                           ~(self.df_all[self.duration].isin(NULLS))]
        #     for irow, row in sub.iterrows():
        #         self.df_all.loc[irow, self.x[1]] = self._calc_durations(row)

        # Assume that rows with a start date but no end date are milestones and set end date = start date
        self.df_all.loc[(~self.df_all[self.x[0]].isin(NULLS)) & (self.df_all[self.x[1]].isin(NULLS)), self.x[1]] = \
            self.df_all[self.x[0]]

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
        df = df.copy()
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
            df_all = pd.DataFrame(self.df_all.groupby(self.y)[self.x[0]].min())
            df_all[self.x[1]] = self.df_all.groupby(self.y)[self.x[1]].max()
            df_all = df_all.reindex(idx).reset_index()

            # check for matches in the subset
            df = pd.merge(df_all, df, how='left', indicator='Exist')
            df.loc[df.Exist != 'both', self.x[1]] = np.datetime64('NaT')  # set to NaT for sorting
            del df['Exist']

        return df
