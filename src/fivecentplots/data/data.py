import pdb
from natsort import natsorted
import scipy.stats as ss
import pandas as pd
import numpy as np
from .. import utilities
utl = utilities

db = pdb.set_trace


class AxisError(Exception):
    def __init__(self, *args, **kwargs):
        """Invalid axis kwargs error."""
        Exception.__init__(self, *args, **kwargs)


class DataError(Exception):
    def __init__(self, *args, **kwargs):
        """Invalid DataFrame error."""
        Exception.__init__(self, *args, **kwargs)


class GroupingError(Exception):
    def __init__(self, *args, **kwargs):
        """Invalid grouping error."""
        Exception.__init__(self, *args, **kwargs)


class RangeError(Exception):
    def __init__(self, *args, **kwargs):
        """Invalid range limit error."""
        Exception.__init__(self, *args, **kwargs)


class Data:
    def __init__(self, name: str, req: list = ['x', 'y'], opt: list = [], fcpp: dict = {}, **kwargs):
        """Base Data class to deal with operations applied to the input data
        (i.e., non-plotting operations)

        Args:
            name: string name of the plot type
            req: list of the xyz axes that are required for the plot type
            opt: list of the xyz axes that are optional for the plot type
            fcpp: theme-file kwargs
            kwargs: user-defined keyword args
        """

        # defaults
        self.req = req
        self.opt = opt
        self.name = name

        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        # Default axis attributes
        self.auto_cols = False
        self.auto_scale = utl.kwget(kwargs, self.fcpp, 'auto_scale', True)
        self.axs = ['x', 'x2', 'y', 'y2', 'z']
        self.ax_scale = kwargs.get('ax_scale', None)
        self.ax2_scale = kwargs.get('ax2_scale', self.ax_scale)
        # validate list? repeated list and all to lower
        self._ax_limit_pad(**kwargs)
        self.error_bars = False
        self.fit = kwargs.get('fit', False)
        self.fit_range_x = utl.kwget(kwargs, self.fcpp, 'fit_range_x', None)
        self.fit_range_y = utl.kwget(kwargs, self.fcpp, 'fit_range_y', None)
        self.interval = utl.validate_list(kwargs.get('perc_int', kwargs.get('nq_int', kwargs.get('conf_int', False))))
        self.legend = None
        self.legend_vals = None
        self.pivot = False
        self.ranges = None
        self.share_col = utl.kwget(kwargs, self.fcpp, 'share_col', False)
        self.share_row = utl.kwget(kwargs, self.fcpp, 'share_row', False)
        self.share_x = utl.kwget(kwargs, self.fcpp, 'share_x', True)
        self.share_x2 = utl.kwget(kwargs, self.fcpp, 'share_x2', True)
        self.share_y = utl.kwget(kwargs, self.fcpp, 'share_y', True)
        self.share_y2 = utl.kwget(kwargs, self.fcpp, 'share_y2', True)
        self.share_z = utl.kwget(kwargs, self.fcpp, 'share_z', True)
        if self.share_row or self.share_col:
            self.share_x = False
            self.share_y = False
            self.share_z = False
        if kwargs.get('wrap', None) is not None:
            self.share_x = True
            self.share_y = True
        if kwargs.get('wrap', None) == 'y' or kwargs.get('wrap', None) == 'x':
            self.share_x = kwargs.get('share_x', True)
            self.share_y = kwargs.get('share_y', True)
        self.sort = utl.kwget(kwargs, self.fcpp, 'sort', True)
        self.stacked = False
        self.swap = utl.kwget(kwargs, self.fcpp, 'swap', False)
        self.trans_x = kwargs.get('trans_x', None)
        self.trans_x2 = kwargs.get('trans_x2', None)
        self.trans_y = kwargs.get('trans_y', None)
        self.trans_y2 = kwargs.get('trans_y2', None)
        self.trans_z = kwargs.get('trans_z', None)
        self.twin_x = kwargs.get('twin_x', False)
        self.twin_y = kwargs.get('twin_y', False)
        if self.twin_x == self.twin_y and self.twin_x:
            raise AxisError('cannot simultaneously twin x and y axes')
        self.xmin = utl.RepeatedList(kwargs.get('xmin', [None]), 'xmin')
        self.x2min = utl.RepeatedList(kwargs.get('x2min', [None]), 'x2min')
        self.xmax = utl.RepeatedList(kwargs.get('xmax', [None]), 'xmax')
        self.x2max = utl.RepeatedList(kwargs.get('x2max', [None]), 'x2max')
        self.ymin = utl.RepeatedList(kwargs.get('ymin', [None]), 'ymin')
        self.y2min = utl.RepeatedList(kwargs.get('y2min', [None]), 'y2min')
        self.ymax = utl.RepeatedList(kwargs.get('ymax', [None]), 'ymax')
        self.y2max = utl.RepeatedList(kwargs.get('y2max', [None]), 'y2max')
        self.zmin = utl.RepeatedList(kwargs.get('zmin', [None]), 'zmin')
        self.zmax = utl.RepeatedList(kwargs.get('zmax', [None]), 'zmax')

        # Update share
        if len(self.xmin) > 1 or len(self.xmax) > 1:
            self.share_x = False
        if len(self.x2min) > 1 or len(self.x2max) > 1:
            self.share_x2 = False
        if len(self.ymin) > 1 or len(self.ymax) > 1:
            self.share_y = False
        if len(self.y2min) > 1 or len(self.y2max) > 1:
            self.share_y2 = False

        # Define DataFrames
        self.df_all = self._check_df(kwargs['df'])
        self.df_fig = None
        self.df_sub = None
        self.changes = pd.DataFrame()  # used with boxplots
        self.indices = pd.DataFrame()  # used with boxplots

        # Get the x, y, and (optional) axis column names and error check
        self.x = utl.validate_list(kwargs.get('x'))
        self.x_vals = [f for f in self.x] if self.x else None
        self.y = utl.validate_list(kwargs.get('y'))
        self.y_vals = [f for f in self.y] if self.y else None
        self.z = utl.validate_list(kwargs.get('z'))
        self.x = self._check_xyz('x')
        self.y = self._check_xyz('y')
        self.z = self._check_xyz('z')
        self.x2 = []
        self.y2 = []
        if self.twin_x:
            self.y2 = [self.y[1]]
            self.y = [self.y[0]]
        if self.twin_y:
            self.x2 = [self.x[1]]
            self.x = [self.x[0]]

        # Ref line
        self.ref_line = kwargs.get('ref_line', None)
        if isinstance(self.ref_line, pd.Series):
            self.df_all['Ref Line'] = self.ref_line

        # Stats
        self.stat = kwargs.get('stat', None)
        self.stat_val = kwargs.get('stat_val', None)
        if self.stat_val is not None and self.stat_val not in self.df_all.columns:
            raise DataError('stat_val column "%s" not in DataFrame' % self.stat_val)
        self.stat_idx = []
        self.lcl = kwargs.get('lcl', [])
        self.ucl = kwargs.get('ucl', [])

        # Special for hist
        normalize = utl.kwget(kwargs, self.fcpp, ['hist_normalize', 'normalize'],
                              kwargs.get('normalize', False))
        kde = utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'],
                        kwargs.get('kde', False))
        if normalize or kde:
            self.norm = True
        else:
            self.norm = False
        self.bins = utl.kwget(kwargs, self.fcpp, ['hist_bins', 'bins'],
                              kwargs.get('bins', 20))

        # Apply an optional filter to the data
        self.filter = kwargs.get('filter', None)
        self._filter_data(kwargs)

        # Define rc grouping column names
        self.col = kwargs.get('col', None)
        self.col_vals = None
        if self.col is not None:
            if self.col == 'x':
                self.col_vals = [f for f in self.x]
            else:
                self.col = self._check_group_columns('col',
                                                     kwargs.get('col', None))
        self.row = kwargs.get('row', None)
        self.row_vals = None
        if self.row is not None:
            if self.row == 'y':
                self.row_vals = [f for f in self.y]
            else:
                self.row = self._check_group_columns('row',
                                                     kwargs.get('row', None))
        self.wrap = kwargs.get('wrap', None)
        self.wrap_vals = None
        if self.wrap is not None:
            if self.wrap == 'y':
                self.wrap_vals = [f for f in self.y]
            elif self.wrap == 'x':
                self.wrap_vals = [f for f in self.x]
            else:
                self.wrap = self._check_group_columns('wrap', self.wrap)
        self.groups = self._check_group_columns('groups', kwargs.get('groups', None))
        self._check_group_errors()
        self.ncols = kwargs.get('ncol', 0)
        self.ncol = 1
        self.nleg_vals = 0
        self.nrow = 1
        self.nwrap = 0
        self.ngroups = 0

        # Define legend grouping column names (legends are common to a figure, not an rc subplot)
        if 'legend' in kwargs.keys():
            if kwargs['legend'] is True:
                self.legend = True
            elif kwargs['legend'] is False:
                self.legend = False
            else:
                self.legend = self._check_group_columns('legend',
                                                        kwargs.get('legend', None))
        elif not self.twin_x and self.y is not None and len(self.y) > 1:
            self.legend = True

        # Define figure grouping column names
        if 'fig_groups' in kwargs.keys():
            self.fig = self._check_group_columns('fig',
                                                 kwargs.get('fig_groups', None))
        else:
            self.fig = self._check_group_columns('fig', kwargs.get('fig', None))
        self.fig_vals = None

        # Make sure groups, legend, and fig_groups are not the same
        if self.legend and self.groups and self.name != 'box':
            if not isinstance(self.legend, bool):
                self._check_group_matching('legend', 'groups')
        if self.legend and self.fig:
            self._check_group_matching('legend', 'fig')
        if self.groups and self.fig:
            self._check_group_matching('groups', 'fig')

        # Define all the columns in use
        self.cols_all = []
        self.cols_all += self.x if self.x is not None else []
        self.cols_all += self.x2 if self.x2 is not None else []
        self.cols_all += self.y if self.y is not None else []
        self.cols_all += self.y2 if self.y2 is not None else []
        self.cols_all += self.z if self.z is not None else []
        self.cols_all += self.col if self.col is not None else []
        self.cols_all += self.row if self.row is not None else []
        self.cols_all += self.wrap if self.wrap is not None else []
        if isinstance(self.legend, list):
            self.cols_all += self.legend

        # Add all non-dataframe kwargs to self
        del kwargs['df']  # for memory
        self.kwargs = kwargs

        # Swap x and y axes
        if self.swap:
            self.swap_xy()

        # Transform data
        self.transform()

        # Other kwargs
        for k, v in kwargs.items():
            if not hasattr(self, k):  # k not in ['df', 'func', 'x', 'y', 'z']:
                setattr(self, k, v)

    @property
    def _groupers(self) -> list:
        """Get all grouping values."""
        props = ['row', 'col', 'wrap', 'groups', 'legend', 'fig']
        grouper = []

        for prop in props:
            if getattr(self, prop) not in ['x', 'y', None]:
                grouper += utl.validate_list(getattr(self, prop))

        return list(set(grouper))

    def _add_range(self, ir: int, ic: int, ax: str, label: str, value: [None, float]):
        """Add a range value unless it already exists.

        Args:
            ir: current axes row index
            ic: current axes column index
            ax: name of the axis ('x', 'y', etc)
            label: range type name ('min', 'max')
            value: value of the range
        """
        key = '{}{}'.format(ax, label)
        if key not in self.ranges[ir, ic].keys() or \
                self.ranges[ir, ic][key] is None:
            self.ranges[ir, ic][key] = value

    def _add_ranges_none(self, ir: int, ic: int):
        """Add None for all range values.

        Args:
            ir: current axes row index
            ic: current axes column index
        """
        for ax in ['x', 'x2', 'y', 'y2', 'z']:
            for mm in ['min', 'max']:
                self._add_range(ir, ic, ax, mm, None)

    def _ax_limit_pad(self, **kwargs):
        """Set padding limits for axis.

        Args:
            kwargs: user-defined keyword args
        """
        self.ax_limit_padding = utl.kwget(kwargs, self.fcpp, 'ax_limit_padding', 0.05)
        for ax in ['x', 'x2', 'y', 'y2', 'z']:
            if not hasattr(self, 'ax_limit_padding_%smin' % ax):
                setattr(self, 'ax_limit_padding_%smin' % ax,
                        utl.kwget(kwargs, self.fcpp,
                                  'ax_limit_padding_%smin' % ax, self.ax_limit_padding))
            if not hasattr(self, 'ax_limit_padding_%smax' % ax):
                setattr(self, 'ax_limit_padding_%smax' % ax,
                        utl.kwget(kwargs, self.fcpp,
                                  'ax_limit_padding_%smax' % ax, self.ax_limit_padding))

    def _check_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the dataframe.

        Args:
            df: input DataFrame

        Returns:
            copy of the now-validated DataFrame (copy avoids column name changes
            with data filtering)
        """
        if df is None:
            raise DataError('Must provide a DataFrame for plotting!')

        if len(df) == 0:
            raise DataError('DataFrame is empty.  Nothing to plot!')

        return df.copy()

    def _check_group_columns(self, group_type: str, col_names: [str, list, None]):
        """Check wrap/row/column grouping variables for errors.

        Args:
            group_type: type of grouping (row, col, leg, wrap)
            col_names: name(s) of the column by which to group or None

        """
        # Force list type
        values = utl.validate_list(col_names)

        # Check that each value exists in the dataframe
        if values is None:
            return

        for val in values:
            if val not in self.df_all.columns:
                raise GroupingError('Grouping column "%s" is not in the DataFrame!' % val)

        # Check for no groups
        if len(list(self.df_all.groupby(values).groups.keys())) == 0:
            raise GroupingError(f'There are no unique groups in the data for the {group_type}=[{", ".join(values)}]')

        # Check for wrap with twinning
        if group_type == 'wrap' and col_names is not None and self.twin_y:
            raise GroupingError('Wrap plots do not support twinning of the y-axis. '
                                'Please consider a row vs column plot instead.')
        if group_type == 'wrap' and col_names is not None and self.twin_x:
            raise GroupingError('Wrap plots do not support twinning of the x-axis. '
                                'Please consider a row vs column plot instead.')

        return values

    def _check_group_errors(self):
        """Check for common errors related to grouping keywords."""
        if self.row and len(self.row) > 1 or self.col and len(self.col) > 1:
            error = 'Only one value can be specified for "%s"' % ('row' if self.row else 'col')
            raise GroupingError(error)

        if self.row is not None and self.row == self.col:
            raise GroupingError('Row and column values must be different!')

        if self.wrap is not None and (self.col or self.row):
            error = 'Cannot combine "wrap" grouping with "%s"' % ('col' if self.col else 'row')
            raise GroupingError(error)

        if self.groups is not None:
            for val in ['row', 'col', 'wrap']:
                if not getattr(self, val):
                    continue
                if len(set(getattr(self, val)) & set(self.groups)) > 0:
                    error = f'"{val}" value(s) cannot also be specified as value in "groups"'
                    raise GroupingError(error)

    def _check_group_matching(self, group1: str, group2: str):
        """Check to make sure certain group column values are not the same.

        Args:
            group1: attr name of first grouping column
            group2: attr name of second grouping column
        """
        equal = set(str(getattr(self, group1))) == set(str(getattr(self, group2)))

        if equal:
            raise GroupingError('%s and %s grouping columns cannot be the same!'
                                % (group1, group2))

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.

        Args:
            xyz: name of variable to check

        TODO: add option to recast non-float/datetime column as categorical str
        """
        if xyz not in self.req and xyz not in self.opt:
            return

        if xyz in self.opt and getattr(self, xyz) is None:
            return

        vals = getattr(self, xyz)

        if vals is None and xyz not in self.opt:
            raise AxisError('Must provide a column name for "%s"' % xyz)

        for val in vals:
            if val not in self.df_all.columns:
                raise DataError('No column named "%s" found in DataFrame' % val)

            # Check case
            try:
                self.df_all[val] = self.df_all[val].astype(float)
                continue
            except ValueError:
                pass
            try:
                self.df_all[val] = self.df_all[val].astype('datetime64[ns]')
                # if all are 00:00:00 time, leave only date
                if len(self.df_all.loc[self.df_all[val].dt.hour != 0, val]) == 0 and \
                        len(self.df_all.loc[self.df_all[val].dt.minute != 0, val]) == 0 and \
                        len(self.df_all.loc[self.df_all[val].dt.second != 0, val]) == 0:
                    self.df_all[val] = pd.DatetimeIndex(self.df_all[val]).date
                continue
            except:  # noqa
                continue
            #     raise AxisError('Could not convert x-column "%s" to float or '
                #  'datetime.' % val)

        # Check for axis errors
        if self.twin_x and len(self.y) != 2:
            raise AxisError('twin_x error! %s y values were specified but two are required' % len(self.y))
        if self.twin_x and len(self.x) > 1:
            raise AxisError('twin_x error! only one x value can be specified')
        if self.twin_y and len(self.x) != 2:
            raise AxisError('twin_y error! %s x values were specified but two are required' % len(self.x))
        if self.twin_y and len(self.y) > 1:
            raise AxisError('twin_y error! only one y value can be specified')

        return vals

    def _filter_data(self, kwargs):
        """Apply an optional filter to the data.

        Args:
            kwargs: user-defined keyword args
        """
        if self.filter:
            self.df_all = utl.df_filter(self.df_all, self.filter)
            if len(self.df_all) == 0:
                raise DataError('DataFrame is empty after applying filter')

    def _get_auto_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-scale the plot data.

        Args:
            df: either self.df_fig or self.df_rc

        Returns:
            updated df with potentially reduced range of data
        """
        if not self.auto_scale:
            return df

        # get the max/min autoscale values
        for ax in self.axs:
            for mm in ['min', 'max']:
                for ir, ic, _ in self._get_subplot_index():
                    key = '{}{}'.format(ax, mm)
                    if ir == 0 and ic == 0 and key in self.ranges[ir][ic].keys():
                        auto_scale_val = self.ranges[ir][ic][key]
                    elif key in self.ranges[ir][ic].keys():
                        if mm == 'min':
                            auto_scale_val = min(auto_scale_val, self.ranges[ir][ic][key])
                        else:
                            auto_scale_val = max(auto_scale_val, self.ranges[ir][ic][key])
                    else:
                        auto_scale_val = None
                if isinstance(auto_scale_val, str) or auto_scale_val is None:
                    continue

                axx = getattr(self, ax)
                for col in axx:
                    if mm == 'min':
                        df = df[df[col] >= auto_scale_val]
                    else:
                        df = df[df[col] <= auto_scale_val]

        return df

    def _get_data_ranges(self):
        """Default data range calculator by subplot.
        --> Some plot types need to override this func with a custom calc.
        """
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self._get_auto_scale(df_fig)

        # Apply shared axes
        for ir, ic, plot_num in self._get_subplot_index():
            for ax in self.axs:
                # Share axes (use self.df_fig to get global min/max)
                if getattr(self, 'share_%s' % ax) and (ir > 0 or ic > 0):
                    self._add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])
                elif getattr(self, 'share_%s' % ax):
                    vals = self._get_data_range(ax, df_fig, plot_num)
                    self._add_range(ir, ic, ax, 'min', vals[0])
                    self._add_range(ir, ic, ax, 'max', vals[1])

                # Share row
                elif self.share_row and self.row is not None and ic > 0 and self.row != 'y':
                    self._add_range(ir, ic, ax, 'min', self.ranges[ir, 0]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[ir, 0]['%smax' % ax])

                elif self.share_row and self.row is not None:
                    if self.row == 'y':
                        df_sub = pd.DataFrame(columns=[self.x[0], self.y[ir]], index=range(len(df_fig)))
                        df_sub[self.x[0]] = df_fig[self.x[0]].values
                        df_sub[self.y[ir]] = df_fig[self.y[ir]].values
                        vals = self._get_data_range(ax, df_sub, plot_num)
                    else:
                        vals = self._get_data_range(ax, df_fig[self.df_fig[self.row[0]] == self.row_vals[ir]], plot_num)
                    self._add_range(ir, ic, ax, 'min', vals[0])
                    self._add_range(ir, ic, ax, 'max', vals[1])

                # Share col
                elif self.share_col and self.col is not None and ir > 0 and self.col != 'x':
                    self._add_range(ir, ic, ax, 'min', self.ranges[0, ic]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[0, ic]['%smax' % ax])
                elif self.share_col and self.col is not None:
                    if self.col == 'x':
                        df_sub = pd.DataFrame(columns=[self.x[ic], self.y[0]], index=range(len(df_fig)))
                        df_sub[self.x[ic]] = df_fig[self.x[ic]].values
                        df_sub[self.y[0]] = df_fig[self.y[0]].values
                        vals = self._get_data_range(ax, df_sub, plot_num)
                    else:
                        vals = self._get_data_range(ax, df_fig[df_fig[self.col[0]] == self.col_vals[ic]], plot_num)
                    self._add_range(ir, ic, ax, 'min', vals[0])
                    self._add_range(ir, ic, ax, 'max', vals[1])

                # subplot level when not shared
                else:
                    df_rc = self._subset(ir, ic)

                    # Empty rc
                    if len(df_rc) == 0:  # this doesn't exist yet!
                        self._add_range(ir, ic, ax, 'min', None)
                        self._add_range(ir, ic, ax, 'max', None)
                        continue

                    # Not shared or wrap by x or y
                    elif not getattr(self, 'share_%s' % ax) or \
                            (self.wrap is not None and self.wrap == 'y' or self.wrap == 'x'):
                        vals = self._get_data_range(ax, df_rc, plot_num)
                        self._add_range(ir, ic, ax, 'min', vals[0])
                        self._add_range(ir, ic, ax, 'max', vals[1])

                    # # Make them all equal to window [0,0]
                    # elif ir > 0 or ic > 0:
                    #     self._add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                    #     self._add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])

    def _get_data_ranges_user_defined(self):
        """Get user defined range values that were set by kwargs."""
        for ir, ic, plot_num in self._get_subplot_index():
            for ax in self.axs:
                for mm in ['min', 'max']:
                    # User defined
                    key = '{}{}'.format(ax, mm)
                    val = getattr(self, key)[plot_num]
                    if val is not None and not isinstance(val, str):
                        self._add_range(ir, ic, ax, mm, val)

    def get_interval_confidence(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> None:
        """Calculate confidence intervals point by point for a curve.

        Args:
            df: data subset
            x: x column name
            y: y column name
            kwargs: user-defined keyword args

        Returns:
            None but calculates ucl and lcl data columns
        """
        if float(self.interval[0]) > 1:
            self.interval[0] = float(self.interval[0]) / 100
        stat = pd.DataFrame()
        stat['mean'] = df[[x, y]].groupby(x).mean().reset_index()[y]
        stat['count'] = df[[x, y]].groupby(x).count().reset_index()[y]
        stat['std'] = df[[x, y]].groupby(x).std().reset_index()[y]
        stat['sderr'] = stat['std'] / np.sqrt(stat['count'])
        stat['ucl'] = np.nan
        stat['lcl'] = np.nan
        for irow, row in stat.iterrows():
            if row['std'] == 0:
                conf = [row['mean'], row['mean']]
            else:
                conf = ss.t.interval(self.interval[0], int(row['count']) - 1,
                                     loc=row['mean'], scale=row['sderr'])
            stat.loc[irow, 'ucl'] = conf[1]
            stat.loc[irow, 'lcl'] = conf[0]

        self.stat_idx = df.groupby(x).mean().index
        self.lcl = stat['lcl']
        self.ucl = stat['ucl']

    def get_interval_nq(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> None:
        """Calculate normal quantile intervals point by point for a curve.

        Args:
            df: data subset
            x: x column name
            y: y column name
            kwargs: user-defined keyword args

        Returns:
            None but calculates ucl and lcl data columns
        """
        # figure out the nq range kwargs
        decimals = max(utl.get_decimals(self.interval[0]),
                       utl.get_decimals(self.interval[1]))
        if self.interval[0] >= -3 and self.interval[1] <= 3:
            step_inner = 1 / 10**decimals
            step_tail = 1
        else:
            step_inner = 1
            step_tail = 1 / 10**decimals

        # also what about not in range?
        lcl, ucl = [], []
        lower = df[[x, y]].groupby(x).quantile(0.5)  # is this the best way?

        for idx, (nn, gg) in enumerate(df[[x, y]].groupby(x)):
            nq = utl.nq(gg[y], step_inner=step_inner, step_tail=step_tail)
            nq.Sigma = nq.Sigma.apply(lambda x: round(x, decimals))
            if self.interval[0] in nq.Sigma.values:
                lcl += [nq.loc[nq.Sigma == self.interval[0], 'Value'].iloc[0]]
            else:
                closest = nq['Sigma'].sub(self.interval[0]).abs().idxmin()
                sigma = nq.Sigma.iloc[closest]
                if idx == 0:
                    print(f'IntervalCalc: sigma={self.interval[0]} not available; using closest value of {sigma}')
                lcl += [nq.Value.iloc[closest]]
            if self.interval[1] in nq.Sigma.values:
                ucl += [nq.loc[nq.Sigma == self.interval[1], 'Value'].iloc[0]]
            else:
                closest = nq['Sigma'].sub(self.interval[1]).abs().idxmin()
                sigma = nq.Sigma.iloc[closest]
                if idx == 0:
                    print(f'IntervalCalc: sigma={self.interval[1]} not available; using closest value of {sigma}')
                ucl += [nq.Value.iloc[closest]]

        self.lcl = pd.Series(lcl)
        self.ucl = pd.Series(ucl)
        self.stat_idx = lower.reset_index()[x]

    def get_interval_percentile(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> None:
        """Calculate percentile intervals point by point for a curve.

        Args:
            df: data subset
            x: x column name
            y: y column name
            kwargs: user-defined keyword args

        Returns:
            None but calculates ucl and lcl data columns

        """
        lower = df[[x, y]].groupby(x).quantile(self.interval[0])
        self.lcl = lower.reset_index()[y]
        self.ucl = df[[x, y]].groupby(x).quantile(self.interval[1]).reset_index()[y]
        self.stat_idx = lower.reset_index()[x]

    def _get_data_range(self, ax: str, df: pd.DataFrame, plot_num: int) -> tuple:
        """Determine the min/max values for a given axis based on user inputs.

        Args:
            ax: name of the axis ('x', 'y', etc)
            df: data table to use for range calculation
            plot_num: index number of the current subplot

        Returns:
            min, max tuple
        """
        if not hasattr(self, ax) or getattr(self, ax) in [None, []]:
            return None, None
        elif self.col == 'x' and self.share_x and ax == 'x':
            cols = self.x_vals
        elif self.col == 'x' and self.share_col and ax == 'x':
            cols = [f for f in self.x_vals if f in df.columns]
        elif self.row == 'y' and self.share_y and ax == 'y':
            cols = self.y_vals
        elif self.row == 'y' and self.share_row and ax == 'y':
            cols = [f for f in self.y_vals if f in df.columns]
        else:
            cols = getattr(self, ax)

        # # Groupby for stats
        # df = self._get_stat_groupings(df)

        # Get the dataframe values for this axis
        dfax = df[cols]

        # Check dtypes
        dtypes = dfax.dtypes.unique()
        if 'str' in dtypes or 'object' in dtypes or 'datetime64[ns]' in dtypes:
            return None, None

        # Calculate actual min / max vals for the axis
        if self.ax_scale in ['log%s' % ax, 'loglog', 'semilog%s' % ax, 'log']:
            axmin = dfax[dfax > 0].stack().min()
            axmax = dfax.stack().max()
            axdelta = np.log10(axmax) - np.log10(axmin)
        else:
            axmin = dfax.stack().min()
            axmax = dfax.stack().max()
            axdelta = axmax - axmin
        if axdelta <= 0:
            axmin -= 0.1 * axmin
            axmax += 0.1 * axmax

        # Check user-specified min values
        vmin = getattr(self, '%smin' % ax)[plot_num]
        if vmin is not None and 'iqr' in str(vmin).lower():
            factor = str(vmin).split('*')
            factor = float(factor[0])
            if 'box' not in self.name or self.groups is None:
                q1 = dfax.quantile(0.25).min()
                q3 = dfax.quantile(0.75).max()
                iqr = factor * (q3 - q1)
                vmin = q1 - iqr
            else:
                q1 = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(0.25)[cols].reset_index()
                q3 = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(0.75)[cols].reset_index()
                iqr = factor * (q3[cols] - q1[cols])
                vmin = (q1[cols] - iqr[cols]).min().iloc[0]
        elif vmin is not None and 'q' in str(vmin).lower():
            xq = float(str(vmin).lower().replace('q', '')) / 100
            if self.groups is None:
                vmin = dfax.quantile(xq).min()
            elif 'box' in self.name:
                vmin = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(xq)[cols].min().iloc[0]
            else:
                vmin = df[self.groups + cols].groupby(self.groups) \
                    .quantile(xq)[cols].min().iloc[0]
        elif vmin is not None:
            vmin = vmin
        elif getattr(self, 'ax_limit_padding_%smin' % ax) is not None:
            if self.ax_scale in ['log%s' % ax, 'loglog',
                                 'semilog%s' % ax, 'log']:
                axmin = np.log10(axmin) - \
                    getattr(self, 'ax_limit_padding_%smin' % ax) * axdelta
                vmin = 10**axmin
            else:
                axmin -= getattr(self, 'ax_limit_padding_%smin' % ax) * axdelta
                vmin = axmin
        else:
            vmin = None

        # Check user-specified max values
        vmax = getattr(self, '%smax' % ax)[plot_num]
        if vmax is not None and 'iqr' in str(vmax).lower():
            factor = str(vmax).split('*')
            factor = float(factor[0])
            if 'box' not in self.name or self.groups is None:
                q1 = dfax.quantile(0.25).min()
                q3 = dfax.quantile(0.75).max()
                iqr = factor * (q3 - q1)
                vmax = q3 + iqr
            else:
                q1 = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(0.25)[cols].reset_index()
                q3 = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(0.75)[cols].reset_index()
                iqr = factor * (q3[cols] - q1[cols])
                # should this be referred to median?
                vmax = (q3[cols] + iqr[cols]).max().iloc[0]
        elif vmax is not None and 'q' in str(vmax).lower():
            xq = float(str(vmax).lower().replace('q', '')) / 100
            if self.groups is None:
                vmax = dfax.quantile(xq).max()
            elif 'box' in self.name:  # move to data.box.py?
                vmax = df[self._groupers + cols].groupby(self._groupers) \
                    .quantile(xq)[cols].max().iloc[0]
            else:
                vmax = df[self.groups + cols].groupby(self.groups) \
                    .quantile(xq)[cols].max().iloc[0]
        elif vmax is not None:
            vmax = vmax
        elif getattr(self, 'ax_limit_padding_%smax' % ax) is not None:
            if self.ax_scale in ['log%s' % ax, 'loglog',
                                 'semilog%s' % ax, 'log']:
                axmax = np.log10(axmax) + \
                    getattr(self, 'ax_limit_padding_%smax' % ax) * axdelta
                vmax = 10**axmax
            else:
                axmax += getattr(self, 'ax_limit_padding_%smax' % ax) * axdelta
                vmax = axmax
        else:
            vmax = None

        if type(vmin) in [float, np.float32, np.float64] and np.isnan(vmin):
            vmin = None
        if type(vmax) in [float, np.float32, np.float64] and np.isnan(vmax):
            vmax = None

        return vmin, vmax

    def get_df_figure(self):
        """Generator to subset the main DataFrame based on fig_item grouping.

        Yields:
            figure index (None if no self.fig_vals)
            figure value (i.e., unique value in the self.fig DataFrame column)
            figure column name
            self
        """
        self._get_fig_groupings()

        if not self.fig_vals:
            # no fig grouping
            self._get_legend_groupings(self.df_all)
            self._get_rc_groupings(self.df_all)
            self.df_fig = self.df_all
            self._get_data_ranges()

            yield None, None, None, self

        else:
            # with fig grouping
            for ifig, fig_val in enumerate(self.fig_vals):
                if isinstance(fig_val, tuple):
                    for ig, gg in enumerate(fig_val):
                        if ig == 0:
                            self.df_fig = self.df_all[self.df_all[self.fig_groups[ig]] == gg].copy()
                        else:
                            self.df_fig = self.df_fig[self.df_fig[self.fig_groups[ig]] == gg]
                elif isinstance(self.fig_groups, list):
                    self.df_fig = self.df_all[self.df_all[self.fig_groups[0]] == fig_val].copy()
                else:
                    self.df_fig = self.df_all[self.df_all[self.fig_groups] == fig_val].copy()

                self._get_legend_groupings(self.df_fig)
                self._get_rc_groupings(self.df_fig)
                self._get_data_ranges()
                yield ifig, fig_val, self.fig, self

        self.df_fig = None

    def _get_fig_groupings(self):
        """Determine the figure grouping levels."""
        if self.fig:
            self.fig_vals = list(self.df_all.groupby(self.fig).groups.keys())

    def get_fit_data(self, ir: int, ic: int, df: pd.DataFrame, x: str,
                     y: str) -> [pd.DataFrame, list, float]:
        """Make new columns of fit data.

        Args:
            ir: current axes row index
            ic: current axes column index
            df: data subset
            x: x column name
            y: y column name

        Returns:
            updated DataFrame
            fit coefficients list
            rsq (for poly fit only)
        """
        df2 = df.copy()
        df['%s Fit' % x] = np.nan
        df['%s Fit' % y] = np.nan

        if self.fit is True or isinstance(self.fit, int):
            # Set range of the fit
            if isinstance(self.fit_range_x, list):
                df2 = df2[(df2[x] >= self.fit_range_x[0])
                          & (df2[x] <= self.fit_range_x[1])].copy()
                if self.ranges[ir, ic]['ymin'] is not None:
                    df2 = df2[(df2[y]) >= self.ranges[ir, ic]['ymin']]
                if self.ranges[ir, ic]['ymax'] is not None:
                    df2 = df2[(df2[y]) <= self.ranges[ir, ic]['ymax']]
            elif isinstance(self.fit_range_y, list):
                df2 = df2[(df2[y] >= self.fit_range_y[0])
                          & (df2[y] <= self.fit_range_y[1])].copy()
                if self.ranges[ir, ic]['xmin'] is not None:
                    df2 = df2[(df2[x]) >= self.ranges[ir, ic]['xmin']]
                if self.ranges[ir, ic]['xmax'] is not None:
                    df2 = df2[(df2[x]) <= self.ranges[ir, ic]['xmax']]
            else:
                df2 = df2.copy()
                if self.ranges[ir, ic]['xmin'] is not None:
                    df2 = df2[(df2[x]) >= self.ranges[ir, ic]['xmin']]
                if self.ranges[ir, ic]['xmax'] is not None:
                    df2 = df2[(df2[x]) <= self.ranges[ir, ic]['xmax']]
                if self.ranges[ir, ic]['ymin'] is not None:
                    df2 = df2[(df2[y]) >= self.ranges[ir, ic]['ymin']]
                if self.ranges[ir, ic]['ymax'] is not None:
                    df2 = df2[(df2[y]) <= self.ranges[ir, ic]['ymax']]

            # Convert to arrays
            xx = np.array(df2[x])
            yy = np.array(df2[y])
            if len(xx) == 0 or len(yy) == 0:
                return df, np.ones(int(self.fit)) * np.nan, 0

            # Fit the polynomial
            coeffs = np.polyfit(xx, yy, int(self.fit))

            # Find R^2
            yval = np.polyval(coeffs, xx)
            ybar = yy.sum() / len(yy)
            ssreg = np.sum((yval - ybar)**2)
            sstot = np.sum((yy - ybar)**2)
            rsq = ssreg / sstot

            # Add fit line
            df['%s Fit' % x] = np.linspace(self.ranges[ir, ic]['xmin'],
                                           self.ranges[ir, ic]['xmax'], len(df))
            df['%s Fit' % y] = np.polyval(coeffs, df['%s Fit' % x])

            return df, coeffs, rsq

        # TODO:: put spline code here
        # if str(self.fit).lower() == 'spline':

        #     return df, [], np.nan

    def _get_legend_groupings(self, df: pd.DataFrame):
        """Determine the legend groupings.

        Args:
            df: data subset
        """
        if self.legend is True and self.twin_x or self.legend is True and len(self.y) > 1:
            self.legend_vals = self.y + self.y2
            self.nleg_vals = len(self.y + self.y2)
            return
        elif self.legend is True and self.twin_y:
            self.legend_vals = self.x + self.x2
            self.nleg_vals = len(self.x + self.x2)
            return

        if not self.legend:
            return

        leg_all = []

        if self.legend is True:
            self.legend = None  # no option for legend here so disable
            return

        if self.legend:
            if isinstance(self.legend, str) and ' | ' in self.legend:
                self.legend = self.legend.split(' | ')
            if isinstance(self.legend, list):
                for ileg, leg in enumerate(self.legend):
                    if ileg == 0:
                        temp = df[leg].copy()
                    else:
                        temp = temp.map(str) + ' | ' + df[leg].map(str)
                self.legend = ' | '.join(self.legend)
                df[self.legend] = temp
            if self.sort:
                legend_vals = natsorted(list(df.groupby(self.legend).groups.keys()))
            else:
                legend_vals = list(df.groupby(self.legend, sort=False).groups.keys())
            self.nleg_vals = len(legend_vals)
        else:
            legend_vals = [None]
            self.nleg_vals = 0

        for leg in legend_vals:
            if not self.x or self.name == 'gantt':
                selfx = [None]
            else:
                selfx = self.x + self.x2
            if not self.y:
                selfy = [None]
            else:
                selfy = self.y + self.y2
            for xx in selfx:
                for yy in selfy:
                    leg_all += [(leg, xx, yy)]

        leg_df = pd.DataFrame(leg_all, columns=['Leg', 'x', 'y'])

        # if leg specified
        if not (leg_df.Leg.isnull()).all():
            leg_df['names'] = list(leg_df.Leg)

        # if more than one y axis and leg specified
        if self.wrap == 'y' or self.wrap == 'x':
            leg_df = leg_df.drop(self.wrap, axis=1).drop_duplicates()
            leg_df[self.wrap] = self.wrap
        elif self.row == 'y':
            del leg_df['y']
            leg_df = leg_df.drop_duplicates().reset_index(drop=True)
        elif self.col == 'x':
            del leg_df['x']
            leg_df = leg_df.drop_duplicates().reset_index(drop=True)
        elif len(leg_df.y.unique()) > 1 and not (leg_df.Leg.isnull()).all() \
                and len(leg_df.x.unique()) == 1:
            leg_df['names'] = leg_df.Leg.map(str) + ' | ' + leg_df.y.map(str)

        # if more than one x and leg specified
        if 'names' not in leg_df.columns:
            leg_df['names'] = leg_df.x
        elif 'x' in leg_df.columns and len(leg_df.x.unique()) > 1 \
                and not self.twin_x:
            leg_df['names'] = \
                leg_df['names'].map(str) + ' | ' + \
                leg_df.y.map(str) + ' / ' + leg_df.x.map(str)

        leg_df = leg_df.set_index('names')
        self.legend_vals = leg_df.reset_index()

    def get_plot_data(self, df: pd.DataFrame):
        """Generator to subset into discrete sets of data for each curve.

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
            xx = [] if not self.x else self.x + self.x2
            yy = [] if not self.y else self.y + self.y2
            lenx = 1 if not self.x else len(xx)
            leny = 1 if not self.y else len(yy)
            vals = pd.DataFrame({'x': self.x if not self.x else xx * leny,
                                 'y': self.y if not self.y else yy * lenx})

            for iline, row in vals.iterrows():
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
                    iline = self.wrap_vals.index(leg)

                yield iline, df, row['x'], row['y'], \
                    None if self.z is None else self.z[0], leg, twin, len(vals)

        else:
            for iline, row in self.legend_vals.iterrows():
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
                    df2 = df[df[self.legend] == row['Leg']].copy()

                # Filter out all nan data
                if row['x'] and row['x'] in df2.columns and len(df2[row['x']].dropna()) == 0 \
                        or row['y'] and row['y'] in df2.columns and len(df2[row['y']].dropna()) == 0:
                    continue

                # Set twin ax status
                twin = False
                if (row['x'] != self.legend_vals.loc[0, 'x'] and self.twin_y) \
                        or (row['y'] != self.legend_vals.loc[0, 'y'] and self.twin_x):
                    twin = True
                yield iline, df2, row['x'], row['y'], \
                    None if self.z is None else self.z[0], row['names'], \
                    twin, len(self.legend_vals)

    def _get_rc_groupings(self, df: pd.DataFrame):
        """Determine the row and column or wrap grid groupings.

        Args:
            df: data subset
        """
        # Set up wrapping (wrap option overrides row/col)
        if self.wrap:
            if self.wrap_vals is None:  # this broke something but removing will cause other failures
                if self.sort:
                    self.wrap_vals = natsorted(list(df.groupby(self.wrap).groups.keys()))
                else:
                    self.wrap_vals = list(df.groupby(self.wrap, sort=False).groups.keys())
            if self.ncols == 0:
                rcnum = int(np.ceil(np.sqrt(len(self.wrap_vals))))
            else:
                rcnum = self.ncols if self.ncols <= len(self.wrap_vals) else len(self.wrap_vals)
            self.ncol = rcnum
            self.nrow = int(np.ceil(len(self.wrap_vals) / rcnum))
            self.nwrap = len(self.wrap_vals)

        # Non-wrapping option
        else:
            # Set up the row grouping
            if self.col:
                if self.col_vals is None:
                    if self.sort:
                        self.col_vals = natsorted(list(df.groupby(self.col).groups.keys()))
                    else:
                        self.col_vals = list(df.groupby(self.col, sort=False).groups.keys())
                self.ncol = len(self.col_vals)

            if self.row:
                if self.row_vals is None:
                    if self.sort:
                        self.row_vals = natsorted(list(df.groupby(self.row).groups.keys()))
                    else:
                        self.row_vals = list(df.groupby(self.row, sort=False).groups.keys())
                self.nrow = len(self.row_vals)

        if self.ncol == 0:
            raise GroupingError('Cannot make subplot(s): number of columns is 0')  # can this ever happen?
        if self.nrow == 0:
            raise GroupingError('Cannot make subplot(s): number of rows is 0')

        self.ranges = self._range_dict()

    def _get_subplot_index(self):
        """Get an index for each subplot based on row x column numbers

        Yields:
            ir: subplot row index
            ic: subplot column index
            int index value
        """
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                yield ir, ic, utl.plot_num(ir, ic, self.ncol) - 1

    def get_rc_subset(self):
        """Subset the data by the row/col/wrap values.

        Yields:
            ir: subplot row index
            ic: subplot column index
            row/col data subset
        """
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.df_rc = self._subset(ir, ic)

                # Handle empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

                # Yield the subset
                yield ir, ic, self.df_rc

        self.df_sub = None

    def get_stat_data(self, df: pd.DataFrame, x: str, y: str) -> [pd.DataFrame, None]:
        """Get a stat subset from input data.

        Args:
            df: data subset
            x: x-axis column name
            y: y-axis column name

        Returns:
            DataFrame with the new calculated stats or None
        """
        if not self.stat:
            return pd.DataFrame()

        if 'q' in self.stat:
            df = df.select_dtypes(exclude=['object'])  # only needed for older versions of pandas
        df_stat = df.groupby(x if not self.stat_val else self.stat_val)
        if 'q' in self.stat:
            q = float(self.stat.replace('q', ''))
            if q > 1:
                q = q / 100
            return df_stat.quantile(q)
        else:
            try:
                return getattr(df_stat, self.stat)().reset_index()
            except AttributeError:
                print('stat "%s" is not supported...skipping stat calculation' % self.stat)
                return None

    def _range_dict(self):
        """Make a list of empty dicts for axes range limits."""
        ranges = np.array([[None] * self.ncol] * self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                ranges[ir, ic] = {}

        return ranges

    def _subset(self, ir: int, ic: int) -> pd.DataFrame:
        """Handles creation of a new data subset based on the type of plot selected.

        Args:
            ir: subplot row index
            ic: subplot column index

        Returns:
            pandas DataFrame subset
        """
        # Wrap plot
        if self.wrap is not None:
            df = self._subset_wrap(ir, ic)

        # Non-wrap plot
        else:
            df = self._subset_row_col(ir, ic)

        # Optional plot specific subset modification
        df = self._subset_modify(ir, ic, df)

        return df

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Optional function in a Data childe class user to perform any additional
        DataFrame subsetting that may be required

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        return df

    def _subset_row_col(self, ir: int, ic: int) -> pd.DataFrame:
        """For row/column plots (no wrap), select the revelant subset from
        self.df_fig.

        Args:
            ir: subplot row index
            ic: subplot column index

        Returns:
            self.df_fig DataFrame subset based on row/column values
        """
        if self.row == 'y':
            self.y = utl.validate_list(self.row_vals[ir])
        if self.col == 'x':
            self.x = utl.validate_list(self.col_vals[ic])
        if self.row not in [None, 'y'] and self.col not in [None, 'x']:
            row = self.row_vals[ir]
            col = self.col_vals[ic]
            return self.df_fig[(self.df_fig[self.row[0]] == row) & (self.df_fig[self.col[0]] == col)].copy()
        elif self.row not in [None, 'y'] and (not self.col or self.col in [None, 'x']):
            row = self.row_vals[ir]
            return self.df_fig[(self.df_fig[self.row[0]] == row)].copy()
        elif self.col not in [None, 'x'] and (not self.row or self.row in [None, 'y']):
            col = self.col_vals[ic]
            return self.df_fig[(self.df_fig[self.col[0]] == col)].copy()
        else:
            return self.df_fig.copy()

    def _subset_wrap(self, ir: int, ic: int) -> pd.DataFrame:
        """For wrap plots, select the revelant subset from self.df_fig.

        Args:
            ir: subplot row index
            ic: subplot column index

        Returns:
            self.df_fig DataFrame subset based on self.wrap value
        """
        if ir * self.ncol + ic > self.nwrap - 1:
            return pd.DataFrame()
        elif self.wrap == 'y':
            # NOTE: can we drop these validate calls for speed?
            self.y = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) \
                + (self.y if self.y is not None else []) \
                + (self.groups if self.groups is not None else []) \
                + (utl.validate_list(self.legend)
                   if self.legend not in [None, True, False] else [])
            return self.df_fig[cols]
        elif self.wrap == 'x':
            self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend)
                    if self.legend is not None else [])
            return self.df_fig[cols]
        else:
            if self.sort:
                self.wrap_vals = \
                    natsorted(list(self.df_fig.groupby(self.wrap).groups.keys()))
            else:
                self.wrap_vals = list(self.df_fig.groupby(self.wrap, sort=False).groups.keys())
            wrap = dict(zip(self.wrap,
                        utl.validate_list(self.wrap_vals[ir * self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

    def swap_xy(self):
        """Swap the x and y axis attributes."""
        # Axis values
        x = self.x
        x2 = self.x2
        self.x = self.y
        self.x2 = self.y2
        self.y = x
        self.y2 = x2

        # Trans
        trans_x = self.trans_x
        trans_x2 = self.trans_x2
        self.trans_x = self.trans_y
        self.tras_x2 = self.trans_y2
        self.trans_y = trans_x
        self.trans_y2 = trans_x2

    def swap_xy_ranges(self):
        """Swap the x and y range values (used in case of horizontal plots)."""
        for ir, ic, plot_num in self._get_subplot_index():
            xmin = self.ranges[ir, ic]['xmin']
            xmax = self.ranges[ir, ic]['xmax']
            x2min = self.ranges[ir, ic]['x2min']
            x2max = self.ranges[ir, ic]['x2max']
            self.ranges[ir, ic]['xmin'] = self.ranges[ir, ic]['ymin']
            self.ranges[ir, ic]['xmax'] = self.ranges[ir, ic]['ymax']
            self.ranges[ir, ic]['x2min'] = self.ranges[ir, ic]['y2min']
            self.ranges[ir, ic]['x2max'] = self.ranges[ir, ic]['y2max']
            self.ranges[ir, ic]['ymin'] = xmin
            self.ranges[ir, ic]['ymax'] = xmax
            self.ranges[ir, ic]['y2min'] = x2min
            self.ranges[ir, ic]['y2max'] = x2max

    def transform(self):
        """Transform x, y, or z data by unique group."""
        # Possible tranformations
        transform = any([self.trans_x, self.trans_x2, self.trans_y,
                         self.trans_y2, self.trans_z])
        if not transform:
            return

        # Container for transformed data
        df = pd.DataFrame()

        # Transform by unique group
        groups_all = self._groupers
        if len(groups_all) > 0:
            groups = self.df_all.groupby(groups_all)
        else:
            groups = [self.df_all]
        for group in groups:
            if isinstance(group, tuple):
                gg = group[1]
            else:
                gg = group

            axis = ['x', 'y', 'z']

            for ax in axis:
                vals = getattr(self, ax)
                if not vals:
                    continue
                for ival, val in enumerate(vals):
                    if getattr(self, 'trans_%s' % ax) == 'abs':
                        gg.loc[:, val] = abs(gg[val])
                    elif getattr(self, 'trans_%s' % ax) == 'negative' \
                            or getattr(self, 'trans_%s' % ax) == 'neg':
                        gg.loc[:, val] = -gg[val]
                    elif getattr(self, 'trans_%s' % ax) == 'nq':
                        if ival == 0:
                            gg = utl.nq(gg[val], val, **self.kwargs)
                    elif getattr(self, 'trans_%s' % ax) == 'inverse' \
                            or getattr(self, 'trans_%s' % ax) == 'inv':
                        gg.loc[:, val] = 1 / gg[val]
                    elif (isinstance(getattr(self, 'trans_%s' % ax), tuple)
                            or isinstance(getattr(self, 'trans_%s' % ax), list)) \
                            and getattr(self, 'trans_%s' % ax)[0] == 'pow':
                        gg.loc[:,
                               val] = gg[val]**getattr(self, 'trans_%s' % ax)[1]
                    elif getattr(self, 'trans_%s' % ax) == 'flip':
                        maxx = gg.loc[:, val].max()
                        gg.loc[:, val] -= maxx
                        gg.loc[:, val] = abs(gg[val])

            if isinstance(group, tuple):
                vals = group[0] if isinstance(group[0], tuple) else [group[0]]
                for k, v in dict(zip(groups_all, vals)).items():
                    gg[k] = v

            df = pd.concat([df, gg])

        self.df_all = df
