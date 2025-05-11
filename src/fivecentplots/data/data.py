import pdb
from natsort import natsorted
import scipy.stats as ss
import pandas as pd
import numpy as np
import numpy.typing as npt
import datetime
from typing import Union
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
        self.ax_scale = kwargs.get('ax_scale', None)
        self.ax2_scale = kwargs.get('ax2_scale', self.ax_scale)
        self.axs = ['x', 'x2', 'y', 'y2', 'z']  # all possible axes
        self._ax_limit_pad(**kwargs)
        self.cbar = utl.kwget(kwargs, self.fcpp, 'cbar', False)
        self.error_bars = False
        self.fit = kwargs.get('fit', False)
        self.fit_range_x = utl.kwget(kwargs, self.fcpp, 'fit_range_x', None)
        self.fit_range_y = utl.kwget(kwargs, self.fcpp, 'fit_range_y', None)
        self.imgs = utl.kwget(kwargs, self.fcpp, 'imgs', None)  # used only for image based plotting
        self.interval = utl.validate_list(kwargs.get('perc_int', kwargs.get('nq_int', kwargs.get('conf_int', False))))
        self.ignore_dates = kwargs.get('ignore_dates', False)
        self.legend = None
        self.legend_vals = None
        self.pivot = False
        self.ranges = None
        self.sample = int(utl.kwget(kwargs, self.fcpp, 'sample', 1))
        self.share_col = utl.kwget(kwargs, self.fcpp, 'share_col', False)
        self.share_row = utl.kwget(kwargs, self.fcpp, 'share_row', False)
        self.share_x = utl.kwget(kwargs, self.fcpp, 'share_x', True)
        self.share_x2 = utl.kwget(kwargs, self.fcpp, 'share_x2', True)
        self.share_y = utl.kwget(kwargs, self.fcpp, 'share_y', True)
        self.share_y2 = utl.kwget(kwargs, self.fcpp, 'share_y2', True)
        self.share_z = utl.kwget(kwargs, self.fcpp, 'share_z', True)
        if utl.kwget(kwargs, self.fcpp, 'cbar_shared', False):
            self.share_z = True
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
        if self.imgs is not None:
            # Quantile zmin/zmax should be independent
            for mm in ['zmin', 'zmax']:
                if kwargs.get(mm) is None:
                    continue
                zmms = utl.validate_list(kwargs.get(mm))
                for zmm in zmms:
                    if 'q' in str(zmm):
                        kwargs[mm] = zmms * int((len(self.imgs) / len(zmms)))
                        self.share_z = False
                        break
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

        # Validate range limits
        for limit in [ax + lim for ax in self.axs for lim in ['min', 'max']]:
            setattr(self, limit, utl.RepeatedList(kwargs.get(limit, [None]), limit))
            self._check_range_limit_format(limit)

        # Option to flip min/max range limits (used by some plot types)
        for ax in self.axs:
            setattr(self, f'invert_range_limits_{ax}', kwargs.get(f'invert_range_limits_{ax}', False))

        # Disable axes sharing if user provides
        for ax in self.axs:
            if len(getattr(self, f'{ax}min')) > 1 or len(getattr(self, f'{ax}max')) > 1:
                setattr(self, f'share_{ax}', False)

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
        self.axs_on = [f for f in self.axs if getattr(self, f) not in [None, []]]  # axes present in this plot only

        # Reference line
        self.ref_line = kwargs.get('ref_line', None)
        if isinstance(self.ref_line, pd.Series):
            self.df_all['Ref Line'] = self.ref_line

        # Stats
        self.stat = kwargs.get('stat', None)
        self.stat_val = kwargs.get('stat_val', None)
        if self.stat_val is not None and self.stat_val not in self.df_all.columns:
            raise DataError(f'stat_val column "{self.stat_val}" not in DataFrame')
        self.stat_idx = []
        self.lcl = kwargs.get('lcl', [])
        self.ucl = kwargs.get('ucl', [])

        # Apply an optional filter to the data
        self.filter = kwargs.get('filter', None)
        self._filter_data(kwargs)

        # Make sure no duplicate indices
        if self.df_all.index.has_duplicates:
            self.df_all = self.df_all.reset_index(drop=True)

        # Define rc grouping column names
        self.col = kwargs.get('col', None)
        self.col_vals = None
        if self.col is not None:
            if self.col == 'x':
                self.col_vals = [f for f in self.x]
            else:
                self.col = self._check_group_columns('col', kwargs.get('col', None))
        self.row = kwargs.get('row', None)
        self.row_vals = None
        if self.row is not None:
            if self.row == 'y':
                self.row_vals = [f for f in self.y]
            else:
                self.row = self._check_group_columns('row', kwargs.get('row', None))
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

        # Define legend grouping column names (legends are shared by a figure, not an rc subplot)
        if 'legend' in kwargs.keys():
            if kwargs['legend'] is True:
                self.legend = True
            elif kwargs['legend'] is False:
                self.legend = False
            else:
                self.legend = self._check_group_columns('legend', kwargs.get('legend', None))
        elif not self.twin_x and self.y is not None and len(self.y) > 1:
            self.legend = True

        # Define figure grouping column names
        if 'fig_groups' in kwargs.keys():
            self.fig = self._check_group_columns('fig', kwargs.get('fig_groups', None))
        else:
            self.fig = self._check_group_columns('fig', kwargs.get('fig', None))
        self._get_fig_groupings()
        self.fig_vals = None

        # Make sure groups, legend, and fig_groups are not the same
        if self.legend and self.groups and self.name != 'box':
            if not isinstance(self.legend, bool):
                self._check_group_matching('legend', 'groups')
        if self.legend and self.fig:
            self._check_group_matching('legend', 'fig')
        if self.groups and self.fig:
            self._check_group_matching('groups', 'fig')

        # Define all the columns in use (used for saving the data.df_all)
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

        # Add all non-DataFrame kwargs to self
        del kwargs['df']  # for memory
        self.kwargs = kwargs

        # Swap x and y axes
        if self.swap:
            self.swap_xy()

        # Transform data
        self.transform()  # DO WE WANT TO DO THIS HERE?

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
            if getattr(self, prop) not in ['x', 'y', None, True, False]:
                grouper += utl.validate_list(getattr(self, prop))

        return list(set(grouper))

    def _add_range(self, ir: int, ic: int, ax: str, label: str, value: Union[None, float]):
        """Add a range limit value unless it already exists.

        Args:
            ir: current axes row index
            ic: current axes column index
            ax: name of the axis ('x', 'y', etc)
            label: range type name ('min', 'max')
            value: value of the range
        """
        self.ranges[f'{ax}{label}'][ir, ic] = value

    def _add_ranges_all_none(self, ir: int, ic: int):
        """Add None for all range values.

        Args:
            ir: current axes row index
            ic: current axes column index
        """
        for ax in self.axs_on:
            for mm in ['min', 'max']:
                self._add_range(ir, ic, ax, mm, None)

    def _ax_limit_pad(self, **kwargs):
        """
        Set padding limits for axis.  Default is for ranges to go 5% beyond the (max - min) range.

        Args:
            kwargs: user-defined keyword args
        """
        self.ax_limit_padding = utl.kwget(kwargs, self.fcpp, 'ax_limit_padding', 0.05)
        for ax in self.axs:
            for mm in ['min', 'max']:
                key = f'ax_limit_padding_{ax}{mm}'
                if not hasattr(self, key):
                    setattr(self, key, utl.kwget(kwargs, self.fcpp, key, self.ax_limit_padding))

    def _check_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the DataFrame.

        Args:
            df: input DataFrame

        Returns:
            copy of the now-validated DataFrame (copy avoids column name changes with data filtering)
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

        # Check that each value exists in the DataFrame
        if values is None:
            return

        for val in values:
            if val not in self.df_all.columns:
                raise GroupingError(f'Grouping column "{val}" is not in the DataFrame!')

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
            error = f'Only one value can be specified for "{("row" if len(self.row) > 1 else "col")}"'
            raise GroupingError(error)

        if self.row is not None and self.row == self.col:
            raise GroupingError('Row and column values must be different!')

        if self.wrap is not None and (self.col or self.row):
            error = f'Cannot combine "wrap" grouping with "{("col" if self.col else "row")}"'
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
            raise GroupingError(f'"{group1}" and "{group2}" grouping columns cannot be the same!')

    def _check_range_limit_format(self, limit):
        """Validate axis ranges.

        Args:
            limit:  user-defined RepeatedList for an axis limit
        """
        limit_obj = getattr(self, limit)

        for limit_val in limit_obj.values:
            if isinstance(limit_val, str):
                if limit_val[0].lower() == 'q' and (float(limit_val[1:]) < 0 or float(limit_val[1:]) > 100):
                    raise DataError(f'"{limit}" must be a float, int, or a valid quantile string starting with a "q" '
                                    'followed by a number between 0 and 1 [current value: "{limit_val}"]')
                elif limit_val[0].lower() != 'q' and limit_val[-4:].lower() != '*iqr':
                    raise DataError(f'"{limit}" must be a float, int, or valid quantile/quartile string '
                                    f'[current value: "{limit_val}"]')
            elif isinstance(limit_val, pd.DataFrame) or isinstance(limit_val, pd.Series):
                raise DataError(f'"{limit}" must be a float, int, or valid quantile/quartile string '
                                f'[current value has type "{type(limit_val)}"]')

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.

        Args:
            xyz: name of variable to check
        """
        if xyz not in self.req and xyz not in self.opt:
            return

        if xyz in self.opt and getattr(self, xyz) is None:
            return

        vals = getattr(self, xyz)

        # Allow plotting by the DataFrame index
        if vals is not None and 'index' in vals and self.name == 'xy':
            self.df_all.index.name = 'index'
        if vals is not None and self.df_all.index.name in vals and self.name == 'xy':
            self.df_all = self.df_all.reset_index()

        if vals is None and xyz not in self.opt:
            raise AxisError(f'Must provide a column name for "{xyz}"')

        for val in vals:
            if self.imgs is None and val not in self.df_all.columns:
                raise DataError(f'No column named "{val}" found in DataFrame')
            elif self.imgs is not None and val not in self.imgs[list(self.imgs.keys())[0]].columns:  # error here?
                raise DataError(f'No column named "{val}" found in DataFrame')

            # Check case (non-image)
            try:
                if self.imgs is None:
                    self.df_all[val] = self.df_all[val].astype(float)
                continue
            except (ValueError, TypeError):
                pass
            try:
                if self.imgs is None and not self.ignore_dates:
                    self.df_all[val] = self.df_all[val].astype('datetime64[ns]')
                    self._strip_timestamp(val)
                    continue
            except:  # noqa
                continue

        # Check for axis errors
        if self.twin_x and len(self.y) != 2:
            raise AxisError(f'twin_x error! {len(self.y)} y values were specified but two are required')
        if self.twin_x and len(self.x) > 1:
            raise AxisError('twin_x error! only one x value can be specified')
        if self.twin_y and len(self.x) != 2:
            raise AxisError(f'twin_y error! {len(self.x)} x values were specified but two are required')
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

    def _convert_q_range_limits(self, ax: str, key: str, data_set: Union[np.ndarray, pd.DataFrame],
                                plot_num: int, mm: str):
        """
        Convert 'q' based range limits to hard numbers

        Args:
            ax: current axis
            key: range key (xmin, ymax, etc.)
            data_set: current data set
            plot_num: current plot number
            mm: 'min' or 'max'

        """

        user_limits = getattr(self, key).values

        # Limit is in a RepeatedList and not explictly defined
        if plot_num + 1 > len(user_limits):
            return

        # Get user limit for this plot
        user_limit = user_limits[plot_num]

        # Convert to numpy array
        if isinstance(data_set, pd.DataFrame):
            cols_found = [f for f in getattr(self, ax) if f in data_set.columns]
            if len(cols_found) == 0:
                return

            # Columns found
            data_set = data_set[getattr(self, ax)]
            data_set = data_set[getattr(self, ax)].stack().values

        # IQR style limit
        if isinstance(user_limit, str) and 'iqr' in user_limit.lower():
            factor = float(str(user_limit.lower()).split('*')[0].rstrip())
            q1 = np.quantile(data_set, 0.25)
            q3 = np.quantile(data_set, 0.75)
            iqr = factor * (q3 - q1)
            if mm == 'min':
                getattr(self, key).values[plot_num] = q1 - iqr
            else:
                getattr(self, key).values[plot_num] = q3 + iqr

        # Quantile limit
        elif isinstance(user_limit, str) and user_limit[0] == 'q':
            if '.' in user_limit:
                xq = float(str(user_limit).lower().replace('q', ''))
            else:
                xq = float(str(user_limit).lower().replace('q', '')) / 100
            getattr(self, key).values[plot_num] = np.quantile(data_set, xq)

    def _get_auto_scale(self,
                        data_set: Union[pd.DataFrame, npt.NDArray],
                        plot_num: int) -> Union[pd.DataFrame, npt.NDArray]:
        """Auto-scale the plot data.

        Args:
            df: either self.df_fig or self.df_rc
            plot_num: plot number

        Returns:
            updated df with potentially reduced range of data
        """
        # data_set = data_set.copy()  # do we need it?

        # Get the max/min autoscale values
        for ax in self.axs_on:
            for mm in ['min', 'max']:
                user_limit = getattr(self, f'{ax}{mm}')[plot_num]
                if user_limit is not None:
                    self._convert_q_range_limits(ax, f'{ax}{mm}', data_set, plot_num, mm)
                    user_limit = getattr(self, f'{ax}{mm}')[plot_num]

                    if not self.auto_scale:
                        continue

                    if not isinstance(data_set, pd.DataFrame):
                        # no autoscale for numpy array??
                        continue

                    for col in getattr(self, ax):
                        if col not in data_set.columns or len(data_set) == 0:
                            continue

                        if isinstance(user_limit, datetime.date) or isinstance(user_limit, datetime.datetime):
                            # Ensure dtypes are aligned for the filtering
                            try:
                                data_set[col] = data_set[col].astype('datetime64[ns]')
                            except:  # noqa
                                raise DataError(f'Column "{col}" could not be cast to datetime dtype')
                            try:
                                user_limit = pd.to_datetime(user_limit)
                            except:  # noqa
                                raise DataError(f'User limit "{user_limit}" could not be cast to datetime dtype')

                        cols = getattr(self, ax)
                        if mm == 'min':
                            mask = (data_set[cols] >= user_limit)
                            if len(mask[mask.all(axis=1)]) < len(data_set):
                                idx = data_set.loc[mask.sum(axis=1) > 0].index
                                data_set = data_set.loc[idx]
                        else:
                            if data_set[col].dtype == 'datetime64[ns]' and \
                                    len(data_set[data_set[col] <= user_limit]) == 0:
                                data_set[col] = user_limit
                            else:
                                mask = (data_set[cols] <= user_limit)
                                if len(mask[mask.all(axis=1)]) < len(data_set):
                                    idx = data_set.loc[mask.sum(axis=1) > 0].index
                                    data_set = data_set.loc[idx]

            vmin, vmax = getattr(self, f'{ax}min')[plot_num], getattr(self, f'{ax}max')[plot_num]
            if vmin is not None and vmax is not None and vmin >= vmax:
                raise DataError(f'{ax}min must be less than {ax}max [{vmin} >= {vmax}]')

            if len(data_set) == 0:
                raise DataError('No data found after applying user-specfied min/max values')

        return data_set

    def _get_data_range(self, ax: str, data_set: Union[pd.DataFrame, npt.NDArray], plot_num: int) -> tuple:
        """Determine the min/max values for a given axis based on user inputs.

        Args:
            ax: name of the axis ('x', 'y', etc)
            data_set: data to use for range calculation
            plot_num: index number of the current subplot

        Returns:
            min, max tuple
        """
        if len(data_set) == 0:
            return None, None

        # Case: data is a pd.DataFrame - separate out values of interest and address special dtype issues
        if isinstance(data_set, pd.DataFrame):
            if len([f for f in getattr(self, ax) if f not in data_set.columns]) > 0:
                return None, None
            vals = data_set[getattr(self, ax)].stack().values  # convert to numpy array
            dtypes = data_set[getattr(self, ax)].dtypes.unique()

            # Check dtypes
            if 'datetime64[ns]' in dtypes or all([isinstance(f, datetime.date) for f in vals]):
                vmin, vmax = None, None
                if getattr(self, f'{ax}min')[plot_num] is not None:
                    vmin = getattr(self, f'{ax}min')[plot_num]
                else:
                    vmin = np.min(vals)
                if getattr(self, f'{ax}max')[plot_num] is not None:
                    vmax = getattr(self, f'{ax}max')[plot_num]
                else:
                    vmax = np.max(vals)
                return np.datetime64(vmin), np.datetime64(vmax)

            elif 'str' in dtypes or 'object' in dtypes:
                return None, None

        # Case: data_set is np.array
        else:
            if len(data_set) == 0:
                return None, None
            vals = data_set

        # Calculate the data (max - min) and account for ax_scale type
        if self.imgs is not None and len(vals.shape) >= 2 and ax == 'x':
            # x-axis of image data
            axmin = 0
            axmax = vals.shape[1]
            axdelta = axmax - axmin
        elif self.imgs is not None and len(vals.shape) >= 2 and ax == 'y':
            # y-axis of image data
            axmin = 0
            axmax = vals.shape[0]
            axdelta = axmax - axmin
        elif self.ax_scale in [f'log{ax}', 'loglog', f'semilog{ax}', 'log']:
            # log data
            axmin = np.min(vals[vals > 0])
            axmax = np.max(vals)
            axdelta = np.log10(axmax) - np.log10(axmin)
        elif len(vals) == 0:
            return None, None
        else:
            # anything else
            axmin = np.min(vals)
            axmax = np.max(vals)
            axdelta = axmax - axmin
        if axdelta is not None and axdelta <= 0:
            axmin -= self.ax_limit_padding * axmin
            axmax += self.ax_limit_padding * axmax

        # Get min value
        if getattr(self, f'{ax}min')[plot_num] is None and getattr(self, f'ax_limit_padding_{ax}min') is not None:
            if self.ax_scale in [f'log{ax}', 'loglog', f'semilog{ax}', 'log']:
                axmin = np.log10(axmin) - getattr(self, f'ax_limit_padding_{ax}min') * axdelta
                vmin = 10**axmin
            else:
                axmin -= getattr(self, f'ax_limit_padding_{ax}min') * axdelta
                vmin = axmin
        else:
            vmin = getattr(self, f'{ax}min')[plot_num]

        # Get max value
        if getattr(self, f'{ax}max')[plot_num] is None and getattr(self, f'ax_limit_padding_{ax}max') is not None:
            if self.ax_scale in [f'log{ax}', 'loglog', f'semilog{ax}', 'log']:
                axmax = np.log10(axmax) + getattr(self, f'ax_limit_padding_{ax}max') * axdelta
                vmax = 10**axmax
            else:
                axmax += getattr(self, f'ax_limit_padding_{ax}max') * axdelta
                vmax = axmax
        else:
            vmax = getattr(self, f'{ax}max')[plot_num]

        # if vmin >= vmax:
        #     raise DataError(f'{ax}min must be less than {ax}max [{vmin} >= {vmax}]')

        # Make sure vmin != vmax
        if vmin is not None and vmax is not None and vmin == vmax:
            vmin -= self.ax_limit_padding * vmin
            vmax += self.ax_limit_padding * vmax

        return vmin, vmax

    def get_data_ranges(self):
        """Calculate data range limits for a given figure."""
        # For only 1 subplot, ranges are already set
        if self.ncol == 1 and self.nrow == 1:
            return

        rr = self._range_dict()  # new range dict with updates based on subplot contents
        for ax in self.axs_on:
            # Case 1: share_[ax] = True
            if getattr(self, f'share_{ax}'):
                mmin = self.ranges[f'{ax}min'][np.not_equal(self.ranges[f'{ax}min'], None)]
                if len(mmin) > 0:
                    rr[f'{ax}min'][np.equal(rr[f'{ax}min'], None)] = mmin.min()
                mmax = self.ranges[f'{ax}max'][np.not_equal(self.ranges[f'{ax}max'], None)]
                if len(mmax) > 0:
                    rr[f'{ax}max'][np.equal(rr[f'{ax}max'], None)] = mmax.max()

            # Case 2: share_row = True
            elif self.share_row and self.row is not None:  # and self.row != 'y':
                for irow in range(0, self.nrow):
                    mmin = self.ranges[f'{ax}min'][irow, :][np.not_equal(self.ranges[f'{ax}min'][irow, :], None)]
                    if len(mmin) > 0:
                        rr[f'{ax}min'][irow, :] = mmin.min()
                    mmax = self.ranges[f'{ax}max'][irow, :][np.not_equal(self.ranges[f'{ax}max'][irow, :], None)]
                    if len(mmax) > 0:
                        rr[f'{ax}max'][irow, :] = mmax.max()

            # Case 3: share_col
            elif self.share_col and self.col is not None:
                for icol in range(0, self.ncol):
                    mmin = self.ranges[f'{ax}min'][:, icol][np.not_equal(self.ranges[f'{ax}min'][:, icol], None)]
                    if len(mmin) > 0:
                        rr[f'{ax}min'][:, icol] = mmin.min()
                    mmax = self.ranges[f'{ax}max'][:, icol][np.not_equal(self.ranges[f'{ax}min'][:, icol], None)]
                    if len(mmax) > 0:
                        rr[f'{ax}max'][:, icol] = mmax.max()

            # Case 4: no sharing
            else:
                rr[f'{ax}min'] = self.ranges[f'{ax}min']
                rr[f'{ax}max'] = self.ranges[f'{ax}max']

        # Overwrite previous ranges
        self.ranges = rr

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
        if self.fit is True or isinstance(self.fit, int):
            df2 = df.copy()
            df = df.copy()
            df[f'{x} Fit'] = np.nan
            df[f'{y} Fit'] = np.nan

            # Set range of the fit
            if isinstance(self.fit_range_x, list):
                df2 = df2[(df2[x] >= self.fit_range_x[0])
                          & (df2[x] <= self.fit_range_x[1])].copy()
                if self.ranges['ymin'][ir, ic] is not None:
                    df2 = df2[(df2[y]) >= self.ranges['ymin'][ir, ic]]
                if self.ranges['ymax'][ir, ic] is not None:
                    df2 = df2[(df2[y]) <= self.ranges['ymax'][ir, ic]]
            elif isinstance(self.fit_range_y, list):
                df2 = df2[(df2[y] >= self.fit_range_y[0])
                          & (df2[y] <= self.fit_range_y[1])].copy()
                if self.ranges['xmin'][ir, ic] is not None:
                    df2 = df2[(df2[x]) >= self.ranges['xmin'][ir, ic]]
                if self.ranges['xmax'][ir, ic] is not None:
                    df2 = df2[(df2[x]) <= self.ranges['xmax'][ir, ic]]
            else:
                df2 = df2.copy()
                if self.ranges['xmin'][ir, ic] is not None:
                    df2 = df2[(df2[x]) >= self.ranges['xmin'][ir, ic]]
                if self.ranges['xmax'][ir, ic] is not None:
                    df2 = df2[(df2[x]) <= self.ranges['xmax'][ir, ic]]
                if self.ranges['ymin'][ir, ic] is not None:
                    df2 = df2[(df2[y]) >= self.ranges['ymin'][ir, ic]]
                if self.ranges['ymax'][ir, ic] is not None:
                    df2 = df2[(df2[y]) <= self.ranges['ymax'][ir, ic]]

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
            df[f'{x} Fit'] = np.linspace(self.ranges['xmin'][ir, ic], self.ranges['xmax'][ir, ic], len(df))
            df[f'{y} Fit'] = np.polyval(coeffs, df[f'{x} Fit'])

            return df, coeffs, rsq

        # TODO:: put spline code here
        # if str(self.fit).lower() == 'spline':

        #     return df, [], np.nan

    def _get_legend_groupings(self, df: pd.DataFrame):
        """Determine the legend groupings.

        Args:
            df: data subset
        """
        if self.legend_vals is not None:
            # Only do this function once
            return
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
                    self.wrap_vals = [f[0] for f in df.groupby(self.wrap, sort=False)]
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
                        self.col_vals = natsorted(df.groupby(self.col, sort=False).groups.keys())
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

    def get_subplot_index(self):
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
                plot_num = utl.plot_num(ir, ic, self.ncol) - 1

                # Handle empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

                # Find data ranges for this subset
                else:
                    df_rc = self._get_auto_scale(self.df_rc, plot_num)  # is this copy ok??
                    for ax in self.axs_on:
                        if getattr(self, ax) is None:
                            continue
                        vals = self._get_data_range(ax, df_rc, plot_num)
                        if getattr(self, f'invert_range_limits_{ax}'):
                            self._add_range(ir, ic, ax, 'min', vals[1])
                            self._add_range(ir, ic, ax, 'max', vals[0])
                        else:
                            self._add_range(ir, ic, ax, 'min', vals[0])
                            self._add_range(ir, ic, ax, 'max', vals[1])
                    self._range_overrides(ir, ic, self.df_rc)

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
                return getattr(df_stat, self.stat)(numeric_only=True).reset_index()
            except AttributeError:
                print(f'stat "{self.stat}" is not supported...skipping stat calculation')
                return None

    def _kwargs_groupers(self, kwargs) -> list:
        """Get all grouping values from kwargs"""
        props = ['row', 'cols', 'wrap', 'groups', 'legend', 'fig']
        grouper = []

        for prop in props:
            if kwargs.get(prop, None) not in ['x', 'y', None]:
                grouper += utl.validate_list(kwargs.get(prop))

        return list(set(grouper))

    def _range_overrides(self, ir: int, ic: int, df_rc: pd.DataFrame):
        """Optional method to allow re-calculations of subset ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            df_rc: data subset
        """
        pass

    def _range_dict(self):
        """Make a list of empty dicts for axes range limits."""
        ranges = {}
        for ax in self.axs_on:
            for mm in ['min', 'max']:
                ranges[f'{ax}{mm}'] = np.array([[None] * self.ncol] * self.nrow)
        return ranges

    def _strip_timestamp(self, val: str):
        """
        If all are 00:00:00 time, leave only date

        Args:
            val: column name
        """
        if len(self.df_all.loc[self.df_all[val].dt.hour != 0, val]) == 0 and \
                len(self.df_all.loc[self.df_all[val].dt.minute != 0, val]) == 0 and \
                len(self.df_all.loc[self.df_all[val].dt.second != 0, val]) == 0:
            self.df_all[val] = pd.DatetimeIndex(self.df_all[val]).date

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
        """
        Optional function in a Data childe class user to perform any additional DataFrame subsetting that may
        be required

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        return df

    def _subset_row_col(self, ir: int, ic: int) -> pd.DataFrame:
        """
        For row/column plots (no wrap), select the revelant subset from self.df_fig.

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
            # return self.df_fig.copy()  # why copy?
            return self.df_fig

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
            wrap = dict(zip(self.wrap, utl.validate_list(self.wrap_vals[ir * self.ncol + ic])))
            mask = pd.concat([self.df_fig[x[0]].eq(x[1]) for x in wrap.items()], axis=1).all(axis=1)
            return self.df_fig[mask]

    def swap_xy(self):
        """
        Swap the x and y axis attributes. This is a convenience function to avoid having to rewrite a long list of
        axis-specific kwargs.
        """
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
        for ir, ic, plot_num in self.get_subplot_index():
            xmin = self.ranges['xmin'][ir, ic]
            xmax = self.ranges['xmax'][ir, ic]
            x2min = self.ranges['x2min'][ir, ic]
            x2max = self.ranges['x2max'][ir, ic]
            self.ranges['xmin'][ir, ic] = self.ranges['ymin'][ir, ic]
            self.ranges['xmax'][ir, ic] = self.ranges['ymax'][ir, ic]
            self.ranges['x2min'][ir, ic] = self.ranges['y2min'][ir, ic]
            self.ranges['x2max'][ir, ic] = self.ranges['y2max'][ir, ic]
            self.ranges['ymin'][ir, ic] = xmin
            self.ranges['ymax'][ir, ic] = xmax
            self.ranges['y2min'][ir, ic] = x2min
            self.ranges['y2max'][ir, ic] = x2max

    def transform(self):
        """Transform x, y, or z data by unique group."""
        # Possible tranformations
        transform = any([self.trans_x, self.trans_x2, self.trans_y, self.trans_y2, self.trans_z])
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
                    if getattr(self, f'trans_{ax}') == 'abs':
                        gg.loc[:, val] = abs(gg[val])
                    elif getattr(self, f'trans_{ax}') == 'negative' or getattr(self, f'trans_{ax}') == 'neg':
                        gg.loc[:, val] = -gg[val]
                    elif getattr(self, f'trans_{ax}') == 'nq':
                        if self.imgs is None:
                            gg = utl.nq(gg[val], val, **self.kwargs)
                        else:
                            gg = utl.nq(self.imgs[gg.index[0]][val], val, **self.kwargs)
                    elif getattr(self, f'trans_{ax}') == 'inverse' or getattr(self, f'trans_{ax}') == 'inv':
                        gg.loc[:, val] = 1 / gg[val]
                    elif (isinstance(getattr(self, f'trans_{ax}'), tuple)
                            or isinstance(getattr(self, f'trans_{ax}'), list)) \
                            and getattr(self, f'trans_{ax}')[0] == 'pow':
                        gg.loc[:, val] = gg[val]**getattr(self, f'trans_{ax}')[1]
                    elif getattr(self, f'trans_{ax}') == 'flip':
                        maxx = gg.loc[:, val].max()
                        gg.loc[:, val] -= maxx
                        gg.loc[:, val] = abs(gg[val])

            if isinstance(group, tuple):
                vals = group[0] if isinstance(group[0], tuple) else [group[0]]
                for k, v in dict(zip(groups_all, vals)).items():
                    gg[k] = v

            df = pd.concat([df, gg])

        self.df_all = df
