import pandas as pd
import numpy as np
from . import utilities as utl
import scipy.stats as ss

try:
    from natsort import natsorted
except:
    natsorted = sorted

import pdb
db = pdb.set_trace

REQUIRED_VALS = {'plot_xy': ['x', 'y'],
                 'plot_bar': ['x', 'y'],
                 'plot_box': ['y'],
                 'plot_hist': ['x'],
                 'plot_contour': ['x', 'y', 'z'],
                 'plot_heatmap': [],
                 'plot_nq': []
                }
OPTIONAL_VALS = {'plot_xy': [],
                 'plot_bar': [],
                 'plot_box': [],
                 'plot_hist': [],
                 'plot_contour': [],
                 'plot_heatmap': ['x', 'y', 'z'],
                 'plot_nq': ['x'],
                }


class AxisError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class DataError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class GroupingError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


class Data:
    def __init__(self, plot_func='xy', **kwargs):

        # Reload default file
        self.fcpp, dummy, dummy2 = utl.reload_defaults(kwargs.get('theme', None))

        # Set the plot type
        self.plot_func = plot_func

        # Default axis attributes
        self.auto_cols = False
        self.auto_scale = utl.kwget(kwargs, self.fcpp, 'auto_scale', True)
        if self.plot_func in ['plot_heatmap', 'plot_hist']:
            self.auto_scale = False
        #self.ax_scale = utl.RepeatedList(kwargs.get('ax_scale', [None]), 'ax_scale')
        self.ax_scale = kwargs.get('ax_scale', None)
        self.ax2_scale = kwargs.get('ax2_scale', self.ax_scale)
        # validate list? repeated list and all to lower
        self.ax_limit_pad(**kwargs)
        self.conf_int = kwargs.get('conf_int', False)
        self.error_bars = False
        self.fit = kwargs.get('fit', False)
        self.fit_range_x = utl.kwget(kwargs, self.fcpp, 'fit_range_x', None)
        self.fit_range_y = utl.kwget(kwargs, self.fcpp, 'fit_range_y', None)
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
        if kwargs.get('wrap', None) is not None:
            self.share_x = True
            self.share_y = True
        if kwargs.get('wrap', None) == 'y' or kwargs.get('wrap', None) == 'x':
            self.share_x = kwargs.get('share_x', True)
            self.share_y = kwargs.get('share_y', True)
        if self.plot_func in ['plot_box']:
            self.share_x = False
        self.sort = utl.kwget(kwargs, self.fcpp, 'sort', True)
        self.stacked = False
        if self.plot_func == 'plot_bar':
            self.stacked = utl.kwget(kwargs, self.fcpp, 'bar_stacked',
                                     kwargs.get('stacked', False))
        elif self.plot_func == 'plot_hist':
            self.stacked = utl.kwget(kwargs, self.fcpp, 'hist_stacked',
                                     kwargs.get('stacked', False))
        self.swap = utl.kwget(kwargs, self.fcpp, 'swap', False)
        self.trans_df_fig = False
        self.trans_df_rc = False
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
        self.df_all = self.check_df(kwargs['df'])
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
        self.x = self.check_xyz('x')
        self.y = self.check_xyz('y')
        self.z = self.check_xyz('z')
        self.x2 = []
        self.y2 = []
        if self.twin_x:
            if len(self.y) < 2:
                raise AxisError('twin_x requires two y-axis columns')
            self.y2 = [self.y[1]]
            self.y = [self.y[0]]
        if self.twin_y:
            if len(self.x) < 2:
                raise AxisError('twin_y requires two x-axis columns')
            self.x2 = [self.x[1]]
            self.x = [self.x[0]]
        if self.plot_func == 'plot_box':
            self.x = ['x']
            self.df_all['x'] = 1
        if self.plot_func == 'plot_heatmap':
            if not self.x and not self.y and not self.z:
                self.x = ['Column']
                self.y = ['Row']
                self.z = ['Value']
                self.auto_cols = True
            else:
                self.pivot = True
        if self.plot_func == 'plot_bar' and \
                utl.kwget(kwargs, self.fcpp, 'bar_error_bars',
                          kwargs.get('error_bars', False)):
            self.error_bars = True
        if self.plot_func == 'plot_nq':
            if not self.x:
                self.x = ['Value']
                self.df_all = pd.DataFrame(self.df_all.stack())
                self.df_all.columns = self.x
            self.trans_x = 'nq'
            self.y = ['Sigma']

        # Ref line
        self.ref_line = kwargs.get('ref_line', None)
        if type(self.ref_line) is pd.Series:
            self.df_all['Ref Line'] = self.ref_line

        # Stats
        self.stat = kwargs.get('stat', None)
        self.stat_val = kwargs.get('stat_val', None)
        if self.stat_val is not None and self.stat_val not in self.df_all.columns:
            raise DataError('stat_val column "%s" not in DataFrame' % self.stat_val)
        self.stat_idx = []
        self.lcl = []
        self.ucl = []

        # Special for hist
        normalize = utl.kwget(kwargs, self.fcpp, 'hist_normalize', kwargs.get('normalize', False))
        kde=utl.kwget(kwargs, self.fcpp, 'hist_kde', kwargs.get('kde', False))
        if normalize or kde:
            self.norm = True
        else:
            self.norm = False
        self.bins = utl.kwget(kwargs, self.fcpp, 'hist_bins', kwargs.get('bins', 20))

        # Apply an optional filter to the data
        self.filter = kwargs.get('filter', None)
        if self.filter:
            self.df_all = utl.df_filter(self.df_all, self.filter)
            if len(self.df_all) == 0:
                raise DataError('DataFrame is empty after applying filter')

        # Define rc grouping column names
        self.col = kwargs.get('col', None)
        self.col_vals = None
        if self.col is not None:
            if self.col == 'x':
                self.col_vals = [f for f in self.x]
            else:
                self.col = self.check_group_columns('col',
                                                    kwargs.get('col', None))
        self.row = kwargs.get('row', None)
        self.row_vals = None
        if self.row is not None:
            if self.row == 'y':
                self.row_vals = [f for f in self.y]
            else:
                self.row = self.check_group_columns('row',
                                                    kwargs.get('row', None))
        self.wrap = kwargs.get('wrap', None)
        self.wrap_vals = None
        if self.wrap is not None:
            if self.wrap == 'y':
                self.wrap_vals = [f for f in self.y]
            elif self.wrap == 'x':
                self.wrap_vals = [f for f in self.x]
            else:
                self.wrap = self.check_group_columns('wrap', self.wrap)
        self.groups = self.check_group_columns('groups', kwargs.get('groups', None))
        self.check_group_errors()
        self.ncols = kwargs.get('ncol', 0)
        self.ncol = 1
        self.nleg_vals = 0
        self.nrow = 1
        self.nwrap = 0
        self.ngroups = 0

        # Define legend grouping column names (legends are common to a figure,
        #   not an rc subplot)
        if 'legend' in kwargs.keys():
            if kwargs['legend'] is True:
                self.legend = True
            elif kwargs['legend'] is False:
                self.legend = False
            else:
                self.legend = self.check_group_columns('legend',
                                                       kwargs.get('legend', None))
        elif not self.twin_x and self.y is not None and len(self.y) > 1:
            self.legend = True

        # Define figure grouping column names
        if 'fig_groups' in kwargs.keys():
            self.fig = self.check_group_columns('fig',
                                                kwargs.get('fig_groups', None))
        else:
            self.fig = self.check_group_columns('fig', kwargs.get('fig', None))
        self.fig_vals = None

        # Make sure groups, legend, and fig_groups are not the same
        if self.legend and self.groups and self.plot_func != 'plot_box':
            if type(self.legend) is not bool:
                self.check_group_matching('legend', 'groups')
        if self.legend and self.fig:
            self.check_group_matching('legend', 'fig')
        if self.groups and self.fig:
            self.check_group_matching('groups', 'fig')

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
        if type(self.legend) is list:
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
            if not hasattr(self, k):  # k not in ['df', 'plot_func', 'x', 'y', 'z']:
                setattr(self, k, v)

    def ax_limit_pad(self, **kwargs):
        """
        Set padding limits for axis
        """

        if self.plot_func in ['plot_contour', 'plot_heatmap']:
            self.ax_limit_padding = kwargs.get('ax_limit_padding', None)
        elif self.plot_func in ['plot_hist']:
            self.ax_limit_padding = kwargs.get('ax_limit_padding', 0)
            self.ax_limit_padding_y_max = kwargs.get('ax_limit_padding', 0.05)
        else:
            self.ax_limit_padding = utl.kwget(kwargs, self.fcpp, 'ax_limit_padding', 0.05)
        for ax in ['x', 'x2', 'y', 'y2', 'z']:
            if not hasattr(self, 'ax_limit_padding_%s_min' % ax):
                setattr(self, 'ax_limit_padding_%s_min' % ax,
                        utl.kwget(kwargs, self.fcpp,
                                  'ax_limit_padding_%s_min' % ax, self.ax_limit_padding))
            if not hasattr(self, 'ax_limit_padding_%s_max' % ax):
                setattr(self, 'ax_limit_padding_%s_max' % ax,
                        utl.kwget(kwargs, self.fcpp,
                                  'ax_limit_padding_%s_max' % ax, self.ax_limit_padding))

    def check_df(self, df):
        """
        Validate the dataframe
        """

        if df is None:
            raise DataError('Must provide a DataFrame called "df" '
                            'for plotting!')

        if len(df) == 0:
            raise DataError('DataFrame is empty.  Nothing to plot!')

        return df.copy()

    def check_group_columns(self, group_type, col_names):
        """
        Check wrap/row/column grouping variables for errors

        Args:
            group_type (str): type of grouping (row, col, leg, wrap)
            col_name (str): name of the column by which to group

        """

        # Force list type
        values = utl.validate_list(col_names)

        # Check that each value exists in the dataframe
        if values is None:
            return

        for val in values:
            if val not in self.df_all.columns:
                raise GroupingError('Grouping column "%s" is not '
                                    'in the DataFrame!' % val)

        # Check for no groups
        if len(natsorted(list(self.df_all.groupby(values).groups.keys()))) == 0:
            raise GroupingError('The number of unique groups in the data for '
                                "the %s=['%s'] is 0" %
                                (group_type, ', '.join(values)))

        # Check for wrap with twiny
        if group_type == 'wrap' and col_names is not None and self.twin_y:
            raise GroupingError('Wrap plots do not support twinning of the y-axis. '
                                'Please consider a row vs column plot instead.')

        return values

    def check_group_matching(self, group1, group2):
        """
        Check to make sure certain group column values are not the same

        Args:
            group1 (str): attr name of first grouping column
            group2 (str): attr name of second grouping column

        """

        equal = set(str(getattr(self, group1))) == set(str(getattr(self, group2)))

        if equal:
            raise GroupingError('%s and %s grouping columns cannot be the same!'
                                % (group1, group2))

    def check_group_errors(self):
        """
        Check for common errors related to grouping keywords
        """

        if self.row and len(self.row) > 1 or self.col and len(self.col) > 1:
            error = 'Only one value can be specified for "%s"' % ('row' if self.row else 'col')
            raise GroupingError(error)

        if self.row is not None and self.row == self.col:
            raise GroupingError('Row and column values must be different!')

        if self.wrap is not None and (self.col or self.row):
            error = 'Cannot combine "wrap" grouping with "%s"' % \
                    ('col' if self.col else 'row')
            raise GroupingError(error)

        if self.groups is not None and \
                ((self.row and self.row[0] in self.groups) or \
                 (self.col and self.col[0] in self.groups)):
            error = '"%s" value cannot also be specified as a "group" value' % \
                    ('col' if self.col else 'row')
            raise GroupingError(error)

        if self.groups is not None and self.wrap is not None:
            if len(list(set(self.wrap) & set(self.groups))) > 0:
                error = '"%s" value cannot also be specified as a "group" value' % \
                        ('col' if self.col else 'row')
                raise GroupingError(error)

    def check_xyz(self, xyz):
        """
        Validate the name and column data provided for x, y, and/or z
        Args:
            xyz (str): name of variable to check
        TODO:
            add option to recast non-float/datetime column as categorical str
        """

        if xyz not in REQUIRED_VALS[self.plot_func] and \
                xyz not in OPTIONAL_VALS[self.plot_func]:
            return

        if xyz in OPTIONAL_VALS[self.plot_func] and getattr(self, xyz) is None:
            return None

        vals = getattr(self, xyz)

        if vals is None and xyz not in OPTIONAL_VALS[self.plot_func]:
            raise AxisError('Must provide a column name for "%s"' % xyz)

        for val in vals:
            if val not in self.df_all.columns:
                raise DataError('No column named "%s" found in DataFrame' % val)

            # Check case
            if self.plot_func == 'plot_heatmap':
                continue
            try:
                self.df_all[val] = self.df_all[val].astype(float)
                continue
            except:
                pass
            try:
                self.df_all[val] = self.df_all[val].astype('datetime64[ns]')
                continue
            except:
                continue
            #     raise AxisError('Could not convert x-column "%s" to float or '
                                #  'datetime.' % val)

        # Check for axis errors
        if self.twin_x and len(self.y) != 2:
            raise AxisError('twin_x error! %s y values were specified but'
                            ' two are required' % len(self.y))
        if self.twin_x and len(self.x) > 1:
            raise AxisError('twin_x error! only one x value can be specified')
        if self.twin_y and len(self.x) != 2:
            raise AxisError('twin_y error! %s x values were specified but'
                            ' two are required' % len(self.x))
        if self.twin_y and len(self.y) > 1:
            raise AxisError('twin_y error! only one y value can be specified')

        return vals

    def get_all_groups(self, df):
        """
        Generator to get all possible allowed groups of data

        Args:
            df:

        Returns:

        """

        group_cols = ['row', 'col', 'wrap', 'leg']
        groups = [getattr(self, f) for f in group_cols
                  if hasattr(self, f) and getattr(self, f) is not None]

        for i, (nn, gg) in enumerate(df.groupby(groups)):
            yield i, nn, self.transform(gg.copy())

    def get_box_index_changes(self):
        """
        Make a DataFrame that shows when groups vals change; used for grouping labels

        Args:
            df (pd.DataFrame): grouping values
            num_groups (int): number of unique groups

        Returns:
            new DataFrame with 1's showing where group levels change for each row of df
        """

        # Check for nan columns
        if self.groups is not None:
            for group in self.groups:
                if len(self.df_rc[group].dropna()) == 0:
                    self.groups.remove(group)
                    print('Column "%s" is all NaN and will be excluded from plot' % group)

        # Get the changes df
        if self.groups is None:
            groups = [(None, self.df_rc.copy())]
            self.ngroups = 0
        else:
            groups = self.df_rc.groupby(self.groups, sort=self.sort)
            self.ngroups = groups.ngroups

        # Order the group labels with natsorting
        gidx = []
        for i, (nn, g) in enumerate(groups):
            gidx += [nn]
        if self.sort:
            gidx = natsorted(gidx)
        self.indices = pd.DataFrame(gidx)
        self.changes = self.indices.copy()

        # Set initial level to 1
        for col in self.indices.columns:
            self.changes.loc[0, col] = 1

        # Determines values for all other rows
        for i in range(1, self.ngroups):
            for col in self.indices.columns:
                if self.indices[col].iloc[i-1] == self.indices[col].iloc[i]:
                    self.changes.loc[i, col] = 0
                else:
                    self.changes.loc[i, col] = 1

    def get_conf_int(self, df, x, y, **kwargs):
        """
        Calculate and draw confidence intervals around a curve

        Args:
            df:
            x:
            y:
            ax:
            color:
            kw:

        Returns:

        """

        if not self.conf_int:
            return

        if str(self.conf_int).lower() == 'range':
            ymin = df.groupby(x).min()[y]
            self.stat_idx = ymin.index
            self.lcl = ymin.reset_index(drop=True)
            self.ucl = df.groupby(x).max()[y].reset_index(drop=True)

        else:
            if float(self.conf_int) > 1:
                self.conf_int = float(self.conf_int)/100
            stat = pd.DataFrame()
            stat['mean'] = df[[x, y]].groupby(x).mean().reset_index()[y]
            stat['count'] = df[[x, y]].groupby(x).count().reset_index()[y]
            stat['std'] = df[[x, y]].groupby(x).std().reset_index()[y]
            stat['sderr'] = stat['std'] / np.sqrt(stat['count'])
            stat['ucl'] = np.nan
            stat['lcl'] = np.nan
            for irow, row in stat.iterrows():
                if row['std'] == 0:
                    conf = [0, 0]
                else:
                    conf = ss.t.interval(self.conf_int, int(row['count'])-1,
                                        loc=row['mean'], scale=row['sderr'])
                stat.loc[irow, 'ucl'] = conf[1]
                stat.loc[irow, 'lcl'] = conf[0]

            self.stat_idx = df.groupby(x).mean().index
            self.lcl = stat['lcl']
            self.ucl = stat['ucl']

    def get_data_range(self, ax, df, ir, ic):
        """
        Determine the min/max values for a given axis based on user inputs

        Args:
            axis (str): x, x2, y, y2, z
            df (pd.DataFrame): data table to use for range calculation

        Returns:
            min, max tuple
        """

        if not hasattr(self, ax) or getattr(self, ax) in [None, []]:
            return None, None
        elif self.col == 'x' and self.share_x and ax == 'x':
            cols = self.x_vals
        elif self.row == 'y' and self.share_y and ax == 'y':
            cols = self.y_vals
        else:
            cols = getattr(self, ax)

        # Heatmap special case
        if self.plot_func == 'plot_heatmap':
            if getattr(self, ax) == ['Column']:
                vmin = min([f for f in df.columns if type(f) is int])
                vmax = max([f for f in df.columns if type(f) is int])
            elif getattr(self, ax) == ['Row']:
                vmin = min([f for f in df.index if type(f) is int])
                vmax = max([f for f in df.index if type(f) is int])
            elif getattr(self, ax) == ['Value'] and self.auto_cols:
                vmin = df.min().min()
                vmax = df.max().max()
            elif ax not in ['x2', 'y2', 'z']:
                vmin = 0
                vmax = len(df[getattr(self, ax)].drop_duplicates())
            elif ax not in ['x2', 'y2']:
                vmin = df[getattr(self, ax)].min().iloc[0]
                vmax = df[getattr(self, ax)].max().iloc[0]
            else:
                vmin = None
                vmax = None
            plot_num = utl.plot_num(ir, ic, self.ncol)
            if getattr(self, '%smin' % ax).get(plot_num):
                vmin = getattr(self, '%smin' % ax).get(plot_num)
            if getattr(self, '%smax' % ax).get(plot_num):
                vmax = getattr(self, '%smax' % ax).get(plot_num)
            if type(vmin) is str:
                vmin = None
            if type(vmax) is str:
                vmax = None
            return vmin, vmax

        # Groupby for stats
        if self.stat is not None and 'only' in self.stat:
            stat_groups = []
            vals_2_chk = ['stat_val', 'legend', 'col', 'row', 'wrap']
            for v in vals_2_chk:
                if getattr(self, v) is not None:
                    stat_groups += utl.validate_list(getattr(self, v))

        # Account for any applied stats
        if self.stat is not None and 'only' in self.stat \
                and 'median' in self.stat:
            df = df.groupby(stat_groups).median().reset_index()
        elif self.stat is not None and 'only' in self.stat:
            df = df.groupby(stat_groups).mean().reset_index()

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
            axdelta = np.log10(axmax)-np.log10(axmin)
        else:
            axmin = dfax.stack().min()
            axmax = dfax.stack().max()
            axdelta = axmax-axmin
        if axdelta <= 0:
            axmin -= 0.1*axmin
            axmax += 0.1*axmax

        # Check user-specified min values
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1
        vmin = getattr(self, '%smin' % ax).get(plot_num)
        if vmin is not None and 'iqr' in str(vmin).lower():
            factor = str(vmin).split('*')
            if len(factor) == 1:
                factor = 1
            else:
                factor = float(factor[0])
            if 'box' not in self.plot_func or self.groups is None:
                q1 = dfax.quantile(0.25).min()
                q3 = dfax.quantile(0.75).max()
                iqr = factor*(q3 - q1)
                vmin = q1 - iqr
            else:
                q1 = df[self.groupers + cols].groupby(self.groupers) \
                        .quantile(0.25)[cols].reset_index()
                q3 = df[self.groupers + cols].groupby(self.groupers) \
                         .quantile(0.75)[cols].reset_index()
                iqr = factor*(q3[cols] - q1[cols])
                vmin = (q1[cols] - iqr[cols]).min().iloc[0]
        elif vmin is not None and 'q' in str(vmin).lower():
            xq = float(str(vmin).lower().replace('q', ''))/100
            if self.groups is None:
                vmin = dfax.quantile(xq).min()
            elif 'box' in self.plot_func:
                vmin = df[self.groupers + cols].groupby(self.groupers) \
                        .quantile(xq)[cols].min().iloc[0]
            else:
                vmin = df[self.groups + cols].groupby(self.groups) \
                        .quantile(xq)[cols].min().iloc[0]
        elif vmin is not None:
            vmin = vmin
        # elif self.plot_func == 'plot_hist' and ax == 'y':
        #     vmin = int(axmin)
        elif getattr(self, 'ax_limit_padding_%s_min' % ax) is not None:
            if self.ax_scale in ['log%s' % ax, 'loglog',
                                 'semilog%s' % ax, 'log']:
                axmin = np.log10(axmin) - \
                        getattr(self, 'ax_limit_padding_%s_min' % ax) * axdelta
                vmin = 10**axmin
            else:
                axmin -= getattr(self, 'ax_limit_padding_%s_min' % ax) * axdelta
                vmin = axmin
        else:
            vmin = None

        # Check user-specified max values
        vmax = getattr(self, '%smax' % ax).get(plot_num)
        if vmax is not None and 'iqr' in str(vmax).lower():
            factor = str(vmax).split('*')
            if len(factor) == 1:
                factor = 1
            else:
                factor = float(factor[0])
            if 'box' not in self.plot_func or self.groups is None:
                q1 = dfax.quantile(0.25).min()
                q3 = dfax.quantile(0.75).max()
                iqr = factor*(q3 - q1)
                vmax = q3 + iqr
            else:
                q1 = df[self.groupers + cols].groupby(self.groupers) \
                          .quantile(0.25)[cols].reset_index()
                q3 = df[self.groupers + cols].groupby(self.groupers) \
                         .quantile(0.75)[cols].reset_index()
                iqr = factor*(q3[cols] - q1[cols])
                vmax = (q3[cols] + iqr[cols]).max().iloc[0]  # should this be referred to median?
        elif vmax is not None and 'q' in str(vmax).lower():
            xq = float(str(vmax).lower().replace('q', ''))/100
            if self.groups is None:
                vmax = dfax.quantile(xq).max()
            elif 'box' in self.plot_func:
                vmax = df[self.groupers + cols].groupby(self.groupers) \
                        .quantile(xq)[cols].max().iloc[0]
            else:
                vmax = df[self.groups + cols].groupby(self.groups) \
                        .quantile(xq)[cols].max().iloc[0]
        elif vmax is not None:
            vmax = vmax
        elif getattr(self, 'ax_limit_padding_%s_max' % ax) is not None:
            if self.ax_scale in ['log%s' % ax, 'loglog',
                                 'semilog%s' % ax, 'log']:
                axmax = np.log10(axmax) + \
                        getattr(self, 'ax_limit_padding_%s_max' % ax) * axdelta
                vmax = 10**axmax
            else:
                axmax += getattr(self, 'ax_limit_padding_%s_max' % ax) * axdelta
                vmax = axmax
        else:
            vmax = None

        if type(vmin) in [float, np.float32, np.float64] and np.isnan(vmin):
            vmin = None
        if type(vmax) in [float, np.float32, np.float64] and np.isnan(vmax):
            vmax = None

        return vmin, vmax

    def get_data_ranges(self, ir, ic):
        """
        Get the data ranges

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        """

        df_fig = self.df_fig.copy()
        df_rc = self.df_rc.copy()
        plot_num = utl.plot_num(ir, ic, self.ncol) - 1

        if self.auto_scale:
            # Filter down by axis limits that have been specified by the user
            limits = ['xmin', 'xmax', 'x2min', 'x2max', 'ymin', 'ymax',
                      'y2min', 'y2max']

            fixed = [f for f in limits
                     if getattr(self, f).get(plot_num) is not None]

            for f in fixed:
                ax = f[0:-3]
                side = f[-3:]
                ax = getattr(self, ax)
                lim = getattr(self, f).get(plot_num)

                if ax is None:
                    continue

                for axx in ax:
                    # Adjust the dataframe by the limits
                    if type(lim) is str or lim is None:
                        continue
                    if side == 'min':
                        if len(df_fig) > 0 and \
                                df_fig[axx].dtype not in ['object', 'str']:
                            df_fig = df_fig[df_fig[axx] >= lim]
                        if len(df_rc) > 0 and \
                                df_rc[axx].dtype not in ['object', 'str']:
                            df_rc = df_rc[df_rc[axx] >= lim]
                    else:
                        if len(df_fig) > 0 and \
                                df_fig[axx].dtype not in ['object', 'str']:
                            df_fig = df_fig[df_fig[axx] <= lim]
                        if len(df_rc) > 0 and \
                                df_rc[axx].dtype not in ['object', 'str']:
                            df_rc = df_rc[df_rc[axx] <= lim]

        # Iterate over axis
        axs = ['x', 'x2', 'y', 'y2', 'z']
        for iax, ax in enumerate(axs):
            if ax == 'z':
                df_fig = self.df_fig.copy()
                df_rc = self.df_rc.copy()
            if self.plot_func == 'plot_hist' and ax == 'y':
                self.get_data_ranges_hist(ir, ic)
                continue
            elif self.plot_func == 'plot_bar':
                self.get_data_ranges_bar(ir, ic)
                continue
            if getattr(self, 'share_%s' % ax) and ir == 0 and ic == 0:
                vals = self.get_data_range(ax, df_fig, ir, ic)
                self.ranges[ir, ic]['%smin' % ax] = vals[0]
                self.ranges[ir, ic]['%smax' % ax] = vals[1]
            elif self.share_row and self.row is not None:
                if self.row == 'y':
                    vals = self.get_data_range(ax, df_fig, ir, ic)
                else:
                    vals = self.get_data_range(ax,
                        df_fig[df_fig[self.row[0]] == self.row_vals[ir]], ir, ic)
                self.ranges[ir, ic]['%smin' % ax] = vals[0]
                self.ranges[ir, ic]['%smax' % ax] = vals[1]
            elif self.share_col and self.col is not None:
                if self.col == 'x':
                    vals = self.get_data_range(ax, df_fig, ir, ic)
                else:
                    vals = self.get_data_range(ax,
                        df_fig[df_fig[self.col[0]] == self.col_vals[ic]], ir, ic)
                self.ranges[ir, ic]['%smin' % ax] = vals[0]
                self.ranges[ir, ic]['%smax' % ax] = vals[1]
            elif len(df_rc) == 0:
                self.ranges[ir, ic]['%smin' % ax] = None
                self.ranges[ir, ic]['%smax' % ax] = None
            elif not getattr(self, 'share_%s' % ax):
                vals = self.get_data_range(ax, df_rc, ir, ic)
                self.ranges[ir, ic]['%smin' % ax] = vals[0]
                self.ranges[ir, ic]['%smax' % ax] = vals[1]
            elif self.wrap is not None and self.wrap == 'y' or self.wrap == 'x':
                vals = self.get_data_range(ax, df_rc, ir, ic)
                self.ranges[ir, ic]['%smin' % ax] = vals[0]
                self.ranges[ir, ic]['%smax' % ax] = vals[1]
            else:
                self.ranges[ir, ic]['%smin' % ax] = \
                    self.ranges[0, 0]['%smin' % ax]
                self.ranges[ir, ic]['%smax' % ax] = \
                    self.ranges[0, 0]['%smax' % ax]

    def get_data_ranges_bar(self, ir, ic):
        """
        Get the data ranges for bar plot data

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        """
        df_bar = pd.DataFrame()
        plot_num = utl.plot_num(ir, ic, self.ncol)

        # y-axis
        if self.share_y and ir == 0 and ic == 0:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                if len(df_rc) == 0:
                    break
                for iline, df, x, y, z, leg_name, twin, ngroups in self.get_plot_data(df_rc):
                    yy = df.groupby(df[self.x[0]]).mean()[self.y[0]]
                    if self.error_bars:
                        yys = df.groupby(df[self.x[0]]).std()[self.y[0]] + yy
                    else:
                        yys = pd.DataFrame()
                    df_bar = pd.concat([df_bar, yy, yys])
            if self.stacked:
                df_bar = df_bar.groupby(df_bar.index).sum()
            df_bar.columns = self.y
            vals = self.get_data_range('y', df_bar, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif self.share_row:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                df_row = df_rc[df_rc[self.row[0]] == self.row_vals[ir]].copy()
                for iline, df, x, y, z, leg_name, twin, ngroups in self.get_plot_data(df_row):
                    yy = df.groupby(df[self.x[0]]).mean()[self.y[0]]
                    if self.error_bars:
                        yys = df.groupby(df[self.x[0]]).std()[self.y[0]] + yy
                    else:
                        yys = pd.DataFrame()
                    df_bar = pd.concat([df_bar, yy, yys])
            if self.stacked:
                df_bar = df_bar.groupby(df_bar.index).sum()
            vals = self.get_data_range('y', df_bar, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif self.share_col:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                df_col = df_rc[df_rc[self.col[0]] == self.col_vals[ic]]
                for iline, df, x, y, z, leg_name, twin in self.get_plot_data(df_col):
                    yy = df.groupby(df[self.x[0]]).mean()[self.y[0]]
                    if self.error_bars:
                        yys = df.groupby(df[self.x[0]]).std()[self.y[0]] + yy
                    else:
                        yys = pd.DataFrame()
                    df_bar = pd.concat([df_bar, yy, yys])
            if self.stacked:
                df_bar = df_bar.groupby(df_bar.index).sum()
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif not self.share_y:
            for iline, df, x, y, z, leg_name, twin in self.get_plot_data(self.df_rc):
                yy = df.groupby(df[self.x[0]]).mean()[self.y[0]]
                if self.error_bars:
                    yys = df.groupby(df[self.x[0]]).std()[self.y[0]] + yy
                else:
                    yys = pd.DataFrame()
                df_bar = pd.concat([df_bar, yy, yys])
            if self.stacked:
                df_bar = df_bar.groupby(df_bar.index).sum()
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        else:
            self.ranges[ir, ic]['ymin'] = self.ranges[0, 0]['ymin']
            self.ranges[ir, ic]['ymax'] = self.ranges[0, 0]['ymax']

        if self.xmin.get(plot_num) is None:
            self.ranges[ir, ic]['xmin'] = None
        else:
            self.ranges[ir, ic]['xmin'] = self.xmin.get(plot_num)
        self.ranges[ir, ic]['x2min'] = None
        self.ranges[ir, ic]['y2min'] = None
        if self.xmax.get(plot_num) is None:
            self.ranges[ir, ic]['xmax'] = None
        else:
            self.ranges[ir, ic]['xmax'] = self.xmax.get(plot_num)
        self.ranges[ir, ic]['x2max'] = None
        self.ranges[ir, ic]['y2max'] = None

        if self.ymin.values == [None] and self.ranges[ir, ic]['ymin'] > 0:
            self.ranges[ir, ic]['ymin'] > 0

    def get_data_ranges_hist(self, ir, ic):
        """
        Get the data ranges

        Args:
            ir (int): subplot row index
            ic (int): subplot col index

        """

        self.y = ['Counts']
        df_hist = pd.DataFrame()
        plot_num = utl.plot_num(ir, ic, self.ncol)

        if self.share_y and ir == 0 and ic == 0:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                if len(df_rc) == 0:
                    break
                for iline, df, x, y, z, leg_name, twin, ngroups in self.get_plot_data(df_rc):
                    counts = np.histogram(df[self.x[0]], bins=self.bins, normed=self.norm)[0]
                    df_hist = pd.concat([df_hist, pd.DataFrame({self.y[0]: counts})])
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif self.share_row:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                df_row = df_rc[df_rc[self.row[0]] == self.row_vals[ir]].copy()
                for iline, df, x, y, z, leg_name, twin, ngroups in self.get_plot_data(df_row):
                    counts = np.histogram(df[self.x[0]], bins=self.bins, normed=self.norm)[0]
                    df_hist = pd.concat([df_hist, pd.DataFrame({self.y[0]: counts})])
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif self.share_col:
            for iir, iic, df_rc in self.get_rc_subset(self.df_fig):
                df_col = df_rc[df_rc[self.col[0]] == self.col_vals[ic]]
                for iline, df, x, y, z, leg_name, twin in self.get_plot_data(df_col):
                    counts = np.histogram(df[self.x[0]], bins=self.bins, normed=self.norm)[0]
                    df_hist = pd.concat([df_hist, pd.DataFrame({self.y[0]: counts})])
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        elif not self.share_y:
            for iline, df, x, y, z, leg_name, twin in self.get_plot_data(self.df_rc):
                counts = np.histogram(df[self.x[0]], bins=self.bins, normed=self.norm)[0]
                df_hist = pd.concat([df_hist, pd.DataFrame({self.y[0]: counts})])
            vals = self.get_data_range('y', df_hist, ir, ic)
            self.ranges[ir, ic]['ymin'] = vals[0]
            self.ranges[ir, ic]['ymax'] = vals[1]
        else:
            self.ranges[ir, ic]['ymin'] = self.ranges[0, 0]['ymin']
            self.ranges[ir, ic]['ymax'] = self.ranges[0, 0]['ymax']
        self.y = None

    def get_df_figure(self):
        """
        Generator to subset the main DataFrame based on fig_item grouping

        Args:
            fig_item (str): figure grouping value
            kw (dict): kwargs dict

        Returns:
            DataFrame subset
        """

        self.get_fig_groupings()

        if not self.fig_vals:
            self.get_legend_groupings(self.df_all)
            self.get_rc_groupings(self.df_all)
            self.df_fig = self.df_all
            self.trans_df_rc = False
            for ir, ic, df_rc in self.get_rc_subset(self.df_fig, True):
                continue
            yield None, None, None, self.df_fig, self

        else:
            for ifig, fig_val in enumerate(self.fig_vals):
                self.trans_df_fig = False
                if type(fig_val) is tuple:
                    for ig, gg in enumerate(fig_val):
                        if ig == 0:
                            self.df_fig = self.df_all[self.df_all[self.fig_groups[ig]] == gg].copy()
                        else:
                            self.df_fig = self.df_fig[self.df_fig[self.fig_groups[ig]] == gg]
                elif self.fig_groups is not None:
                    if type(self.fig_groups) is list:
                        self.df_fig = self.df_all[self.df_all[self.fig_groups[0]] == fig_val].copy()
                    else:
                        self.df_fig = self.df_all[self.df_all[self.fig_groups] == fig_val].copy()
                else:
                    self.df_fig = self.df_all

                self.get_legend_groupings(self.df_fig)
                self.get_rc_groupings(self.df_fig)

                for ir, ic, df_rc in self.get_rc_subset(self.df_fig, True):
                    continue
                yield ifig, fig_val, self.fig, self.df_fig, self

        self.df_fig = None

    def get_fig_groupings(self):
        """
        Determine the figure grouping levels
        """

        if self.fig:
            self.fig_vals = list(self.df_all.groupby(self.fig).groups.keys())

    def get_fit_data(self, ir, ic, df, x, y):
        """
        Make columns of fitted data

        Args:
            df (pd.DataFrame): main DataFrame
            x (str): x-column name
            y (str): y-column name

        Returns:
            updated DataFrame and rsq (for poly fit only)
        """

        df2 = df.copy()
        df['%s Fit' % x] = np.nan
        df['%s Fit' % y] = np.nan

        if not self.fit:
            return df, np.nan

        if self.fit == True or type(self.fit) is int:

            if type(self.fit_range_x) is list:
                df2 = df2[(df2[x] >= self.fit_range_x[0]) & \
                          (df2[x] <= self.fit_range_x[1])].copy()
                if self.ranges[ir, ic]['ymin'] is not None:
                    df2 = df2[(df2[y]) >= self.ranges[ir, ic]['ymin']]
                if self.ranges[ir, ic]['ymax'] is not None:
                    df2 = df2[(df2[y]) <= self.ranges[ir, ic]['ymax']]
            elif type(self.fit_range_y) is list:
                df2 = df2[(df2[y] >= self.fit_range_y[0]) & \
                          (df2[y] <= self.fit_range_y[1])].copy()
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

            xx = np.array(df2[x])
            yy = np.array(df2[y])
            if len(xx) == 0 or len(yy) == 0:
                return df, np.ones(int(self.fit)) * np.nan, 0

            # Fit the polynomial
            coeffs = np.polyfit(xx, yy, int(self.fit))

            # Find R^2
            yval = np.polyval(coeffs, xx)
            ybar = yy.sum()/len(yy)
            ssreg = np.sum((yval-ybar)**2)
            sstot = np.sum((yy-ybar)**2)
            rsq = ssreg/sstot

            # Add fit line
            df['%s Fit' % x] = np.linspace(self.ranges[ir, ic]['xmin'],
                                           self.ranges[ir, ic]['xmax'], len(df))
            df['%s Fit' % y] = np.polyval(coeffs, df['%s Fit' % x])

            return df, coeffs, rsq

        if str(self.fit).lower() == 'spline':

            # PUT SPLINE CODE HERE

            return df, [], np.nan

    def get_legend_groupings(self, df):
        """
        Determine the legend groupings

        Args:
            df (pd.DataFrame):  data being plotted

        Returns:
            updated kwargs dict
        """

        if self.legend == True and self.twin_x \
                or self.legend == True and len(self.y) > 1:
            self.legend_vals = self.y + self.y2
            self.nleg_vals = len(self.y + self.y2)
            return
        elif self.legend == True and self.twin_y:
            self.legend_vals = self.x + self.x2
            self.nleg_vals = len(self.x + self.x2)
            return

        if not self.legend:
            return

        leg_all = []

        if self.legend == True:
            self.legend = None  # no option for legend here so disable
            return

        if self.legend:
            if type(self.legend) is str and ' | ' in self.legend:
                self.legend = self.legend.split(' | ')
            if type(self.legend) is list:
                for ileg, leg in enumerate(self.legend):
                    if ileg == 0:
                        temp = df[leg].copy()
                    else:
                        temp = temp.map(str) + ' | ' + df[leg].map(str)
                self.legend = ' | '.join(self.legend)
                df[self.legend] = temp
            if self.sort:
                legend_vals = \
                    natsorted(list(df.groupby(self.legend).groups.keys()))
            else:
                legend_vals = \
                    list(df.groupby(self.legend, sort=False).groups.keys())
            self.nleg_vals = len(legend_vals)
        else:
            legend_vals = [None]
            self.nleg_vals = 0

        for leg in legend_vals:
            if not self.x:
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
        if not (leg_df.Leg==None).all():
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
        elif len(leg_df.y.unique()) > 1 and not (leg_df.Leg==None).all() \
                and len(leg_df.x.unique()) == 1:
            leg_df['names'] = leg_df.Leg.map(str) + ' | ' + leg_df.y.map(str)

        # if more than one x and leg specified
        if 'names' not in leg_df.columns:
            leg_df['names'] = leg_df.x
        elif 'x' in leg_df.columns and len(leg_df.x.unique()) > 1 \
                and not self.twin_x:
            leg_df['names'] = \
                leg_df['names'].map(str) + ' | ' + leg_df.y.map(str) + ' / ' + leg_df.x.map(str)

        leg_df = leg_df.set_index('names')
        self.legend_vals = leg_df.reset_index()

    def get_plot_data(self, df):
        """
        Generator to subset into discrete sets of data for each curve

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            subset
        """

        if type(self.legend_vals) != pd.DataFrame:
            xx = [] if not self.x else self.x + self.x2
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

    def get_rc_groupings(self, df):
        """
        Determine the row and column or wrap grid groupings

        Args:
            df (pd.DataFrame):  data being plotted; usually a subset of
                self.df_all
        """

        # Set up wrapping (wrap option overrides row/col)
        if self.wrap:
            if self.wrap_vals is None:  # this broke something but removing will cause other failures
                if self.sort:
                    self.wrap_vals = \
                        natsorted(list(df.groupby(self.wrap).groups.keys()))
                else:
                    self.wrap_vals = \
                        list(df.groupby(self.wrap, sort=False).groups.keys())
            if self.ncols == 0:
                rcnum = int(np.ceil(np.sqrt(len(self.wrap_vals))))
            else:
                rcnum = self.ncols if self.ncols <= len(self.wrap_vals) \
                        else len(self.wrap_vals)
            self.ncol = rcnum
            self.nrow = int(np.ceil(len(self.wrap_vals)/rcnum))
            self.nwrap = len(self.wrap_vals)

        # Non-wrapping option
        else:
            # Set up the row grouping
            if self.col:
                if self.col_vals is None:
                    if self.sort:
                        self.col_vals = \
                            natsorted(list(df.groupby(self.col).groups.keys()))
                    else:
                        self.col_vals = \
                            list(df.groupby(self.col, sort=False).groups.keys())
                self.ncol = len(self.col_vals)

            if self.row:
                if self.row_vals is None:
                    if self.sort:
                        self.row_vals = \
                            natsorted(list(df.groupby(self.row).groups.keys()))
                    else:
                        self.row_vals = \
                            list(df.groupby(self.row, sort=False).groups.keys())
                self.nrow = len(self.row_vals)

        if self.ncol == 0:
            raise GroupingError('Cannot make subplot(s): '
                                'number of columns is 0')
        if self.nrow == 0:
            raise GroupingError('Cannot make subplot(s): '
                                'number of rows is 0')

        self.ranges = np.array([[None]*self.ncol]*self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.ranges[ir, ic] = {}

    def get_rc_subset(self, df, ranges=False):
        """
        Subset the data by the row/col values

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            subset DataFrame
        """

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                if self.wrap is not None:
                    if ir*self.ncol + ic > self.nwrap-1:
                        self.df_rc = pd.DataFrame()
                    elif self.wrap == 'y':
                        self.y = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
                        cols = (utl.validate_list(self.x) if self.x is not None else []) + \
                               (utl.validate_list(self.y) if self.y is not None else []) + \
                               (utl.validate_list(self.groups) if self.groups is not None else []) + \
                               (utl.validate_list(self.legend) if self.legend not in [None, True, False] else [])
                        self.df_rc = df[cols]
                    elif self.wrap == 'x':
                        self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
                        cols = (utl.validate_list(self.x) if self.x is not None else []) + \
                               (utl.validate_list(self.y) if self.y is not None else []) + \
                               (utl.validate_list(self.groups) if self.groups is not None else []) + \
                               (utl.validate_list(self.legend) if self.legend is not None else [])
                        if self.plot_func == 'plot_hist':
                            cols = [f for f in cols if f != 'Counts']
                        self.df_rc = df[cols]
                    else:
                        if self.sort:
                            self.wrap_vals = \
                                natsorted(list(df.groupby(self.wrap).groups.keys()))
                        else:
                            self.wrap_vals = list(df.groupby(self.wrap, sort=False).groups.keys())
                        wrap = dict(zip(self.wrap,
                                    utl.validate_list(self.wrap_vals[ir*self.ncol + ic])))
                        self.df_rc = df.loc[(df[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()
                else:
                    if self.row == 'y':
                        self.y = utl.validate_list(self.row_vals[ir])
                    if self.col == 'x':
                        self.x = utl.validate_list(self.col_vals[ic])
                    if self.row not in [None, 'y'] \
                            and self.col not in [None, 'x']:
                        row = self.row_vals[ir]
                        col = self.col_vals[ic]
                        self.df_rc = df[(df[self.row[0]] == row) &
                                        (df[self.col[0]] == col)].copy()
                    elif self.row not in [None, 'y'] and not self.col:
                        row = self.row_vals[ir]
                        self.df_rc = df[(df[self.row[0]] == row)].copy()
                    elif self.col not in [None, 'x'] and not self.row:
                        col = self.col_vals[ic]
                        self.df_rc = df[(df[self.col[0]] == col)].copy()
                    elif self.col not in [None, 'x'] and self.row in [None, 'y']:
                        col = self.col_vals[ic]
                        self.df_rc = df[(df[self.col[0]] == col)].copy()
                    elif self.col in [None, 'x'] and self.row not in [None, 'y']:
                        row = self.row_vals[ir]
                        self.df_rc = df[(df[self.row[0]] == row)].copy()
                    else:
                        self.df_rc = df

                # Reshaping
                if self.plot_func == 'plot_heatmap':
                    if self.pivot:
                        # Reshape if input dataframe is stacked
                        self.df_rc = pd.pivot_table(self.df_rc, values=self.z[0],
                                                    index=self.y[0], columns=self.x[0])
                    if self.sort:
                        cols = natsorted(self.df_rc.columns)
                    else:
                        cols = self.df_rc.columns
                    self.df_rc = self.df_rc[cols]
                    if self.sort:
                        self.df_rc.index = natsorted(self.df_rc.index)

                    # Set limits
                    plot_num = utl.plot_num(ir, ic, self.ncol) - 1
                    if not self.xmin.get(plot_num):
                        self.xmin.values[plot_num] = -0.5
                    if not self.xmax.get(plot_num):
                        self.xmax.values[plot_num] = \
                            len(self.df_rc.columns) - 0.5
                    if self.ymin.get(plot_num) is not None \
                            and self.ymax.get(plot_num) is not None \
                            and self.ymin.get(plot_num) < self.ymax.get(plot_num):
                        ymin = self.ymin.get(plot_num)
                        self.ymin.values[plot_num] = self.ymax.get(plot_num)
                        self.ymax.values[plot_num] = ymin
                    if not self.ymax.get(plot_num):
                        self.ymax.values[plot_num] = -0.5
                    if not self.ymin.get(plot_num):
                        self.ymin.values[plot_num] = len(self.df_rc) - 0.5
                    if self.x == ['Column'] and self.auto_cols:
                        self.df_rc = self.df_rc[[f for f in self.df_rc.columns
                                                 if f >= self.xmin.get(plot_num)]]
                        self.df_rc = self.df_rc[[f for f in self.df_rc.columns
                                                 if f <= self.xmax.get(plot_num)]]
                    if self.y == ['Row'] and self.auto_cols:
                        self.df_rc = self.df_rc.loc[[f for f in self.df_rc.index
                                                     if f >= self.ymax.get(plot_num)]]
                        self.df_rc = self.df_rc.loc[[f for f in self.df_rc.index
                                                     if f <= self.ymin.get(plot_num)]]
                    dtypes = [int, np.int32, np.int64]
                    if self.df_rc.index.dtype in dtypes and list(self.df_rc.index) != \
                            [f + self.df_rc.index[0] for f in range(0, len(self.df_rc.index))]:
                        self.df_rc.index = self.df_rc.index.astype('O')
                    if self.df_rc.columns.dtype == 'object':
                        ddtypes = list(set([type(f) for f in self.df_rc.columns]))
                        if all(f in dtypes for f in ddtypes):
                            self.df_rc.columns = [np.int64(f) for f in self.df_rc.columns]
                    elif self.df_rc.columns.dtype in dtypes and list(self.df_rc.columns) != \
                            [f + self.df_rc.columns[0] for f in range(0, len(self.df_rc.columns))]:
                        self.df_rc.columns = self.df_rc.columns.astype('O')
                    if self.x[0] in self.df_fig.columns:
                        self.num_x = len(self.df_fig[self.x].drop_duplicates())
                    else:
                        self.num_x = None
                    if self.y[0] in self.df_fig.columns:
                        self.num_y = len(self.df_fig[self.y].drop_duplicates())
                    else:
                        self.num_y = None

                # Deal with empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

                # Calculate axis ranges
                if ranges:
                    self.get_data_ranges(ir, ic)

                # Get boxplot changes DataFrame
                if 'box' in self.plot_func and len(self.df_rc) > 0:  # think we are doing this twice
                    self.get_box_index_changes()
                    self.ranges[ir, ic]['xmin'] = 0.5
                    self.ranges[ir, ic]['xmax'] = len(self.changes) + 0.5

                # Yield the subset
                yield ir, ic, self.df_rc

        self.df_sub = None

    def get_stat_data(self, df, x, y):
        """
        Get a stat subset from input data

        Args:
            df (pd.DataFrame): input data
            x (str): x-column name
            y (str): y-column name

        """

        if not self.stat:
            return pd.DataFrame()

        df_stat = df.groupby(x if not self.stat_val else self.stat_val)
        try:
            if 'q' in self.stat:
                q = float(self.stat.replace('q', ''))
                if q > 1: q = q / 100
                return df_stat.quantile(q)
            else:
                return getattr(df_stat, self.stat)().reset_index()
        except:
            print('stat "%s" is not supported...skipping stat calculation' % self.stat)
            return None

    @property
    def groupers(self):
        """
        Get all grouping values
        """

        props = ['row', 'col', 'wrap', 'groups', 'legend', 'fig']
        grouper = []

        for prop in props:
            if getattr(self, prop) not in ['x', 'y', None]:
                grouper += utl.validate_list(getattr(self, prop))

        return list(set(grouper))

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df

    def swap_xy(self):
        """
        Swap the x and y axis
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

    def transform(self):
        """
        Transform x, y, or z data by unique group
        """

        # Possible tranformations
        transform = any([self.trans_x, self.trans_x2, self.trans_y,
                         self.trans_y2, self.trans_z])
        if not transform:
            return

        # Container for transformed data
        df = pd.DataFrame()

        # Transform by unique group
        groups_all = self.groupers
        if len(groups_all) > 0:
            groups = self.df_all.groupby(groups_all)
        else:
            groups = [self.df_all]
        for group in groups:
            if type(group) is tuple:
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
                        gg.loc[:, val] = 1/gg[val]
                    elif (type(getattr(self, 'trans_%s' % ax)) is tuple \
                            or type(getattr(self, 'trans_%s' % ax)) is list) \
                            and getattr(self, 'trans_%s' % ax)[0] == 'pow':
                        gg.loc[:, val] = gg[val]**getattr(self, 'trans_%s' % ax)[1]
                    elif getattr(self, 'trans_%s' % ax) == 'flip':
                        maxx = gg.loc[:, val].max()
                        gg.loc[:, val] -= maxx
                        gg.loc[:, val] = abs(gg[val])

            if type(group) is tuple:
                vals = group[0] if type(group[0]) is tuple else [group[0]]
                for k, v in dict(zip(groups_all, vals)).items():
                    gg[k] = v

            df = pd.concat([df, gg])

        self.df_all = df
