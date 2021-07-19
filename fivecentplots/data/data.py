import pandas as pd
import numpy as np
from .. import utilities
utl = utilities
import scipy.stats as ss

try:
    from natsort import natsorted
except:
    natsorted = sorted

import pdb
db = pdb.set_trace


def local_groupers(kwargs):
    props = ['row', 'col', 'wrap', 'groups', 'legend', 'fig']
    grouper = []

    for prop in props:
        if prop in kwargs.keys() and kwargs[prop] not in ['x', 'y', None]:
            grouper += utl.validate_list(kwargs[prop])

    return grouper


def reshape_2D(kwargs):
    """
    Reshape 2D image data to be suitable for certain non-imshow plot types

    Args:
        kwargs (dict): user-input keyword dict

    Returns:
        updated kwargs

    """

    kwargs['x'] = ['Value']
    lg = local_groupers(kwargs)
    if len(lg) > 0:
        kwargs['df'] = kwargs['df'].set_index(lg)
        kwargs['df'] = pd.DataFrame(kwargs['df'].stack())
        kwargs['df'].columns = kwargs['x']
        kwargs['df'] = kwargs['df'].reset_index()
    else:
        kwargs['df'] = kwargs['df'][utl.df_int_cols(kwargs['df'])]
        kwargs['df'] = pd.DataFrame(kwargs['df'].stack())
        kwargs['df'].columns = kwargs['x']

    return kwargs


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
    def __init__(self, name='xy', req=['x', 'y'], opt=[], **kwargs):

        # defaults
        self.req = req
        self.opt = opt
        self.name = name

        # Reload default file
        self.fcpp, dummy, dummy2 = utl.reload_defaults(kwargs.get('theme', None))

        # Default axis attributes
        self.auto_cols = False
        self.auto_scale = utl.kwget(kwargs, self.fcpp, 'auto_scale', True)
        self.axs = ['x', 'x2', 'y', 'y2', 'z']
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
        normalize = utl.kwget(kwargs, self.fcpp, ['hist_normalize', 'normalize'],
                              kwargs.get('normalize', False))
        kde=utl.kwget(kwargs, self.fcpp, ['hist_kde', 'kde'],
                      kwargs.get('kde', False))
        if normalize or kde:
            self.norm = True
        else:
            self.norm = False
        self.bins = utl.kwget(kwargs, self.fcpp, ['hist_bins', 'bins'],
                              kwargs.get('bins', 20))

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
        if self.legend and self.groups and self.name != 'box':
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
            if not hasattr(self, k):  # k not in ['df', 'func', 'x', 'y', 'z']:
                setattr(self, k, v)

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

    def _get_data_ranges(self):
        """
        Default data range calculator by subplot
        --> some plot types need a custom calc
        """

        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self.get_auto_scale(df_fig)

        # Apply shared axes
        for ir, ic, plot_num in self.get_subplot_index():
            for ax in self.axs:
                # Share axes (use self.df_fig to get global min/max)
                if getattr(self, 'share_%s' % ax):
                    vals = self.get_data_range(ax, df_fig, plot_num)
                    self.add_range(ir, ic, ax, 'min', vals[0])
                    self.add_range(ir, ic, ax, 'max', vals[1])
                elif getattr(self, 'share_%s' % ax) and (ir > 0 or ic > 0):
                    self.add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                    self.add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])

                # Share row
                elif self.share_row and self.row is not None:
                    if self.row == 'y':
                        vals = self.get_data_range(ax, df_fig, plot_num)
                    else:
                        vals = self.get_data_range(ax,
                            df_fig[self.df_fig[self.row[0]] == self.row_vals[ir]], plot_num)
                    self.add_range(ir, ic, ax, 'min', vals[0])
                    self.add_range(ir, ic, ax, 'max', vals[1])
                elif self.share_row and self.row is not None and ic > 0:
                    self.add_range(ir, ic, ax, 'min', self.ranges[ir, 0]['%smin' % ax])
                    self.add_range(ir, ic, ax, 'max', self.ranges[ir, 0]['%smax' % ax])

                # Share col
                elif self.share_col and self.col is not None:
                    if self.col == 'x':
                        vals = self.get_data_range(ax, df_fig, ir, ic)
                    else:
                        vals = self.get_data_range(ax,
                            df_fig[df_fig[self.col[0]] == self.col_vals[ic]], plot_num)
                    self.add_range(ir, ic, ax, 'min', vals[0])
                    self.add_range(ir, ic, ax, 'max', vals[1])
                elif self.share_col and self.col is not None and ir > 0:
                    self.add_range(ir, ic, ax, 'min', self.ranges[0, ic]['%smin' % ax])
                    self.add_range(ir, ic, ax, 'max', self.ranges[0, ic]['%smax' % ax])

                # subplot level when not shared
                else:
                    df_rc = self.subset(ir, ic)

                    # Empty rc
                    if len(df_rc) == 0:  # this doesn't exist yet!
                        self.add_range(ir, ic, ax, 'min', None)
                        self.add_range(ir, ic, ax, 'max', None)
                        continue

                    # Not shared or wrap by x or y
                    elif not getattr(self, 'share_%s' % ax) or \
                            (self.wrap is not None and \
                                self.wrap == 'y' or \
                                self.wrap == 'x'):
                        vals = self.get_data_range(ax, df_rc, plot_num)
                        self.add_range(ir, ic, ax, 'min', vals[0])
                        self.add_range(ir, ic, ax, 'max', vals[1])

                    # Make them all equal to 0,0
                    elif ir > 0 or ic > 0:
                        self.add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                        self.add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])

    def _get_data_ranges_user_defined(self):
        """
        Get user defined range values
        """

        for ir, ic, plot_num in self.get_subplot_index():
            for ax in self.axs:
                for mm in ['min', 'max']:
                    # User defined
                    key = '{}{}'.format(ax, mm)
                    val = getattr(self, key).get(plot_num)
                    if val is not None and type(val) is not str:
                        self.add_range(ir, ic, ax, mm, val)

    def _subset_modify(self, df, ir, ic):
        """
        Perform any additional DataFrame subsetting (only on specific plot types)
        """

        return df

    def add_range(self, ir, ic, ax, label, value):
        """
        Add a range value unless it already exists
        """

        key = '{}{}'.format(ax, label)
        if key not in self.ranges[ir, ic].keys() or \
                self.ranges[ir, ic][key] is None:
            self.ranges[ir, ic][key] = value

    def add_ranges_none(self, ir, ic):
        """
        Add None for all range values
        """

        self.add_range(ir, ic, 'x', 'min', None)
        self.add_range(ir, ic, 'x2', 'min', None)
        self.add_range(ir, ic, 'y', 'min', None)
        self.add_range(ir, ic, 'y2', 'min', None)
        self.add_range(ir, ic, 'z', 'min', None)
        self.add_range(ir, ic, 'x', 'max', None)
        self.add_range(ir, ic, 'x2', 'max', None)
        self.add_range(ir, ic, 'y', 'max', None)
        self.add_range(ir, ic, 'y2', 'max', None)
        self.add_range(ir, ic, 'z', 'max', None)

    def ax_limit_pad(self, **kwargs):
        """
        Set padding limits for axis
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

        if xyz not in self.req and xyz not in self.opt:
            return

        if xyz in self.opt and getattr(self, xyz) is None:
            return None

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

    def get_auto_scale(self, df):
        """
        Auto-scale the plot data

        Args:
            df (pd.DataFrame): either self.df_fig or self.df_rc

        Returns:
            updated df with potentially reduced range of data

        """

        if not self.auto_scale:
            return df

        # get the max/min autoscale values
        for ax in self.axs:
            for mm in ['min', 'max']:
                for ir, ic, _ in self.get_subplot_index():
                    key = '{}{}'.format(ax, mm)
                    if ir==0 and ic==0 and key in self.ranges[ir][ic].keys():
                        auto_scale_val = self.ranges[ir][ic][key]
                    elif key in self.ranges[ir][ic].keys():
                        if mm == 'min':
                            auto_scale_val = min(auto_scale_val, self.ranges[ir][ic][key])
                        else:
                            auto_scale_val = max(auto_scale_val, self.ranges[ir][ic][key])
                    else:
                        auto_scale_val = None
                if type(auto_scale_val) is str or auto_scale_val is None:
                    continue

                axx = getattr(self, ax)
                for col in axx:
                    if mm == 'min':
                        df = df[df[col] >= auto_scale_val]
                    else:
                        df = df[df[col] <= auto_scale_val]

        return df

    def get_conf_int(self, df, x, y, **kwargs):
        """
        Calculate confidence intervals around a curve

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

    def get_data_range(self, ax, df, plot_num):
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

        # Groupby for stats
        df = self.get_stat_groupings(df)

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
        vmin = getattr(self, '%smin' % ax).get(plot_num)
        if vmin is not None and 'iqr' in str(vmin).lower():
            factor = str(vmin).split('*')
            if len(factor) == 1:
                factor = 1
            else:
                factor = float(factor[0])
            if 'box' not in self.name or self.groups is None:
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
            elif 'box' in self.name:
                vmin = df[self.groupers + cols].groupby(self.groupers) \
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
        vmax = getattr(self, '%smax' % ax).get(plot_num)
        if vmax is not None and 'iqr' in str(vmax).lower():
            factor = str(vmax).split('*')
            if len(factor) == 1:
                factor = 1
            else:
                factor = float(factor[0])
            if 'box' not in self.name or self.groups is None:
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
            elif 'box' in self.name:  # move to data.box.py?
                vmax = df[self.groupers + cols].groupby(self.groupers) \
                        .quantile(xq)[cols].max().iloc[0]
            else:
                vmax = df[self.groups + cols].groupby(self.groups) \
                        .quantile(xq)[cols].max().iloc[0]
            #getattr(self, '%smax' % ax).values[plot_num]
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
            # no fig grouping
            self.get_legend_groupings(self.df_all)
            self.get_rc_groupings(self.df_all)
            self.df_fig = self.df_all
            self.get_data_ranges()

            yield None, None, None, self

        else:
            # with fig grouping
            for ifig, fig_val in enumerate(self.fig_vals):
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
                self.get_data_ranges()

                yield ifig, fig_val, self.fig, self

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

        self.ranges = self.range_dict()

    def get_subplot_index(self):

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                yield ir, ic, utl.plot_num(ir, ic, self.ncol) - 1

    def get_rc_subset(self):
        """
        Subset the data by the row/col/wrap values

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            subset DataFrame
        """

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.df_rc = self.subset(ir, ic)

                # Handle empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

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

    def get_stat_groupings(self, df):
        """
        Groupby for stat/stat_vals

        Args:
            df (pd.DataFrame):  subset

        Returns:
            updated DataFrame
        """

        if self.stat is not None and 'only' in self.stat:
            stat_groups = []
            vals_2_chk = ['stat_val', 'legend', 'col', 'row', 'wrap']
            for v in vals_2_chk:
                if getattr(self, v) is not None:
                    stat_groups += utl.validate_list(getattr(self, v))

        if self.stat is not None and 'only' in self.stat \
                and 'median' in self.stat:
            df = df.groupby(stat_groups).median().reset_index()
        elif self.stat is not None and 'only' in self.stat:
            df = df.groupby(stat_groups).mean().reset_index()

        return df

    def range_dict(self):
        """
        Make a list of empty dicts for axes range limits
        """

        ranges = np.array([[None]*self.ncol]*self.nrow)
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                ranges[ir, ic] = {}

        return ranges

    def see(self):
        """
        Prints a readable list of class attributes
        """

        df = pd.DataFrame({'Attribute':list(self.__dict__.copy().keys()),
             'Name':[str(f) for f in self.__dict__.copy().values()]})
        df = df.sort_values(by='Attribute').reset_index(drop=True)

        return df

    def subset(self, ir, ic, apply_ranges=False):
        # Wrap plot
        if self.wrap is not None:
            df = self.subset_wrap(ir, ic)

        # Non-wrap plot
        else:
            df = self.subset_row_col(ir, ic)

        # Optional plot specific subset modification
        df = self.subset_modify(df, ir, ic)

        if not apply_ranges:
            return df

        # apply any known ranges
        for k, v in self.ranges[ir, ic].items():
            if v is not None:
                if k[1] == '2':
                    ax = k[0:2]
                else:
                    ax = k[0]
                vv = getattr(self, ax)
                attr = k[-3:]
                for vvv in vv:
                    if attr == 'min':
                        df = df.loc[df[vvv]>=v]
                    else:
                        df = df.loc[df[vvv]<=v]

        return df

    def subset_row_col(self, ir, ic):
        """
        For row/column plots (no wrap), select the revelant subset from self.df_fig
        """

        if self.row == 'y':
            self.y = utl.validate_list(self.row_vals[ir])
        if self.col == 'x':
            self.x = utl.validate_list(self.col_vals[ic])
        if self.row not in [None, 'y'] \
                and self.col not in [None, 'x']:
            row = self.row_vals[ir]
            col = self.col_vals[ic]
            return self.df_fig[(self.df_fig[self.row[0]] == row) &
                               (self.df_fig[self.col[0]] == col)].copy()
        elif self.row not in [None, 'y'] and \
                (not self.col or self.col in [None, 'x']):
            row = self.row_vals[ir]
            return self.df_fig[(self.df_fig[self.row[0]] == row)].copy()
        elif self.col not in [None, 'x'] and \
                (not self.row or self.row in [None, 'y']):
            col = self.col_vals[ic]
            return self.df_fig[(self.df_fig[self.col[0]] == col)].copy()
        else:
            return self.df_fig.copy()

    def _subset_wrap(self, ir, ic):
        """
        For wrap plots, select the revelant subset from self.df_fig
        """

        if ir * self.ncol + ic > self.nwrap-1:
            return pd.DataFrame()
        elif self.wrap == 'y':
            # can we drop these validate calls for speed
            self.y = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend) if self.legend not in [None, True, False] else [])
            return self.df_fig[cols]
        elif self.wrap == 'x':
            self.x = utl.validate_list(self.wrap_vals[ic + ir * self.ncol])
            cols = (self.x if self.x is not None else []) + \
                   (self.y if self.y is not None else []) + \
                   (self.groups if self.groups is not None else []) + \
                   (utl.validate_list(self.legend) if self.legend is not None else [])
            return self.df_fig[cols]
        else:
            if self.sort:
                self.wrap_vals = \
                    natsorted(list(self.df_fig.groupby(self.wrap).groups.keys()))
            else:
                self.wrap_vals = list(self.df_fig.groupby(self.wrap, sort=False).groups.keys())
            wrap = dict(zip(self.wrap,
                        utl.validate_list(self.wrap_vals[ir*self.ncol + ic])))
            return self.df_fig.loc[(self.df_fig[list(wrap)] == pd.Series(wrap)).all(axis=1)].copy()

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
