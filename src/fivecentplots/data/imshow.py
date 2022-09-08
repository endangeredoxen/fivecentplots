from . import data
import pdb
import pandas as pd
from .. import utilities
utl = utilities
db = pdb.set_trace


class ImShow(data.Data):
    def __init__(self, fcpp: dict = {}, **kwargs):
        """ImShow-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            fcpp: theme-file kwargs
            kwargs: user-defined keyword args
        """
        name = 'imshow'
        req = []
        opt = []
        kwargs['df'] = utl.df_from_array2d(kwargs['df'])
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', None)

        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, dummy, dummy2 = utl.reload_defaults(kwargs.get('theme', None))

        # Color plane splitting
        cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
        if cfa is not None:
            kwargs['df'] = utl.split_color_planes(kwargs['df'], cfa)

        # auto stretching
        self._stretch(kwargs)

        # check for invalid axis options
        vals = ['twin_x', 'twin_y']
        for val in vals:
            if val in kwargs:
                raise data.AxisError(f'{val} is not a valid option for imshow plots')

        # check for invalid grouping options
        if 'row' in kwargs and kwargs['row'] == 'y':
            raise data.GroupingError('Cannot group row by "y" for imshow plots')
        if 'col' in kwargs and kwargs['col'] == 'x':
            raise data.GroupingError('Cannot group col by "x" for imshow plots')
        if 'wrap' in kwargs and kwargs['wrap'] == 'y':
            raise data.GroupingError('Cannot wrap by "y" for imshow plots')
        if 'legend' in kwargs and kwargs['legend'] is not None:
            raise data.GroupingError('legend not available for imshow plots')

        super().__init__(name, req, opt, self.fcpp, **kwargs)

        # overrides
        self.auto_scale = False

        self.auto_cols = True
        self.x = ['Column']
        self.y = ['Row']
        self.z = ['Value']

        self.df_all = utl.df_int_cols_convert(self.df_all)

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.  For imshow, there are no req or opt
        values needed.

        Args:
            xyz: name of variable to check
        """
        return getattr(self, xyz)

    def _get_data_range(self, ax: str, df: pd.DataFrame, plot_num: int) -> tuple:
        """Determine the min/max values for a given axis based on user inputs.
        ImShow-specific modifications applied.

        Args:
            ax: name of the axis ('x', 'y', etc)
            df: data table to use for range calculation
            plot_num: index number of the current subplot

        Returns:
            min, max tuple
        """
        if not hasattr(self, ax) or getattr(self, ax) in [None, []]:
            return None, None
        else:
            cols = getattr(self, ax)

        # imshow special case
        df = df.dropna(axis=1, how='all')
        groups = self._groupers
        if getattr(self, ax) == ['Column']:
            vmin = 0
            if len(groups) > 0:
                cols = df.iloc[0].dropna().index
                cols = [f for f in cols if f not in groups]
                vmax = len(cols)
            else:
                vmax = len(df.columns)
        elif getattr(self, ax) == ['Row']:
            vmin = 0
            if len(groups) > 0:
                vmax = df.groupby(self._groupers).size().max()
            else:
                vmax = len(df.index)
        elif getattr(self, ax) == ['Value']:  # and self.auto_cols:
            vmin = df[utl.df_int_cols(df)].min().min()
            vmax = df[utl.df_int_cols(df)].max().max()

        return vmin, vmax

    def _get_data_ranges(self):
        """ImShow-specific data range calculator by subplot."""
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self._get_auto_scale(df_fig)

        # Apply shared axes
        for ir, ic, plot_num in self._get_subplot_index():
            for ax in self.axs:
                # Share axes (use self.df_fig to get global min/max)
                if getattr(self, 'share_%s' % ax) and ir == 0 and ic == 0:
                    vals = self._get_data_range(ax, df_fig, plot_num)
                    self._add_range(ir, ic, ax, 'min', vals[0])
                    self._add_range(ir, ic, ax, 'max', vals[1])
                elif getattr(self, 'share_%s' % ax) and (ir > 0 or ic > 0):
                    self._add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])

                # Share row
                elif self.share_row and self.row is not None and ic > 0:
                    self._add_range(ir, ic, ax, 'min', self.ranges[ir, 0]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[ir, 0]['%smax' % ax])
                elif self.share_row and self.row is not None:
                    vals = self._get_data_range(ax, df_fig[self.df_fig[self.row[0]] == self.row_vals[ir]], plot_num)
                    self._add_range(ir, ic, ax, 'min', vals[0])
                    self._add_range(ir, ic, ax, 'max', vals[1])

                # Share col
                elif self.share_col and self.col is not None and ir > 0:
                    self._add_range(ir, ic, ax, 'min', self.ranges[0, ic]['%smin' % ax])
                    self._add_range(ir, ic, ax, 'max', self.ranges[0, ic]['%smax' % ax])
                elif self.share_col and self.col is not None:
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
                            (self.wrap is not None
                             and self.wrap == 'y'
                             or self.wrap == 'x'):
                        vals = self._get_data_range(ax, df_rc, plot_num)
                        self._add_range(ir, ic, ax, 'min', vals[0])
                        self._add_range(ir, ic, ax, 'max', vals[1])

                    # Make them all equal to 0,0
                    elif ir > 0 or ic > 0:
                        self._add_range(ir, ic, ax, 'min', self.ranges[0, 0]['%smin' % ax])
                        self._add_range(ir, ic, ax, 'max', self.ranges[0, 0]['%smax' % ax])

        # some extras
        width = len(self.df_fig.dropna(axis=1, how='all').columns)
        height = len(self.df_fig.dropna(axis=0, how='all').index)
        self.wh_ratio = width / height

        for ir, ic, plot_num in self._get_subplot_index():
            # invert ymin and ymax
            temp = self.ranges[ir, ic]['ymax']
            self.ranges[ir, ic]['ymax'] = self.ranges[ir, ic]['ymin']
            self.ranges[ir, ic]['ymin'] = temp

            # get the ratio of width to height for figure size
            width = self.ranges[ir, ic]['xmax'] - self.ranges[ir, ic]['xmin']
            height = self.ranges[ir, ic]['ymin'] - self.ranges[ir, ic]['ymax']
            self.wh_ratio = max(self.wh_ratio, width / height)

    def get_rc_subset(self):
        """Subset the ImShow data by the row/col/wrap values.

        Yields:
            ir: subplot row index
            ic: subplot column index
            row/col data subset
        """
        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.df_rc = self._subset(ir, ic)

                # imshow addition
                self.df_rc.index.astype = int
                cols = utl.df_int_cols(self.df_rc)
                self.df_rc = self.df_rc[cols]
                self.df_rc.columns.astype = int

                # Deal with empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

                # Yield the subset
                yield ir, ic, self.df_rc

        self.df_sub = None

    def _stretch(self, kwargs):
        """Perform contrast strectching on an image

        Args:
            kwargs: user-defined keyword args
        """
        stretch = utl.kwget(kwargs, self.fcpp, 'stretch', kwargs.get('stretch', None))
        if stretch is not None:
            stretch = utl.validate_list(stretch)
            if len(stretch) == 1:
                stretch = [stretch[0], stretch[0]]
            uu = kwargs['df'].stack().mean()
            ss = kwargs['df'].stack().std()
            kwargs['zmin'] = uu - abs(stretch[0]) * ss
            kwargs['zmax'] = uu + stretch[1] * ss
