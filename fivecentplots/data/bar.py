from . import data
import pdb
import pandas as pd
from .. import utilities
utl = utilities
db = pdb.set_trace


class Bar(data.Data):
    def __init__(self, **kwargs):
        """Barplot-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        name = 'bar'

        super().__init__(name, **kwargs)

        # overrides
        self.stacked = utl.kwget(kwargs, self.fcpp, ['bar_stacked', 'stacked'],
                                 kwargs.get('stacked', False))
        if utl.kwget(kwargs, self.fcpp, ['bar_error_bars', 'error_bars'],
                     kwargs.get('error_bars', False)):
            self.error_bars = True

    def _filter_data(self, kwargs):
        """Apply an optional filter to the data but allow showing of all groups
        with kwarg `show_all_groups`

        Args:
            kwargs: user-defined keyword args
        """
        if kwargs.get('show_all_groups', False):
            groups = self.df_all[self.x[0]].unique()

        if self.filter:
            self.df_all = utl.df_filter(self.df_all, self.filter)
            if len(self.df_all) == 0:
                raise data.DataError('DataFrame is empty after applying filter')

        if kwargs.get('show_all_groups', False):
            missing = [f for f in groups if f not in self.df_all[self.x[0]].unique()]
            temp = pd.DataFrame(columns=self.df_all.columns, index=range(0, len(missing)))
            for mm in missing:
                temp[self.x[0]] = mm
                temp[self.y[0]] = 0
            self.df_all = pd.concat([self.df_all, temp]).reset_index(drop=True)

    def _get_data_ranges(self):
        """Barplot-specific data range calculator by subplot."""
        # First get any user defined range values and apply optional auto scaling
        df_fig = self.df_fig.copy()  # use temporarily for setting ranges
        self._get_data_ranges_user_defined()
        df_fig = self._get_auto_scale(df_fig)
        df_bar = pd.DataFrame()

        for ir, ic, plot_num in self._get_subplot_index():
            # y-axis
            groupby = self.x + self._groupers
            df_rc = self._subset(ir, ic)

            if len(df_rc) == 0:
                self._add_ranges_none(ir, ic)
                break
            if self.share_y and ir == 0 and ic == 0:
                df_rc = df_fig
            elif self.share_row:
                df_rc = df_rc[df_rc[self.row[0]] == self.row_vals[ir]].copy()
            elif self.share_col:
                df_rc = df_rc[df_rc[self.col[0]] == self.col_vals[ic]].copy
            elif self.share_y and ir > 0 or ic > 0:
                self._add_range(ir, ic, 'y', 'min', self.ranges[0, 0]['ymin'])
                self._add_range(ir, ic, 'y', 'max', self.ranges[0, 0]['ymax'])
                continue
            elif self.wrap is not None:
                df_rc = df_fig
                groupby += self.wrap

            # sum along the bar groups
            yy = df_rc.groupby(groupby).sum()[self.y[0]].reset_index()

            # add error bar std dev
            if self.error_bars:
                yys = df_rc.groupby(groupby).std()[self.y[0]].reset_index()
                yy[self.y[0]] += yys[self.y[0]]

            df_bar = pd.concat([df_bar, yy])

            # stacked case
            if self.stacked:
                df_bar = df_bar.groupby(self.x[0]).sum()

            # get the ranges
            vals = self._get_data_range('y', df_bar, plot_num)
            if any(df_bar[self.y[0]].values < 0):
                self._add_range(ir, ic, 'y', 'min', vals[0])
            else:
                self._add_range(ir, ic, 'y', 'min', 0)
            self._add_range(ir, ic, 'y', 'max', vals[1])

        for ir, ic, plot_num in self._get_subplot_index():
            # other axes
            self._add_range(ir, ic, 'x', 'min', None)
            self._add_range(ir, ic, 'x2', 'min', None)
            self._add_range(ir, ic, 'y2', 'min', None)
            self._add_range(ir, ic, 'x', 'max', None)
            self._add_range(ir, ic, 'x2', 'max', None)
            self._add_range(ir, ic, 'y2', 'max', None)
