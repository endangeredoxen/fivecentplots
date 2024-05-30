from . import data
import pdb
import pandas as pd
import numpy as np
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

        # For image data, grouping information is stored in kwargs['df'] but the actual image arrays are kwargs['imgs']
        # which is a dict of numpy.ndarray's or 2D dataframes
        kwargs['df'], kwargs['imgs'] = utl.img_data_format(kwargs)

        self.channels = kwargs['df'].iloc[0]['channels']
        if self.channels == 1:
            self.shape = (kwargs['df'].iloc[0]['rows'], kwargs['df'].iloc[0]['cols'])
        else:
            self.shape = (kwargs['df'].iloc[0]['rows'], kwargs['df'].iloc[0]['cols'], self.channels)

        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', None)

        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        self.wh_ratio = utl.kwget(kwargs, self.fcpp, ['wh_ratio'], kwargs.get('wh_ratio', None))

        # Color plane splitting
        cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
        if cfa is not None and self.channels == 1:
            kwargs['df'], kwargs['imgs'] = utl.split_color_planes_wrapper(kwargs['df'], kwargs['imgs'], cfa)

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
        #self.z = [f for f in self.imgs[list(self.imgs.keys())[0]].columns if f not in ['Row', 'Column']]
        if self.channels > 1 and hasattr(self, 'cbar') and self.cbar:
            raise ValueError('Colorbar option not available for 3D image data')
            self.cbar = False

        # Update valid axes
        self.axs = [f for f in ['x', 'x2', 'y', 'y2', 'z'] if getattr(self, f) not in [None, []]]

        # auto stretching
        self._stretch(kwargs)

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

        # imshow special case
        if getattr(self, ax) in [['Column'], ['Row']]:
            vmin = 0
            if 'Plane' in df.columns:
                df = df.dropna(axis=1, how='all')
                vmax = df.groupby(self._groupers)[getattr(self, ax)[0]].unique().str.len().max()
            else:
                vmax = df[getattr(self, ax)[0]].max() + 1
        else:
            # Can these even happen any more?
            vmin = df[getattr(self, ax)].min()
            vmax = df[getattr(self, ax)].max()

        return vmin, vmax

    def _zmin(self, df, zmin):
        if isinstance(zmin, str) and 'q' in zmin:
            qq = float(zmin[1:])
            if qq > 1:
                qq /= 100
            return np.quantile(df, qq)
        else:
            return df.min()

    def _zmax(self, df, zmax):
        if isinstance(zmax, str) and 'q' in zmax:
            qq = float(zmax[1:])
            if qq > 1:
                qq /= 100
            return np.quantile(df, qq)
        else:
            return df.max()

    def _get_data_ranges(self):
        """ImShow-specific data range calculator by subplot."""
        # Get user ranges
        self._get_data_ranges_user_defined()

        # Get the image indices associated with the figure
        idx = list(self.df_fig.index)

        # Apply x/y axes ranges
        for ir, ic, plot_num in self._get_subplot_index():
            # x- and y-axis min limits (always zero unless user overrides via kwargs)
            self._add_range(ir, ic, 'x', 'min', 0)
            self._add_range(ir, ic, 'y', 'min', 0)

            # x- and y-axis max limits (size of image rows and cols unless user overrides via kwargs)
            self._add_range(ir, ic, 'x', 'max', self.df_fig.loc[idx].cols.max())
            if len(self._groupers) > 0:
                self._add_range(ir, ic, 'y', 'max', self.df_fig.loc[idx].rows.max())
            else:
                self._add_range(ir, ic, 'y', 'max', self.df_fig.loc[idx].rows.sum())

            # Invert y (do it here in case user-defined a y-range)
            temp = self.ranges[ir, ic]['ymax']
            self.ranges[ir, ic]['ymax'] = self.ranges[ir, ic]['ymin']
            self.ranges[ir, ic]['ymin'] = temp

        # Get the z-axis ranges for each subplot
        _zmin = np.array([[None] * self.ncol] * self.nrow)
        _zmax = np.array([[None] * self.ncol] * self.nrow)
        for ir, ic, plot_num in self._get_subplot_index():
            df = self._subset(ir, ic)
            if len(df) == 0:
                # Empty dataframe
                _zmin[ir, ic] = np.nan
                _zmax[ir, ic] = np.nan
                continue

            # zmin's
            if isinstance(self.zmin[plot_num], str) and 'q' in self.zmin[plot_num]:
                qq = float(self.zmin[plot_num][1:])
                if qq > 1:
                    qq /= 100
                _zmin[ir, ic] = np.quantile(df, qq)
            else:
                _zmin[ir, ic] = df.min()

            # zmax's
            if isinstance(self.zmax[plot_num], str) and 'q' in self.zmax[plot_num]:
                qq = float(self.zmax[plot_num][1:])
                if qq > 1:
                    qq /= 100
                _zmax[ir, ic] = np.quantile(df, qq)
            else:
                _zmax[ir, ic] = df.max()

        # Apply the z-ranges as specified by the user
        for ir, ic, plot_num in self._get_subplot_index():
            if len(self.zmin) > 1 or not any([self.share_z, self.share_row, self.share_col]):
                self._add_range(ir, ic, 'z', 'min', _zmin[ir, ic])
            elif self.share_z:
                self._add_range(ir, ic, 'z', 'min', np.nanmin(_zmin))
            elif self.share_row:
                self._add_range(ir, ic, 'z', 'min', np.nanmin(_zmin[ir, :]))
            else:
                self._add_range(ir, ic, 'z', 'min', np.nanmin(_zmin[:, ic]))

            if len(self.zmax) > 1 or not any([self.share_z, self.share_row, self.share_col]):
                self._add_range(ir, ic, 'z', 'max', _zmax[ir, ic])
            elif self.share_z:
                self._add_range(ir, ic, 'z', 'max', np.nanmax(_zmax))
            elif self.share_row:
                self._add_range(ir, ic, 'z', 'max', np.nanmax(_zmax[ir, :]))
            else:
                self._add_range(ir, ic, 'z', 'max', np.nanmax(_zmax[:, ic]))

        # if self.share_z and len(self.zmin) == 1:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'min', _zmin.min())
        # elif self.share_row:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'min', _zmin[ir, :].min())
        # elif self.share_col:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'min', _zmin[:, ic].min())


        # elif len(self.zmin) > 1:  # overrides share_z
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'min', self.zmin[plot_num - 1])

        # # Share row zmin
        # elif self.share_row and not self.wrap:
        #     for irv, rv in enumerate(self.row_vals):
        #         zmin = 1E20
        #         idx_row = self.df_fig.loc[self.df_fig[self.row[0]] == rv].index
        #         for ii in idx_row:
        #             zmin = min(zmin, self.imgs[ii].min())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if ir == irv:
        #                 self._add_range(ir, ic, 'z', 'min', zmin)

        # # Share row and wrap zmin
        # elif self.share_row and self.wrap:
        #     for ir in range(0, self.nrow):
        #         zmin = 1E20
        #         idx_row = range(0 + ir * self.ncol, min(self.ncol + ir * self.ncol, len(self.imgs.keys())))
        #         for ii in idx_row:
        #             zmin = min(zmin, self.imgs[ii].min())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if plot_num in idx_row:
        #                 self._add_range(ir, ic, 'z', 'min', zmin)

        # # Share col zmin
        # elif self.share_col and not self.wrap:
        #     for icv, cv in enumerate(self.col_vals):
        #         zmin = 1E20
        #         idx_col = self.df_fig.loc[self.df_fig[self.col[0]] == cv].index
        #         for ii in idx_col:
        #             zmin = min(zmin, self.imgs[ii].min())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if ic == icv:
        #                 self._add_range(ir, ic, 'z', 'min', zmin)

        # # Share col and wrap zmin
        # elif self.share_col and self.wrap:
        #     for ic in range(0, self.ncol):
        #         zmin = 1E20
        #         idx_col = list(self.imgs.keys())[ic * (self.ncol - 1)::self.ncol]
        #         for ii in idx_col:
        #             zmin = min(zmin, self.imgs[ii].min())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if plot_num in idx_col:
        #                 self._add_range(ir, ic, 'z', 'min', zmin)

        # # No sharing zmin
        # else:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         df_rc = self._subset(ir, ic)

        #         # Empty rc
        #         if len(df_rc) == 0:  # this doesn't exist yet!
        #             self._add_range(ir, ic, 'z', 'min', None)
        #             continue

        #         # Existing rc
        #         else:  # doesn't support the case of list of 'q's or q in % (check if data supports that)
        #             if isinstance(self.zmin[0], str) and 'q' in self.zmin[0]:
        #                 self._add_range(ir, ic, 'z', 'min', np.quantile(df_rc, float(self.zmin[0][1:])))
        #             else:
        #                 self._add_range(ir, ic, 'z', 'min', df_rc.min())

        # # zmax
        # if self.share_z and len(self.zmax) == 1:
        #     zmax = 0
        #     for ii in idx:
        #         zmax = max(zmax, self.imgs[ii].max())
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'max', zmax)
        # elif len(self.zmax) > 1:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         self._add_range(ir, ic, 'z', 'max', self.zmax[plot_num - 1])

        # # Share row zmax
        # elif self.share_row and not self.wrap:
        #     for irv, rv in enumerate(self.row_vals):
        #         zmax = 0
        #         idx_row = self.df_fig.loc[self.df_fig[self.row[0]] == rv].index
        #         for ii in idx_row:
        #             zmax = max(zmax, self.imgs[ii].max())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if ir == irv:
        #                 self._add_range(ir, ic, 'z', 'max', zmax)

        # # Share row and wrap zmax
        # elif self.share_row and self.wrap:
        #     for ir in range(0, self.nrow):
        #         zmax = 0
        #         idx_row = range(0 + ir * self.ncol, min(self.ncol + ir * self.ncol, len(self.imgs.keys())))
        #         for ii in idx_row:
        #             zmax = max(zmax, self.imgs[ii].max())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if plot_num in idx_row:
        #                 self._add_range(ir, ic, 'z', 'max', zmax)

        # # Share col zmax
        # elif self.share_col and not self.wrap:
        #     for icv, cv in enumerate(self.col_vals):
        #         zmax = 0
        #         idx_col = self.df_fig.loc[self.df_fig[self.col[0]] == cv].index
        #         for ii in idx_col:
        #             zmax = max(zmax, self.imgs[ii].max())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if ic == icv:
        #                 self._add_range(ir, ic, 'z', 'max', zmax)

        # # Share col and wrap zmax
        # elif self.share_col and self.wrap:
        #     for ic in range(0, self.ncol):
        #         zmax = 0
        #         idx_col = list(self.imgs.keys())[ic * (self.ncol - 1)::self.ncol]
        #         for ii in idx_col:
        #             zmax = max(zmax, self.imgs[ii].max())
        #         for ir, ic, plot_num in self._get_subplot_index():
        #             if plot_num in idx_col:
        #                 self._add_range(ir, ic, 'z', 'max', zmax)

        # # subplot level when not shared
        # else:
        #     for ir, ic, plot_num in self._get_subplot_index():
        #         df_rc = self._subset(ir, ic)

        #         # Empty rc
        #         if len(df_rc) == 0:  # this doesn't exist yet!
        #             self._add_range(ir, ic, 'z', 'max', None)
        #             continue

        #         # Existing rc
        #         else:  # doesn't support the case of list of 'q's or q in % (check if data supports that)
        #             if isinstance(self.zmin[0], str) and 'q' in self.zmax[0]:
        #                 self._add_range(ir, ic, 'z', 'max', np.quantile(df_rc, float(self.zmax[0][1:])))
        #             else:
        #                 self._add_range(ir, ic, 'z', 'max', df_rc.max())

        # Matplotlib imshow extent offset fix
        if self.engine == 'mpl':
            for ir, ic, plot_num in self._get_subplot_index():
                for val in self.ranges[ir, ic]:
                    if (val[0] == 'x' or val[0] == 'y') and self.ranges[ir, ic][val] is not None:
                        self.ranges[ir, ic][val] -= 0.5

        # Update some size parameters based the new self.df_fig
        if self.channels > 1:
            self.shape = (self.df_fig.loc[idx]['rows'].max(), self.df_fig.loc[idx, 'cols'].iloc[0], self.channels)
        elif len(self._groupers) > 0:
            self.shape = (self.df_fig.loc[idx]['rows'].max(), self.df_fig.loc[idx, 'cols'].iloc[0])
        else:
            self.shape = (self.df_fig.loc[idx]['rows'].sum(), self.df_fig.loc[idx, 'cols'].iloc[0])
        width = self.ranges[ir, ic]['xmax'] - self.ranges[ir, ic]['xmin']
        height = self.ranges[ir, ic]['ymin'] - self.ranges[ir, ic]['ymax']
        if self.wh_ratio is None:
            self.wh_ratio = width / height

    def _stretch(self, kwargs):
        """Perform contrast stretching on an image

        Args:
            kwargs: user-defined keyword args
        """
        stretch = utl.kwget(kwargs, self.fcpp, 'stretch', kwargs.get('stretch', None))
        if stretch is not None:
            stretch = utl.validate_list(stretch)
            if len(stretch) == 1:
                stretch = [stretch[0], stretch[0]]
            all_data = np.concatenate(list(self.imgs.values()))
            uu = all_data.mean()
            ss = all_data.std()
            self.zmin.values[0] = uu - abs(stretch[0]) * ss
            self.zmax.values[0] = uu + stretch[1] * ss

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the actual image data based on the df_group subset

        Args:
            ir: subplot row index
            ic: subplot column index
            df: df_groups subset based on self._subset

        Returns:
            updated DataFrame with image data
        """
        if len(df.index) == 0:
            return df

        subset_dict = {key: value for key, value in self.imgs.items() if key in list(df.index)}
        return np.concatenate(list(subset_dict.values()), 1)
