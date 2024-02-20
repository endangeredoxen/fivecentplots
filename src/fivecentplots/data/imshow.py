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

        # For image data, grouping information is stored in kwargs['df'] but the actual image arrays are in
        # the self.imgs dict
        kwargs['df'], kwargs['imgs'] = utl.img_df_transform(kwargs['df'])

        self.channels = kwargs['df'].loc[0, 'channels']
        if self.channels == 1:
            self.shape = (kwargs['df'].loc[0, 'rows'], kwargs['df'].loc[0, 'cols'])
        else:
            self.shape = (kwargs['df'].loc[0, 'rows'], kwargs['df'].loc[0, 'cols'], self.channels)

        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', None)

        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

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
        self.z = [f for f in self.imgs[0].columns if f not in ['Row', 'Column']]
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

    def _get_data_ranges(self):
        """ImShow-specific data range calculator by subplot."""
        # Get user ranges
        self._get_data_ranges_user_defined()

        # Get the image indices associated with the figure
        idx = list(self.df_fig.index)

        # Apply shared axes
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

        # z-axis limits
        # Share z
        if getattr(self, 'share_z'):
            zmin = 1E20
            zmax = 0
            for ii in idx:
                zmin = min(zmin, self.imgs[ii][self.z].min().min())
                zmax = max(zmax, self.imgs[ii][self.z].max().max())
            for ir, ic, plot_num in self._get_subplot_index():
                self._add_range(ir, ic, 'z', 'min', zmin)
                self._add_range(ir, ic, 'z', 'max', zmax)

        # Share row
        elif self.share_row and not self.wrap:
            for irv, rv in enumerate(self.row_vals):
                zmin = 1E20
                zmax = 0
                idx_row = self.df_fig.loc[self.df_fig[self.row[0]] == rv].index
                for ii in idx_row:
                    zmin = min(zmin, self.imgs[ii][self.z].min().min())
                    zmax = max(zmax, self.imgs[ii][self.z].max().max())
                for ir, ic, plot_num in self._get_subplot_index():
                    if ir == irv:
                        self._add_range(ir, ic, 'z', 'min', zmin)
                        self._add_range(ir, ic, 'z', 'max', zmax)

        elif self.share_row and self.wrap:
            for ir in range(0, self.nrow):
                zmin = 1E20
                zmax = 0
                idx_row = range(0 + ir * self.ncol, min(self.ncol + ir * self.ncol, len(self.imgs.keys())))
                for ii in idx_row:
                    zmin = min(zmin, self.imgs[ii][self.z].min().min())
                    zmax = max(zmax, self.imgs[ii][self.z].max().max())
                for ir, ic, plot_num in self._get_subplot_index():
                    if plot_num in idx_row:
                        self._add_range(ir, ic, 'z', 'min', zmin)
                        self._add_range(ir, ic, 'z', 'max', zmax)

        # Share col
        elif self.share_col and not self.wrap:
            for icv, cv in enumerate(self.col_vals):
                zmin = 1E20
                zmax = 0
                idx_col = self.df_fig.loc[self.df_fig[self.col[0]] == cv].index
                for ii in idx_col:
                    zmin = min(zmin, self.imgs[ii][self.z].min().min())
                    zmax = max(zmax, self.imgs[ii][self.z].max().max())
                for ir, ic, plot_num in self._get_subplot_index():
                    if ic == icv:
                        self._add_range(ir, ic, 'z', 'min', zmin)
                        self._add_range(ir, ic, 'z', 'max', zmax)

        elif self.share_col and self.wrap:
            for ic in range(0, self.ncol):
                zmin = 1E20
                zmax = 0
                idx_col = list(self.imgs.keys())[ic * (self.ncol - 1)::self.ncol]
                for ii in idx_col:
                    zmin = min(zmin, self.imgs[ii][self.z].min().min())
                    zmax = max(zmax, self.imgs[ii][self.z].max().max())
                for ir, ic, plot_num in self._get_subplot_index():
                    if plot_num in idx_col:
                        self._add_range(ir, ic, 'z', 'min', zmin)
                        self._add_range(ir, ic, 'z', 'max', zmax)

        # subplot level when not shared
        else:
            for ir, ic, plot_num in self._get_subplot_index():
                df_rc = self._subset(ir, ic)

                # Empty rc
                if len(df_rc) == 0:  # this doesn't exist yet!
                    self._add_range(ir, ic, 'z', 'min', None)
                    self._add_range(ir, ic, 'z', 'max', None)
                    continue

                # Existing rc
                else:
                    self._add_range(ir, ic, 'z', 'min', df_rc[self.z].min().min())
                    self._add_range(ir, ic, 'z', 'max', df_rc[self.z].max().max())

        # Matplotlib imshow extent offset fix
        if self.engine == 'mpl':
            for ir, ic, plot_num in self._get_subplot_index():
                for val in self.ranges[ir, ic]:
                    if ('x' in val or 'y' in val) and self.ranges[ir, ic][val] is not None:
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
        self.wh_ratio = max(self.shape[1] / self.shape[0], width / height)

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
            vals = []
            for k, v in self.imgs.items():
                vals += [v[self.z].values.T]
            vals = np.concatenate(vals)
            uu = vals.mean()
            ss = vals.std()
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

        return pd.concat([self.imgs[x] for x in list(df.index)])
