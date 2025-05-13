from . import data
import pdb
import pandas as pd
import numpy as np
from natsort import natsorted
from .. import utilities
utl = utilities
db = pdb.set_trace


class ImShow(data.Data):
    name = 'imshow'
    req = []
    opt = []
    url = 'imshow.html'

    def __init__(self, fcpp: dict = {}, **kwargs):
        """ImShow-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            fcpp: theme-file kwargs
            kwargs: user-defined keyword args
        """
        # Get theme defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        # Format input image data
        #   kwargs['df']:
        #       * Single image only:
        #           - 2D numpy array of pixel data
        #           - OR a 2D pd.DataFrame of pixel data
        #       * Multiple images:
        #           - pd.DataFrame with 1 row per image
        #           - row index value must match a key in the kwargs['imgs'] dict
        #           - other columns in this DataFrame are grouping columns
        #   kwargs['imgs']:
        #       * Single image only:
        #           - Not defined or used
        #       * Multiple images:
        #           - dict of the actual image data; dict key must match a row index value in kwargs['df']
        kwargs['df'], kwargs['imgs'] = utl.img_data_format(kwargs)

        # Determine if image data is 2D grayscale or 3D/4D RGB/RGBA and define number of channels
        self.channels = kwargs['df'].iloc[0]['channels']

        # Set the image shape; all images must have same shape
        if self.channels == 1:
            self.shape = (kwargs['df'].iloc[0]['rows'], kwargs['df'].iloc[0]['cols'])
        else:
            self.shape = (kwargs['df'].iloc[0]['rows'], kwargs['df'].iloc[0]['cols'], self.channels)

        # Set a custom width to height ratio
        self.wh_ratio = utl.kwget(kwargs, self.fcpp, ['wh_ratio'], kwargs.get('wh_ratio', None))

        # Optional color plane splitting
        self.cfa = utl.kwget(kwargs, self.fcpp, 'cfa', kwargs.get('cfa', None))
        if self.cfa is not None and self.channels == 1:
            kwargs['df'], kwargs['imgs'] = utl.split_color_planes_wrapper(kwargs['df'], kwargs['imgs'], self.cfa)

        # kwargs overrrides
        kwargs['ax_limit_padding'] = kwargs.get('ax_limit_padding', 0)  # don't pad range limits by default
        self.wrap_reference = kwargs.get('wrap_reference', {})

        # Catch invalid axis options
        vals = ['twin_x', 'twin_y'] + [f for f in kwargs if f[0:2] == 'x2'] + [f for f in kwargs if f[0:2] == 'y2']
        for val in vals:
            if val in kwargs:
                raise data.AxisError(f'{val} is not a valid option for imshow plots')

        # Check for invalid grouping options
        if 'row' in kwargs and kwargs['row'] == 'y':
            raise data.GroupingError('Cannot group row by "y" for imshow plots')
        if 'col' in kwargs and kwargs['col'] == 'x':
            raise data.GroupingError('Cannot group col by "x" for imshow plots')
        if 'wrap' in kwargs and kwargs['wrap'] == 'y':
            raise data.GroupingError('Cannot wrap by "y" for imshow plots')
        if 'legend' in kwargs and kwargs['legend'] is not None:
            raise data.GroupingError('legend not available for imshow plots')
        if 'wrap_reference' in kwargs:
            error = False
            if not isinstance(kwargs['wrap_reference'], dict):
                error = True
            keys = kwargs['wrap_reference'].keys()
            if len(keys) == 0 or len(keys) > 2:
                error = True
            title_key = [f for f in keys if f != 'position'][0]
            if not isinstance(kwargs['wrap_reference'][title_key], np.ndarray):
                error = True
            if error:
                raise data.GroupingError('wrap_reference must be a dictionary with at least one key '
                                         'containing an image array')
        if 'wrap_reference' in kwargs and 'wrap' not in kwargs:
            raise data.GroupingError('wrap_reference can only be used with a wrap plot')

        # Super data.Data
        super().__init__(self.name, self.req, self.opt, self.fcpp, **kwargs)

        # overrides
        self.auto_scale = False
        self.axs_on = ['x', 'y', 'z']  # not defined by data.Data because they were not in kwargs
        self.x = ['Column']
        self.y = ['Row']
        self.z = ['Value']
        if self.channels > 1 and hasattr(self, 'cbar') and self.cbar:
            raise ValueError('Colorbar option not available for 3D image data')
            self.cbar = False
        self.invert_range_limits_y = True

        # auto stretching
        self._stretch(kwargs)

    def _check_xyz(self, xyz: str):
        """Validate the name and column data provided for x, y, and/or z.  For imshow, there are no req or opt
        values needed.

        Args:
            xyz: name of variable to check
        """
        return getattr(self, xyz)

    def _range_overrides(self, ir, ic, df_rc):
        """imshow-specific modifications."""
        # Prevent cropping beyond limits
        if self.ranges['xmin'][ir, ic] < 0:
            self.ranges['xmin'][ir, ic] = 0
        if self.ranges['xmax'][ir, ic] > df_rc.shape[1]:
            self.ranges['xmax'][ir, ic] = df_rc.shape[1]
        if self.ranges['ymax'][ir, ic] < 0:
            self.ranges['ymax'][ir, ic] = 0
        if self.ranges['ymin'][ir, ic] > df_rc.shape[0]:
            self.ranges['ymin'][ir, ic] = df_rc.shape[0]

        # Update the wh_ratio
        if not self.wh_ratio:
            if self.ranges['xmax'][ir, ic] is not None and self.ranges['xmin'][ir, ic] is not None:
                width = self.ranges['xmax'][ir, ic] - self.ranges['xmin'][ir, ic]
            else:
                width = df_rc.shape[1]
            if self.ranges['ymin'][ir, ic] is not None and self.ranges['ymax'][ir, ic] is not None:
                height = self.ranges['ymin'][ir, ic] - self.ranges['ymax'][ir, ic]
            else:
                height = df_rc.shape[0]
            self.wh_ratio = width / height

        # Matplotlib imshow extent offset fix
        if self.engine == 'mpl':
            for limit in ['xmin', 'xmax', 'ymin', 'ymax']:
                if self.ranges[limit][ir, ic] is not None:
                    self.ranges[limit][ir, ic] -= 0.5

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

    def _filter_data(self, kwargs):
        """Apply an optional filter to the data.

        Args:
            kwargs: user-defined keyword args
        """
        data.Data._filter_data(self, kwargs)

        if len(self.wrap_reference) > 0:
            # Figure out the new data
            position = self.wrap_reference.get('position', 'first')
            title = [f for f in self.wrap_reference.keys() if f != 'position'][0]
            wrap_cols = utl.validate_list(kwargs.get('wrap'))
            if self.sort:
                wrap_cols = natsorted(wrap_cols)
                self.df_all = self.df_all.sort_values(wrap_cols)
                self.sort = False

            # Add the wrap reference image
            self.imgs['wrap_reference'] = self.wrap_reference[title]
            wrap_row = pd.DataFrame(columns=self.df_all.columns, index=['wrap_reference'])
            wrap_row[wrap_cols[0]] = title
            for icol in range(1, len(wrap_cols)):
                wrap_row[wrap_cols[icol]] = 'wrap_reference_999'  # special code to drop this later
            wrap_row['rows'] = self.wrap_reference[title].shape[0]
            wrap_row['cols'] = self.wrap_reference[title].shape[1]
            if len(self.wrap_reference[title].shape) == 3:
                wrap_row['channels'] = 3
            else:
                wrap_row['channels'] = 2
            wrap_row = wrap_row.fillna('')
            if position == 'last':
                self.df_all = pd.concat([self.df_all, wrap_row])
            else:
                self.df_all = pd.concat([wrap_row, self.df_all])
