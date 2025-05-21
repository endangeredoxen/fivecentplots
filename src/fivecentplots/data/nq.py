from . import data
import pdb
from .. import utilities
import pandas as pd
import numpy as np
utl = utilities
db = pdb.set_trace


class NQ(data.Data):
    name = 'nq'
    req = []
    opt = ['x']
    url = 'nq.html'

    def __init__(self, fcpp: dict = {}, **kwargs):
        """NQ-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        **Note that the actual nq calculation is performed in Data.transform

        Args:
            kwargs: user-defined keyword args
        """
        # Set defaults
        if fcpp:
            self.fcpp = fcpp.copy()
        else:
            self.fcpp, _, _, _ = utl.reload_defaults(kwargs.get('theme', None))

        # NQ supports image data and regular non-image data
        if 'imgs' in kwargs and isinstance(kwargs['imgs'], dict) and len(kwargs['imgs']) > 0 \
                or isinstance(kwargs['df'], np.ndarray) and len(kwargs['df'].shape) > 1:
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

            # Other parameters
            self.channels = kwargs['df'].iloc[0]['channels']

            # Reformat RGBA
            if self.channels > 1:
                # Need to reformat the group and image DataFrames
                groups = self._kwargs_groupers(kwargs)
                if 'Channel' in groups:
                    imgs = {}
                    for ii, (k, v) in enumerate(kwargs['imgs'].items()):
                        # Separate the RGB columns into separate images
                        for icol, col in enumerate(['R', 'G', 'B']):
                            imgs[3 * ii + icol] = v[:, :, icol]

                # Update the grouping table and image DataFrame dict
                kwargs['imgs'] = imgs
                kwargs['df'] = pd.merge(kwargs['df'], pd.DataFrame({'Channel': ['R', 'G', 'B']}), how='cross')
                kwargs['df']['channels'] = 1
            else:
                # Stack? need some kind of luma or way to select just one color channel
                pass

        else:
            kwargs['imgs'] = None
            self.channels = -1

        super().__init__(self.name, self.req, self.opt, **kwargs)

        # NQ parameters
        self.sigma = kwargs.get('sigma', None)
        self.tail = abs(kwargs.get('tail', 3))
        self.step_tail = kwargs.get('step_tail', 0.2)
        self.step_inner = kwargs.get('step_inner', 0.5)
        self.percentiles = kwargs.get('percentiles', False)

        # Column overrides
        self.x = ['Value']
        if self.percentiles:
            self.y = ['Percent']
            # self.ax_scale = 'prob' --> TODO: figure out the scale for prob plot
        else:
            self.y = ['Sigma']
        self.axs_on = ['x', 'y']

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        if self.legend is None:
            if self.imgs is not None:
                # New image format
                subset_dict = {key: value for key, value in self.imgs.items() if key in list(df.index)}
                df_sub = np.concatenate(list(subset_dict.values()), 1)
            elif self.x[0] not in df.columns:
                # pd.DataFrame image data
                df_sub = df[utl.df_int_cols(df)]
            else:
                df_sub = df.copy()

            df_sub = utl.nq(df_sub, self.x[0], sigma=self.sigma, tail=self.tail, step_tail=self.step_tail,
                            step_inner=self.step_inner, percentiles=self.percentiles)
            return df_sub

        else:
            df_sub = []
            for iline, row in self.legend_vals.iterrows():
                if self.imgs is not None:
                    idx = list(df.loc[df[self.legend] == row['Leg']].index)
                    subset_dict = {key: value for key, value in self.imgs.items() if key in idx}
                    temp = np.concatenate(list(subset_dict.values()), 1)
                else:
                    temp = df.loc[df[self.legend] == row['Leg']]
                    del temp[self.legend]
                temp = utl.nq(temp, self.x[0], sigma=self.sigma, tail=self.tail, step_tail=self.step_tail,
                              step_inner=self.step_inner, percentiles=self.percentiles)
                temp[self.legend] = row['Leg']
                df_sub += [temp]

            return pd.concat(df_sub)
