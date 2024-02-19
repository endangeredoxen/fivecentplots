from . import data
import pdb
from .. import utilities
import pandas as pd
utl = utilities
db = pdb.set_trace


class NQ(data.Data):
    def __init__(self, **kwargs):
        """NQ-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        **Note that the actual nq calculation is performed in Data.transform

        Args:
            kwargs: user-defined keyword args
        """
        name = 'nq'
        req = []
        opt = ['x']

        # Check if input is image data
        try:
            # For image data, grouping information is stored in kwargs['df'] but the actual image arrays are in
            # the self.imgs dict
            kwargs['df'], kwargs['imgs'] = utl.img_df_transform(kwargs['df'])
            self.channels = kwargs['df'].loc[0, 'channels']

        except TypeError:
            # This might be a problem if the intent is passing image data but it is malformatted
            kwargs['imgs'] = None
            self.channels = -1

        # Reformat RGBA
        if kwargs['imgs'] is not None and self.channels > 1:
            # Need to reformat the group and image DataFrames
            imgs = {}
            for ii, (k, v) in enumerate(kwargs['imgs'].items()):
                # Separate the RGB columns into separate images
                for icol, col in enumerate(['R', 'G', 'B']):
                    imgs[3 * ii + icol] = v[['Row', 'Column', col]]
                    imgs[3 * ii + icol].columns = ['Row', 'Column', 'Value']

            # Update the grouping table and image dataframe dict
            kwargs['df'] = pd.merge(kwargs['df'], pd.DataFrame({'Channel': ['R', 'G', 'B']}), how='cross')
            kwargs['imgs'] = imgs

        # Image data x-column will always be reformatted to "Value"
        if kwargs['imgs'] is not None:
            kwargs['x'] = ['Value']

        kwargs['trans_x'] = 'nq'

        super().__init__(name, req, opt, **kwargs)

        # overrides
        if kwargs.get('percentiles'):
            self.y = ['Percent']
            # self.ax_scale = 'prob' --> TODO: figure out the scale for prob plot
        else:
            self.y = ['Sigma']

        # Update valid axes
        self.axs = [f for f in ['x', 'x2', 'y', 'y2', 'z'] if getattr(self, f) not in [None, []]]
