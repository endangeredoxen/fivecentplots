from . import data
import pdb
from .. import utilities
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
        try:
            kwargs['df'], self.shape = utl.img_df_from_array_or_df(kwargs['df'])
        except ValueError:
            pass

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
