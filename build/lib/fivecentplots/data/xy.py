from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class XY(data.Data):
    name = 'xy'
    url = 'plot.html'

    def __init__(self, **kwargs):
        """XY plot-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        super().__init__(self.name, **kwargs)
