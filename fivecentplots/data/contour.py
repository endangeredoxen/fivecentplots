from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class Contour(data.Data):
    def __init__(self, **kwargs):
        """Contour-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        name = 'contour'
        req = ['x', 'y', 'z']
        opt = []
        kwargs['ax_limit_padding_zmax'] = 0
        kwargs['ax_limit_padding_zmin'] = 0

        super().__init__(name, req, opt, **kwargs)

        self.auto_scale = False
