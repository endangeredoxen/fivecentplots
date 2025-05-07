from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class Contour(data.Data):
    name = 'contour'
    req = ['x', 'y', 'z']
    opt = []
    url = 'contour.html'

    def __init__(self, **kwargs):
        """Contour-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        kwargs['ax_limit_padding_zmax'] = 0
        kwargs['ax_limit_padding_zmin'] = 0

        super().__init__(self.name, self.req, self.opt, **kwargs)

        self.auto_scale = False
