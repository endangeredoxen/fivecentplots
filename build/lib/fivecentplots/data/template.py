from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class YourPlot(data.Data):
    name = ''
    req = []
    opt = []

    def __init__(self, **kwargs):

        super().__init__(self.name, self.req, self.opt, **kwargs)

        # overrides
