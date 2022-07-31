from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class YourPlot(data.Data):
    def __init__(self, **kwargs):

        name = ''
        req = []
        opt = []

        super().__init__(name, req, opt, **kwargs)

        # overrides
