from . import data
import pdb
from .. import utilities
utl = utilities
db = pdb.set_trace


class XY(data.Data):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        # overrides

    def get_data_ranges(self):

        self._get_data_ranges()

    def subset_modify(self, df, ir, ic):

        return self._subset_modify(df, ir, ic)

    def subset_wrap(self, ir, ic):

        return self._subset_wrap(ir, ic)
