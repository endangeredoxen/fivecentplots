from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
utl = utilities
db = pdb.set_trace


class Contour(data.Data):
    def __init__(self, **kwargs):

        name = 'contour'
        req = ['x', 'y', 'z']
        opt = []
        kwargs['ax_limit_padding_zmax'] = 0
        kwargs['ax_limit_padding_zmin'] = 0

        super().__init__(name, req, opt, **kwargs)

        self.auto_scale = False

    def get_data_ranges(self):

        self._get_data_ranges()

    def subset_modify(self, df, ir, ic):

        return self._subset_modify(df, ir, ic)

    def subset_wrap(self, ir, ic):

        return self._subset_wrap(ir, ic)