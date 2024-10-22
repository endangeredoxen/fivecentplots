from . import data
import pandas as pd
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

        self.hist = utl.kwget(kwargs, self.fcpp, ['hist', 'hist_on'], False)

    def _get_rc_groupings(self, df: pd.DataFrame):
        """Determine the row and column or wrap grid groupings.

        Args:
            df: data subset
        """
        data.Data._get_rc_groupings(self, df)

        if self.hist:
            self.ncol *= 2
            self.nrow *= 2
            self.ranges = self._range_dict()

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Optional function in a Data childe class user to perform any additional
        DataFrame subsetting that may be required

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        if self.hist:
            # Disable bottom right corner subplot
            if ir % 2 == 1 and ic % 2 == 1:
                return pd.DataFrame()
            # Calculate the histogram data
            elif ir % 2 == 1:
                db()
            elif ic % 2 == 1:
                db()
            else:
                return df
        else:
            return df