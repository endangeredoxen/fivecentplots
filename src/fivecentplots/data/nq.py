from . import data
import pdb
import pandas as pd
from .. import utilities
utl = utilities
db = pdb.set_trace


def local_groupers(kwargs):
    """Get all the valid grouping columns.

    Args:
        kwargs: user-defined keyword args

    Returns:
        list of valid groups
    """
    props = ['row', 'col', 'wrap', 'groups', 'legend', 'fig']
    grouper = []

    for prop in props:
        if prop in kwargs.keys() and kwargs[prop] not in ['x', 'y', None]:
            grouper += utl.validate_list(kwargs[prop])

    return grouper


def reshape_2D(kwargs):
    """
    Reshape 2D image data to be suitable for certain non-imshow plot
    types.

    Args:
        kwargs: user-defined keyword args

    Returns:
        updated kwargs

    """

    kwargs['x'] = ['Value']
    lg = local_groupers(kwargs)
    if len(lg) > 0:
        kwargs['df'] = kwargs['df'].set_index(lg)
        kwargs['df'] = pd.DataFrame(kwargs['df'].stack())
        kwargs['df'].columns = kwargs['x']
        kwargs['df'] = kwargs['df'].reset_index()
    else:
        kwargs['df'] = kwargs['df'][utl.df_int_cols(kwargs['df'])]
        kwargs['df'] = pd.DataFrame(kwargs['df'].stack())
        kwargs['df'].columns = kwargs['x']

    return kwargs


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
        kwargs['df'] = utl.df_from_array2d(kwargs['df'])

        if not kwargs.get('x', None):
            kwargs = reshape_2D(kwargs)
        kwargs['trans_x'] = 'nq'

        super().__init__(name, req, opt, **kwargs)

        # overrides
        self.y = ['Sigma']
