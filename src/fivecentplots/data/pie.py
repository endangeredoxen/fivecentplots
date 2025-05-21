from . import data
import pdb
import pandas as pd
from .. import utilities
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Pie(data.Data):
    name = 'pie'
    req = ['x', 'y']
    opt = []
    url = 'pie.html'

    def __init__(self, **kwargs):
        """Pie-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        super().__init__(self.name, self.req, self.opt, **kwargs)

        # check for invalid axis options
        vals = ['twin_x', 'twin_y']
        for val in vals:
            if getattr(self, val):
                raise data.AxisError(f'{val} is not a valid option for pie charts')

        # check for invalid grouping options
        if self.row == 'y':
            raise data.GroupingError('Cannot group row by "y" for pie charts')
        if self.col == 'x':
            raise data.GroupingError('Cannot group col by "x" for pie charts')
        if self.wrap == 'y':
            raise data.GroupingError('Cannot wrap by "y" for pie charts')
        if self.legend not in [True, None]:
            raise data.GroupingError('legend can only equal True, False, or None for pie charts')

    def _get_data_ranges(self):
        """Pie-specific data range calculator by subplot."""
        for ir, ic, plot_num in self.get_subplot_index():
            self.ranges['xmin'][ir, ic] = -1
            self.ranges['xmax'][ir, ic] = 1
            self.ranges['ymin'][ir, ic] = -1
            self.ranges['ymax'][ir, ic] = 1

            self.ranges['x2min'][ir, ic] = None
            self.ranges['x2max'][ir, ic] = None
            self.ranges['y2min'][ir, ic] = None
            self.ranges['y2max'][ir, ic] = None
            self.ranges['zmin'][ir, ic] = None
            self.ranges['zmax'][ir, ic] = None

    def _get_legend_groupings(self, df: pd.DataFrame):
        """Determine the legend groupings.

        Args:
            df: data subset
        """
        if not self.legend:
            return

        leg_all = []

        # custom for pie plot
        self.legend = self.x[0]
        if self.sort:
            legend_vals = natsorted(list(df.groupby(self.legend).groups.keys()))
        else:
            legend_vals = list(df.groupby(self.legend, sort=False).groups.keys())
        self.nleg_vals = len(legend_vals)

        for leg in legend_vals:
            leg_all += [(leg, self.x[0], self.y[0])]
        leg_df = pd.DataFrame(leg_all, columns=['Leg', 'x', 'y'])

        # if leg specified
        if not (leg_df.Leg.isnull()).all():
            leg_df['names'] = list(leg_df.Leg)

        leg_df = leg_df.set_index('names')
        self.legend_vals = leg_df.reset_index()
