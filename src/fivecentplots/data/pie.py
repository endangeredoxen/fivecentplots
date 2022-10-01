from . import data
import pdb
import pandas as pd
from .. import utilities
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Pie(data.Data):
    def __init__(self, **kwargs):
        """Pie-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        name = 'pie'
        req = ['x', 'y']
        opt = []

        super().__init__(name, req, opt, **kwargs)

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
        for ir, ic, plot_num in self._get_subplot_index():
            self.ranges[ir, ic]['xmin'] = -1
            self.ranges[ir, ic]['xmax'] = 1
            self.ranges[ir, ic]['ymin'] = -1
            self.ranges[ir, ic]['ymax'] = 1

            self.ranges[ir, ic]['x2min'] = None
            self.ranges[ir, ic]['x2max'] = None
            self.ranges[ir, ic]['y2min'] = None
            self.ranges[ir, ic]['y2max'] = None
            self.ranges[ir, ic]['zmin'] = None
            self.ranges[ir, ic]['zmax'] = None

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
