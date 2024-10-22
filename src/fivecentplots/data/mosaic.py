from . import data, bar, box, contour, gantt, heatmap, hist, imshow, nq, pie, xy
import pdb
import numpy as np
from .. import utilities
utl = utilities
db = pdb.set_trace


class Mosaic(data.Data):
    name = 'mosaic'
    url = ''
    data_lut = {'bar': bar.Bar,
                'boxplot': box.Box,
                'contour': contour.Contour,
                'gantt': gantt.Gantt,
                'heatmap': heatmap.Heatmap,
                'imshow': imshow.ImShow,
                'nq': nq.NQ,
                'pie': pie.Pie,
                'xy': xy.XY}

    def __init__(self, **kwargs):
        """Make objects for the various plot types"""
        self.kwargs = kwargs

        # Make a data object for each plot in the mosaic plot
        self.data_objs = {}
        self.num_plots = len(utl.numeric_keys(kwargs))
        for nk in utl.numeric_keys(kwargs):
            try:
                self.data_objs[nk] = self.data_lut[kwargs[nk]['plot_func']](fcpp=kwargs['fcpp'], **kwargs[nk])
            except data.DataError as e:
                raise data.DataError(f'For \"{kwargs[nk]["plot_func"]}\" plot at mosaic plot index {nk}:\n' + \
                                     f'{" " * len("DataError: ")}{e}')


    def __getitem__(self, index):
        return self.data_objs[str(index)]

    def __len__(self):
        return len(self.data_objs)

    def __setitem__(self, index, value):
        self.data_objs[str(index)] = value

    def get_data_ranges(self):
        """Calculate data range limits for a given figure."""
        for dobj in self.data_objs:
            self.data_objs[dobj].get_data_ranges()

    def get_df_figure(self):
        """Mosaic plot does not allow fig_item grouping so override the normal generator here.

        Yields:
            figure index (None if no self.fig_vals)
            figure value (i.e., unique value in the self.fig DataFrame column)
            figure column name
            self
        """
        # TODO: error for fig_groups or fig in kwargs for this

        heights, widths = [], []
        for idx, dobj in self.data_objs.items():
            # no fig grouping
            dobj._get_legend_groupings(dobj.df_all)
            dobj._get_rc_groupings(dobj.df_all)
            dobj.df_fig = dobj.df_all

            # override ncol, nrow
            if dobj.ncols == 0:
                rcnum = int(np.ceil(np.sqrt(self.num_plots)))
            else:
                rcnum = dobj.ncols if dobj.ncols <= self.num_plots else self.num_plots
            dobj.ncol = rcnum
            dobj.nrow = int(np.ceil(self.num_plots / rcnum))
            dobj.ranges = dobj._range_dict()

        # add attributes back to self for convenience
        self.ncol = dobj.ncol
        self.nrow = dobj.nrow

        yield None, None, None, self

    def get_rc_subset(self):
        """Subset the data by the row/col/wrap values.

        Yields:
            ir: subplot row index
            ic: subplot column index
            plot_num: numeric plot number
            row/col data subset
        """
        for ir in range(0, self[0].nrow):
            for ic in range(0, self[0].ncol):
                plot_num = utl.plot_num(ir, ic, self[0].ncol) - 1

                if plot_num > len(self) - 1:
                    yield ir, ic, plot_num, []
                else:
                    dobj = self[plot_num]

                # Get the data subset
                dobj.df_rc = dobj._subset(ir, ic)

                # Handle empty dfs
                if len(dobj.df_rc) == 0:
                    dobj.df_rc = pd.DataFrame()

                # Find data ranges for this subset
                else:
                    df_rc = dobj._get_auto_scale(dobj.df_rc, plot_num)

                    for ax in dobj.axs_on:
                        if getattr(dobj, ax) is None:
                            continue
                        vals = dobj._get_data_range(ax, df_rc, plot_num)
                        if getattr(dobj, f'invert_range_limits_{ax}'):
                            dobj._add_range(ir, ic, ax, 'min', vals[1])
                            dobj._add_range(ir, ic, ax, 'max', vals[0])
                        else:
                            dobj._add_range(ir, ic, ax, 'min', vals[0])
                            dobj._add_range(ir, ic, ax, 'max', vals[1])
                    dobj._range_overrides(ir, ic, dobj.df_rc)

                # Yield the subset
                yield ir, ic, plot_num, dobj.df_rc

        self.df_sub = None
