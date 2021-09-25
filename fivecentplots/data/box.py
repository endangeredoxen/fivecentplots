from . import data
import pdb
import pandas as pd
import numpy as np
from .. import utilities
import scipy.stats as ss
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Box(data.Data):
    def __init__(self, **kwargs):

        name = 'box'
        req = ['y']
        opt = []

        super().__init__(name, req, opt, **kwargs)

    def get_box_index_changes(self):
        """
        Make a DataFrame that shows when groups vals change; used for grouping labels

        Args:
            df (pd.DataFrame): grouping values
            num_groups (int): number of unique groups

        Returns:
            new DataFrame with 1's showing where group levels change for each row of df
        """

        # Check for nan columns
        if self.groups is not None:
            for group in self.groups:
                if len(self.df_rc[group].dropna()) == 0:
                    self.groups.remove(group)
                    print('Column "%s" for a subplot is all NaN and will be excluded from plot' % group)

        # Get the changes df
        if self.groups is None or self.groups == []:
            groups = [(None, self.df_rc.copy())]
            self.ngroups = 0
        else:
            groups = self.df_rc.groupby(self.groups, sort=self.sort)
            self.ngroups = groups.ngroups

        # Order the group labels with natsorting
        gidx = []
        for i, (nn, g) in enumerate(groups):
            gidx += [nn]
        if self.sort:
            gidx = natsorted(gidx)
        self.indices = pd.DataFrame(gidx)
        self.changes = self.indices.copy()

        # Set initial level to 1
        for col in self.indices.columns:
            self.changes.loc[0, col] = 1

        # Determines values for all other rows
        for i in range(1, self.ngroups):
            for col in self.indices.columns:
                if self.indices[col].iloc[i-1] == self.indices[col].iloc[i]:
                    self.changes.loc[i, col] = 0
                else:
                    self.changes.loc[i, col] = 1

        return True

    def get_data_ranges(self):

        self._get_data_ranges()

    def get_rc_subset(self):
        """
        Subset the data by the row/col/wrap values

        Args:
            df (pd.DataFrame): main DataFrame

        Returns:
            subset DataFrame
        """

        for ir in range(0, self.nrow):
            for ic in range(0, self.ncol):
                self.df_rc = self.subset(ir, ic)

                # Plot specific subsetting
                self.subset_modify(self.df_rc, ir, ic)

                # Deal with empty dfs
                if len(self.df_rc) == 0:
                    self.df_rc = pd.DataFrame()

                # Get boxplot changes DataFrame
                if 'box' in self.name and len(self.df_rc) > 0:  # think we are doing this twice
                    if (self.groups is not None and self.groups != []) and \
                            len(self.df_rc.groupby(self.groups)) == 0:
                        continue
                    self.get_box_index_changes()
                    self.ranges[ir, ic]['xmin'] = 0.5
                    self.ranges[ir, ic]['xmax'] = len(self.changes) + 0.5

                # Yield the subset
                yield ir, ic, self.df_rc

        self.df_sub = None

    def subset_modify(self, df, ir, ic):

        return self._subset_modify(df, ir, ic)

    def subset_wrap(self, ir, ic):

        return self._subset_wrap(ir, ic)