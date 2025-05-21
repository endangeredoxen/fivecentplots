from . import data
import pdb
import pandas as pd
from .. import utilities
from natsort import natsorted
utl = utilities
db = pdb.set_trace


class Box(data.Data):
    name = 'box'
    req = ['y']
    opt = []
    url = 'boxplot.html'

    def __init__(self, **kwargs):
        """Boxplot-specific Data class to deal with operations applied to the
        input data (i.e., non-plotting operations)

        Args:
            kwargs: user-defined keyword args
        """
        super().__init__(self.name, self.req, self.opt, **kwargs)

        self.axs_on = ['x', 'y']  # x still exists but is covered up by label boxes

    def _get_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return the groupby keys of a DataFrame

        Args:
            df: input DataFrame

        Returns:
            DataFrame of the groupby keys
        """
        return pd.DataFrame(df.groupby(self.groups).groups.keys())

    def get_box_index_changes(self):
        """
        Make a DataFrame that shows when groups vals change; used for grouping labels

        Args:
            df (pd.DataFrame): grouping values
            num_groups (int): number of unique groups
        """
        if len(self.df_rc) == 0:
            return

        # Check for nan columns
        if self.groups is not None:
            for group in self.groups:
                if len(self.df_rc[group].dropna()) == 0:
                    self.groups.remove(group)
                    print(f'Column "{group}" for at least one subplot is all NaN and will be excluded from plot')

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
                if self.indices[col].iloc[i - 1] == self.indices[col].iloc[i]:
                    self.changes.loc[i, col] = 0
                else:
                    self.changes.loc[i, col] = 1

    def _range_overrides(self, ir: int, ic: int, df_rc: pd.DataFrame):
        """Modify boxplot ranges.

        Args:
            ir: subplot row index
            ic: subplot column index
            df_rc: data subset
        """
        if (self.groups is not None and self.groups != []) and \
                self.df_rc.groupby(self.groups, dropna=False).ngroups == 0:
            return
        self.get_box_index_changes()
        self.ranges['xmin'][ir, ic] = 0.5
        self.ranges['xmax'][ir, ic] = len(self.changes) + 0.5

    def _subset_modify(self, ir: int, ic: int, df: pd.DataFrame) -> pd.DataFrame:
        """Modify the subset to deal with share x axis range.

        Args:
            ir: subplot row index
            ic: subplot column index
            df: data subset

        Returns:
            modified DataFrame subset
        """
        # Don't do anything for axes sharing if just one subplot
        if self.nrow == 1 and self.ncol == 1:
            return df

        if self.share_x and self.groups is not None and len(df) > 0:
            df1 = self._get_groups(self.df_all)
            df2 = self._get_groups(df)
            dfm = pd.merge(df1, df2, how='outer',
                           suffixes=('', '_y'), indicator=True)
            missing = dfm[dfm['_merge'] == 'left_only'][dfm.columns[0:-1]]
            missing.columns = self.groups
            df = pd.concat([df, missing])

        return df
