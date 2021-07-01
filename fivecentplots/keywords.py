### Keyword docstrings
import pandas as pd
import os
import pdb
db = pdb.set_trace
osjoin = os.path.join
cur_dir = os.path.dirname(__file__)
from distutils.version import LooseVersion
#PATH = osjoin(cur_dir, r'doc\_static\docstrings')


def make_docstrings():

    if LooseVersion(pd.__version__) >= LooseVersion('1.2'):
        kw = pd.read_excel(osjoin(cur_dir, 'keywords.xlsx'), engine='openpyxl', sheet_name=None)
    else:
        kw = pd.read_excel(osjoin(cur_dir, 'keywords.xlsx'), sheet_name=None)

    for k, v in kw.items():
        kw[k] = kw[k].replace('`', '', regex=True)
        kw[k]['Keyword'] = kw[k]['Keyword'].apply(lambda x: str(x).split(':')[-1])
        if 'Example' in kw[k].columns:
            kw[k]['Example'] = kw[k]['Example'].apply(lambda x: 'see online docs'
                                                    if '.html' in str(x) else x)
        else:
            kw[k]['Example'] = 'None'
        nans = kw[k][kw[k]['Keyword']=='nan']
        if len(nans) > 0:
            kw[k] = kw[k].dropna()
            for irow, row in nans.iterrows():
                row = row.dropna()
                idx = kw[k].index[kw[k].index < irow][-1]
                for col in row.index:
                    if col != 'Keyword':
                        kw[k].loc[idx, col] += ' | ' + row[col]

    return kw

