import imageio
import fivecentplots as fcp
import numpy as np
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
import pytest
osjoin = os.path.join
db = pdb.set_trace


@pytest.fixture
def df():
    return pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')


def test_ci(df):
    np.testing.assert_almost_equal(utl.ci(df['I [A]']), (0.1422958885037856, 0.18313061803216213))
    assert np.isnan(utl.ci(pd.Series())[0])


def test_dfkwarg(df):
    kwargs = utl.dfkwarg(df, {})
    assert 'df' in kwargs

    kwargs = utl.dfkwarg(np.zeros((5, 5)), {})
    assert 'df' in kwargs

    kwargs = utl.dfkwarg(1, {})
    assert kwargs['df'] is None


def test_df_filter(df):
    dff = utl.df_filter(df, 'Substrate=="Si" & Target Wavelength==460 & Boost Level==0.2')
    assert len(dff) == 102
