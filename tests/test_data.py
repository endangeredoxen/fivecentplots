import pytest
import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.data.data as data
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
osjoin = os.path.join
db = pdb.set_trace


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')


def test_exceptions():
    # AxisError
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, twin_y=True)

    # GroupingError
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], groups='Marky Mark')
    with pytest.raises(data.GroupingError):
        fcp.boxplot(df_box, y='Value', groups=['Funky Bunch'])

