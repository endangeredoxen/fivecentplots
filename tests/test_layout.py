import pytest
import fivecentplots as fcp
import pandas as pd
import os
import pdb
from pathlib import Path
import fivecentplots.data.data as data
import fivecentplots.engines.layout as layout
import numpy as np
osjoin = os.path.join
db = pdb.set_trace


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_interval.csv')
df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
df_gantt = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt.csv')


def test_init_intervals():
    with pytest.raises(ValueError):
        fcp.plot(df, x='x', y='y', perc_int=[0.25])


def test_element():
    ele = layout.Element()
    with pytest.raises(ValueError):
        ele.size_all = [1, 1]
    with pytest.raises(ValueError):
        ele.size_all_bg = [1, 1]