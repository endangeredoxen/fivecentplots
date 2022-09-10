import pytest
import fivecentplots as fcp
import pandas as pd
import os
import pdb
from pathlib import Path
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

    ele = layout.Element(df=pd.DataFrame, size=None)
    assert 'df' not in ele.kwargs.keys()
    ele.size = [3, 4]
    assert ele._size_orig == [3, 4]
    assert ele.size_inches == [0.03, 0.04]
    ele.on = False
    assert ele.size_inches == [0, 0]


def test_legend_element():
    leg = layout.Legend_Element()
    expected = pd.DataFrame(columns=['Key', 'Curve', 'LineType'])
    pd.testing.assert_frame_equal(leg._values, expected)

    leg = layout.Legend_Element(legend=True, sort=True)
    leg.values = pd.DataFrame({'Key': 'hi', 'Curve': 'bye', 'LineType': 'fire'}, index=[0])
    expected = pd.DataFrame({'Key': 'hi', 'Curve': 'bye', 'LineType': 'fire'}, index=[0])
    pd.testing.assert_frame_equal(leg.values, expected)


def test_object_array():
    obj = layout.ObjectArray()
    np.testing.assert_array_equal(obj._obj, np.array([]))

    obj.obj = np.ones(5)
    np.testing.assert_array_equal(obj.obj, np.ones(5))

    obj.obj = np.array([3, 2, 1])
    assert obj[6] == 2.0
    assert len(obj) == 8

    obj.reshape(4, 2)
    assert obj.obj[2, 1] == 3.0
