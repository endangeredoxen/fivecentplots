import pytest
import fivecentplots as fcp
import pandas as pd
import os
import pdb
from pathlib import Path
import fivecentplots.data.data as data
import numpy as np
osjoin = os.path.join
db = pdb.set_trace


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
df_gantt = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt.csv')


def test_AxisError():
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, twin_y=True)
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage', y='I [A]', twin_y=True)
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]', 'Boost Level'], twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.plot(df, y='Voltage', x=['Voltage', 'I [A]', 'Boost Level'], twin_y=True)
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_y=True, wrap='Boost Level')
    with pytest.raises(data.AxisError):
        fcp.plot(df, x='Voltage')
    with pytest.raises(data.AxisError):
        fcp.plot(df, y=['Voltage', 'I [A]'], x=['Voltage', 'I [A]'], twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.plot(df, y=['Voltage', 'I [A]'], x=['Voltage', 'I [A]'], twin_y=True)


def test_DataError():
    with pytest.raises(data.DataError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], stat_val='Marky Mark')
    with pytest.raises(data.DataError):
        fcp.plot('hi', x='Voltage', y=['Voltage', 'I [A]'])
    with pytest.raises(data.DataError):
        fcp.plot(pd.DataFrame(columns=['Voltage', 'I [A]']), x='Voltage', y=['Voltage', 'I [A]'])
    with pytest.raises(data.DataError):
        fcp.plot(df, x='Voltage', y='Boom')
    with pytest.raises(data.DataError):
        fcp.plot(df, x='Voltage', y='I [A]', filter='Temperature [C] == 99')
    with pytest.raises(data.DataError):
        fcp.gantt(df_gantt, x='Start', y='Task')
    with pytest.raises(data.DataError):
        df_gantt['Start2'] = 'hi'
        fcp.gantt(df_gantt, x=['Start2', 'Stop'], y='Task')
    with pytest.raises(data.DataError):
        df_gantt['Stop2'] = 'hi'
        fcp.gantt(df_gantt, x=['Start', 'Stop2'], y='Task')


def test_GroupingError():
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], groups='Marky Mark')
    with pytest.raises(data.GroupingError):
        fcp.boxplot(df_box, y='Value', groups=['Funky Bunch'])
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, wrap='Boost Level')
    with pytest.raises(data.GroupingError):
        fcp.plot(df, y='Voltage', x=['Voltage', 'I [A]'], twin_y=True, wrap='Boost Level')
    with pytest.raises(data.GroupingError):
        df['hi'] = np.nan
        fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], wrap='hi')
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y='I [A]', row=['Boost Level', 'Temperature [C]'])
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y='I [A]', row='Boost Level', col='Boost Level')
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y='I [A]', wrap='Boost Level', col='Temperature [C]')
    with pytest.raises(data.GroupingError):
        fcp.plot(df, x='Voltage', y='I [A]', legend='Boost Level', fig='Boost Level')
    with pytest.raises(data.GroupingError):
        fcp.boxplot(df_box, y='Value', groups='Sample', fig='Sample')
    with pytest.raises(data.GroupingError):
        fcp.boxplot(df_box, y='Value', groups=['Sample', 'Batch', 'Region'], wrap='Region')
    with pytest.raises(data.GroupingError):
        fcp.boxplot(df_box, y='Value', groups=['Sample', 'Batch', 'Region'], col='Region')
