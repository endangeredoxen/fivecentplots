import pytest
import fivecentplots as fcp
import pandas as pd
import os
import pdb
from pathlib import Path
osjoin = os.path.join
db = pdb.set_trace


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_interval.csv')


def test_deprecated():
    kwargs = {'leg_groups': 'alive', 'ylabel': 'jeremey', 'sharex': 'nothing_man'}
    kwargs = fcp.deprecated(kwargs)
    assert list(sorted(kwargs.keys())) == ['label_y', 'legend', 'share_x']


def test_paste_kwargs():
    kwargs = {'leg_groups': 'alive', 'ylabel': 'jeremey', 'sharex': 'nothing_man'}
    fcp.paste_kwargs(kwargs)


def test_bad_engine():
    with pytest.raises(fcp.EngineError):
        fcp.plot(pd.DataFrame(), engine='v8', inline=False, save=False, show=False)


def test_filepaths():
    filepath = fcp.plot(df, x='x', y='y', return_filename=True, inline=False, save=True, show=False)
    vals = filepath.split(os.sep)
    assert vals[-2] == Path.cwd().name
    assert vals[-1] == 'y vs x.png'
    os.remove('y vs x.png')


def test_save_data():
    fcp.plot(df, x='x', y='y', filter='x < 0', save_data='custom.csv', inline=False, save=False, show=False)
    output = pd.read_csv('custom.csv')
    assert len(output) == 174
    os.remove('custom.csv')

    fcp.plot(df, x='x', y='y', filter='x < 0', save_data=True, inline=False, save=False, show=False)
    output = pd.read_csv('y vs x.csv')
    assert len(output) == 174
    os.remove('y vs x.csv')


def test_debug(capsys):
    fcp.plot(df, x='x', y='y', inline=False, save=False, show=False, debug_size=True)
    out, err = capsys.readouterr()
    raw_vals = out.replace('\n   ', '\n').split('\n')
    vals = {}
    for rv in raw_vals:
        kv = rv.split(' = ')
        if len(kv) == 2:
            vals[kv[0]] = int(float(kv[1].replace('\n', '')))
    assert vals['self.fig.size[0]'] == 493
    assert vals['_labtick_x'] == 51
