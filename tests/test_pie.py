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
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'pie'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
df.loc[df.pH < 0, 'pH'] = -df.pH

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')


# Other
def make_all(start=None, stop=None):
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__], start=start, stop=stop)


def show_all(only_fails=True, start=None):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__], start=start)


SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


# plt_ functions can be used directly outside of pytest for debug
def plt_angle(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('angle', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), start_angle=0, percents=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name.with_suffix('.png'), save=not bm, inline=False,
            jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_basic_no_sort(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic_no_sort', make_reference, REFERENCE)

    # Make the plot
    df_ = df.copy()
    df_.loc[df_.Liquid == 'Orange juice', 'pH'] *= -1
    fcp.pie(df_, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name.with_suffix('.png'), save=not bm, inline=False,
            jitter=False, sort=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_donut(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('donut', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name.with_suffix('.png'), save=not bm, inline=False,
            jitter=False,
            inner_radius=0.5, percents_distance=0.75)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_explode(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('explode', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', explode=(0, 0.1), start_angle=90,
            alpha=0.85, percents=True, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_explode_all(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('explode_all', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', explode=('all', 0.1),
            start_angle=90, alpha=0.85, percents=True, filename=name.with_suffix('.png'), save=not bm, inline=False,
            jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', start_angle=90, alpha=0.85,
            legend=True, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend_unsort(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend_unsort', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', start_angle=90, alpha=0.85,
            legend=True, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, sort=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend_rc(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend_rc', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, col='Measurement',
            row='T [C]', legend=True, ax_size=[250, 250],
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend_wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, wrap='Measurement',
            legend=True, ax_size=[250, 250],
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_percents(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('percents', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, percents=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_shadow(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shadow', make_reference, REFERENCE)

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), shadow=True, start_angle=90, percents=False,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_basic(benchmark):
    plt_basic()
    benchmark(plt_basic, True)


def test_basic_no_sort(benchmark):
    plt_basic_no_sort()
    benchmark(plt_basic_no_sort, True)


def test_donut(benchmark):
    plt_donut()
    benchmark(plt_donut, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


def test_legend_rc(benchmark):
    plt_legend_rc()
    benchmark(plt_legend_rc, True)


def test_legend_wrap(benchmark):
    plt_legend_wrap()
    benchmark(plt_legend_wrap, True)


def test_percents(benchmark):
    plt_percents()
    benchmark(plt_percents, True)


def test_explode(benchmark):
    plt_explode()
    benchmark(plt_explode, True)


def test_explode_all(benchmark):
    plt_explode_all()
    benchmark(plt_explode_all, True)


def test_shadow(benchmark):
    plt_shadow()
    benchmark(plt_shadow, True)


def test_angle(benchmark):
    plt_angle()
    benchmark(plt_angle, True)


def test_invalid():
    with pytest.raises(data.AxisError):
        fcp.pie(df, x='Liquid', y=['pH', 'Measurement'], twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.pie(df, y='Liquid', x=['pH', 'Measurement'], twin_y=True)
    with pytest.raises(data.GroupingError):
        fcp.pie(df, y='Liquid', x='pH', row='y')
    with pytest.raises(data.GroupingError):
        fcp.pie(df, y='Liquid', x='pH', wrap='y')
    with pytest.raises(data.GroupingError):
        fcp.pie(df, y='Liquid', x='pH', col='x')
    with pytest.raises(data.GroupingError):
        fcp.pie(df, y='Liquid', x='pH', legend='Measurement')


if __name__ == '__main__':
    pass
