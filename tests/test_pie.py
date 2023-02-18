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

test = 'pie'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
df.loc[df.pH < 0, 'pH'] = -df.pH

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')


# Other
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def make_all():
    """
    Remake all test master images
    """

    if not MASTER.exists():
        os.makedirs(MASTER)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all(only_fails=True):
    """
    Run the show=True option on all plt functions
    """

    if not MASTER.exists():
        os.makedirs(MASTER)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        if only_fails:
            try:
                member[1]()
            except AssertionError:
                member[1](show=True)
                db()
        else:
            member[1](show=True)
            db()


# plt_ functions can be used directly outside of pytest for debug
def plt_basic(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_master') if master else 'basic'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name + '.png', save=not bm, inline=False,
            jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_basic_no_sort(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_no_sort_master') if master else 'basic_no_sort'

    # Make the plot
    df_ = df.copy()
    df_.loc[df_.Liquid == 'Orange juice', 'pH'] *= -1
    fcp.pie(df_, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name + '.png', save=not bm, inline=False,
            jitter=False, sort=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_donut(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'donut_master') if master else 'donut'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, filename=name + '.png', save=not bm, inline=False,
            jitter=False,
            inner_radius=0.5, percents_distance=0.75)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', start_angle=90, alpha=0.85,
            legend=True, filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend_unsort(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_unsort_master') if master else 'legend_unsort'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', start_angle=90, alpha=0.85,
            legend=True, filename=name + '.png', save=not bm, inline=False, jitter=False, sort=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend_rc(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_rc_master') if master else 'legend_rc'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, col='Measurement',
            row='T [C]', legend=True, ax_size=[250, 250],
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_wrap_master') if master else 'legend_wrap'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, wrap='Measurement',
            legend=True, ax_size=[250, 250],
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_percents(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'percents_master') if master else 'percents'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            start_angle=90, alpha=0.85, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_explode(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'explode_master') if master else 'explode'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', explode=(0, 0.1), start_angle=90,
            alpha=0.85, percents=True, filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_explode_all(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'explode_all_master') if master else 'explode_all'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25', explode=('all', 0.1),
            start_angle=90, alpha=0.85, percents=True, filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_shadow(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'shadow_master') if master else 'shadow'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), shadow=True, start_angle=90, percents=False,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_angle(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'angle_master') if master else 'angle'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), start_angle=0, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


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
