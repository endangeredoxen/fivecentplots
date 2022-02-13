
import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os
import sys
import pdb
import platform
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace
if platform.system() != 'Windows':
    print('Warning!  Image test files generated in windows.  Compatibility with linux/mac may vary')

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__),
                'tests', 'test_images', MPL,  'pie.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__),
                 'tests', 'fake_data_bar.csv'))
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

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all():
    """
    Run the show=True option on all plt functions
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


# plt_ functions can be used directly outside of pytest for debug
def plt_basic(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_master') if master else 'basic'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            startangle=90, alpha=0.85, filename=name + '.png', save=not bm, inline=False,
            jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_donut(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'donut_master') if master else 'donut'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            startangle=90, alpha=0.85, filename=name + '.png', save=not bm, inline=False,
            jitter=False,
            innerradius=0.5, pctdistance=0.75)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            startangle=90, alpha=0.85, legend=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
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
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
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
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_percents(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'percents_master') if master else 'percents'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            startangle=90, alpha=0.85, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_explode(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'explode_master') if master else 'explode'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), startangle=90, alpha=0.85, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_shadow(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'shadow_master') if master else 'shadow'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), shadow=True, startangle=90, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_angle(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'angle_master') if master else 'angle'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            explode=(0, 0.1), startangle=0, percents=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_basic(benchmark):
    plt_basic()
    benchmark(plt_basic, True)


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


def test_shadow(benchmark):
    plt_shadow()
    benchmark(plt_shadow, True)


def test_angle(benchmark):
    plt_angle()
    benchmark(plt_angle, True)


if __name__ == '__main__':
    pass
