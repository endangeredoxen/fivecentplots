
import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb, platform
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace
if platform.system() != 'Windows':
    raise utl.PlatformError()

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL,  'barplot.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_bar.csv'))
df.loc[df.pH < 0, 'pH'] = -df.pH

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')


# Other
SHOW = False


def make_all():
    """
    Remake all test master images
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all():
    """
    Remake all test master images
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


def test_basic(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_master') if master else 'basic'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            startangle=90, alpha=0.85, filename=name + '.png', 
            inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_donut(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'donut_master') if master else 'donut'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            startangle=90, alpha=0.85, filename=name + '.png', 
            inline=False, jitter=False,
            innerradius=0.5, pctdistance=0.75)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            startangle=90, alpha=0.85, legend=True,
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_legend_rc(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_rc_master') if master else 'legend_rc'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, col='Measurement', 
            row='T [C]', legend=True, ax_size=[250, 250],
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_legend_wrap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_wrap_master') if master else 'legend_wrap'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, wrap='Measurement', 
            legend=True, ax_size=[250, 250],
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_percents(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'percents_master') if master else 'percents'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            startangle=90, alpha=0.85, percents=True,
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_explode(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'explode_master') if master else 'explode'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            explode=(0,0.1), startangle=90, alpha=0.85, percents=True,
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_shadow(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'shadow_master') if master else 'shadow'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            explode=(0,0.1), shadow=True, startangle=90, percents=True,
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_angle(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'angle_master') if master else 'angle'

    # Make the plot
    fcp.pie(df, x='Liquid', y='pH', show=SHOW, 
            filter='Measurement=="A" & T [C]==25', 
            explode=(0,0.1), startangle=0, percents=True,
            filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare

