
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
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'misc.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data.csv'))

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


def test_text_box_single(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_single_master') if master else 'text_box_single'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
         filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
         text='Die (-1,2) shows best response', text_position=[0, 380],
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


def test_text_box_single_style(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_single_style_master') if master else 'text_box_single_style'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows\nbest response', text_position=[10, 340], text_font_size=20,
             text_edge_color='#FF0000', text_font_color='#00FF00', text_fill_color='#ffffff',
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


def test_text_box_multiple(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_multiple_master') if master else 'text_box_multiple'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text=['Die (-1,2) shows best response', '(c) 2019', 'Boom!'],
             text_position=[[10, 379], [10, 10], [320, 15]], text_font_color=['#000000', '#FF00FF'],
             text_font_size=[14, 8, 18], text_fill_color='#FFFFFF',
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


def test_text_box_position_figure(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_position_figure_master') if master else 'text_box_position_figure'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[110, 440], text_coordinate='figure',
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


def test_text_box_position_data(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_position_data_master') if master else 'text_box_position_data'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[0.903, 1.2], text_coordinate='data',
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


def test_text_box_position_units(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'text_box_position_units_master') if master else 'text_box_position_units'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[0, 0.95], text_units='relative',
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
