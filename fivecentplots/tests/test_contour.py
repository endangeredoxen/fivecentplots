
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
    print('Warning!  Image test files generated in windows.  Compatibility with linux/mac may vary')

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'contour.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_contour.csv'))

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')

# Other
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False

if platform.system() != 'Windows':
    print('Warning!  Image test files generated in windows.  Compatibility with linux/mac may vary')


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
    fcp.contour(df=df, x='X', y='Y', z='Value', filled=False,
                cbar=False, ax_size=[400, 400], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=40, show_points=True,
                filename=name + '.png',
                marker_edge_color='#000000', marker_fill_color='#000000')

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


def test_basic_rc(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_rc_master') if master else 'basic_rc'

    # Make the plot
    fcp.contour(df=df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=False,
                cbar=False, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250,250], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=40, show_points=True,
                filename=name + '.png',
                marker_edge_color='#000000', marker_fill_color='#000000')

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


def test_filled(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_master') if master else 'filled'

    # Make the plot
    fcp.contour(df=df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250,250], show=SHOW,
                label_rc_font_size=12, levels=40,
                filename=name + '.png')

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_filled_no_share(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_no_share_master') if master else 'filled_no_share'

    # Make the plot
    fcp.contour(df=df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250,250], show=SHOW,
                label_rc_font_size=12, levels=40, share_z=False,
                filename=name + '.png')

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_filled_separate(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_separate_master') if master else 'filled_separate'

    # Make the plot
    fcp.contour(df=df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250,250], show=SHOW,
                label_rc_font_size=12, levels=40, separate_labels=True,
                filename=name + '.png')

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_filled_range(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_range_master') if master else 'filled_range'

    # Make the plot
    fcp.contour(df=df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250,250], show=SHOW,
                label_rc_font_size=12, zmin=1, zmax=3, levels=40,
                filename=name + '.png')

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


if __name__ == '__main__':
    pass