import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
osjoin = os.path.join
db = pdb.set_trace

test = 'contour'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')

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
    fcp.contour(df, x='X', y='Y', z='Value', filled=False,
                cbar=False, ax_size=[400, 400], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=40, show_points=True,
                filename=name + '.png', save=not bm, inline=False,
                marker_edge_color='#000000', marker_fill_color='#000000')

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


def plt_basic_rc(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_rc_master') if master else 'basic_rc'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=False,
                cbar=False, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=40, show_points=True,
                filename=name + '.png', save=not bm, inline=False,
                marker_edge_color='#000000', marker_fill_color='#000000')

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


def plt_basic_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_wrap_master') if master else 'basic_wrap'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', wrap=['Batch', 'Experiment'], filled=False,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=40, show_points=True, filename=name + '.png', save=not bm, inline=False,
                marker_edge_color='#000000', marker_fill_color='#000000')

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


def plt_filled(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_master') if master else 'filled'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=40,
                filename=name + '.png', save=not bm, inline=False)

    if bm:
        return

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


def plt_filled_no_share(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_no_share_master') if master else 'filled_no_share'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=40, share_z=False, tick_labels_major_z_rotation=45,
                filename=name + '.png', save=not bm, inline=False)

    if bm:
        return

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


def plt_filled_separate(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_separate_master') if master else 'filled_separate'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=40, separate_labels=True,
                filename=name + '.png', save=not bm, inline=False)

    if bm:
        return

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


def plt_filled_range(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'filled_range_master') if master else 'filled_range'

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, zmin=1, zmax=3, levels=40,
                filename=name + '.png', save=not bm, inline=False)

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


def test_basic_rc(benchmark):
    plt_basic_rc()
    benchmark(plt_basic_rc, True)


def test_basic_wrap(benchmark):
    plt_basic_wrap()
    benchmark(plt_basic_wrap, True)


def test_filled(benchmark):
    plt_filled()
    benchmark(plt_filled, True)


def test_filled_no_share(benchmark):
    plt_filled_no_share()
    benchmark(plt_filled_no_share, True)


def test_filled_separate(benchmark):
    plt_filled_separate()
    benchmark(plt_filled_separate, True)


def test_filled_range(benchmark):
    plt_filled_range()
    benchmark(plt_filled_range, True)


if __name__ == '__main__':
    pass
