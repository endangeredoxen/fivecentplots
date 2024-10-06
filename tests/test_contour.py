import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'contour'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')

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
def plt_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', filled=False, cbar=False, ax_size=[400, 400], show=SHOW,
                contour_width=2, label_rc_font_size=12, levels=30, show_points=True, filename=name.with_suffix('.png'),
                save=not bm, inline=False, marker_edge_color='#000000', marker_fill_color='#000000')

    if bm:
        return

    if not show:
        # Axis width
        utl.unit_test_measure_axes(name, 70, None, 402, None, 1, alias=True)
        # Margins
        utl.unit_test_measure_margin(name, 70, 115, left=80, right=10, top=10, bottom=76, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_basic_rc(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic_rc', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=False,
                cbar=False, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=30, show_points=True,
                filename=name.with_suffix('.png'), save=not bm, inline=False,
                marker_edge_color='#000000', marker_fill_color='#000000')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_basic_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic_wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', wrap=['Batch', 'Experiment'], filled=False,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW, contour_width=2,
                label_rc_font_size=12, levels=30, show_points=True, filename=name.with_suffix('.png'), save=not bm,
                inline=False, marker_edge_color='#000000', marker_fill_color='#000000',
                ax_edge_width=5)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_filled(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('filled', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=30,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_filled_no_share(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('filled_no_share', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=30, share_z=False, tick_labels_major_z_rotation=45,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_filled_range(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('filled_range', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, zmin=1, zmax=3, levels=30,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


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


def test_filled_range(benchmark):
    plt_filled_range()
    benchmark(plt_filled_range, True)


if __name__ == '__main__':
    pass
