import pytest
import imageio.v3 as imageio
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

test = 'heatmap'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
img_cat = utl.img_grayscale_deprecated(imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'))

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
def plt_cat_cell_size(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('cat_cell_size', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, data_labels=True,
                heatmap_font_color='#aaaaaa', show=SHOW, tick_labels_major_y_edge_width=0,
                ws_ticks_ax=5, cell_size=100,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_cat_no_label(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('cat_no_label', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, show=SHOW,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_cat_label(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('cat_label', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, data_labels=True,
                heatmap_font_color='#aaaaaa', show=SHOW, tick_labels_major_y_edge_width=0,
                ws_ticks_ax=5, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_cat_non_uniform(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('cat_non-uniform', make_reference, REFERENCE)

    # Make the plot
    df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
    fcp.heatmap(df2, x='X', y='Y', z='Value', row='Batch', col='Experiment',
                cbar=True, show=SHOW, share_z=True, ax_size=[400, 400],
                data_labels=False, label_rc_font_size=12, filter='Batch==103', cmap='viridis',
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_heatmap(bm=False, make_reference=False, show=False):
    print('deprecated: plt_heatmap')
    return

    name = utl.unit_test_get_img_name('heatmap', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600],
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_heatmap_stretched(bm=False, make_reference=False, show=False):

    print('deprecated: plt_heatmap_stretched')
    return

    name = utl.unit_test_get_img_name('heatmap_stretched', make_reference, REFERENCE)

    # Make the plot
    uu = img_cat.stack().mean()
    ss = img_cat.stack().std()
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600],
                zmin=uu - 3 * ss, zmax=uu + 3 * ss,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_heatmap_zoomed(bm=False, make_reference=False, show=False):

    print('deprecated: plt_heatmap_zoomed')
    return

    name = utl.unit_test_get_img_name('heatmap_zoomed', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=700, xmax=1100,
                ymin=300, ymax=400,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_cat_no_label(benchmark):
    plt_cat_no_label()
    benchmark(plt_cat_no_label, True)


def test_cat_label(benchmark):
    plt_cat_label()
    benchmark(plt_cat_label, True)


def test_cat_cell_size(benchmark):
    plt_cat_cell_size()
    benchmark(plt_cat_cell_size, True)


def test_cat_non_uniform(benchmark):
    plt_cat_non_uniform()
    benchmark(plt_cat_non_uniform, True)


@pytest.mark.skip(reason="deprecated; use imshow")
def test_heatmap(benchmark):
    plt_heatmap()
    benchmark(plt_heatmap, True)


@pytest.mark.skip(reason="deprecated; use imshow")
def test_heatmap_stretched(benchmark):
    plt_heatmap_stretched()
    benchmark(plt_heatmap_stretched, True)


@pytest.mark.skip(reason="deprecated; use imshow")
def test_heatmap_zoomed(benchmark):
    plt_heatmap_zoomed()
    benchmark(plt_heatmap_zoomed, True)


def test_invalid():
    with pytest.raises(data.AxisError):
        fcp.heatmap(img_cat, twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.heatmap(img_cat, twin_y=True)
    with pytest.raises(data.GroupingError):
        fcp.heatmap(img_cat, row='y')
    with pytest.raises(data.GroupingError):
        fcp.heatmap(img_cat, wrap='y')
    with pytest.raises(data.GroupingError):
        fcp.heatmap(img_cat, col='x')
    with pytest.raises(data.GroupingError):
        fcp.heatmap(img_cat, legend=True)


if __name__ == '__main__':
    pass
