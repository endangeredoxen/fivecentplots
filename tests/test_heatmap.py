import pytest
import imageio
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

test = 'heatmap'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
img_cat = utl.img_grayscale(imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'))

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
    Remake all test master images
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
def plt_cat_no_label(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'cat_no_label_master') if master else 'cat_no_label'

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, show=SHOW,
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


def plt_cat_label(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'cat_label_master') if master else 'cat_label'

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, data_labels=True,
                heatmap_font_color='#aaaaaa', show=SHOW, tick_labels_major_y_edge_width=0,
                ws_ticks_ax=5, filename=name + '.png', save=not bm, inline=False)

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


def plt_cat_cell_size(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'cat_cell_size_master') if master else 'cat_cell_size'

    # Make the plot
    fcp.heatmap(df, x='Category', y='Player', z='Average', cbar=True, data_labels=True,
                heatmap_font_color='#aaaaaa', show=SHOW, tick_labels_major_y_edge_width=0,
                ws_ticks_ax=5, cell_size=100,
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


def plt_cat_non_uniform(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'cat_non-uniform_master') if master else 'cat_non-uniform'

    # Make the plot
    df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
    fcp.heatmap(df2, x='X', y='Y', z='Value', row='Batch', col='Experiment',
                cbar=True, show=SHOW, share_z=True, ax_size=[400, 400],
                data_labels=False, label_rc_font_size=12, filter='Batch==103', cmap='viridis',
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


def plt_heatmap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'heatmap_master') if master else 'heatmap'

    # Make the plot
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600],
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


def plt_heatmap_stretched(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'heatmap_stretched_master') if master else 'heatmap_stretched'

    # Make the plot
    uu = img_cat.stack().mean()
    ss = img_cat.stack().std()
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600],
                zmin=uu - 3 * ss, zmax=uu + 3 * ss,
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


def plt_heatmap_zoomed(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'heatmap_zoomed_master') if master else 'heatmap_zoomed'

    # Make the plot
    fcp.heatmap(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=700, xmax=1100,
                ymin=300, ymax=400,
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


def test_heatmap(benchmark):
    plt_heatmap()
    benchmark(plt_heatmap, True)


def test_heatmap_stretched(benchmark):
    plt_heatmap_stretched()
    benchmark(plt_heatmap_stretched, True)


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
