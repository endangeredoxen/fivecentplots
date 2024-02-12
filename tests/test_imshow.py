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
mpl.use('agg')

test = 'imshow'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
img_cat_orig = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')
img_cat = utl.img_grayscale(img_cat_orig)
img_cp = utl.rgb2bayer(imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_color_planes.png'))
cp = utl.split_color_planes(img_cp, as_dict=True)
cp['r'] *= 0.5
cp['b'] -= 50
cp['gr'][cp['gr'] < 25] = 25
for ii in cp:
    cp[ii] = cp[ii].reset_index(drop=True)
    cp[ii]['Plane'] = ii
    cp[ii]['Green?'] = True if 'g' in ii else False
img_rc = pd.concat(cp)
img_rc.loc[img_rc.Value < 0, 'Value'] = 0


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
def plt_imshow(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_master') if master else 'imshow'

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600],
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


def plt_imshow_rotate(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_rotate_master') if master else 'imshow_rotate'

    # Make the plot
    fcp.imshow(img_cat.T, cmap='inferno', cbar=True, ax_size=[600, 600],
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


def plt_imshow_no_cbar(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_no_cbar_master') if master else 'imshow_no_cbar'

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=False, ax_size=[600, 600],
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


def plt_imshow_tick_labels(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_tick_labels_master') if master else 'imshow_tick_labels'

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], tick_labels_major=True,
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


def plt_imshow_stretched(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_stretched_master') if master else 'imshow_stretched'

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], stretch=3,
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


def plt_imshow_zoomed(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_zoomed_master') if master else 'imshow_zoomed'

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=700, xmax=1100,
               ymin=300, ymax=400, filename=name + '.png', save=not bm, inline=False)

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


def plt_col(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'col_master') if master else 'col'

    fcp.imshow(img_rc, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name + '.png', save=not bm,
               inline=False, share_z=False, cbar=True)

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


def plt_share_col(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'share_col_master') if master else 'share_col'

    fcp.imshow(img_rc, cmap='inferno', ax_size=[300, 300], row='Plane', filename=name + '.png', save=not bm,
               inline=False, col='Green?', cbar=True, label_rc_fill_color='#00FF00',
               label_rc_font_color='#FF0000', share_col=True)

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


def plt_share_row(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'share_row_master') if master else 'share_row'

    fcp.imshow(img_rc, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name + '.png', save=not bm,
               inline=False, row='Green?', cbar=True, label_rc_fill_color='#00FF00',
               label_rc_font_color='#FF0000', share_row=True)

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


def plt_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_master') if master else 'wrap'

    fcp.imshow(img_cp, cmap='inferno', ax_size=[300, 300], cfa='rggb',
               filename=name + '.png', save=not bm, inline=False, wrap='Plane')

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


def plt_wrap_one(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_one_master') if master else 'wrap_one'

    fcp.imshow(img_cp, cmap='inferno', ax_size=[300, 300], cfa='rggb', filter='Plane=="gb"',
               filename=name + '.png', save=not bm, inline=False, wrap='Plane')

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


def plt_wrap_combos(bm=False, master=False, remove=True, show=False):

    def compare_with_master(master, show, name):
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

    img_all = pd.DataFrame()
    img_test = pd.DataFrame(utl.img_grayscale(img_cat_orig).to_numpy()[300:600, 800:1100])
    for i in range(0, 6):
        temp = img_test.copy() * (1 + 2 * i / 10)
        temp['Number'] = f'Image {i}'
        img_all = pd.concat([img_all, temp])

    # 1 x 1
    name = osjoin(MASTER, 'wrap_combos_1x1_master') if master else 'wrap_combos_1x1'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0"]')
    compare_with_master(master, show, name)

    # 1 x 2
    name = osjoin(MASTER, 'wrap_combos_1x2_master') if master else 'wrap_combos_1x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 5"]')
    compare_with_master(master, show, name)

    # 1 x 3
    name = osjoin(MASTER, 'wrap_combos_1x3_master') if master else 'wrap_combos_1x3'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_master(master, show, name)

    # 2 x 3
    name = osjoin(MASTER, 'wrap_combos_2x3_master') if master else 'wrap_combos_2x3'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False)
    compare_with_master(master, show, name)

    # 3 x 1
    name = osjoin(MASTER, 'wrap_combos_3x1_master') if master else 'wrap_combos_3x1'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_master(master, show, name)

    # 3 x 2
    name = osjoin(MASTER, 'wrap_combos_3x2_master') if master else 'wrap_combos_3x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, cmap=['inferno', 'gray'], share_col=True)
    compare_with_master(master, show, name)

    # 4 x 2
    name = osjoin(MASTER, 'wrap_combos_4x2_master') if master else 'wrap_combos_4x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name + '.png', save=not bm, inline=False, cmap=['gray', 'inferno'], share_row=True)
    compare_with_master(master, show, name)


def plt_wrap_combos_cbar(bm=False, master=False, remove=True, show=False):

    def compare_with_master(master, show, name):
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

    img_all = pd.DataFrame()
    img_test = pd.DataFrame(utl.img_grayscale(img_cat_orig).to_numpy()[300:600, 800:1100])
    for i in range(0, 6):
        temp = img_test.copy() * (1 + 2 * i / 10)
        temp['Number'] = f'Image {i}'
        img_all = pd.concat([img_all, temp])

    # # 1 x 1
    # name = osjoin(MASTER, 'wrap_combos_1x1_cbar_master') if master else 'wrap_combos_1x1_cbar'
    # fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
    #            filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0"]')
    # compare_with_master(master, show, name)

    # # 1 x 2
    # name = osjoin(MASTER, 'wrap_combos_1x2_cbar_master') if master else 'wrap_combos_1x2_cbar'
    # fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
    #            filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 5"]')
    # compare_with_master(master, show, name)

    # # 1 x 3
    # name = osjoin(MASTER, 'wrap_combos_1x3_cbar_master') if master else 'wrap_combos_1x3_cbar'
    # fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
    #            filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    # compare_with_master(master, show, name)

    # # 2 x 3
    # name = osjoin(MASTER, 'wrap_combos_2x3_cbar_master') if master else 'wrap_combos_2x3_cbar'
    # fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
    #            filename=name + '.png', save=not bm, inline=False)
    # compare_with_master(master, show, name)

    # # 3 x 1
    # name = osjoin(MASTER, 'wrap_combos_3x1_cbar_master') if master else 'wrap_combos_3x1_cbar'
    # fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
    #            filename=name + '.png', save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    # compare_with_master(master, show, name)

    # 3 x 2
    name = osjoin(MASTER, 'wrap_combos_3x2_cbar_master') if master else 'wrap_combos_3x2_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name + '.png', save=not bm, inline=False, cmap=['inferno', 'gray'], share_col=True)
    compare_with_master(master, show, name)

    # 4 x 2
    name = osjoin(MASTER, 'wrap_combos_4x2_cbar_master') if master else 'wrap_combos_4x2_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name + '.png', save=not bm, inline=False, cmap=['gray', 'inferno'], share_row=True)
    compare_with_master(master, show, name)

# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_imshow(benchmark):
    plt_imshow()
    benchmark(plt_imshow, True)


def test_imshow_rotate(benchmark):
    plt_imshow_rotate()
    benchmark(plt_imshow_rotate, True)


def test_imshow_no_cbar(benchmark):
    plt_imshow_no_cbar()
    benchmark(plt_imshow_no_cbar, True)


def test_imshow_tick_labels(benchmark):
    plt_imshow_tick_labels()
    benchmark(plt_imshow_tick_labels, True)


def test_imshow_stretched(benchmark):
    plt_imshow_stretched()
    benchmark(plt_imshow_stretched, True)


def test_imshow_zoomed(benchmark):
    plt_imshow_zoomed()
    benchmark(plt_imshow_zoomed, True)


def test_col(benchmark):
    plt_col()
    benchmark(plt_col, True)


def test_share_col(benchmark):
    plt_share_col()
    benchmark(plt_share_col, True)


def test_share_row(benchmark):
    plt_share_row()
    benchmark(plt_share_row, True)


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


def test_wrap_one(benchmark):
    plt_wrap_one()
    benchmark(plt_wrap_one, True)


def test_invalid():
    with pytest.raises(data.AxisError):
        fcp.imshow(img_rc, twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.imshow(img_rc, twin_y=True)
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_rc, row='y')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_rc, wrap='y')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_rc, col='x')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_rc, legend=True)


if __name__ == '__main__':
    pass
