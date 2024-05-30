import pytest
import imageio.v3 as imageio
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.data.data as data
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
db = pdb.set_trace
mpl.use('agg')

test = 'imshow'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

### Test images
# RGB cat
img_cat_orig = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')

# RGB cat in grayscale
img_cat = utl.img_grayscale(img_cat_orig)

# RGB split by color plane and pixel values modified by plane
img_cp_orig = utl.rgb2bayer(imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_color_planes.png'))
cp = utl.split_color_planes(img_cp_orig, as_dict=True)
cp['r'] *= 0.5
cp['b'] -= 50
cp['b'][cp['b'] < 0] = 255
cp['gr'][cp['gr'] < 25] = 25
img_cp = pd.DataFrame({'Plane': cp.keys(), 'Green?': [False, True, True, False]}, index=cp)

# RGB cat in grayscale with grouping columns for row, col, wrap tests
img_all = pd.DataFrame()
img_test = pd.DataFrame(utl.img_grayscale(img_cat_orig).to_numpy()[300:600, 800:1100])
img_test.loc[[0, 299]] = 100
img_test[[0, 299]] = 100
img_test.loc[3, range(3, 297)] = 0
img_test.loc[range(3, 297), 3] = 0
img_test.loc[296, range(3, 297)] = 0
img_test.loc[range(3, 297), 296] = 0
groups = ['hi', 'there', 'you', 'are', 'something', 'special', 'buddy', 'boy']
for i in range(0, 6):
    temp = img_test.copy() * (1 + 2 * i / 10)
    temp['Number'] = f'Image {i}'
    for gg in groups:
        temp[gg] = 'dr crusher'
    img_all = pd.concat([img_all, temp])

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')

# Other
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def make_all():
    """
    Remake all test make_reference images
    """

    if not REFERENCE.exists():
        os.makedirs(REFERENCE)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](make_reference=True)
        print('done!')


def show_all(only_fails=True):
    """
    Run the show=True option on all plt functions
    """

    if not REFERENCE.exists():
        os.makedirs(REFERENCE)
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
def plt_imshow(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, ax_size=[600, 600],
               filename=name, save=not bm, inline=False, timer=False)

    if bm:
        return

    if show == False:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_cbar(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_cbar', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', ax_size=[600, 600], label_font_size=18, cbar=True,
               label_y_text='Rowgs', label_y_edge_width=7, label_y_edge_color='#ff00ff',
               label_x_text='Coljs', label_x_edge_width=15, label_x_edge_color='#ff00ff',
               label_z_edge_width=4, label_z_edge_color='#ff0000', label_z_text='Valyue', label_z_fill_color='#00ff00',
               filename=name, save=not bm, inline=False, timer=False)

    if bm:
        return

    if show == False:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
        utl.unit_test_measure_margin(name, 170, 365, left=10, right=10, bottom=10)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rgb(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat_orig, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if show == False:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rgb_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rgb', make_reference, REFERENCE)

    # Make the plot
    imgs = {}
    imgs[0] = img_cat_orig
    imgs[1] = ((1 - img_cat_orig / 255) * 255).astype(np.uint8)
    df = pd.DataFrame({'case': ['original', 'inverse']}, index=[0, 1])
    fcp.imshow(df, imgs=imgs, wrap='case', filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if show == False:
        utl.unit_test_measure_axes(name, 40, None, 800, alias=False)
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rotate(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rotate', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat.T, cmap='inferno', cbar=True, ax_size=[600, 600],
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_tick_labels(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_tick_labels', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], tick_labels_major=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return

    if show == False:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_stretched(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_stretched', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], stretch=3,
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if show == False:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
        utl.unit_test_measure_margin(name, 170, 365, left=10, top=10, bottom=10, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_zoomed(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_zoomed', make_reference, REFERENCE)

    # Make the plot
    xmin, xmax, ymin, ymax, size_x, size_y = 700, 1100, 300, 400, 600, 600
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[size_x, size_y], xmin=xmin, xmax=xmax,
               ymin=ymin, ymax=ymax, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if show == False:
        utl.unit_test_measure_axes(name, 40, 100, size_x, int((size_x / (xmax - xmin) * (ymax - ymin))), 1, alias=False)
        utl.unit_test_measure_margin(name, 50, 100, left=10, bottom=10, alias=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=False, cbar=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_shared_cbar(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_shared_cbar', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=False, cbar=True, cbar_shared=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_share_z(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_share_z', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='gray', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=True, cbar=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_z_user_range(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_z_user_range', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='viridis', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, zmin=[-100, 100], zmax=[400, 500, 600, 700], cbar=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_quantiles(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_share_z', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, zmin=['q0.35', 'q10', 'q0.33', 0], zmax=['q0.36', 'q90'], share_z=True,
               cbar=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_share_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('share_col', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], row='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, col='Green?', cbar=True, label_rc_fill_color='#00FF00',
               label_rc_font_color='#FF0000', share_col=True, title='power up', title_fill_color='#eeddcc')

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_share_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('share_row', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, row='Green?', cbar=True, label_rc_fill_color='#00FF00',
               label_rc_font_color='#FF0000', share_row=True)

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap', make_reference, REFERENCE)

    fcp.imshow(img_cp_orig, cmap='inferno', ax_size=[300, 300], cfa='rggb',
               filename=name.with_suffix('.png'), save=not bm, inline=False, wrap='Plane')

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_one(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap_one', make_reference, REFERENCE)

    fcp.imshow(img_cp_orig, cmap='inferno', ax_size=[300, 300], cfa='rggb', filter='Plane=="gb"',
               label_wrap_edge_color='#0000ff', filename=name.with_suffix('.png'), save=not bm, inline=False, wrap='Plane')

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_combos(bm=False, make_reference=False, show=False):

    def compare_with_reference(make_reference, show, name):
        if make_reference:
            return
        elif show == -1:
            utl.show_file(name + '.png')
        elif show:
            utl.show_file(REFERENCE / name + '_reference.png')
            utl.show_file(name + '.png')
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png', show=True)
        else:
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png')
            if remove:
                os.remove(name + '.png')

            assert not compare

    # # 1 x 1
    # name = REFERENCE / 'col_combos_1x1_reference' if make_reference else 'col_combos_1x1'
    # fcp.imshow(img_all, ax_size=[300, 300], col='Number', label_rc_edge_width=2,
    #            ax_edge_width=5, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
    #            filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')
    # compare_with_reference(make_reference, show, name)

    # 1 x 2
    name = REFERENCE / 'col_combos_1x2_reference' if make_reference else 'col_combos_1x2'
    fcp.imshow(img_all, ax_size=[300, 300], col='Number', label_rc_edge_width=1,
               ax_edge_width=5, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 1"]')
    compare_with_reference(make_reference, show, name)

    # 1 x 3
    # name = REFERENCE / 'col_combos_1x3_reference' if make_reference else 'col_combos_1x3'
    # fcp.imshow(img_all, ax_size=[250, 250], col='Number', label_rc_edge_width=1,
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
    #            filename=name.with_suffix('.png'), save=not bm, inline=False,
    #            filter='Number in ["Image 0", "Image 1", "Image 5"]')
    # compare_with_reference(make_reference, show, name)

    # # 1 x 3b
    # name = REFERENCE / 'col_combos_1x3b_reference' if make_reference else 'col_combos_1x3b'
    # fcp.imshow(img_all, ax_size=[250, 250], col='Number', label_rc_edge_width=1,
    #            ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
    #            filename=name.with_suffix('.png'), save=not bm, inline=False,
    #            filter='Number in ["Image 0", "Image 1", "Image 5"]')
    # compare_with_reference(make_reference, show, name)


def plt_row_combos(bm=False, make_reference=False, show=False):

    def compare_with_reference(make_reference, show, name):
        if make_reference:
            return
        elif show == -1:
            utl.show_file(name + '.png')
        elif show:
            utl.show_file(REFERENCE / name + '_reference.png')
            utl.show_file(name + '.png')
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png', show=True)
        else:
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png')
            if remove:
                os.remove(name + '.png')

            assert not compare

    # 1 x 1
    name = REFERENCE / 'row_combos_1x1_reference' if make_reference else 'row_combos_1x1'
    fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
               ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')
    compare_with_reference(make_reference, show, name)

    # 2 x 1
    name = REFERENCE / 'row_combos_2x1_reference' if make_reference else 'row_combos_2x1'
    fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
               ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 1"]')
    compare_with_reference(make_reference, show, name)

    # # 3 x 1
    # name = REFERENCE / 'row_combos_3x1_reference' if make_reference else 'row_combos_3x1'
    # fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
    #            ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
    #            filename=name.with_suffix('.png'), save=not bm, inline=False,
    #            filter='Number in ["Image 0", "Image 1", "Image 5"]')
    # compare_with_reference(make_reference, show, name)

    # # 3 x 1b
    # name = REFERENCE / 'row_combos_3x1b_reference' if make_reference else 'row_combos_3x1b'
    # fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
    #            ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
    #            filename=name.with_suffix('.png'), save=not bm, inline=False,
    #            filter='Number in ["Image 0", "Image 1", "Image 5"]')
    # compare_with_reference(make_reference, show, name)


def plt_wrap_combos(bm=False, make_reference=False, show=False):

    def compare_with_reference(make_reference, show, name):
        if make_reference:
            return
        elif show == -1:
            utl.show_file(name + '.png')
        elif show:
            utl.show_file(REFERENCE / name + '_reference.png')
            utl.show_file(name + '.png')
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png', show=True)
        else:
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png')
            if remove:
                os.remove(name + '.png')

            assert not compare

    img_all = pd.DataFrame()
    img_test = pd.DataFrame(utl.img_grayscale(img_cat_orig).to_numpy()[300:600, 800:1100])
    groups = ['hi', 'there', 'you', 'are', 'something', 'special', 'buddy', 'boy']
    for i in range(0, 6):
        temp = img_test.copy() * (1 + 2 * i / 10)
        temp['Number'] = f'Image {i}'
        for gg in groups:
            temp[gg] = 'dr crusher'
        img_all = pd.concat([img_all, temp])

    # 1 x 1
    name = REFERENCE / 'wrap_combos_1x1_reference' if make_reference else 'wrap_combos_1x1'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')
    compare_with_reference(make_reference, show, name)

    # 1 x 2
    name = REFERENCE / 'wrap_combos_1x2_reference' if make_reference else 'wrap_combos_1x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 5"]')
    compare_with_reference(make_reference, show, name)

    # 1 x 3
    name = REFERENCE / 'wrap_combos_1x3_reference' if make_reference else 'wrap_combos_1x3'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_reference(make_reference, show, name)

    # 2 x 3
    name = REFERENCE / 'wrap_combos_2x3_reference' if make_reference else 'wrap_combos_2x3'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False)
    compare_with_reference(make_reference, show, name)

    # 3 x 1
    name = REFERENCE / 'wrap_combos_3x1_reference' if make_reference else 'wrap_combos_3x1'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_reference(make_reference, show, name)

    # 3 x 2
    name = REFERENCE / 'wrap_combos_3x2_reference' if make_reference else 'wrap_combos_3x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['inferno', 'gray'], share_col=True)
    compare_with_reference(make_reference, show, name)

    # 4 x 2
    name = REFERENCE / 'wrap_combos_4x2_reference' if make_reference else 'wrap_combos_4x2'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
               filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['gray', 'inferno'], share_row=True)
    compare_with_reference(make_reference, show, name)


def plt_wrap_combos_cbar(bm=False, make_reference=False, show=False):

    def compare_with_reference(make_reference, show, name):
        if make_reference:
            return
        elif show == -1:
            utl.show_file(name + '.png')
        elif show:
            utl.show_file(REFERENCE / name + '_reference.png')
            utl.show_file(name + '.png')
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png', show=True)
        else:
            compare = utl.img_compare(name + '.png', REFERENCE / name + '_reference.png')
            if remove:
                os.remove(name + '.png')

            assert not compare

    # img_all = pd.DataFrame()
    # img_test = pd.DataFrame(utl.img_grayscale(img_cat_orig).to_numpy()[300:600, 800:1100])
    # for i in range(0, 6):
    #     temp = img_test.copy() * (1 + 2 * i / 10)
    #     temp['Number'] = f'Image {i}'
    #     img_all = pd.concat([img_all, temp])

    # 1 x 1
    name = REFERENCE / 'wrap_combos_1x1_cbar_reference' if make_reference else 'wrap_combos_1x1_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')
    compare_with_reference(make_reference, show, name)

    # 1 x 2
    name = REFERENCE / 'wrap_combos_1x2_cbar_reference' if make_reference else 'wrap_combos_1x2_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 5"]')
    compare_with_reference(make_reference, show, name)

    # 1 x 3
    name = REFERENCE / 'wrap_combos_1x3_cbar_reference' if make_reference else 'wrap_combos_1x3_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_reference(make_reference, show, name)

    # 2 x 3
    name = REFERENCE / 'wrap_combos_2x3_cbar_reference' if make_reference else 'wrap_combos_2x3_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, zmin=[50, 100], zmax=[200, 300])
    compare_with_reference(make_reference, show, name)

    # 3 x 1
    name = REFERENCE / 'wrap_combos_3x1_cbar_reference' if make_reference else 'wrap_combos_3x1_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0", "Image 2", "Image 4"]')
    compare_with_reference(make_reference, show, name)

    # 3 x 2
    name = REFERENCE / 'wrap_combos_3x2_cbar_reference' if make_reference else 'wrap_combos_3x2_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['inferno', 'gray'], share_col=True)
    compare_with_reference(make_reference, show, name)

    # 4 x 2
    name = REFERENCE / 'wrap_combos_4x2_cbar_reference' if make_reference else 'wrap_combos_4x2_cbar'
    fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
               ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['gray', 'inferno'], share_row=True)
    compare_with_reference(make_reference, show, name)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_imshow(benchmark):
    plt_imshow()
    benchmark(plt_imshow, True)


def test_imshow_cbar(benchmark):
    plt_imshow_cbar()
    benchmark(plt_imshow_cbar, True)


def test_imshow_rgb(benchmark):
    plt_imshow_rgb()
    benchmark(plt_imshow_rgb, True)


def test_imshow_rgb_wrap(benchmark):
    plt_imshow_rgb_wrap()
    benchmark(plt_imshow_rgb_wrap, True)


def test_imshow_rotate(benchmark):
    plt_imshow_rotate()
    benchmark(plt_imshow_rotate, True)


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


def test_col_shared_cbar(benchmark):
    plt_col_shared_cbar()
    benchmark(plt_col_shared_cbar, True)


def test_col_share_z(benchmark):
    plt_share_col()
    benchmark(plt_col_share_z, True)


def test_col_z_user_range(benchmark):
    plt_col_z_user_range()
    benchmark(plt_col_z_user_range, True)


def test_col_combos(benchmark):
    plt_col_combos()
    benchmark(plt_col_combos, True)


def test_row_combos(benchmark):
    plt_row_combos()
    benchmark(plt_row_combos, True)


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
    ## BAD DATA INPUT FOR ARRAY VS DF


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


def test_wrap_combos(benchmark):
    plt_wrap_combos()
    # benchmark(plt_wrap_combos, True)


def test_wrap_combos_cbar(benchmark):
    plt_wrap_combos_cbar()
    # benchmark(plt_wrap_combos_cbar, True)


if __name__ == '__main__':
    pass
