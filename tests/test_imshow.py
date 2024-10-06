import pytest
import imageio.v3 as imageio
import fivecentplots as fcp
import pandas as pd
import numpy as np
import sys
import pdb
from pathlib import Path
import fivecentplots.data.data as data
import fivecentplots.utilities as utl
import matplotlib as mpl
db = pdb.set_trace
mpl.use('agg')

test = 'imshow'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Test images
# RGB cat
img_cat_orig = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')
img_color_bars = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_color_bars.png')

# RGB cat in grayscale
img_cat = utl.img_grayscale_deprecated(img_cat_orig)

# RGB split by color plane and pixel values modified by plane
img_cp_orig = utl.rgb2bayer(imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_color_planes.png'))
cp = utl.split_color_planes(img_cp_orig, as_dict=True)
cp['r'] = cp['r'] * 0.5
cp['b'] -= 50
cp['b'][cp['b'] < 0] = 250
cp['b'][cp['b'] == 255] = 250
cp['gr'][cp['gr'] < 25] = 25
img_cp = pd.DataFrame({'Plane': cp.keys(), 'Green?': [False, True, True, False]}, index=cp)

# RGB cat in grayscale with grouping columns for row, col, wrap tests
img_all = pd.DataFrame()
img_test = pd.DataFrame(utl.img_grayscale_deprecated(img_cat_orig).to_numpy()[300:600, 800:1100])
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
def make_all(start=None, stop=None):
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__], start=start, stop=stop)


def show_all(only_fails=True, start=None):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__], start=start)


SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


# plt_ functions can be used directly outside of pytest for debug
def plt_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=False, cbar=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 90, 300, 4, alias=False)
        utl.unit_test_measure_margin(name, 'c', 150, left=10, right=10, top=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_cbar(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_cbar', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=False, cbar=True)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 90, 300, 4, cbar=True, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_combos(bm=False, make_reference=False, show=False):

    enabled = ['1x1', '1x2', '1x3', '2x3', '3x1', '3x2', '4x2', '4x2b']
    # enabled = ['1x3']

    # 1 x 1
    if '1x1' in enabled:
        name = utl.unit_test_get_img_name('col_combos_1x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[300, 300], col='Number', label_rc_edge_width=3,
                   ax_edge_width=5, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, 20, 150, left=10, right=10, top=10, bottom=10, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 2
    if '1x2' in enabled:
        name = utl.unit_test_get_img_name('col_combos_1x2', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[300, 300], col='Number', label_rc_edge_width=4,
                   ax_edge_width=6, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, 20, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 100, 312, 2, alias=True)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 18, 312, 2, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 3
    if '1x3' in enabled:
        name = utl.unit_test_get_img_name('col_combos_1x3', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], col='Number', label_rc_edge_width=1,
                   ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, 12, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 100, 252, 3, alias=True)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 14, 252, 3, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 3b
    if '1x3b' in enabled:
        name = utl.unit_test_get_img_name('col_combos_1x3b', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], col='Number', label_rc_edge_width=1,
                   ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, bottom=10, alias=False)
            # Label margin
            utl.unit_test_measure_margin(name, 12, 150, left=10, right=10, top=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 100, 250, 3, alias=False)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 14, 250, 3, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 3c
    if '1x3c' in enabled:
        name = utl.unit_test_get_img_name('col_combos_1x3c', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], col='Number', label_rc_edge_width=0,
                   ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, bottom=10, alias=False)
            # Label margin
            utl.unit_test_measure_margin(name, 12, 150, left=10, right=10, top=10, alias=False)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 100, 250, 3, alias=False)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 14, 250, 3, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_shared_cbar(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_shared_cbar', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=False, cbar=True, cbar_shared=True)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 90, 300, 4, alias=False)
        utl.unit_test_measure_margin(name, 'c', 150, left=10, top=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_share_z(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_share_z', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='gray', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, share_z=True, cbar=True)

    if bm:
        return

    if not show:
        # Axes widths
        utl.unit_test_measure_axes_cols(name, 54, 300, 4, alias=False, cbar=True)
        # Label widths
        utl.unit_test_measure_axes_cols(name, 13, 300, 4, alias=False)
        # Margins
        utl.unit_test_measure_margin(name, 70, 150, left=10, right=81, top=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_z_user_range(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_z_user_range', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='viridis', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               share_z=True, save=not bm, inline=False, zmin=[-100, 100], zmax=[400, 500, 600, 700], cbar=True)

    if bm:
        return

    if not show:
        # Axes widths
        utl.unit_test_measure_axes_cols(name, 54, 300, 4, alias=False, cbar=True)
        # Label widths
        utl.unit_test_measure_axes_cols(name, 13, 300, 4, alias=False)
        # Margins
        utl.unit_test_measure_margin(name, 70, 150, left=10, right=88, top=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col_quantiles(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_quantiles', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, zmin=['q0.35', 'q10', 'q1', 0], zmax=['q0.36', 'q90'], share_z=True,
               cbar=True)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, ax_size=[600, 600], filename=name, save=not bm, inline=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_cbar(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_cbar', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', ax_size=[600, 600], label_font_size=18, cbar=True,
               label_y_text='Rowgs', label_y_edge_width=7, label_y_edge_color='#ff00ff',
               label_x_text='Coljs', label_x_edge_width=15, label_x_edge_color='#ff00ff',
               label_z_edge_width=4, label_z_edge_color='#ff0000', label_z_text='Valyue', label_z_fill_color='#00ff00',
               filename=name, save=not bm, inline=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
        utl.unit_test_measure_margin(name, 170, 365, left=10, right=10, bottom=10)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rgb(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat_orig, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rgb_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rgb_wrap', make_reference, REFERENCE)

    # Make the plot
    imgs = {}
    imgs[0] = img_cat_orig
    imgs[1] = ((1 - img_cat_orig / 255) * 255).astype(np.uint8)
    df = pd.DataFrame({'case': ['this is the cat pirate named Meow Gary Gary Gar Gar',
                                "this is the inverse Gary"]}, index=[0, 1])
    fcp.imshow(df, imgs=imgs, wrap='case', filename=name.with_suffix('.png'), save=not bm, inline=False,
               title_wrap_font_size=50, label_wrap_font_size=24)

    if bm:
        return
    if not show:
        utl.unit_test_measure_axes(name, 150, None, 800, alias=False)
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rotate(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rotate', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat.T, cmap='inferno', cbar=True, ax_size=[600, 600],
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_tick_labels(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_tick_labels', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], tick_labels_major=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_tick_and_axes_labels(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_tick_and_axes_labels', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], tick_labels_major=True,
               filename=name.with_suffix('.png'), save=not bm, inline=False,
               label_y='Row', label_x='Column')

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_stretched(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_stretched', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[600, 600], stretch=3,
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if not show:
        utl.unit_test_measure_axes(name, 40, 100, 600, 300, 1, alias=False)
        utl.unit_test_measure_margin(name, 170, 365, left=10, top=10, bottom=10, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_zoomed(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_zoomed', make_reference, REFERENCE)

    # Make the plot
    xmin, xmax, ymin, ymax, size_x, size_y = 700, 1100, 300, 400, 600, 600
    fcp.imshow(img_cat, cmap='inferno', cbar=True, ax_size=[size_x, size_y], xmin=xmin, xmax=xmax,
               ymin=ymin, ymax=ymax, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    if not show:
        utl.unit_test_measure_axes(name, 40, 100, size_x, int((size_x / (xmax - xmin) * (ymax - ymin))), 1, alias=False)
        utl.unit_test_measure_margin(name, 50, 100, left=10, bottom=10, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_combos(bm=False, make_reference=False, show=False):

    enabled = ['1x1', '2x1', '3x1', '3x1b', '3x1c']
    # enabled = ['3x1c']

    # 1 x 1
    if '1x1' in enabled:
        name = utl.unit_test_get_img_name('row_combos_1x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=3,
                   ax_edge_width=5, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, None, 310, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 150, 260, 1, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 2 x 1
    if '2x1' in enabled:
        name = utl.unit_test_get_img_name('row_combos_2x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=4,
                   ax_edge_width=6, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, 20, 310, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_rows(name, 'c', 262, 2, alias=True)
            # Label widths
            utl.unit_test_measure_axes_rows(name, 313, 262, 2, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 1
    if '3x1' in enabled:
        name = utl.unit_test_get_img_name('row_combos_3x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
                   ax_edge_width=1, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, None, 300, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_rows(name, 'c', 252, 3, alias=True)
            # Label widths
            utl.unit_test_measure_axes_rows(name, 300, 252, 3, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 1b
    if '3x1b' in enabled:
        name = utl.unit_test_get_img_name('row_combos_3x1b', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=1,
                   ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, top=10, bottom=10, alias=False)
            # Label margin
            utl.unit_test_measure_margin(name, None, 300, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_rows(name, 'c', 250, 3, alias=False)
            # Label widths
            utl.unit_test_measure_axes_rows(name, 300, 250, 3, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 1c
    if '3x1c' in enabled:
        name = utl.unit_test_get_img_name('row_combos_3x1c', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], row='Number', label_rc_edge_width=0,
                   ax_edge_width=0, ax_edge_color='#ff0000', label_rc_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 1", "Image 5"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, top=10, bottom=10, alias=False)
            # Label margin
            utl.unit_test_measure_margin(name, 150, 296, left=10, right=10, top=10, bottom=10, alias=False)
            # Axes widths
            utl.unit_test_measure_axes_rows(name, 'c', 250, 3, alias=False)
            # Label widths
            utl.unit_test_measure_axes_rows(name, 296, 250, 3, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_share_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('share_col', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], row='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, col='Green?', cbar=True, label_rc_fill_color='#00FF00',
               label_rc_font_color='#FF0000', share_col=True, title='power up', title_fill_color='#eeddcc')

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_rows(name, 80, 225, 2, alias=False)
        utl.unit_test_measure_axes_cols(name, 150, 300, 1, alias=False)
        utl.unit_test_measure_axes_cols(name, 440, 300, 1, alias=False)
        utl.unit_test_measure_axes_cols(name, 680, 300, 1, alias=False)
        utl.unit_test_measure_axes_cols(name, 920, 300, 1, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_share_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('share_row', make_reference, REFERENCE)

    fcp.imshow(img_cp, imgs=cp, cmap='inferno', ax_size=[300, 300], col='Plane', filename=name.with_suffix('.png'),
               save=not bm, inline=False, row='Green?', cbar=True, label_rc_fill_color='#00FF00',
               title='power up', title_fill_color='#eeddcc', title_span='fig',
               label_rc_font_color='#FF0000', share_row=True)

    if bm:
        return

    if not show:
        # Label margins
        utl.unit_test_measure_margin(name, 48, None, left=10, right=161, alias=False)
        # Label widths
        utl.unit_test_measure_axes_cols(name, 48, 300, 4, alias=False)
        # Axes margins
        utl.unit_test_measure_margin(name, 180, None, left=10, right=10, alias=False)
        # Axes widths
        utl.unit_test_measure_axes_cols(name, 180, 300, 2, alias=False)
        # Axes heights
        utl.unit_test_measure_axes_rows(name, 110, 225, 1, alias=False)
        utl.unit_test_measure_axes_rows(name, 480, 225, 1, alias=False)
        utl.unit_test_measure_axes_rows(name, 860, 225, 1, alias=False)
        utl.unit_test_measure_axes_rows(name, 1220, 225, 1, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap', make_reference, REFERENCE)

    fcp.imshow(img_cp_orig, cmap='inferno', ax_size=[300, 300], cfa='rggb',
               filename=name.with_suffix('.png'), save=not bm, inline=False, wrap='Plane')

    if bm:
        return

    if not show:
        # Margins
        utl.unit_test_measure_margin(name, 100, 100, left=10, right=10, top=10, bottom=10, alias=False)
        # Label widths
        utl.unit_test_measure_axes_cols(name, 42, 300 * 2, 1, alias=False)
        # Axes widths
        utl.unit_test_measure_axes_cols(name, 180, 300 * 2, 1, alias=False)
        # Title width
        utl.unit_test_measure_axes_cols(name, 13, 300 * 2, 1, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_combos(bm=False, make_reference=False, show=False):

    enabled = ['1x1', '1x2', '1x3', '2x3', '3x1', '3x2', '4x2', '4x2b']
    enabled = ['1x3']

    img_all = pd.DataFrame()
    img_test = pd.DataFrame(utl.img_grayscale_deprecated(img_cat_orig).to_numpy()[300:600, 800:1100])
    groups = ['hi', 'there', 'you', 'are', 'something', 'special', 'buddy', 'boy']
    for i in range(0, 6):
        temp = img_test.copy() * (1 + 2 * i / 10)
        temp['Number'] = f'Image {i}'
        for gg in groups:
            temp[gg] = 'dr crusher'
        img_all = pd.concat([img_all, temp])

    # 1 x 1
    if '1x1' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=0, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, filter='Number in ["Image 0"]')

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=False)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 100, 250, 1, alias=False)
            # Label width
            utl.unit_test_measure_margin(name, 45, None, left=10, right=10, alias=False)
            # Title margin
            utl.unit_test_measure_margin(name, 13, None, left=10, right=10, alias=False)
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 2
    if '1x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x2', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=3, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', title_wrap_edge_width=3,
                   filename=name.with_suffix('.png'), save=not bm, inline=False, label_wrap_edge_width=7,
                   filter='Number in ["Image 0", "Image 5"]', ws_col=20)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 256, 2, alias=True)
            # Label width
            utl.unit_test_measure_axes_cols(name, 58, 256, 2, alias=True)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=True)
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 3
    if '1x3' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x3', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=2, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, title_wrap_edge_width=4,
                   filter='Number in ["Image 0", "Image 2", "Image 4"]', label_wrap_edge_width=3)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 758, 1, alias=True)
            # Label width
            utl.unit_test_measure_axes_cols(name, 58, 758, 1, alias=True)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=True)
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 2 x 3
    if '2x3' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_2x3', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, title_wrap_edge_width=3)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 754, 1, alias=True)
            # Axes + label + title height
            utl.unit_test_measure_axes_rows(name, 300, 250 * 2 + 4 + 2 * 30 + 30 + 6, 1, alias=True)
            # Label width
            utl.unit_test_measure_axes_cols(name, 50, 754, 1, alias=False)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=True)
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 1
    if '3x1' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_3x1', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
                   ax_edge_width=2, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 2", "Image 4"]')
        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, alias=True)
            # Label margin
            utl.unit_test_measure_margin(name, 45, None, left=10, right=10, alias=False)
            # Title margin
            utl.unit_test_measure_margin(name, 13, None, left=10, right=10, alias=False)
            # Axes + label + title height
            utl.unit_test_measure_axes_rows(name, 70, 250 * 3 + 2 * 6 + 3 * 30 + 30 - 1, 1, alias=True)
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 2
    if '3x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_3x2', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
                   ax_edge_width=4, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['inferno', 'gray'],
                   share_col=True, label_wrap_edge_width=3, title_wrap_edge_width=2)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, top=10, bottom=10, alias=True)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 250 * 2 + 4 * 3, 1, alias=True)
            # Axes + label + title height
            utl.unit_test_measure_axes_rows(name, 300, 250 * 3 + 4 * 6 + 3 * 30 + 6 * 3 + 30 + 4, 1, alias=True)
            # Label width
            utl.unit_test_measure_axes_cols(name, 50, 250 * 2 + 4 * 3, 1, alias=True)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 4 x 2
    if '4x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_4x2', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
                   ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['gray', 'inferno'],
                   share_row=True)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, bottom=10, alias=True)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 250 * 4 + 5, 1, alias=True)
            # Axes + label + title height
            utl.unit_test_measure_axes_rows(name, 300, 250 * 2 + 4 + 2 * 30 + 30 - 1, 1, alias=True)
            # Label width
            utl.unit_test_measure_axes_cols(name, 42, 250 * 4 + 5, 1, alias=False)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    if '4x2b' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_4x2_no_edge', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
                   ax_edge_width=0, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff',
                   filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['gray', 'inferno'],
                   share_row=True)

        if not show:
            # Axes margin
            utl.unit_test_measure_margin(name, 150, 150, left=10, right=10, bottom=10, alias=False)
            # Axes width
            utl.unit_test_measure_axes_cols(name, 175, 250 * 4, 1, alias=False)
            # Axes + label + title height
            utl.unit_test_measure_axes_rows(name, 300, 250 * 2 + 2 * 30 + 30, 1, alias=False)
            # Label width
            utl.unit_test_measure_axes_cols(name, 42, 250 * 4, 1, alias=False)
            # Title margin
            utl.unit_test_measure_margin(name, 17, None, left=10, right=10, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_combos_cbar(bm=False, make_reference=False, show=False):

    enabled = ['1x1', '1x2', '1x3', '2x3', '3x1', '3x2', '4x2']
    # enabled = ['3x2']

    img_all = pd.DataFrame()
    img_test = pd.DataFrame(utl.img_grayscale_deprecated(img_cat_orig).to_numpy()[300:600, 800:1100])
    for i in range(0, 6):
        temp = img_test.copy() * (1 + 2 * i / 10)
        temp['Number'] = f'Image {i}'
        img_all = pd.concat([img_all, temp])

    # 1 x 1
    if '1x1' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x1_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=7, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), title_wrap_edge_width=2,
                   label_z_edge_width=1, label_z_edge_color='#555555', save=not bm, inline=False,
                   filter='Number in ["Image 0"]')

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 150, None, left=10, alias=True)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=10, alias=False)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=10, top=10, bottom=10, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 2
    if '1x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x2_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=5, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 5"]', tick_labels_major=True)

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 100, None, left=44, alias=True)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=44, alias=False)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=44, top=10, alias=False)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 42, 260, 2, alias=False)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 100, 260, 2, cbar=True, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 1 x 3
    if '1x3' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_1x3_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=3, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False, title_wrap_edge_width=4,
                   filter='Number in ["Image 0", "Image 2", "Image 4"]', label_wrap_edge_width=3)

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 100, None, left=10, alias=True)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=10, alias=True)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=10, top=10, alias=True)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 55, 256, 3, alias=True)
            # Axes widths
            utl.unit_test_measure_axes_cols(name, 150, 256, 3, cbar=True, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 2 x 3
    if '2x3' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_2x3_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=3, title_wrap_edge_color='aa00ff',
                   ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False, zmin=[50, 100], zmax=[200, 300],
                   label_edge_width=2, title_wrap_edge_width=3)

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 150, None, left=10, alias=True)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=10, right=122, alias=False)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=10, top=10, bottom=10, right=122, alias=True)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 50, 252, 3, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 1
    if '3x1' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_3x1_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=1, title_wrap_edge_color='aa00ff',
                   ax_edge_width=2, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False,
                   filter='Number in ["Image 0", "Image 2", "Image 4"]')
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 3 x 2
    if '3x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_3x2_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=2, title_wrap_edge_color='aa00ff',
                   ax_edge_width=1, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['inferno', 'gray'],
                   share_col=True, label_wrap_edge_width=3, title_wrap_edge_width=2)

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 150, None, left=10, alias=True)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=10, right=120, alias=True)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=10, top=10, bottom=10, right=120, alias=True)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 50, 252, 2, alias=True)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    # 4 x 2
    if '4x2' in enabled:
        name = utl.unit_test_get_img_name('wrap_combos_4x2_cbar', make_reference, REFERENCE)
        fcp.imshow(img_all, ax_size=[250, 250], wrap='Number', ncol=4, title_wrap_edge_color='aa00ff',
                   ax_edge_width=0, ax_edge_color='#ff0000', label_wrap_edge_color='#0000ff', cbar=True,
                   filename=name.with_suffix('.png'), save=not bm, inline=False, cmap=['gray', 'inferno'],
                   share_row=True, title_wrap_edge_width=0, label_wrap_edge_width=0)

        if not show:
            # Axes
            utl.unit_test_measure_margin(name, 150, None, left=10, alias=False)
            # Label
            utl.unit_test_measure_margin(name, 50, None, left=10, right=120, alias=False)
            # Title
            utl.unit_test_measure_margin(name, 25, 150, left=10, top=10, bottom=10, right=120, alias=False)
            # Label widths
            utl.unit_test_measure_axes_cols(name, 42, 250, 4, alias=False)

        return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_one(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap_one', make_reference, REFERENCE)

    fcp.imshow(img_cp_orig, cmap='inferno', ax_size=[300, 300], cfa='rggb', filter='Plane=="gb"',
               label_wrap_edge_color='#0000ff', filename=name.with_suffix('.png'), save=not bm, inline=False,
               wrap='Plane')

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_ws(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap_ws', make_reference, REFERENCE)

    df = pd.DataFrame({'alpha_blend': [1, 10, 50], 'gamma': [1, 2.2, 1.8]}, index=['img0', 'img1', 'img2'])
    imgs = {'img0': img_color_bars,
            'img1': (1.75 * img_color_bars).astype(np.uint8),
            'img2': (0.15 * img_color_bars).astype(np.uint8)}

    fcp.imshow(df, imgs=imgs, wrap=['gamma', 'alpha_blend'], ncol=3, ws_col=15, filename=name.with_suffix('.png'),
               save=not bm, inline=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', left=10, right=10, top=10, bottom=10, alias=False)
        utl.unit_test_measure_axes_cols(name, 175, 400, 3, alias=False, ws=15)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


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


def test_imshow_tick_and_axes_labels(benchmark):
    plt_imshow_tick_and_axes_labels()
    benchmark(plt_imshow_tick_and_axes_labels, True)


def test_imshow_stretched(benchmark):
    plt_imshow_stretched()
    benchmark(plt_imshow_stretched, True)


def test_imshow_zoomed(benchmark):
    plt_imshow_zoomed()
    benchmark(plt_imshow_zoomed, True)


def test_col(benchmark):
    plt_col()
    benchmark(plt_col, True)


def test_col_cbar(benchmark):
    plt_col_cbar()
    benchmark(plt_col_cbar, True)


def test_col_shared_cbar(benchmark):
    plt_col_shared_cbar()
    benchmark(plt_col_shared_cbar, True)


def test_col_share_z(benchmark):
    plt_share_col()
    benchmark(plt_col_share_z, True)


def test_col_z_user_range(benchmark):
    plt_col_z_user_range()
    benchmark(plt_col_z_user_range, True)


def test_col_combos():
    plt_col_combos()


def test_row_combos():
    plt_row_combos()


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
    # BAD DATA INPUT FOR ARRAY VS DF
    with pytest.raises(data.AxisError):
        fcp.imshow(img_test, twin_x=True)
    with pytest.raises(data.AxisError):
        fcp.imshow(img_test, twin_y=True)
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_test, row='y')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_test, wrap='y')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_test, col='x')
    with pytest.raises(data.GroupingError):
        fcp.imshow(img_test, legend=True)


def test_wrap_combos(benchmark):
    plt_wrap_combos()
    # benchmark(plt_wrap_combos, True)


def test_wrap_combos_cbar():
    plt_wrap_combos_cbar()


if __name__ == '__main__':
    pass
