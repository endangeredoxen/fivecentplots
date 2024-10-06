import pytest
import imageio.v3 as imageio
import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import numpy as np
import fivecentplots.data.data as data
import fivecentplots.utilities as utl
import matplotlib as mpl
from io import StringIO
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'hist'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
img_patch = imageio.imread(Path(fcp.__file__).parent / 'test_data/hist_patch.png')
raw = imageio.imread(Path(fcp.__file__).parent / 'test_data/RAW.png')
img_cat_orig = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')
img_cat = utl.img_grayscale_deprecated(img_cat_orig)

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
def plt_grid(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
             inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_no_share(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_no_share', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
             inline=False, save=not bm, filename=name.with_suffix('.png'), share_y=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_share_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_share_col', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
             inline=False, save=not bm, filename=name.with_suffix('.png'), share_col=True)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_share_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_share_row', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
             inline=False, save=not bm, filename=name.with_suffix('.png'), share_row=True)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_horizontal(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('horizontal', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, horizontal=True,
             inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'bbbb')
    dn = 255
    max_count = (img_patch == dn).sum()
    fcp.hist(img, markers=False, ax_scale='logy', ax_size=[600, 400], line_width=2,
             show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'), xmax=dn + 5,
             ax_hlines=max_count, ax_vlines=dn)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_multiple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_multiple_int', make_reference, REFERENCE)

    # With int
    df_img = pd.DataFrame({'image': ['raw', 'raw_inv']}, index=['raw', 'raw_inv'])
    imgs = {'raw': raw, 'raw_inv': ((1 - raw / raw.max()) * raw.max()).astype(np.uint16)}
    fcp.hist(df_img, imgs=imgs, **fcp.HIST, legend='image', line_alpha=0.4,
             show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'))

    if not bm:
        return utl.unit_test_options(make_reference, show, name, REFERENCE)

    name = utl.unit_test_get_img_name('image_multiple_float', make_reference, REFERENCE)

    # With float
    temp_out = StringIO()
    sys.stdout = temp_out
    df_img = pd.DataFrame({'image': ['raw', 'raw_inv']}, index=['raw', 'raw_inv'])
    imgs = {'raw': raw, 'raw_inv': ((1 - raw / raw.max()) * raw.max())}
    fcp.hist(df_img, imgs=imgs, **fcp.HIST, legend='image', line_alpha=0.4,
             show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'))
    sys.stdout = sys.__stdout__

    assert 'Warning: histograms of image data with float values is slow' in temp_out.getvalue()

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_cdf(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_cdf', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'bbbb')
    dn = 255
    max_count = (img_patch == dn).sum()
    fcp.hist(img, cdf=True, **fcp.HIST, show=SHOW, inline=False, save=not bm,
             filename=name.with_suffix('.png'), xmax=dn + 5, ax_hlines=max_count, ax_vlines=dn)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_pdf(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_pdf', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'bbbb')
    dn = 255
    max_count = (img_patch == dn).sum()
    fcp.hist(img, pdf=True, **fcp.HIST, show=SHOW, inline=False, save=not bm,
             filename=name.with_suffix('.png'), xmax=dn + 5, ax_hlines=max_count, ax_vlines=dn)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_legend', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'rggb')
    dnr = 183
    dng = 226
    max_count_r = np.unique(img[::2, ::2], return_counts=True)[1].max()
    max_count_gb = np.unique(img[::2, 1::2], return_counts=True)[1].max()

    fcp.hist(img, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_scale='logy', ax_size=[600, 400],
             legend='Plane', cfa='rggb', line_width=2, colors=fcp.RGGB,
             ax_hlines=[max_count_r, max_count_gb], ax_vlines=[dnr, dng])

    if bm:
        return
    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', right=96, top=10, alias=True)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_legend_cdf(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_legend_cdf', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'rggb')
    dnr = 183
    dng = 226
    max_count_r = np.unique(img[::2, ::2], return_counts=True)[1].max()
    max_count_gb = np.unique(img[::2, 1::2], return_counts=True)[1].max()

    fcp.hist(img, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'), cdf=True,
             markers=False, ax_scale='logy', ax_size=[600, 400],
             legend='Plane', cfa='rggb', line_width=2, colors=fcp.RGGB,
             ax_hlines=[max_count_r, max_count_gb], ax_vlines=[dnr, dng])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_legend_pdf(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_legend_pdf', make_reference, REFERENCE)

    # Make the plot
    img = fcp.utilities.rgb2bayer(img_patch, 'rggb')
    dnr = 183
    dng = 226
    max_count_r = np.unique(img[::2, ::2], return_counts=True)[1].max()
    max_count_gb = np.unique(img[::2, 1::2], return_counts=True)[1].max()

    fcp.hist(img, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_scale='logy', ax_size=[600, 400],
             legend='Plane', cfa='rggb', line_width=2, colors=fcp.RGGB,
             ax_hlines=[max_count_r, max_count_gb], ax_vlines=[dnr, dng])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_image_rgb(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('image_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(img_cat_orig, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_size=[600, 400], legend='Channel', ax_scale='logy',
             line_width=2, colors=fcp.RGB)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_kde(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('kde', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', kde=True, kde_width=2,
             inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_kde_horizontal(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('kde_horizontal', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', kde=True, kde_width=2,
             inline=False, save=not bm, filename=name.with_suffix('.png'), horizontal=True,)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region',
             inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_patch_single(bm=False, make_reference=False, show=False):
    # TODO: deal with tick label near right side (too much?)
    name = utl.unit_test_get_img_name('patch_single', make_reference, REFERENCE)

    # Make the patch
    img_rgb = np.ones([25, 25]).astype(np.uint8)
    fcp.hist(img_rgb, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_size=[600, 400], line_width=2)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_patch_single_log(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('patch_single_log', make_reference, REFERENCE)

    # Make the patch
    img_rgb = np.ones([25, 25]).astype(np.uint8)
    fcp.hist(img_rgb, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_scale='logy', ax_size=[600, 400], line_width=2)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_patch_solid(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('patch_solid', make_reference, REFERENCE)

    # Make the patch
    img_rgb = np.zeros([25, 25]).astype(np.uint8)
    img_rgb[::2, ::2] = 180  # green_red
    img_rgb[1::2, 1::2] = 180  # green_blue
    img_rgb[::2, 1::2] = 10  # red
    img_rgb[1::2, ::2] = 255  # blue

    fcp.hist(img_rgb, show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'),
             markers=False, ax_scale='logy', ax_size=[600, 400], legend='Plane',
             cfa='grbg', line_width=2, xmin=-5, xmax=260, colors=fcp.RGGB)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    if not show:
        utl.unit_test_measure_margin(name, 'c', 'c', right=10, top=10, alias=True)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple_cdf_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple_cdf_row', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', row='Region', cdf=True, **fcp.HIST, show=SHOW, inline=False, save=not bm,
             filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple_cdf_row_shared(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple_cdf_row_shared', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', row='Region', cdf=True, **fcp.HIST, show=SHOW, inline=False, save=not bm,
             filename=name.with_suffix('.png'), share_row=True, xmax=10)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple_no_bars(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple_no_bars', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, inline=False, save=not bm, filename=name.with_suffix('.png'), bars=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_values(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap_values', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', wrap='Batch',
             ax_size=[250, 250], horizontal=True, inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap_names(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap_names', make_reference, REFERENCE)

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    df['Value*3'] = 3 * df['Value']
    fcp.hist(df, x=['Value', 'Value*2', 'Value*3'], wrap='x', show=SHOW, ncol=3, ax_size=[250, 250],
             inline=False, save=not bm, filename=name.with_suffix('.png'))

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_simple(benchmark):
    plt_simple()
    benchmark(plt_simple, True)


def test_simple_no_bars(benchmark):
    plt_simple_no_bars()
    benchmark(plt_simple_no_bars, True)


def test_simple_cdf_row(benchmark):
    plt_simple_cdf_row()
    benchmark(plt_simple_cdf_row, True)


def test_horizontal(benchmark):
    plt_horizontal()
    benchmark(plt_horizontal, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


def test_kde(benchmark):
    plt_kde()
    benchmark(plt_kde, True)


def test_kde_horizontal(benchmark):
    plt_kde_horizontal()
    benchmark(plt_kde_horizontal, True)


def test_grid(benchmark):
    plt_grid()
    benchmark(plt_grid, True)


def test_grid_no_share(benchmark):
    plt_grid_no_share()
    benchmark(plt_grid_no_share, True)


def test_grid_share_col(benchmark):
    plt_grid_share_col()
    benchmark(plt_grid_share_col, True)


def test_grid_share_row(benchmark):
    plt_grid_share_row()
    benchmark(plt_grid_share_row, True)


def test_wrap_values(benchmark):
    plt_wrap_values()
    benchmark(plt_wrap_values, True)


def test_wrap_names(benchmark):
    plt_wrap_names()
    benchmark(plt_wrap_names, True)


def test_image(benchmark):
    plt_image()
    benchmark(plt_image, True)


def test_image_multiple(benchmark):
    plt_image_multiple()
    benchmark(plt_image, True)


def test_image_cdf(benchmark):
    plt_image_cdf()
    benchmark(plt_image_cdf, True)


def test_image_pdf(benchmark):
    plt_image_pdf()
    benchmark(plt_image_pdf, True)


def test_image_legend(benchmark):
    plt_image_legend()
    benchmark(plt_image_legend, True)


def test_image_legend_cdf(benchmark):
    plt_image_legend_cdf()
    benchmark(plt_image_legend_cdf, True)


def test_image_legend_pdf(benchmark):
    plt_image_legend_pdf()
    benchmark(plt_image_legend_pdf, True)


def test_patch_single(benchmark):
    plt_patch_single()
    benchmark(plt_patch_single, True)


def test_patch_single_log(benchmark):
    plt_patch_single_log()
    benchmark(plt_patch_single_log, True)


def test_patch_solid(benchmark):
    plt_patch_solid()
    benchmark(plt_patch_solid, True)


def test_invalid():

    with pytest.raises(data.GroupingError):
        df['Value*2'] = 2 * df.Value
        df['Value*3'] = 3 * df.Value
        fcp.hist(df, x=['Value', 'Value*2', 'Value*3'], wrap='y', ncol=3, ax_size=[250, 250])


if __name__ == '__main__':
    pass
