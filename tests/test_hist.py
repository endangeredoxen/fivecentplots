import imageio
import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import numpy as np
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
osjoin = os.path.join
db = pdb.set_trace

if Path(f'../tests/test_images/mpl_v{mpl.__version__}').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / 'hist.py'
else:
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / 'hist.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
url = 'https://imgs.michaels.com/MAM/assets/1/D730994AF28E498A909A1002BBF38107/img/16F309E5F1CF4742B4AACD8E0CCF08E0/D203087S_1.jpg?fit=inside|1024:1024'  # noqa
imgr = imageio.imread(url)

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


def show_all():
    """
    Remake all test master images
    """

    if not MASTER.exists():
        os.makedirs(MASTER)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


# plt_ functions can be used directly outside of pytest for debug
def plt_simple(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'simple_master') if master else 'simple'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW,
             inline=False, save=not bm, filename=name + '.png')

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


def plt_horizontal(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'horizontal_master') if master else 'horizontal'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, horizontal=True,
             inline=False, save=not bm, filename=name + '.png')

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


def plt_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region',
             inline=False, save=not bm, filename=name + '.png')

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


def plt_kde(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'kde_master') if master else 'kde'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', kde=True, kde_width=2,
             inline=False, save=not bm, filename=name + '.png')

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


def plt_grid(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_master') if master else 'grid'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
             inline=False, save=not bm, filename=name + '.png')

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


def plt_wrap_values(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_values_master') if master else 'wrap_values'

    # Make the plot
    fcp.hist(df, x='Value', show=SHOW, legend='Region', wrap='Batch',
             ax_size=[250, 250], horizontal=True, inline=False, save=not bm, filename=name + '.png')

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


def plt_wrap_names(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_names_master') if master else 'wrap_names'

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    df['Value*3'] = 3 * df['Value']
    fcp.hist(df, x=['Value', 'Value*2', 'Value*3'], wrap='x', show=SHOW, ncol=3, ax_size=[250, 250],
             inline=False, save=not bm, filename=name + '.png')

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


def plt_image(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'image_master') if master else 'image'

    # Make the plot
    img = fcp.utilities.rgb2bayer(imgr, 'bbbb')
    dn = 255
    max_count = (imgr == dn).sum()
    fcp.hist(img, markers=False, ax_scale='logy', ax_size=[600, 400], line_width=2,
             show=SHOW, inline=False, save=not bm, filename=name + '.png', xmax=dn + 5,
             ax_hlines=max_count, ax_vlines=dn)

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


def plt_image_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'image_legend_master') if master else 'image_legend'

    # Make the plot
    img = fcp.utilities.rgb2bayer(imgr, 'rggb')
    dnr = 180
    dng = 230
    max_count_r = (img.loc[::2, img.columns[::2]].stack().values == dnr).sum()
    max_count_gb = (img.loc[1::2, img.columns[::2]
                            ].stack().values == dng).sum()
    fcp.hist(img, show=SHOW, inline=False, save=not bm, filename=name + '.png',
             markers=False, ax_scale='logy', ax_size=[600, 400],
             legend='Plane', cfa='rggb', line_width=2, colors=fcp.BAYER,
             ax_hlines=[max_count_r, max_count_gb], ax_vlines=[dnr, dng])

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


def plt_patch_solid(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'patch_solid_master') if master else 'patch_solid'

    # Make the patch
    img_rgb = np.zeros([25, 25]).astype(np.uint8)
    img_rgb[::2, ::2] = 180  # green_red
    img_rgb[1::2, 1::2] = 180  # green_blue
    img_rgb[::2, 1::2] = 10  # red
    img_rgb[1::2, ::2] = 255  # blue

    fcp.hist(img_rgb, show=SHOW, inline=False, save=not bm, filename=name + '.png',
             markers=False, ax_scale='logy', ax_size=[600, 400], legend='Plane',
             cfa='grbg', line_width=2, xmin=-5, xmax=260, colors=fcp.BAYER)

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
def test_simple(benchmark):
    plt_simple()
    benchmark(plt_simple, True)


def test_horizontal(benchmark):
    plt_horizontal()
    benchmark(plt_horizontal, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


def test_kde(benchmark):
    plt_kde()
    benchmark(plt_kde, True)


def test_grid(benchmark):
    plt_grid()
    benchmark(plt_grid, True)


def test_wrap_values(benchmark):
    plt_wrap_values()
    benchmark(plt_wrap_values, True)


def test_wrap_names(benchmark):
    plt_wrap_names()
    benchmark(plt_wrap_names, True)


def test_image(benchmark):
    plt_image()
    benchmark(plt_image, True)


def test_image_legend(benchmark):
    plt_image_legend()
    benchmark(plt_image_legend, True)


def test_patch_solid(benchmark):
    plt_patch_solid()
    benchmark(plt_patch_solid, True)


if __name__ == '__main__':
    pass
