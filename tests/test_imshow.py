import imageio
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

test = 'imshow'
if Path(f'../tests/test_image').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path(f'tests/test_image').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')

# Other
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False

# Read an image
url = 'https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&rect=0%2C214%2C2000%2C1214&poi=%5B920%2C546%5D&w=2000&h=1000&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F47%2F2020%2F10%2F07%2Fcat-in-pirate-costume-380541532-2000.jpg'  # noqa
imgr = imageio.imread(url)

# Convert to a grayscale DataFrame
img = utl.img_grayscale(imgr)


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
    Run the show=True option on all plt functions
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
def plt_imshow(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_master') if master else 'imshow'

    # Make the plot
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600],
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
    fcp.imshow(img, cmap='inferno', cbar=False, ax_size=[600, 600],
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
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600], tick_labels_major=True,
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
    uu = img.stack().mean()
    ss = img.stack().std()
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600],
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


def plt_imshow_zoomed(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_zoomed_master') if master else 'imshow_zoomed'

    # Make the plot
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=700, xmax=1100,
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


def plt_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_master') if master else 'wrap'

    # Get alt data
    url = 'https://upload.wikimedia.org/wikipedia/commons/2/28/RGB_illumination.jpg'
    imgr = imageio.imread(url)
    raw = fcp.utilities.rgb2bayer(imgr)

    # Make the plot
    fcp.imshow(raw, cmap='inferno', ax_size=[300, 300], cfa='rggb',
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

    # Get alt data
    url = 'https://upload.wikimedia.org/wikipedia/commons/2/28/RGB_illumination.jpg'
    imgr = imageio.imread(url)
    raw = fcp.utilities.rgb2bayer(imgr)

    # Make the plot
    fcp.imshow(raw, cmap='inferno', ax_size=[300, 300], cfa='rggb', filter='Plane=="gb"',
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


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_imshow(benchmark):
    plt_imshow()
    benchmark(plt_imshow, True)


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


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


def test_wrap_one(benchmark):
    plt_wrap_one()
    benchmark(plt_wrap_one, True)


if __name__ == '__main__':
    pass
