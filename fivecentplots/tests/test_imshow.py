import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb, platform
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace
if platform.system() != 'Windows':
    print('Warning!  Image test files generated in windows.  Compatibility with linux/mac may vary')

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'imshow.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_heatmap.csv'))

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')

# Other
SHOW = False

# Read an image
import imageio
url = 'https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&rect=0%2C214%2C2000%2C1214&poi=%5B920%2C546%5D&w=2000&h=1000&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F47%2F2020%2F10%2F07%2Fcat-in-pirate-costume-380541532-2000.jpg'
imgr = imageio.imread(url)

# Convert to a grayscale DataFrame
img = utl.img_grayscale(imgr)


def make_all():
    """
    Remake all test master images
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all():
    """
    Remake all test master images
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


def test_imshow(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_master') if master else 'imshow'

    # Make the plot
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600],
               filename=name + '.png', inline=False)

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


def test_imshow_no_cbar(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_no_cbar_master') if master else 'imshow_no_cbar'

    # Make the plot
    fcp.imshow(img, cmap='inferno', cbar=False, ax_size=[600, 600],
               filename=name + '.png', inline=False)

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


def test_imshow_stretched(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_stretched_master') if master else 'imshow_stretched'

    # Make the plot
    uu = img.stack().mean()
    ss = img.stack().std()
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600], zmin=uu-3*ss, zmax=uu+3*ss,
               filename=name + '.png', inline=False)

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


def test_imshow_zoomed(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'imshow_zoomed_master') if master else 'imshow_zoomed'

    # Make the plot
    fcp.imshow(img, cmap='inferno', cbar=True, ax_size=[600, 600], xmin=700, xmax=1100,
                ymin=300, ymax=400,
                filename=name + '.png', inline=False)

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


def test_wrap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_master') if master else 'wrap'
    
    # Get alt data
    url = 'https://upload.wikimedia.org/wikipedia/commons/2/28/RGB_illumination.jpg'
    imgr = imageio.imread(url)
    raw = fcp.utilities.rgb2bayer(imgr)
    
    # Make the plot
    fcp.imshow(raw, cmap='inferno', ax_size=[300, 300], cfa='rggb',
               filename=name + '.png', inline=False, wrap='Plane')

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


if __name__ == '__main__':
    pass