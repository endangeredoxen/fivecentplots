
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
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'hist.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))
import imageio
url = 'https://imgs.michaels.com/MAM/assets/1/D730994AF28E498A909A1002BBF38107/img/16F309E5F1CF4742B4AACD8E0CCF08E0/D203087S_1.jpg?fit=inside|1024:1024'
imgr = imageio.imread(url)

# Set theme
fcp.set_theme('gray')
# fcp.set_theme('white')

# Other
SHOW = False


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


def test_simple(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'simple_master') if master else 'simple'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW,
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


def test_horizontal(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'horizontal_master') if master else 'horizontal'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW, horizontal=True,
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


def test_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW, legend='Region',
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


def test_kde(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'kde_master') if master else 'kde'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW, legend='Region', kde=True, kde_width=2,
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


def test_grid(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_master') if master else 'grid'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW, legend='Region', col='Batch', row='Sample', ax_size=[250, 250],
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


def test_wrap_values(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_values_master') if master else 'wrap_values'

    # Make the plot
    fcp.hist(df=df, x='Value', show=SHOW, legend='Region', wrap='Batch', ax_size=[250, 250], horizontal=True,
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


def test_wrap_names(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'wrap_names_master') if master else 'wrap_names'

    # Make the plot
    df['Value*2'] = 2*df['Value']
    df['Value*3'] = 3*df['Value']
    fcp.hist(df=df, x=['Value', 'Value*2', 'Value*3'], wrap='x', show=SHOW, ncol=3, ax_size=[250, 250],
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


def test_image(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'image_master') if master else 'image'

    # Make the plot
    img = fcp.utilities.rgb2bayer(imgr, 'bbbb')
    fcp.hist(img, markers=False, ax_scale='logy', ax_size=[600, 400], line_width=2,
             show=SHOW, filename=name + '.png', inline=False)

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


def test_image_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'image_legend_master') if master else 'image_legend'

    # Make the plot
    img = fcp.utilities.rgb2bayer(imgr, 'rggb')
    fcp.hist(img, show=SHOW, filename=name + '.png', inline=False,
             markers=False, ax_scale='logy', ax_size=[600, 400],
             legend='Plane', cfa='rggb', line_width=2, colors=fcp.BAYER)

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