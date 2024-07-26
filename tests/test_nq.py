import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
import imageio.v3 as imageio

osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'imshow'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
img_cat_orig = imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')
img_cat = utl.img_grayscale(img_cat_orig)

# Set theme
fcp.set_theme('gray')

# Other
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def make_all():
    """
    Remake all test master images
    """

    if not REFERENCE.exists():
        os.makedirs(REFERENCE)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all(only_fails=True):
    """
    Remake all test master images
    """

    if not REFERENCE.exists():
        os.makedirs(REFERENCE)
    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'test_' in f[0]]
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


def test_nq(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat, filename=name.with_suffix('.png'))

    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_percentiles(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_percentiles', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat, percentiles=True, filename=name.with_suffix('.png'))

    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_legend(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_legend', make_reference, REFERENCE)

    img1 = img_cat.copy()
    img2 = img_cat.copy()
    img1['State'] = 'Original'
    img2.loc[:, :] /= 2
    img2['State'] = 'Half'
    img = pd.concat([img1, img2])

    # Make the plot
    fcp.nq(img, legend='State', filename=name.with_suffix('.png'))

    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_multiple_no_group(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_multiple_no_group', make_reference, REFERENCE)

    img1 = img_cat.copy()
    img2 = img_cat.copy()
    img1['State'] = 'Original'
    img2.loc[:, :] /= 2
    img2['State'] = 'Half'
    img = pd.concat([img1, img2])

    # Make the plot
    fcp.nq(img, filename=name.with_suffix('.png'))

    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_rgb(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat_orig, legend='Channel', colors=fcp.RGB, filename=name.with_suffix('.png'))

    utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
