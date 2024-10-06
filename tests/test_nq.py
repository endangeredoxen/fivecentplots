import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
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
img_cat = utl.img_grayscale_deprecated(img_cat_orig)

# Set theme
fcp.set_theme('gray')


# Other
def make_all(start=None, stop=None):
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__], start=start, stop=stop)


def show_all(only_fails=True, start=None):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__], start=start)


SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def test_nq(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat, filename=name.with_suffix('.png'))

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_percentiles(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_percentiles', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat, percentiles=True, filename=name.with_suffix('.png'))

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


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

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


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

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_nq_rgb(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(img_cat_orig, legend='Channel', colors=fcp.RGB, filename=name.with_suffix('.png'))

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
