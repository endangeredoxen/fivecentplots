import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
import platform
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace
if platform.system() != 'Windows':
    print('Warning!  Image test files generated in windows.  Compatibility with linux/mac may vary')

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests',
                'test_images', MPL, 'gantt.py')

df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__),
                 'tests', 'fake_data_gantt.csv'))


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

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


# plt_ functions can be used directly outside of pytest for debug
def plt_basic(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_master') if master else 'basic'

    # Make the plot
    fcp.gantt(df=df, x=['Start', 'Stop'], y='Task',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def plt_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_master') if master else 'legend'

    if bm:
        return

    # Make the plot
    fcp.gantt(df=df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

    # Compare with master
    if master:
        return
    elif show:
        utl.show_file(osjoin(MASTER, name + '_master.png'))
        utl.show_file(name + '.png')
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(
            name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_basic(benchmark):
    plt_basic()
    benchmark(plt_basic, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


if __name__ == '__main__':
    pass
