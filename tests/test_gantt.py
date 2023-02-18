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

test = 'gantt'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt.csv')


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


def show_all(only_fails=True):
    """
    Remake all test master images
    """

    if not MASTER.exists():
        os.makedirs(MASTER)
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
def plt_basic(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'basic_master') if master else 'basic'

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_sort_ascending(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'sort_ascending_master') if master else 'sort_ascending'

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', sort='ascending',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_style(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'style_master') if master else 'style'

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task',
              color_by_bar=True, gantt_edge_width=2, gantt_edge_color='#555555',
              gantt_height=0.2, gantt_fill_alpha=1,
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_legend_order_by(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_order_by_master') if master else 'legend_order_by'

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              gantt_tick_labels_x_rotation=45, order_by_legend=True,
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_row(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'row_master') if master else 'row'

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_col(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'col_master') if master else 'col'

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', col='Category', share_x=False,
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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


def plt_rc_missing(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'rc_missing_master') if master else 'rc_missing'

    # Make the plot
    df['Temp'] = 'Boom'
    df.loc[5:, 'Temp'] = 'Boom2'
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category', col='Temp',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', wrap='Category',
              filename=name + '.png', save=not bm, inline=False, ax_size=[600, 400])

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
def test_basic(benchmark):
    plt_basic()
    benchmark(plt_basic, True)


def test_sort_ascending(benchmark):
    plt_sort_ascending()
    benchmark(plt_sort_ascending, True)


def test_style(benchmark):
    plt_style()
    benchmark(plt_style, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


def test_legend_order_by(benchmark):
    plt_legend_order_by()
    benchmark(plt_legend_order_by, True)


def test_row(benchmark):
    plt_row()
    benchmark(plt_row, True)


def test_col(benchmark):
    plt_col()
    benchmark(plt_col, True)


def test_rc_missing(benchmark):
    plt_rc_missing()
    benchmark(plt_rc_missing, True)


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


if __name__ == '__main__':
    pass
