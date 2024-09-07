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
mpl.use('agg')

test = 'gantt'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'


df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt.csv')


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
def plt_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('basic', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_sort_ascending(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('sort_ascending', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', sort='ascending',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_style(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('style', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task',
              color_by_bar=True, gantt_edge_width=2, gantt_edge_color='#555555',
              gantt_height=0.2, gantt_fill_alpha=1,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend_order_by(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend_order_by', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              gantt_tick_labels_x_rotation=45, order_by_legend=True,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('row', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', col='Category', share_x=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_rc_missing(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('rc_missing', make_reference, REFERENCE)

    # Make the plot
    df['Temp'] = 'Boom'
    df.loc[5:, 'Temp'] = 'Boom2'
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category', col='Temp',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', wrap='Category',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)


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
