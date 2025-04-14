import pytest
import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.data as data
import fivecentplots.utilities as utl
import matplotlib as mpl
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'barplot'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/real_data_bar.csv')

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
def plt_col_shared(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col_shared', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, col='Measurement', row='T [C]',
            ax_hlines=0, ax_size=[300, 300], share_col=True, ax_edge_width=0,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return

    if not show:
        # Axis width
        utl.unit_test_measure_axes(name, 70, None, 300, None, 1, alias=False)
        # Col label height
        utl.unit_test_measure_axes(name, None, 130, None, 30, 1, alias=False)
        # Axis height
        utl.unit_test_measure_axes(name, None, 190, None, 300, 1, 45, alias=False)
        # Margins
        utl.unit_test_measure_margin(name, 170, 190, left=80, alias=False)
        utl.unit_test_measure_margin(name, 170, 190, right=10, top=10, bottom=158, alias=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_error(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('error', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, error_bars=True, ymin=0, ymax=35,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, sort=False, color_by='bar')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_horizontal(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('horizontal', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A"', horizontal=True, error_bars=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, legend='Measurement',
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, label_edge_width=10,
            label_edge_color='#000000', legend_font_size=6, legend_title_font_size=18)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_rolling_mean(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('rolling_mean', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df2, x='date', y='cases', show=SHOW, ax_size=[800, 500],
            tick_labels_major_x_rotation=90, rolling_mean=14, legend=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_rolling_mean_styled(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('rolling_mean_styled', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df2, x='date', y='cases', show=SHOW, ax_size=[800, 500],
            tick_labels_major_x_rotation=90, rolling_mean=14, bar_fill_color='#aaaaaa',
            rolling_mean_line_color='#000000', markers=True, marker_size=4,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('row_col', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, col='Measurement', row='T [C]',
            ax_hlines=0, ax_size=[300, 300],
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_shared(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('row_shared', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, col='Measurement', row='T [C]',
            ax_hlines=0, ax_size=[300, 300], share_row=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_stacked(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('stacked', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, stacked=True, legend='Measurement',
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_stacked_horizontal(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('stacked_horizontal', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, stacked=True, legend='Measurement', xmin=0, xmax=41,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, horizontal=True, width=0.3)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_vertical(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('vertical', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25',
            tick_labels_major_x_rotation=90,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_vertical_zero_group(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('vertical_zero_group', make_reference, REFERENCE)

    # Make the plot
    temp = pd.DataFrame({'Liquid': ['Air'], 'pH': [0], 'Measurement': ['A'],
                         'T [C]': [125]}, index=[30])
    fcp.bar(df=pd.concat([df, temp]), x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            tick_labels_major_x_rotation=90, show_all_groups=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, wrap='Measurement', ax_size=[300, 300],
            filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_vertical(benchmark):
    plt_vertical()
    benchmark(plt_vertical, True)


def test_horizontal(benchmark):
    plt_horizontal()
    benchmark(plt_horizontal, True)


def test_error(benchmark):
    plt_error()
    benchmark(plt_error, True)


def test_legend(benchmark):
    plt_legend()
    benchmark(plt_legend, True)


def test_stacked(benchmark):
    plt_stacked()
    benchmark(plt_stacked, True)


def test_stacked_horizontal(benchmark):
    plt_stacked_horizontal()
    benchmark(plt_stacked_horizontal, True)


def test_row_col(benchmark):
    plt_row_col()
    benchmark(plt_row_col, True)


def test_col_shared(benchmark):
    plt_col_shared()
    benchmark(plt_col_shared, True)


def test_row_shared(benchmark):
    plt_row_shared()
    benchmark(plt_row_shared, True)


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


def test_rolling_mean(benchmark):
    plt_rolling_mean()
    benchmark(plt_rolling_mean, True)


def test_rolling_mean_styled(benchmark):
    plt_rolling_mean_styled()
    benchmark(plt_rolling_mean_styled, True)


def test_vertical_zero_group(benchmark):
    plt_vertical_zero_group()
    benchmark(plt_vertical_zero_group, True)


def test_bad_filter():

    with pytest.raises(data.data.DataError):
        fcp.bar(df, x='Liquid', y='pH', filter='Measurement=="Q"')
