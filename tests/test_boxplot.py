import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import numpy as np
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'boxplot'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
df2 = df.copy()
df2.loc[df2.Region == 'Alpha123', 'Region'] = 5 * 'Alpha123'
df2['Lots of Values'] = df2.index % 2 + df2.index
seaborn_url = r'https://raw.githubusercontent.com/mwaskom/seaborn-data/master'
df_crash = pd.read_csv(seaborn_url + '/car_crashes.csv')

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
def plt_grand_means(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grand_means', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, grand_mean=True, grand_median=True,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_column(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_column', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], col='Region', show=SHOW, ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 59, 302, 2)
        utl.unit_test_measure_margin(name, 59, 120, left=74, right=81, bottom=10, alias=True)
        utl.unit_test_measure_margin(name, 59, 120, top=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_row', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], row='Region', show=SHOW, ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_rows(name, 260, 302, 2)
        utl.unit_test_measure_margin(name, 80, 120, left=74, top=10, bottom=10, alias=True)
        utl.unit_test_measure_margin(name, 80, 120, right=41, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Sample', 'Region'], wrap='Batch', show=SHOW, ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_wrap_y(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_y', make_reference, REFERENCE)

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    fcp.boxplot(df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_grid_wrap_y_no_share(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid_y-no-share', make_reference, REFERENCE)

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    fcp.boxplot(df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300], share_y=False,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_auto_size(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_auto_size', make_reference, REFERENCE)

    # Make the plot
    df2 = df.copy()
    df2.Value *= 2
    df2.loc[df2.Sample == 1, 'Sample'] = 4
    df2.loc[df2.Sample == 2, 'Sample'] = 5
    df2.loc[df2.Sample == 3, 'Sample'] = 6
    df3 = df.copy()
    df3.Value *= 3
    df3.loc[df3.Sample == 1, 'Sample'] = 7
    df3.loc[df3.Sample == 2, 'Sample'] = 8
    df3.loc[df3.Sample == 3, 'Sample'] = 9
    df4 = df.copy()
    df4.Value *= 4
    df4.loc[df4.Sample == 1, 'Sample'] = 10
    df4.loc[df4.Sample == 2, 'Sample'] = 11
    df4 = pd.concat([df4, df3, df2, df])
    fcp.boxplot(df4, y='Value', groups=['Batch', 'ID', 'Sample'],  ax_size='auto', label_y_fill_color='#ff0000',
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_auto_size_90(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_auto_size_90', make_reference, REFERENCE)

    # Make the plot
    df2 = df.copy()
    df2.Value *= 2
    df2.loc[df2.Sample == 1, 'Sample'] = 4
    df2.loc[df2.Sample == 2, 'Sample'] = 5
    df2.loc[df2.Sample == 3, 'Sample'] = 6
    df3 = df.copy()
    df3.Value *= 3
    df3.loc[df3.Sample == 1, 'Sample'] = 7
    df3.loc[df3.Sample == 2, 'Sample'] = 8
    df3.loc[df3.Sample == 3, 'Sample'] = 9
    df4 = df.copy()
    df4.Value *= 4
    df4.loc[df4.Sample == 1, 'Sample'] = 10
    df4.loc[df4.Sample == 2, 'Sample'] = 11
    df4 = pd.concat([df4, df3, df2, df])
    fcp.boxplot(df4, y='Value', groups=['Batch', 'ID', 'Sample'],  ax_size='auto', label_y_fill_color='#ff0000',
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False,
                box_group_label_rotation=90)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_auto_size_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_auto_size_wrap', make_reference, REFERENCE)

    # Make the plot
    df2 = df.copy()
    df2.Value *= 2
    df2.loc[df2.Sample == 1, 'Sample'] = 4
    df2.loc[df2.Sample == 2, 'Sample'] = 5
    df2.loc[df2.Sample == 3, 'Sample'] = 6
    df3 = df.copy()
    df3.Value *= 3
    df3.loc[df3.Sample == 1, 'Sample'] = 7
    df3.loc[df3.Sample == 2, 'Sample'] = 8
    df3.loc[df3.Sample == 3, 'Sample'] = 9
    df4 = df.copy()
    df4.Value *= 4
    df4.loc[df4.Sample == 1, 'Sample'] = 10
    df4.loc[df4.Sample == 2, 'Sample'] = 11
    df4 = pd.concat([df4, df3, df2, df])
    fcp.boxplot(df4, y='Value', wrap='Batch', groups=['ID', 'Sample'],  ax_size='auto',
                label_wrap_fill_color='#ff0000', label_wrap_edge_color='#0000ff', label_wrap_edge_width=5,
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False,
                ax_edge_width=7, ax_edge_color='#03333d', filter='Batch != 106',
                title_wrap_edge_width=3, title_wrap_edge_color='#afa500')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_auto_size_crash_simple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_auto_size_crash_simple', make_reference, REFERENCE)

    # Make the plot
    cc = pd.melt(df_crash, id_vars='abbrev', value_vars=['speeding', 'alcohol', 'not_distracted', 'no_previous'],
                 var_name='cause', value_name='accidents')
    fcp.boxplot(cc, y='accidents', groups='cause', ax_size='auto',
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_auto_size_crash_complex(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_auto_size_crash_complex', make_reference, REFERENCE)

    # Make the plot
    cc = pd.melt(df_crash, id_vars='abbrev', value_vars=['speeding', 'alcohol', 'not_distracted', 'no_previous'],
                 var_name='cause', value_name='accidents')
    fcp.boxplot(cc, y='accidents', groups=['abbrev'], row='cause', label_y_fill_color='#ff0000', share_y=False,
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, ax_size='auto')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], legend='Region', show=SHOW,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_legend_lots(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_legend_lots', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df2, y='Value', groups=['Batch', 'Sample'], legend='Lots of Values', show=SHOW,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, legend_edge_color='#000000')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_long(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_long', make_reference, REFERENCE)

    # Make the plot
    df2 = df.copy()
    df2['This is a really long way to say show me your ID sucka'] = df['ID'] * 2
    fcp.boxplot(df2, y='Value', groups=['Batch', 'This is a really long way to say show me your ID sucka', 'Sample'],
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_means(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_means', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, group_means=True,
                ax_edge_width=1, box_group_label_edge_width=1,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 20, 300, left=74, right=81, bottom=10, top=10, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_multiple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_multiple', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample', 'Region'], show=SHOW,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_multiple_nan(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_multiple_nan', make_reference, REFERENCE)

    # Make the plot
    df['Test'] = np.nan
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample', 'Test'], show=SHOW,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False, box_stat_line='q0.5')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_group_single(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('group_single', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups='Batch', show=SHOW, box_whisker=False, box_group_title_font_size=24,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False,
                box_group_label_edge_color='#0000ff')

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 50, 190, left=74, bottom=10, top=10, right=106, alias=True)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_mean_diamonds(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('mean_diamonds', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, mean_diamonds=True, conf_coeff=0.95,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_mean_diamonds_filled(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('mean_diamonds_filled', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch'], show=SHOW, mean_diamonds=True, conf_coeff=0.95,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False,
                mean_diamonds_fill_color='#ff0000', mean_diamonds_edge_color='#0000ff')

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_one_group(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('one_group', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', ymin='q0', ymax='q100',
                show=SHOW, filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False,
                box_stat_line='q50', box_group_label_font_size=24)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return

    if not show:
        utl.unit_test_measure_margin(name, 20, 150, left=74, bottom=10, top=10, right=10, alias=True)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple_groups_none(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple_groups_none', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True, groups=None,
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_simple_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('simple_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True, legend='Batch',
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_violin(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('violin', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_violin_styled(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('violin_styled', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                violin_fill_color='#eaef1a', violin_fill_alpha=1, violin_edge_color='#555555', violin_edge_width=2,
                violin_box_color='#ffffff', violin_whisker_color='#ff0000',
                violin_median_marker='+', violin_median_color='#00ffff', violin_median_size=10,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_violin_box_off(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('violin_box_off', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                violin=True, violin_box_on=False, violin_markers=True, jitter=False,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_stat_mean(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('stat_mean', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='mean', ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_stat_median(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('stat_median', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='median', ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_stat_std_dev(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('stat_std-dev', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='std', ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_dividers(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('dividers', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_divider=False, ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_range_lines(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('range_lines', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_range_lines=False, ax_size=[300, 300],
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_simple(benchmark):
    plt_simple()
    benchmark(plt_simple, True)


def test_simple_groups_none(benchmark):
    plt_simple_groups_none()
    benchmark(plt_simple_groups_none, True)


def test_simple_legend(benchmark):
    plt_simple_legend()
    benchmark(plt_simple_legend, True)


def test_one_group(benchmark):
    plt_one_group()
    benchmark(plt_one_group, True)


def test_group_single(benchmark):
    plt_group_single()
    benchmark(plt_group_single, True)


def test_group_multiple(benchmark):
    plt_group_multiple()
    benchmark(plt_group_multiple, True)


def test_group_multiple_nan(benchmark):
    plt_group_multiple_nan()
    benchmark(plt_group_multiple_nan, True)


def test_group_legend(benchmark):
    plt_group_legend()
    benchmark(plt_group_legend, True)


def test_group_legend_lots(benchmark):
    plt_group_legend_lots()
    benchmark(plt_group_legend_lots, True)


def test_group_auto_size(benchmark):
    plt_group_auto_size()
    benchmark(plt_group_auto_size, True)


def test_group_auto_size_90(benchmark):
    plt_group_auto_size_90()
    benchmark(plt_group_auto_size_90, True)


def test_group_auto_size_wrap(benchmark):
    plt_group_auto_size_wrap()
    benchmark(plt_group_auto_size_wrap, True)


def test_group_auto_size_crash_simple(benchmark):
    plt_group_auto_size_crash_simple()
    benchmark(plt_group_auto_size_crash_simple, True)


def test_group_auto_size_crash_complex(benchmark):
    plt_group_auto_size_crash_complex()
    benchmark(plt_group_auto_size_crash_complex, True)


def test_group_long(benchmark):
    plt_group_long()
    benchmark(plt_group_long, True)


def test_grid_column(benchmark):
    plt_grid_column()
    benchmark(plt_grid_column, True)


def test_grid_row(benchmark):
    plt_grid_row()
    benchmark(plt_grid_row, True)


def test_grid_wrap(benchmark):
    plt_grid_wrap()
    benchmark(plt_grid_wrap, True)


def test_grid_wrap_y(benchmark):
    plt_grid_wrap_y()
    benchmark(plt_grid_wrap_y, True)


def test_grid_wrap_y_no_share(benchmark):
    plt_grid_wrap_y_no_share()
    benchmark(plt_grid_wrap_y_no_share, True)


def test_grand_means(benchmark):
    plt_grand_means()
    benchmark(plt_grand_means, True)


def test_group_means(benchmark):
    plt_group_means()
    benchmark(plt_group_means, True)


def test_mean_diamonds(benchmark):
    plt_mean_diamonds()
    benchmark(plt_mean_diamonds, True)


def test_mean_diamonds_filled(benchmark):
    plt_mean_diamonds_filled()
    benchmark(plt_mean_diamonds_filled, True)


def test_violin(benchmark):
    plt_violin()
    benchmark(plt_violin, True)


def test_violin_styled(benchmark):
    plt_violin_styled()
    benchmark(plt_violin_styled, True)


def test_violin_box_off(benchmark):
    plt_violin_box_off()
    benchmark(plt_violin_box_off, True)


def test_stat_mean(benchmark):
    plt_stat_mean()
    benchmark(plt_stat_mean, True)


def test_stat_median(benchmark):
    plt_stat_median()
    benchmark(plt_stat_median, True)


def test_stat_std_dev(benchmark):
    plt_stat_std_dev()
    benchmark(plt_stat_std_dev, True)


def test_dividers(benchmark):
    plt_dividers()
    benchmark(plt_dividers, True)


def test_range_lines(benchmark):
    plt_range_lines()
    benchmark(plt_range_lines, True)


if __name__ == '__main__':
    pass
