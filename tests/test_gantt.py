import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
import datetime
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import pytest
import fivecentplots.data.data as data
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
df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_gantt_milestone.csv')


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
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('col', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', col='Category', share_x=False, share_y=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',  order_by_legend=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_legend_order_by(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('legend_order_by', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', legend='Assigned',
              gantt_tick_labels_x_rotation=45, order_by_legend=True,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_rc_missing(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('rc_missing', make_reference, REFERENCE)

    # Make the plot
    df['Temp'] = 'Boom'
    df.loc[5:, 'Temp'] = 'Boom2'
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category', col='Temp', share_y=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('row', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', row='Category', share_y=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_sort_ascending(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('sort_ascending', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', sort='ascending', label_y=True,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_style(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('style', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task',
              color_by='bar', gantt_edge_width=2, gantt_edge_color='#555555',
              gantt_height=0.2, gantt_fill_alpha=1, grid_major_y=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df, x=['Start', 'Stop'], y='Task', wrap='Category',
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[600, 400])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_bar_labels(bm=False, make_reference=False, show=False):

    # Make the plots
    name = utl.unit_test_get_img_name('bar_labels_user_xmax', make_reference, REFERENCE)
    fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Description',
              xmax=datetime.datetime(2025, 6, 15),
              workstreams='Workstream', workstreams_label_font_size=12,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])
    if bm:
        return

    utl.unit_test_options(make_reference, show, name, REFERENCE)

    name = utl.unit_test_get_img_name('bar_labels_auto_expand', make_reference, REFERENCE)
    fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Description',
              workstreams='Workstream', workstreams_label_font_size=12,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])
    utl.unit_test_options(make_reference, show, name, REFERENCE)

    name = utl.unit_test_get_img_name('bar_labels_no_expand_no_xmax', make_reference, REFERENCE)
    fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Description',
              auto_expand=False, workstreams='Workstream', workstreams_label_font_size=12,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_milestones_location(bm=False, make_reference=False, show=False):

    locs = [
            'top',
            'right'
            ]

    # Make the plot
    for loc in locs:
        name = utl.unit_test_get_img_name(f'milestones_location_{loc}_no_expand', make_reference, REFERENCE)
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description',
                  xmax=datetime.datetime(2025, 6, 15), milestone_text_location=loc, auto_expand=False,
                  workstreams='Workstream', workstreams_label_font_size=10,
                  workstreams_title_font_size=13, filename=name.with_suffix('.png'), save=not bm, inline=False,
                  ax_size=[900, 400])

        if bm:
            return
        utl.unit_test_options(make_reference, show, name, REFERENCE)

        name = utl.unit_test_get_img_name(f'milestones_location_{loc}_expand', make_reference, REFERENCE)
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description',
                  milestone_text_location=loc, auto_expand=True,
                  workstreams='Workstream', workstreams_label_font_size=10,
                  workstreams_title_font_size=13, filename=name.with_suffix('.png'), save=not bm, inline=False,
                  ax_size=[900, 400])

        utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_today(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('today', make_reference, REFERENCE)

    # Make the plot
    fcp.gantt(df2, x=['Start date', 'End date'], y='Description', today=datetime.datetime(2026, 3, 6),
              today_text='Giddie up', today_color='#ff0000', today_style='--', today_fill_color='#00ff00',
              workstreams='Workstream', workstreams_label_font_size=12, dependencies=False,
              milestone_text=False,
              filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_workstreams_date_type(bm=False, make_reference=False, show=False):

    kw = {
          'clean': {},
          'date_min': {'xmin': datetime.datetime(2025, 12, 10)},
          'date_max': {'xmax': datetime.datetime(2026, 3, 10)},
          'one_month': {'xmin': datetime.datetime(2026, 1, 5), 'xmax': datetime.datetime(2026, 1, 29)},
    }
    plots = [
             ['week'],
             ['month'],
             ['quarter'],
             ['year'],
             ['month-year'],
             ['week', 'month-year', 'quarter'],
             ['quarter-year'],
             ['quarter-year', 'month'],
             ['week', 'month'],
             ['week', 'quarter'],
             ['week', 'year'],
             ['week', 'month', 'quarter'],
             ['week', 'month', 'year'],
             ['week', 'month', 'quarter', 'year'],
             ['month', 'quarter'],
             ['month', 'year'],
             ['quarter', 'year'],
             ['month', 'quarter', 'year'],
    ]

    for plot in plots:
        for k, v in kw.items():
            name = utl.unit_test_get_img_name(f'workstreams_date_type_{"_".join(plot)}_{k}', make_reference, REFERENCE)
            fcp.gantt(df2, x=['Start date', 'End date'], y='Description',
                      date_type=plot, workstreams='Workstream', workstreams_label_font_size=12,
                      filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[1200, 400], **v)
            if bm:
                return
            utl.unit_test_options(make_reference, show, name, REFERENCE)

    # Bad date type errors ADD OTHERS
    name = Path('error')
    with pytest.raises(data.DataError):
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Owner',
                  date_type='dance', workstreams='Workstream', workstreams_label_font_size=12,
                  filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])

    with pytest.raises(data.DataError):
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Owner',
                  date_type=['quarter-year', 'year'], workstreams='Workstream', workstreams_label_font_size=12,
                  filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])

    with pytest.raises(data.DataError):
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description', bar_labels='Owner',
                  date_type=['month-year', 'quarter-year'], workstreams='Workstream', workstreams_label_font_size=12,
                  filename=name.with_suffix('.png'), save=not bm, inline=False, ax_size=[900, 400])


def plt_workstreams_location(bm=False, make_reference=False, show=False):

    locs = [
            'left',
            'right',
            'inline'
            ]

    # Make the plot
    for loc in locs:
        name = utl.unit_test_get_img_name(f'workstreams_location_{loc}', make_reference, REFERENCE)
        fcp.gantt(df2, x=['Start date', 'End date'], y='Description', gantt_date_type=['quarter-year', 'month'],
                  workstreams='Workstream', workstreams_location=loc, workstreams_label_font_size=10,
                  workstreams_title_font_size=13, filename=name.with_suffix('.png'), save=not bm, inline=False,
                  ax_size=[900, 400], match_bar_color=True if loc == 'right' else False)

        if bm:
            return
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


def test_bar_labels(benchmark):
    plt_bar_labels()
    benchmark(plt_bar_labels, True)


def test_milestones_location(benchmark):
    plt_milestones_location()
    benchmark(plt_milestones_location, True)


def test_today(benchmark):
    plt_today()
    benchmark(plt_today, True)


def test_workstreams_date_type(benchmark):
    plt_workstreams_date_type()
    benchmark(plt_workstreams_date_type, True)


def test_workstreams_location(benchmark):
    plt_workstreams_location()
    benchmark(plt_workstreams_location, True)


if __name__ == '__main__':
    pass
