import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace

MPL = utl.get_mpl_version_dir()
if Path(f'../tests/test_images/{MPL}').exists():
    MASTER = Path(f'../tests/test_images/{MPL}') / 'barplot.py'
else:
    MASTER = Path(f'tests/test_images/{MPL}') / 'barplot.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/real_data_bar.csv')

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
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def show_all():
    """
    Run the show=True option on all plt functions
    """

    members = inspect.getmembers(sys.modules[__name__])
    members = [f for f in members if 'plt_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](show=True)
        db()


# plt_ functions can be used directly outside of pytest for debug
def plt_vertical(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'vertical_master') if master else 'vertical'

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A" & T [C]==25',
            tick_labels_major_x_rotation=90,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_vertical_zero_group(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'vertical_zero_group_master') if master else 'vertical_zero_group'

    # Make the plot
    temp = pd.DataFrame({'Liquid': ['Air'], 'pH': [0], 'Measurement': ['A'],
                         'T [C]': [125]}, index=[30])
    fcp.bar(df=pd.concat([df, temp]), x='Liquid', y='pH', show=SHOW,
            filter='Measurement=="A" & T [C]==25',
            tick_labels_major_x_rotation=90, show_all_groups=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_horizontal(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'horizontal_master') if master else 'horizontal'

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, filter='Measurement=="A"', horizontal=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_error(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'error_master') if master else 'error'

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, error_bars=True,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, legend='Measurement',
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_stacked(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stacked_master') if master else 'stacked'

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, stacked=True, legend='Measurement',
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_row_col(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'row_col_master') if master else 'row_col'

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, col='Measurement', row='T [C]',
            ax_hlines=0, ax_size=[300, 300],
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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

    # Make the plot
    fcp.bar(df, x='Liquid', y='pH', show=SHOW, tick_labels_major_x_rotation=90, wrap='Measurement', ax_size=[300, 300],
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_rolling_mean(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'rolling_mean_master') if master else 'rolling_mean'

    # Make the plot
    fcp.bar(df2, x='date', y='cases', show=SHOW, ax_size=[800, 500],
            tick_labels_major_x_rotation=90, rolling_mean=14,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_rolling_mean_styled(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'rolling_mean_styled_master') if master else 'rolling_mean_styled'

    # Make the plot
    fcp.bar(df2, x='date', y='cases', show=SHOW, ax_size=[800, 500],
            tick_labels_major_x_rotation=90, rolling_mean=14, bar_fill_color='#aaaaaa',
            rolling_mean_line_color='#000000', markers=True, marker_size=4,
            filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def test_row_col(benchmark):
    plt_row_col()
    benchmark(plt_row_col, True)


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


def test_rolling_mean(benchmark):
    plt_rolling_mean()
    benchmark(plt_rolling_mean, True)


def test_rolling_mean_styled(benchmark):
    plt_rolling_mean_styled()
    benchmark(plt_rolling_mean_styled, True)
