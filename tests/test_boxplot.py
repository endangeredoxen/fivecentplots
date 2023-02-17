import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
import numpy as np
osjoin = os.path.join
db = pdb.set_trace

test = 'boxplot'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

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
    Run the show=True option on all plt functions
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
def plt_simple(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'simple_master') if master else 'simple'

    # Make the plot
    fcp.boxplot(df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True,
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


def plt_simple_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'simple_legend_master') if master else 'simple_legend'

    # Make the plot
    fcp.boxplot(df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True, legend='Batch',
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


def plt_one_group(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'one_group_master') if master else 'one_group'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', ymin='q0', ymax='q100',
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False, box_stat_line='q50',
                box_group_label_font_size=24)  # font size is wrong!

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


def plt_group_single(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_single_master') if master else 'group_single'

    # Make the plot
    fcp.boxplot(df, y='Value', groups='Batch', show=SHOW, box_whisker=False, box_group_title_font_size=24,
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


def plt_group_multiple(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_multiple_master') if master else 'group_multiple'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW,
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


def plt_group_multiple_nan(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_multiple_nan_master') if master else 'group_multiple_nan'

    # Make the plot
    df['Test'] = np.nan
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample', 'Test'], show=SHOW,
                filename=name + '.png', save=not bm, inline=False, jitter=False, box_stat_line='q0.5')

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


def plt_group_legend(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_legend_master') if master else 'group_legend'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], legend='Region', show=SHOW,
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


def plt_group_legend_lots(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_legend_lots_master') if master else 'group_legend_lots'

    # Make the plot
    fcp.boxplot(df2, y='Value', groups=['Batch', 'Sample'], legend='Lots of Values', show=SHOW,
                filename=name + '.png', save=not bm, inline=False, jitter=False, legend_edge_color='#000000')

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


def plt_group_long(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_long_master') if master else 'group_long'

    # Make the plot
    df2 = df.copy()
    df2['This is a really long way to say show me your ID sucka'] = df['ID'] * 2
    fcp.boxplot(df2, y='Value', groups=['Batch', 'This is a really long way to say show me your ID sucka', 'Sample'],
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_group_auto_size(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_auto_size_master') if master else 'group_auto_size'

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
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_group_auto_size_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_auto_size_wrap_master') if master else 'group_auto_size_wrap'

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
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_group_auto_size_crash_simple(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_auto_size_crash_simple_master') if master else 'group_auto_size_crash_simple'

    # Make the plot
    cc = pd.melt(df_crash, id_vars='abbrev', value_vars=['speeding', 'alcohol', 'not_distracted', 'no_previous'],
                 var_name='cause', value_name='accidents')
    fcp.boxplot(cc, y='accidents', groups='cause', ax_size='auto',
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False)

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


def plt_group_auto_size_crash_complex(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_auto_size_crash_complex_master') if master else 'group_auto_size_crash_complex'

    # Make the plot
    cc = pd.melt(df_crash, id_vars='abbrev', value_vars=['speeding', 'alcohol', 'not_distracted', 'no_previous'],
                 var_name='cause', value_name='accidents')
    fcp.boxplot(cc, y='accidents', groups=['abbrev'], row='cause', label_y_fill_color='#ff0000', share_y=False,
                show=SHOW, filename=name + '.png', save=not bm, inline=False, jitter=False, ax_size='auto')

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


def plt_grid_column(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_column_master') if master else 'grid_column'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], col='Region', show=SHOW, ax_size=[300, 300],
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


def plt_grid_row(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_row_master') if master else 'grid_row'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], row='Region', show=SHOW, ax_size=[300, 300],
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


def plt_grid_wrap(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_wrap_master') if master else 'grid_wrap'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Sample', 'Region'], wrap='Batch', show=SHOW, ax_size=[300, 300],
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


def plt_grid_wrap_y(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_y_master') if master else 'grid_y'

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    fcp.boxplot(df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300],
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


def plt_grid_wrap_y_no_share(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_y-no-share_master') if master else 'grid_y-no-share'

    # Make the plot
    df['Value*2'] = 2 * df['Value']
    fcp.boxplot(df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300], share_y=False,
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


def plt_grand_means(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grand_means_master') if master else 'grand_means'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, grand_mean=True, grand_median=True,
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


def plt_group_means(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_means_master') if master else 'group_means'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, group_means=True,
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


def plt_mean_diamonds(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'mean_diamonds_master') if master else 'mean_diamonds'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, mean_diamonds=True, conf_coeff=0.95,
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


def plt_mean_diamonds_filled(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'mean_diamonds_filled_master') if master else 'mean_diamonds_filled'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch'], show=SHOW, mean_diamonds=True, conf_coeff=0.95,
                filename=name + '.png', save=not bm, inline=False, jitter=False,
                mean_diamonds_fill_color='#ff0000', mean_diamonds_edge_color='#0000ff')

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


def plt_violin(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_master') if master else 'violin'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                filename=name + '.png', save=not bm, inline=False)

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


def plt_violin_styled(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_styled_master') if master else 'violin_styled'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                violin_fill_color='#eaef1a', violin_fill_alpha=1, violin_edge_color='#555555', violin_edge_width=2,
                violin_box_color='#ffffff', violin_whisker_color='#ff0000',
                violin_median_marker='+', violin_median_color='#00ffff', violin_median_size=10,
                filename=name + '.png', save=not bm, inline=False)

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


def plt_violin_box_off(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_box_off_master') if master else 'violin_box_off'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                violin=True, violin_box_on=False, violin_markers=True, jitter=False,
                filename=name + '.png', save=not bm, inline=False)

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


def plt_stat_mean(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_mean_master') if master else 'stat_mean'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='mean', ax_size=[300, 300],
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


def plt_stat_median(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_median_master') if master else 'stat_median'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='median', ax_size=[300, 300],
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


def plt_stat_std_dev(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_std-dev_master') if master else 'stat_std-dev'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='std', ax_size=[300, 300],
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


def plt_dividers(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'dividers_master') if master else 'dividers'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_divider=False, ax_size=[300, 300],
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


def plt_range_lines(bm=False, master=False, remove=True, show=False):

    name = osjoin(MASTER, 'range_lines_master') if master else 'range_lines'

    # Make the plot
    fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_range_lines=False, ax_size=[300, 300],
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
def test_simple(benchmark):
    plt_simple()
    benchmark(plt_simple, True)


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
