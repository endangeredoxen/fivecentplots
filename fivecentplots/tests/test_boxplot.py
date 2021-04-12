
import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb, platform
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
db = pdb.set_trace
if platform.system() != 'Windows':
    raise utl.PlatformError()

MPL = utl.get_mpl_version_dir()
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'boxplot.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))

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
    members = [f for f in members if 'test_' in f[0]]
    for member in members:
        print('Running %s...' % member[0], end='')
        member[1](master=True)
        print('done!')


def test_simple(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'simple_master') if master else 'simple'

    # Make the plot
    fcp.boxplot(df=df, y='Value', show=SHOW, tick_labels_minor=True, grid_minor=True,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_group_single(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_single_master') if master else 'group_single'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups='Batch', show=SHOW,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_group_multiple(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_multiple_master') if master else 'group_multiple'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_group_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_legend_master') if master else 'group_legend'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], legend='Region', show=SHOW,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grid_column(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_column_master') if master else 'grid_column'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], col='Region', show=SHOW, ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grid_row(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_row_master') if master else 'grid_row'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], row='Region', show=SHOW, ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grid_wrap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_wrap_master') if master else 'grid_wrap'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Sample', 'Region'], wrap='Batch', show=SHOW, ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grid_wrap_y(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_y_master') if master else 'grid_y'

    # Make the plot
    df['Value*2'] = 2*df['Value']
    fcp.boxplot(df=df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grid_wrap_y_no_share(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid_y-no-share_master') if master else 'grid_y-no-share'

    # Make the plot
    df['Value*2'] = 2*df['Value']
    fcp.boxplot(df=df, y=['Value', 'Value*2'], groups=['Batch', 'Sample', 'Region'], wrap='y', show=SHOW,
                ax_size=[300, 300], share_y=False,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_grand_means(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grand_means_master') if master else 'grand_means'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, grand_mean=True, grand_median=True,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_group_means(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'group_means_master') if master else 'group_means'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, group_means=True,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_mean_diamonds(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'mean_diamonds_master') if master else 'mean_diamonds'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, mean_diamonds=True, conf_coeff=0.95,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_violin(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_master') if master else 'violin'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_violin_styled(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_styled_master') if master else 'violin_styled'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True,
                violin_fill_color='#eaef1a', violin_fill_alpha=1, violin_edge_color='#555555', violin_edge_width=2,
                violin_box_color='#ffffff', violin_whisker_color='#ff0000',
                violin_median_marker='+', violin_median_color='#00ffff', violin_median_size=10,
                filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_violin_box_off(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'violin_box_off_master') if master else 'violin_box_off'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, violin=True, violin_box_on=False, violin_markers=True, jitter=False,
                filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_stat_mean(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_mean_master') if master else 'stat_mean'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='mean', ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_stat_median(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_median_master') if master else 'stat_median'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='median', ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_stat_std_dev(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'stat_std-dev_master') if master else 'stat_std-dev'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_stat_line='std', ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_dividers(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'dividers_master') if master else 'dividers'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_divider=False, ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_range_lines(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'range_lines_master') if master else 'range_lines'

    # Make the plot
    fcp.boxplot(df=df, y='Value', groups=['Batch', 'Sample'], show=SHOW, box_range_lines=False, ax_size=[300, 300],
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    elif show:
        os.startfile(osjoin(MASTER, name + '_master.png'))
        os.startfile(name + '.png')
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'), show=True)
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


if __name__ == '__main__':
    pass