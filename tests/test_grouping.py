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

test = 'grouping'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df1 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')

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
    members = [f for f in members if 'test_' in f[0]]
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
    members = [f for f in members if 'test_' in f[0]]
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


def test_legend_single(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_single_master') if master else 'legend_single'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_multiple(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_multiple_master') if master else 'legend_multiple'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_multiple_xy(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_multiple_xy_master') if master else 'legend_multiple_xy'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I [A]', 'Voltage'], lines=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_secondary_none(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_secondary_none_master') if master else 'legend_secondary_none'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_secondary_axis(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_secondary_axis_master') if master else 'legend_secondary_axis'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, legend=True, cmap='inferno',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_secondary_axis2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_secondary_axis2_master') if master else 'legend_secondary_axis2'

    # Make the plot
    fcp.plot(df1, y='Voltage', x=['Voltage', 'I [A]'], twin_y=True, legend=True, grid_major_x2=True, grid_minor_x2=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_secondary_column(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_secondary_column_master') if master else 'legend_secondary_column'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name + '.png')

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


def test_legend_position(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_position_master') if master else 'legend_position'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             legend_location=2, save=True, inline=False, filename=name + '.png')

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


def test_legend_position_below(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'legend_position_below_master') if master else 'legend_position_below'

    # Make the plot
    df1.loc[df1.Die == '(1,1)', 'Long Legend'] = 'Sample #ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    df1.loc[df1.Die == '(2,-1)', 'Long Legend'] = 'Sample #RUNFORYOURLIFEWITHME'
    df1.loc[df1.Die == '(-1,2)', 'Long Legend'] = 'Sample #THESKYISANEIGHBORHOOD!!!!!!!!!'
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Long Legend', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             legend_location='below', save=True, inline=False, filename=name + '.png')
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


def test_groups_none(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_none_master') if master else 'groups_none'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Temperature [C]',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name + '.png')

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


def test_groups_enabled(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_enabled_master') if master else 'groups_enabled'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', groups='Die', legend='Temperature [C]',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name + '.png')

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


def test_groups_enabled2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_enabled2_master') if master else 'groups_enabled2'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', groups=['Die', 'Temperature [C]'],
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name + '.png')

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


def test_groups_boxplot(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_boxplot_master') if master else 'groups_boxplot'

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], legend='Region',
                save=True, inline=False, filename=name + '.png', jitter=False)

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


def test_groups_row_col(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_row_col_master') if master else 'groups_row_col'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=14,
             save=True, inline=False, filename=name + '.png')

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


def test_groups_row_col_y(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_row_col_y_master') if master else 'groups_row_col_y'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], legend='Die', col='Boost Level', row='y',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name + '.png')

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


def test_groups_row_col_y_share(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_row_col_y_share_master') if master else 'groups_row_col_y_share'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], legend='Die', col='Boost Level', row='y', share_row=True,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name + '.png')

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


def test_groups_row_col_x(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_row_col_x_master') if master else 'groups_row_col_x'

    # Make the plot
    fcp.plot(df1, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', row='Boost Level', col='x',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name + '.png')

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


def test_groups_row_col_x_share(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_row_col_x_share_master') if master else 'groups_row_col_x_share'

    # Make the plot
    fcp.plot(df1, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', row='Boost Level', col='x', share_col=True,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name + '.png')

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


def test_groups_wrap_unique(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_unique_master') if master else 'groups_wrap_unique'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name + '.png', tick_cleanup='remove')

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


def test_groups_wrap_unique_seperate(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_unique_seperate_master') if master else 'groups_wrap_unique_seperate'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             separate_labels=True, separate_ticks=False, save=True, inline=False, filename=name + '.png')

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


def test_groups_wrap_column_ncol(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_column_ncol_master') if master else 'groups_wrap_column_ncol'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', ncol=2,
             save=True, inline=False, filename=name + '.png')

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


def test_groups_wrap_xy(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_xy_master') if master else 'groups_wrap_xy'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I Set', 'I [A]'], legend='Die', wrap='y',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[325, 325],
             filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name + '.png')

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


def test_groups_wrap_xy2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_xy2_master') if master else 'groups_wrap_xy2'

    # Make the plot
    fcp.plot(df1, y='Voltage', x=['I Set', 'I [A]'], legend='Die', wrap='x',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[325, 325],
             filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name + '.png')

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


def test_groups_wrap_names_no_sharing(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'groups_wrap_names-no-sharing_master') if master else 'groups_wrap_names-no-sharing'

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I Set', 'I [A]'], legend='Die', wrap='y',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[525, 170],
             filter='Substrate=="Si" & Target Wavelength==450', ncol=1, ws_row=0,
             separate_labels=False, separate_ticks=False,
             save=True, inline=False, filename=name + '.png')

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


def test_figure(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'figure_master') if master else 'figure'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', fig_groups='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name + '.png')

    # Compare with master
    if master:
        return
    elif show:
        for die in df1.Die.unique():
            tag = ' where %s=%s' % ('Die', die)
            utl.show_file(osjoin(MASTER, name + '_master' + tag + '.png'))
            utl.show_file(name + tag + '.png')
    else:
        for die in df1.Die.unique():
            tag = ' where %s=%s' % ('Die', die)
            compare = utl.img_compare(name + tag + '.png', osjoin(MASTER, name + '_master' + tag + '.png'))
            if remove:
                os.remove(name + tag + '.png')

            assert not compare


def test_figure2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'figure2_master') if master else 'figure2'

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', fig_groups=['Die', 'Substrate'], wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Target Wavelength==450',
             save=True, inline=False, filename=name + '.png')

    # Compare with master
    if master:
        return
    elif show:
        for die in df1.Die.unique():
            for sub in df1.Substrate.unique():
                tag = ' where %s=%s and where %s=%s' % ('Die', die, 'Substrate', sub)
                utl.show_file(osjoin(MASTER, name + '_master' + tag + '.png'))
                utl.show_file(name + tag + '.png')
    else:
        for die in df1.Die.unique():
            for sub in df1.Substrate.unique():
                tag = ' where %s=%s and where %s=%s' % ('Die', die, 'Substrate', sub)
                compare = utl.img_compare(name + tag + '.png', osjoin(MASTER, name + '_master' + tag + '.png'))
                if remove:
                    os.remove(name + tag + '.png')

                assert not compare


if __name__ == '__main__':
    pass
