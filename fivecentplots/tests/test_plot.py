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
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'plot.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data.csv'))
ts = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_ts.csv'))

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


def test_xy_scatter(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'xy_scatter_master') if master else 'xy_scatter'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=SHOW, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_xy_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'xy_legend_master') if master else 'xy_legend'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_xy_log_scale(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'xy_log-scale_master') if master else 'xy_log-scale'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9, xmax=2.1, grid_minor=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_xy_categorical(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'xy_categorical_master') if master else 'xy_categorical'

    # Make the plot
    fcp.plot(df, x='Die', y='I [A]', show=SHOW,
         filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Voltage==1.5',
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


def test_xy_ts(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'xy_ts_master') if master else 'xy_ts'

    # Make the plot
    fcp.plot(ts, x='Date', y='Happiness Quotient', markers=False, ax_size=[1000, 250],
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


def test_secondary_xy_shared_y(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'secondary-xy_shared-y_master') if master else 'secondary-xy_shared-y'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
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


def test_secondary_xy_shared_x(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'secondary-xy_shared-x_master') if master else 'secondary-xy_shared-x'

    # Make the plot
    fcp.plot(df, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', twin_y=True, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
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


def test_multiple_xy_y(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'multiple-xy_y-only_master') if master else 'multiple-xy_y-only'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_multiple_xy_x(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'multiple-xy_x-only_master') if master else 'multiple-xy_x-only'

    # Make the plot
    fcp.plot(df, x=['Boost Level', 'I [A]'], y='Voltage', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_multiple_xy_both(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'multiple-xy_both_master') if master else 'multiple-xy_both'

    # Make the plot
    fcp.plot(df, x=['Boost Level', 'I [A]'], y=['Voltage', 'Temperature [C]'], legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_row(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid-plots-row_master') if master else 'grid-plots-row'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', row='Boost Level', show=SHOW, ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
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


def test_column(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid-plots-column_master') if master else 'grid-plots-column'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', show=SHOW, ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
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


def test_row_x_column(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid-plots-row-x-column_master') if master else 'grid-plots-row-x-column'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]', show=SHOW,          ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
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


def test_wrap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'grid-plots-wrap_master') if master else 'grid-plots-wrap'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'], show=SHOW,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
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


def test_other_lines(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_lines_master') if master else 'other_lines'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_hlines=[(0, '#FF0000', '--', 3, 1, 'Open'), 1.2], ax_vlines=[0, (1, '#00FF00')],
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


def test_other_lines_df(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_lines_df_master') if master else 'other_lines_df'

    # Make the plot
    df['Open'] = 0
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=True,
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
            ax_hlines=[('Open', '#FF0000', '--', 3, 1), 1.2], ax_vlines=[0, (1, '#00FF00')],
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


def test_other_curve_fitting(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_curve-fitting_master') if master else 'other_curve-fitting'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=4, fit_eqn=True, fit_rsq=True, fit_font_size=9,
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


def test_other_curve_fitting_range(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_curve-fitting-range_master') if master else 'other_curve-fitting-range'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_eqn=True, fit_rsq=True, fit_font_size=9, fit_range_x=[1.3, 2],
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


def test_other_curve_fitting_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_curve-fitting-legend_master') if master else 'other_curve-fitting-legend'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_range_x=[1.3, 2], fit_width=2, fit_style='--',
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


def test_other_curve_fitting_legend2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_curve-fitting-legend2_master') if master else 'other_curve-fitting-legend2'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, wrap='Die', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_range_x=[1.3, 2], fit_width=2, fit_color='#555555', ax_size=[250, 250],
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


def test_other_stat_bad(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_stat-lines-bad_master') if master else 'other_stat-lines-bad'

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='median',
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


def test_other_stat_good(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_stat-lines-good_master') if master else 'other_stat-lines-good'

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='median', stat_val='I Set',
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


def test_other_stat_good_mult(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_stat-lines-good-multipe-y_master') if master else 'other_stat-lines-good-multipe-y'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], show=SHOW, legend=True, stat='median',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_other_conf_int(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_conf-int_master') if master else 'other_conf-int'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             conf_int=0.95,
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


def test_other_ref_line(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_ref-line-y=x_master') if master else 'other_ref-line-y=x'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=df['Voltage'], ref_line_legend_text='y=x', xmin=0, ymin=0, xmax=1.6, ymax=1.6,
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


def test_other_ref_line_mult(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_ref-line_mult_master') if master else 'other_ref-line_mult'

    # Make the plot
    df['2*Voltage'] = 2*df['Voltage']
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
            xmin=0, ymin=0, xmax=1.6, ymax=1.6,
            ref_line=['Voltage', '2*Voltage'], ref_line_legend_text=['y=x', 'y=2*x'], ref_line_style=['-', '--'], ref_line_color=[5,6],
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


def test_other_ref_line_complex(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'other_ref-line-complex_master') if master else 'other_ref-line-complex'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=1.555*df['Voltage']**4-3.451*df['Voltage']**3+2.347*df['Voltage']**2-0.496*df['Voltage']+0.014,
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


if __name__ == '__main__':
    pass