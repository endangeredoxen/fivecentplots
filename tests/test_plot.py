import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import pytest
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'plot'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df_interval = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_interval.csv')
ts = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_ts.csv')

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
def plt_column(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-column', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', show=SHOW, ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        # Axis width
        utl.unit_test_measure_axes(name, 70, None, 227, None, 1, alias=True)
        # Col label height
        utl.unit_test_measure_axes(name, None, 90, None, 30, 1, alias=False)
        # Axis height
        utl.unit_test_measure_axes(name, None, 90, None, 227, 1, 45, alias=True)
        # Margins
        utl.unit_test_measure_margin(name, 70, 115, left=70, right=117, bottom=76, alias=True)
        utl.unit_test_measure_margin(name, 170, 190, top=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_column_no_names(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-column-no-names', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', show=SHOW, ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             label_col_values_only=True, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        # Axis width
        utl.unit_test_measure_axes(name, 70, None, 227, None, 1, alias=True)
        # Col label height
        utl.unit_test_measure_axes(name, None, 90, None, 30, 1, alias=False)
        # Axis height
        utl.unit_test_measure_axes(name, None, 90, None, 227, 1, 45, alias=True)
        # Margins
        utl.unit_test_measure_margin(name, 70, 115, left=70, right=117, bottom=76, alias=True)
        utl.unit_test_measure_margin(name, 170, 190, top=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_multiple_xy_both(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('multiple-xy_both', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x=['Boost Level', 'I [A]'], y=['Voltage', 'Temperature [C]'], legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 60, 240, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 60, 240, top=10, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_multiple_xy_both_label(bm=False, make_reference=False, show=False):
    # Tests layout._set_label_text - both combinations
    name = utl.unit_test_get_img_name('multiple-xy_both_label', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x=['Boost Level', 'I [A]'], y=['Voltage', 'Temperature [C]'], legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             label_x='yep', label_y_text='no way',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_multiple_xy_x(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('multiple-xy_x-only', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x=['Boost Level', 'I [A]'], y='Voltage', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 60, 240, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 60, 240, top=10, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_multiple_xy_y(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('multiple-xy_y-only', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], legend='Die', show=SHOW,
             legend_edge_width=4, legend_edge_color='#000000',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False, xmin=1.3)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 60, 400, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 60, 240, left=84, right=10, bottom=76, top=10, alias=True)
        utl.unit_test_measure_margin(name, 400, 240, right=219, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=4, fit_eqn=True, fit_rsq=True, fit_font_size=9,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, right=10,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=2, fit_eqn=True, fit_rsq=True, fit_font_size=9, fit_range_y=[0, 1], fit_color='#FF0000',
             filename=name.with_suffix('.png'), save=not bm, inline=False, fit_legend_text='smile fit')
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, right=134,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting3(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting3', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=2, fit_eqn=True, fit_rsq=True, fit_font_size=9, fit_range_x=[-1, -1], fit_color='#ff0000',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, right=10,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting_range(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting-range', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_eqn=True, fit_rsq=True, fit_font_size=9, fit_range_x=[1.3, 2],
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, right=96,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting-legend', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_range_x=[1.3, 2], fit_width=2, fit_style='--',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_curve_fitting_legend2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_curve-fitting-legend2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, wrap='Die', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_range_x=[1.3, 2], fit_width=2, fit_color='#555555', ax_size=[250, 250],
             label_wrap_size=22, label_wrap_font_size=12, label_wrap_fill_color='55FF3A',
             label_wrap_font_color='#ff0000', title_wrap_size=25, title_wrap_font_size=10,
             title_wrap_font_color='#0000ff',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_conf_int(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_conf-int', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', title='IV Data', lines=False, show=SHOW, legend=True,
             conf_int=0.95, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_conf_int2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_conf-int2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', title='IV Data', lines=False, show=SHOW,
             conf_int=95, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_conf_int_no_std(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_conf-int_no_std', make_reference, REFERENCE)

    # Make the plot
    df2 = pd.DataFrame({'x': [0, 1, 2], 'y': [2, 4, 6]})
    df2 = pd.concat([df2, df2, df2, df2, df2])
    fcp.plot(df2, x='x', y='y', title='IV Data', lines=False, show=SHOW,
             conf_int=95, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_lcl_only(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_lcl_only', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             lcl=-0.5, ymin=-1, lcl_fill_color='#FF0000')
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=94, top=43, bottom=76, right=10,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_lcl_only_inside(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_lcl_only_inside', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False, control_limit_side='inside',
             lcl=-0.5, ymin=-1, lcl_fill_color='#FF0000')
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=94, top=43, bottom=76, right=10,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_lines(bm=False, make_reference=False, show=False):
    name = utl.unit_test_get_img_name('other_lines', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_hlines=[(0, '#FF0000', '--', 3, 1, 'Open'), 1.2], ax_vlines=[0, (1, '#00FF00')],
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 365, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 365, left=84, top=43, bottom=76, right=118,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_lines_df(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_lines_df', make_reference, REFERENCE)

    # Make the plot
    df['Open'] = 0
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_hlines=[('Open', '#FF0000', '--', 3, 1), 1.2], ax_vlines=[0, (1, '#00FF00')],
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 365, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 365, left=84, top=43, bottom=76, right=118,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_nq_int(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_nq-int', make_reference, REFERENCE)

    # Make the plot
    df_interval['y2'] = -df_interval['y']
    fcp.plot(df_interval, x='x', y=['y', 'y2'], twin_x=True,  title='IV Data', lines=False,
             show=SHOW, ymin='q0.05', ymax='q99',
             nq_int=[-2, 2], filename=name.with_suffix('.png'), save=not bm, inline=False, legend=True)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_nq_int2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_nq-int2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', title='IV Data', lines=False, show=SHOW,
             nq_int=[-4, 4], filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=98, top=43, bottom=76, right=10,
                                     alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_percentile_int(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_percentile-int', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', title='IV Data', lines=False, show=SHOW, ymax='1.5*iqr', ymin='0*iqr',
             perc_int=[0.25, 0.75], filename=name.with_suffix('.png'), save=not bm, inline=False, legend=True)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ref_line(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ref-line-y=x', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=df['Voltage'], ref_line_legend_text='y=x', xmin=0, ymin=0, xmax=1.6, ymax=1.6,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 370, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ref_line_leg(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ref-line_leg-y=x', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW,
             ref_line_color='#FF0000', ref_line_style='--', ref_line_width=2,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=df['Voltage'], ref_line_legend_text='y=x', xmin=0, ymin=0, xmax=1.6, ymax=1.6,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        # utl.unit_test_measure_axes(name, 110, 365, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 365, left=84, top=43, bottom=76)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ref_line_mult(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ref-line_mult', make_reference, REFERENCE)

    # Make the plot
    df['2*Voltage'] = 2 * df['Voltage']
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmin=0, ymin=0, xmax=1.6, ymax=1.6,
             ref_line=['Voltage', '2*Voltage', '3*Voltage'],
             ref_line_legend_text=['y=x', 'y=2*x'], ref_line_style=['-', '--'],
             ref_line_color=[5, 6], filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ref_line_mult2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ref-line_mult2', make_reference, REFERENCE)

    # Make the plot
    df['2*Voltage'] = 2 * df['Voltage']
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmin=0, ymin=0, xmax=1.6, ymax=1.6, ref_line=['Voltage', '2*Voltage'],
             ref_line_style=['-', '--'], ref_line_color=[5, 6], filename=name.with_suffix('.png'), save=not bm,
             inline=False)
    if bm:
        return

    if not show:
        # utl.unit_test_measure_axes(name, 110, 370, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 110, 370, left=84, top=43, bottom=76, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ref_line_complex(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ref-line-complex', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=1.555 * df['Voltage']**4 - 3.451 * df['Voltage']**3
             + 2.347 * df['Voltage']**2 - 0.496 * df['Voltage'] + 0.014,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_stat_bad(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_stat-lines-bad', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='median', filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_stat_bad2(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_stat-lines-bad2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='medianx', filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_stat_good(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_stat-lines-good', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW, groups='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='median', stat_val='I Set', filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_stat_q(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_stat-lines-q', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             stat='q50', stat_val='I Set', markers=False, title_fill_color='#ff0000',
             filename=name.with_suffix('.png'), save=not bm, inline=False, ax_edge_width=5)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_stat_good_mult(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_stat-lines-good-multiple-y', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], show=SHOW, legend=True, stat='median',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ucl_only(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ucl_only', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             ucl=1.0, ymax=3, ucl_fill_alpha=0.8)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ucl_only_inside(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ucl_only_inside', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False, control_limit_side='inside',
             ucl=1.0, ymax=3, ucl_fill_alpha=0.8)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ucl_lcl_inside(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ucl_lcl_inside', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             ucl=1.0, ymax=3, lcl=-0.5, ymin=-1, control_limit_side='inside',
             legend=True)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_other_ucl_lcl_outside(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('other_ucl_lcl_outside', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             ucl=1.0, ymax=3, lcl=-0.5, ymin=-1, ucl_fill_color='#FFFF00',
             legend=True)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', row='Boost Level',
             show=SHOW, ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        # Axis width
        utl.unit_test_measure_axes(name, 70, 165, 227, 227, 1, alias=True)
        # Row label width
        utl.unit_test_measure_axes(name, 70, None, 30, None, 1, skip=300, alias=False)
        # Row label height
        utl.unit_test_measure_axes_rows(name, 334, 227, 3, alias=False)
        # Margins
        utl.unit_test_measure_margin(name, 680, 105, left=70, top=10, bottom=76, alias=True)
        utl.unit_test_measure_margin(name, 680, 105, right=117, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_no_names(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row-no-names', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', row='Boost Level',
             show=SHOW, ax_size=[225, 225], label_row_values_only=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_x_column(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row-x-column', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]', show=SHOW,
             ax_size=[225, 225], label_x_fill_color='#00ffff', label_y_fill_color='#00ff0f',
             filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 340, 227, 3)
        utl.unit_test_measure_margin(name, 430, 145, left=10, bottom=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_x_column_empty(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row-x-column_empty', make_reference, REFERENCE)

    # Make the plot
    df_empty = df.copy()
    df_empty.loc[0, 'Temperature [C]'] = 77
    fcp.plot(df_empty, y=['Voltage', 'I [A]'], x='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13, show=SHOW,
             filename=name.with_suffix('.png'), save=not bm, inline=False, share_x=False, twin_x=True,
             tick_labels_y2_edge_width=1, tick_labels_y2_edge_color='#000000',
             label_y2_edge_width=1, label_y2_edge_color='#000000',
             )
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_x_column_no_names(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row-x-column-no-names', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]', show=SHOW,
             ax_size=[225, 225], label_rc_values_only=True, filter='Substrate=="Si" & Target Wavelength==450',
             label_rc_font_size=13, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_row_x_column_sep_labels(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-row-x-column_sep_labels', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]', show=SHOW,
             ax_size=[225, 225],
             filter='Substrate=="Si" & Target Wavelength==450',
             label_rc_font_size=13, separate_labels=True, share_y=False,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes_cols(name, 375, 227, 3)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_secondary_xy_not_shared_x(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary-xy_not_shared-x', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', twin_y=True, show=SHOW, x2min=[-10, -20],
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25', row='Die',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_secondary_xy_not_shared_y(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary-xy_not_shared-y', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die', y2min=[0, -1],
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25', col='Die',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_secondary_xy_shared_x(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary-xy_shared-x', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', twin_y=True, show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25', row='Die',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             label_x_edge_width=1, label_x_edge_color='#000000',
             label_x2_edge_width=1, label_x2_edge_color='#000000',
             tick_labels_major_x_edge_width=1, tick_labels_major_x_edge_color='#000000',
             tick_labels_major_x2_edge_width=1, tick_labels_major_x2_edge_color='#000000',
             )
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 650, 190, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 650, 285, left=84, top=10, bottom=10, alias=True)
        utl.unit_test_measure_margin(name, 650, 285, right=255, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_secondary_xy_shared_y(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary-xy_shared-y', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25', col='Die',
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             label_y_edge_width=1, label_y_edge_color='#000000',
             label_y2_edge_width=1, label_y2_edge_color='#000000',
             tick_labels_y_edge_width=1, tick_labels_y_edge_color='#000000',
             tick_labels_y2_edge_width=1, tick_labels_y2_edge_color='#000000')
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_wrap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('grid-plots-wrap', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'], show=SHOW,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
             filename=name.with_suffix('.png'), save=not bm, inline=False,
             ax_hlines=[(1, '#FF0000'), (2, '#00FF00'), (3, '#0000FF'), (4, '#FFF000'), (5, '#000FFF'), 6],
             ax_hlines_by_plot=True, ax_hlines_label=['sipping', 'on', 'gin', '&', 'juice', 'yall'])
    if bm:
        return

    if show:
        utl.unit_test_measure_margin(name, 25, 115, top=10, alias=False)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_categorical(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_categorical', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Die', y='I [A]', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Voltage==1.5',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_index(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_index', make_reference, REFERENCE)

    # Make the plot using an unamed index column
    fcp.plot(df, x='index', y='I [A]', legend='Die', show=SHOW, ymax=1.4,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)

    ########################################################################################################
    # Make the plot using an named index column
    name = utl.unit_test_get_img_name('xy_index2', make_reference, REFERENCE)

    df_idx = df.copy()
    df_idx.index.name = 'tupac'
    fcp.plot(df_idx, x='tupac', y='I [A]', legend='Die', show=SHOW, ymax=1.4,
             label_x_edge_width=1, label_x_edge_color='#000000',
             tick_labels_major_x_edge_color='#000000', tick_labels_major_x_edge_width=1,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW, ymax=1.4,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 150, 200, 402, 402, 2, alias=True)
        utl.unit_test_measure_margin(name, 150, 200, left=84, top=15, bottom=76, right=117, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_legend_no_sort(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_legend_no_sort', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW, ymax=1.4, sort=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_log_scale(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_log-scale', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9, xmax=2.1,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             grid_minor=True, filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_scatter(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_scatter', make_reference, REFERENCE)

    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False, ax_edge_width=14,
             label_fill_color='#ff0000', label_font_style='italic', ticks_color='#ff0000',
             tick_labels_major_edge_color='#0000ff', label_edge_color='#000000', label_y_edge_width=2,
             label_x_edge_width=3, ws_fig_label=15, xmin=0.5, label_y_text='How', label_y_font_size=18)
    if bm:
        return

    if not show:
        utl.unit_test_measure_axes(name, 110, 185, 428, 428, 2, alias=True)
        utl.unit_test_measure_margin(name, 265, 270, left=15, top=43, bottom=15, right=10, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_scatter_swap(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_scatter_swap', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, swap=True,
             show=False, filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_ts(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_ts', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(ts, x='Date', y='Happiness Quotient', markers=False, ax_size=[1000, 250],
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_column(benchmark):
    plt_column()
    benchmark(plt_column, True)


@pytest.mark.skipif(sys.version_info < (3, 7), reason='python3.6 is being deprecated')
def test_column_no_names(benchmark):
    plt_column_no_names()
    benchmark(plt_column_no_names, True)


def test_multiple_xy_both(benchmark):
    plt_multiple_xy_both()
    benchmark(plt_multiple_xy_both, True)


def test_multiple_xy_both_label(benchmark):
    plt_multiple_xy_both_label()
    benchmark(plt_multiple_xy_both_label, True)


def test_multiple_xy_x(benchmark):
    plt_multiple_xy_x()
    benchmark(plt_multiple_xy_x, True)


def test_multiple_xy_y(benchmark):
    plt_multiple_xy_y()
    benchmark(plt_multiple_xy_y, True)


def test_other_conf_int(benchmark):
    plt_other_conf_int()
    benchmark(plt_other_conf_int, True)


def test_other_conf_int2(benchmark):
    plt_other_conf_int2()
    benchmark(plt_other_conf_int2, True)


def test_other_conf_int_no_std(benchmark):
    plt_other_conf_int_no_std()
    benchmark(plt_other_conf_int_no_std, True)


def test_other_curve_fitting(benchmark):
    plt_other_curve_fitting()
    benchmark(plt_other_curve_fitting, True)


def test_other_curve_fitting2(benchmark):
    plt_other_curve_fitting2()
    benchmark(plt_other_curve_fitting2, True)


def test_other_curve_fitting3(benchmark):
    plt_other_curve_fitting3()
    benchmark(plt_other_curve_fitting3, True)


def test_other_curve_fitting_legend(benchmark):
    plt_other_curve_fitting_legend()
    benchmark(plt_other_curve_fitting_legend, True)


def test_other_curve_fitting_legend2(benchmark):
    plt_other_curve_fitting_legend2()
    benchmark(plt_other_curve_fitting_legend2, True)


def test_other_curve_fitting_range(benchmark):
    plt_other_curve_fitting_range()
    benchmark(plt_other_curve_fitting_range, True)


def test_other_lcl_only(benchmark):
    plt_other_lcl_only()
    benchmark(plt_other_lcl_only, True)


def test_other_lcl_only_inside(benchmark):
    plt_other_lcl_only_inside()
    benchmark(plt_other_lcl_only_inside, True)


def test_other_lines(benchmark):
    plt_other_lines()
    benchmark(plt_other_lines, True)


def test_other_lines_df(benchmark):
    plt_other_lines_df()
    benchmark(plt_other_lines_df, True)


def test_other_nq_int(benchmark):
    plt_other_nq_int()
    benchmark(plt_other_nq_int, True)


def test_other_nq_int2(benchmark):
    plt_other_nq_int2()
    benchmark(plt_other_nq_int, True)


def test_other_percentile_int(benchmark):
    plt_other_percentile_int()
    benchmark(plt_other_percentile_int, True)


def test_other_ucl_only(benchmark):
    plt_other_ucl_only()
    benchmark(plt_other_ucl_only, True)


def test_other_ucl_only_inside(benchmark):
    plt_other_ucl_only_inside()
    benchmark(plt_other_ucl_only_inside, True)


def test_other_ucl_lcl_inside(benchmark):
    plt_other_ucl_lcl_inside()
    benchmark(plt_other_ucl_lcl_inside, True)


def test_other_ucl_lcl_outside(benchmark):
    plt_other_ucl_lcl_outside()
    benchmark(plt_other_ucl_lcl_outside, True)


def test_other_ref_line_leg(benchmark):
    plt_other_ref_line_leg()
    benchmark(plt_other_ref_line_leg, True)


def test_other_ref_line(benchmark):
    plt_other_ref_line()
    benchmark(plt_other_ref_line, True)


def test_other_ref_line_complex(benchmark):
    plt_other_ref_line_complex()
    benchmark(plt_other_ref_line_complex, True)


def test_other_ref_line_mult(benchmark):
    plt_other_ref_line_mult()
    benchmark(plt_other_ref_line_mult, True)


def test_other_ref_line_mult2(benchmark):
    plt_other_ref_line_mult2()
    benchmark(plt_other_ref_line_mult2, True)


def test_other_stat_bad(benchmark):
    plt_other_stat_bad()
    benchmark(plt_other_stat_bad, True)


def test_other_stat_bad2(benchmark):
    plt_other_stat_bad2()
    benchmark(plt_other_stat_bad2, True)


def test_other_stat_good(benchmark):
    plt_other_stat_good()
    benchmark(plt_other_stat_good, True)


def test_other_stat_q(benchmark):
    plt_other_stat_q()
    benchmark(plt_other_stat_q, True)


def test_other_stat_good_mult(benchmark):
    plt_other_stat_good_mult()
    benchmark(plt_other_stat_good_mult, True)


def test_row(benchmark):
    plt_row()
    benchmark(plt_row, True)


@pytest.mark.skipif(sys.version_info < (3, 7), reason='python3.6 is being deprecated')
def test_row_no_names(benchmark):
    plt_row_no_names()
    benchmark(plt_row_no_names, True)


def test_row_x_column(benchmark):
    plt_row_x_column()
    benchmark(plt_row_x_column, True)


@pytest.mark.skipif(sys.version_info < (3, 7), reason='python3.6 is being deprecated')
def test_row_x_column_no_names(benchmark):
    plt_row_x_column_no_names()
    benchmark(plt_row_x_column_no_names, True)


def test_row_x_column_empty(benchmark):
    plt_row_x_column_empty()
    benchmark(plt_row_x_column_empty, True)


def test_row_x_column_sep_labels(benchmark):
    plt_row_x_column_sep_labels()
    benchmark(plt_row_x_column_sep_labels, True)


def test_secondary_xy_shared_x(benchmark):
    plt_secondary_xy_shared_x()
    benchmark(plt_secondary_xy_shared_x, True)


def test_secondary_xy_not_shared_x(benchmark):
    plt_secondary_xy_not_shared_x()
    benchmark(plt_secondary_xy_not_shared_x, True)


def test_secondary_xy_shared_y(benchmark):
    plt_secondary_xy_shared_y()
    benchmark(plt_secondary_xy_shared_y, True)


def test_secondary_xy_not_shared_y(benchmark):
    plt_secondary_xy_not_shared_y()
    benchmark(plt_secondary_xy_not_shared_y, True)


def test_wrap(benchmark):
    plt_wrap()
    benchmark(plt_wrap, True)


def test_xy_categorical(benchmark):
    plt_xy_categorical()
    benchmark(plt_xy_categorical, True)


def test_xy_index(benchmark):
    plt_xy_index()
    benchmark(plt_xy_index, True)


def test_xy_legend(benchmark):
    plt_xy_legend()
    benchmark(plt_xy_legend, True)


def test_xy_legend_no_sort(benchmark):
    plt_xy_legend_no_sort()
    benchmark(plt_xy_legend_no_sort, True)


def test_xy_log_scale(benchmark):
    plt_xy_log_scale()
    benchmark(plt_xy_log_scale, True)


def test_xy_scatter(benchmark):
    plt_xy_scatter()
    benchmark(plt_xy_scatter, True)


def test_xy_scatter_swap(benchmark):
    plt_xy_scatter_swap()
    benchmark(plt_xy_scatter_swap, True)


def test_xy_ts(benchmark):
    plt_xy_ts()
    benchmark(plt_xy_ts, True)


if __name__ == '__main__':
    pass
