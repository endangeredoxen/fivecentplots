import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'grouping'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df1 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')

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


def test_figure(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('figure', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', fig_groups='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name.with_suffix('.png'))

    for die in df1.Die.unique():
        tag = f' where Die={die}'

        if make_reference:
            new_name = name.parent / (name.stem.replace('_reference', tag + '_reference') + '.png')
            os.rename(name.parent / (name.stem + tag + '.png'), new_name)

        utl.unit_test_options(make_reference, show, name.parent / (name.stem + tag + '.png'), REFERENCE)


def test_figure2(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('figure2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', fig_groups=['Die', 'Substrate'], wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Target Wavelength==450',
             save=True, inline=False, filename=name.with_suffix('.png'))

    for nn, gg in df1.groupby(['Die', 'Substrate']):
        tag = f' where Die={nn[0]} and where Substrate={nn[1]}'

        if make_reference:
            new_name = name.parent / (name.stem.replace('_reference', tag + '_reference') + '.png')
            os.rename(name.parent / (name.stem + tag + '.png'), new_name)

        utl.unit_test_options(make_reference, show, name.parent / (name.stem + tag + '.png'), REFERENCE)


def test_groups_boxplot(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_boxplot', make_reference, REFERENCE)

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], legend='Region',
                save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_enabled(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_enabled', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', groups='Die', legend='Temperature [C]',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_enabled2(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_enabled2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', groups=['Die', 'Temperature [C]'],
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_none(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_none', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Temperature [C]',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_row_col(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_row_col', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=14,
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_row_col_x(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_row_col_x', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', row='Boost Level', col='x',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_row_col_x_share(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_row_col_x_share', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', row='Boost Level', col='x', share_col=True,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_row_col_y(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_row_col_y', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], legend='Die', col='Boost Level', row='y',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_row_col_y_share(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_row_col_y_share', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], legend='Die', col='Boost Level', row='y', share_row=True,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==75',
             label_rc_font_size=14, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_column_ncol(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_column_ncol', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', ncol=2,
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_names_no_sharing(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_names-no-sharing', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I Set', 'I [A]'], legend='Die', wrap='y',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[525, 170],
             filter='Substrate=="Si" & Target Wavelength==450', ncol=1, ws_row=0,
             separate_labels=False, separate_ticks=False,
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_unique(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_unique', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name.with_suffix('.png'), tick_cleanup='remove')
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_unique_seperate(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_unique_seperate', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
             separate_labels=True, separate_ticks=True, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# def test_groups_wrap_unique_seperate2(make_reference=False, remove=True, show=False):

#     name = utl.unit_test_get_img_name('groups_wrap_unique_seperate2', make_reference, REFERENCE)

#     # Make the plot
#     fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
#              ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450',
#              separate_labels=True, separate_ticks=False, save=True, inline=False, filename=name.with_suffix('.png'))
#     return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_xy(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_xy', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I Set', 'I [A]'], legend='Die', wrap='y',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[325, 325],
             filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_groups_wrap_xy2(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('groups_wrap_xy2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, y='Voltage', x=['I Set', 'I [A]'], legend='Die', wrap='x',
             groups=['Boost Level', 'Temperature [C]'], ax_size=[325, 325],
             filter='Substrate=="Si" & Target Wavelength==450',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_multiple(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_multiple', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             save=True, inline=False, filename=name.with_suffix('.png'))

    if not show:
        utl.unit_test_measure_axes(name, 305, 150, 402, 402, 1, alias=True)
        utl.unit_test_measure_margin(name, 305, 150, left=84, top=10, bottom=76, right=155, alias=True)

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_multiple_xy(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_multiple_xy', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['I [A]', 'Voltage'], lines=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_position(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_position', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             legend_location=2, save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_position_below(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_position_below', make_reference, REFERENCE)

    # Make the plot
    df1.loc[df1.Die == '(1,1)', 'Long Legend'] = 'Sample #ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    df1.loc[df1.Die == '(2,-1)', 'Long Legend'] = 'Sample #RUNFORYOURLIFEWITHME'
    df1.loc[df1.Die == '(-1,2)', 'Long Legend'] = 'Sample #THESKYISANEIGHBORHOOD!!!!!!!!!'
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Long Legend', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             legend_location='below', save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_secondary_axis(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_secondary_axis', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, legend=True, cmap='inferno',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_secondary_axis2(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_secondary_axis2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, y='Voltage', x=['Voltage', 'I [A]'], twin_y=True, legend=True, grid_major_x2=True, grid_minor_x2=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_secondary_column(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_secondary_column', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_secondary_none(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_secondary_none', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_legend_single(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('legend_single', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df1, x='Voltage', y='I [A]', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             save=True, inline=False, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
