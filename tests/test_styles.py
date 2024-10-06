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

test = 'styles'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')


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


def test_alpha(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('alpha', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', fig_fill_alpha=0.5,
             legend_fill_color='#FF0000', legend_fill_alpha=0.52,
             ax_fill_color='#FFFFFF', ax_fill_alpha=0.7,
             label_x_fill_color='#0000FF', label_x_fill_alpha=0.2,
             label_y_fill_color='#FF00FF', label_y_fill_alpha=0.2,
             tick_labels_major_fill_color='#AAFB05', tick_labels_major_fill_alpha=0.45,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_alpha_marker(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('alpha_marker', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_alpha_legend_marker(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('alpha_legend_marker', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3, legend_marker_alpha=0.3,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_edge_color(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('edge_color', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_edge_color='#00FF00', legend_edge_color='#FF0000', ax_edge_color='#FFFFFF',
             label_x_edge_color='#0000FF', label_y_edge_color='#FF00FF', label_edge_width=1,
             tick_labels_major_edge_color='#AAFB05', tick_labels_major_edge_width=5,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_fill_color(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('fill_color', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', legend_fill_color='#FF0000', ax_fill_color='#FFFFFF',
             label_x_fill_color='#0000FF', label_y_fill_color='#FF00FF',
             tick_labels_major_fill_color='#AAFB05',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_fonts(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('fonts', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             title='My Oh My This Is Way Too Long of a Title for This Plot What Were You Thinking!',
             title_font_style='italic', label_y_font_size=25, label_y_style='normal',
             label_x_font_weight='normal', tick_labels_major_font='fantasy', filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_hist(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('hist', make_reference, REFERENCE)

    # Make the plot
    df_hist = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.hist(df_hist, x='Value', show=SHOW, legend='Region', hist_fill_alpha=1, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_hist_color(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('hist_color', make_reference, REFERENCE)

    # Make the plot
    df_hist = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.hist(df_hist, x='Value', show=SHOW, legend='Region', hist_fill_alpha=1,
             colors=['#FF0000', '#00FF11'],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_color(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_color', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_color_cmap(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_color_cmap', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             cmap='inferno',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_color_custom(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_color_custom', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=['#FF0000', '#FF7777', '#FFAAAA'], tick_labels_minor=True, tick_labels_minor_fill_color='#ff0000',
             filename=name.with_suffix('.png'), tick_labels_minor_edge_color='#0000ff',
             tick_labels_minor_font_style=['normal', 'italic'], xlabel='volts')
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_color_index(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_color_index', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 3, 3, 6, 6],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_style(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_style', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             lines_alpha=0.33, lines_style='--', lines_width=3,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_style2(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_style2', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             line_alpha=0.33, line_style='--', line_width=3,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_line_style_by_line(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('line_style_by_line', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], markers=False,
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 1, 1, 2, 2],
             lines_width=[3, 1, 3, 1, 3, 1], lines_style=['--', '-'],
             lines_alpha=[0.6, 1],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_boxplot(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot', make_reference, REFERENCE)

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                box_fill_color=[0, 0, 1, 1, 2, 2], box_fill_alpha=0.3, box_edge_width=0,
                marker_edge_color=[0, 0, 1, 1, 2, 2], marker_type=['o', '+'],
                box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_boxplot2(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot2', make_reference, REFERENCE)

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                box_marker_edge_color=[0, 0, 1, 1, 2, 2], marker_size=10,
                box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                box_marker_edge_alpha=0.6, box_marker_fill_alpha=1, box_marker_fill_color=['#00FF00', '#FF0000'],
                box_marker_fill=True, box_marker_type=['o'],
                filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_boxplot3(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot3', make_reference, REFERENCE)

    # Make the plot
    fcp.set_theme('_test', verbose=True)
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                marker_size=10, box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                box_marker_edge_alpha=0.6, box_marker_fill_alpha=1, box_marker_type=['+'], verbose=True,
                filename=name.with_suffix('.png'))
    fcp.set_theme('gray', verbose=True)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_edge(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_edge', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_fill(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_fill', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'], marker_fill_color='#FFFFFF',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_fill_default(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_fill_default', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_fill_alpha(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_fill_alpha', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True, marker_fill_alpha=0.5,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_type(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_type', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', 'd'],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_type_none(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_type_none', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', None, '+', '*', 'B', None],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_size(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_size', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_size_legend(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_size_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2, legend_marker_size=2,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_marker_size_column(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('marker_size_column', make_reference, REFERENCE)

    # Make the plot
    df['marker_size'] = df['Voltage'] * 10
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), inline=False, marker_size='marker_size')
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_spines(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('spines', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_edge_color='#FF0000', spine_left=False, spine_right=False,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_theme_white(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('theme_white', make_reference, REFERENCE)

    # Make the plot
    fcp.set_theme('white')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'))
    fcp.set_theme('gray')  # return to default gray
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_theme_white_fly(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('theme_white_fly', make_reference, REFERENCE)

    # Make the plot
    fcp.set_theme('gray')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], theme='white',
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'))
    fcp.set_theme('gray')  # return to default gray
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_theme_gray(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('theme_gray', make_reference, REFERENCE)

    # Make the plot
    fcp.set_theme('gray')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
