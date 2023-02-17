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

test = 'styles'
if Path('../tests/test_images').exists():
    MASTER = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    MASTER = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    MASTER = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')


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


def test_fill_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'fill_color_master') if master else 'fill_color'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', legend_fill_color='#FF0000', ax_fill_color='#FFFFFF',
             label_x_fill_color='#0000FF', label_y_fill_color='#FF00FF',
             tick_labels_major_fill_color='#AAFB05',
             filename=name + '.png')

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


def test_edge_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'edge_color_master') if master else 'edge_color'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_edge_color='#00FF00', legend_edge_color='#FF0000', ax_edge_color='#FFFFFF',
             label_x_edge_color='#0000FF', label_y_edge_color='#FF00FF',
             tick_labels_major_edge_color='#AAFB05', tick_labels_major_edge_width=5,
             filename=name + '.png')

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


def test_spines(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'spines_master') if master else 'spines'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_edge_color='#FF0000', spine_left=False, spine_right=False,
             filename=name + '.png')

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


def test_alpha(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_master') if master else 'alpha'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', fig_fill_alpha=0.5,
             legend_fill_color='#FF0000', legend_fill_alpha=0.52,
             ax_fill_color='#FFFFFF', ax_fill_alpha=0.7,
             label_x_fill_color='#0000FF', label_x_fill_alpha=0.2,
             label_y_fill_color='#FF00FF', label_y_fill_alpha=0.2,
             tick_labels_major_fill_color='#AAFB05', tick_labels_major_fill_alpha=0.45,
             filename=name + '.png')

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


def test_alpha_marker(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_marker_master') if master else 'alpha_marker'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3,
             filename=name + '.png')

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


def test_alpha_legend_marker(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_legend_marker_master') if master else 'alpha_legend_marker'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3, legend_marker_alpha=0.3,
             filename=name + '.png')

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


def test_line_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_master') if master else 'line_color'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name + '.png')

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


def test_line_color_custom(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_custom_master') if master else 'line_color_custom'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=['#FF0000', '#FF7777', '#FFAAAA'], tick_labels_minor=True, tick_labels_minor_fill_color='#ff0000',
             filename=name + '.png', tick_labels_minor_edge_color='#0000ff',
             tick_labels_minor_font_style=['normal', 'italic'], xlabel='volts')

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


def test_line_color_index(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_index_master') if master else 'line_color_index'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 3, 3, 6, 6],
             filename=name + '.png')

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


def test_line_color_cmap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_cmap_master') if master else 'line_color_cmap'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             cmap='inferno',
             filename=name + '.png')

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


def test_line_style(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style_master') if master else 'line_style'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             lines_alpha=0.33, lines_style='--', lines_width=3,
             filename=name + '.png')

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


def test_line_style2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style2_master') if master else 'line_style2'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             line_alpha=0.33, line_style='--', line_width=3,
             filename=name + '.png')

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


def test_line_style_by_line(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style_by_line_master') if master else 'line_style_by_line'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], markers=False,
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 1, 1, 2, 2],
             lines_width=[3, 1, 3, 1, 3, 1], lines_style=['--', '-'],
             lines_alpha=[0.6, 1],
             filename=name + '.png')

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


def test_marker_edge(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_edge_master') if master else 'marker_edge'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'],
             filename=name + '.png')

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


def test_marker_fill(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_master') if master else 'marker_fill'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'], marker_fill_color='#FFFFFF',
             filename=name + '.png')

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


def test_marker_fill_default(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_default_master') if master else 'marker_fill_default'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True,
             filename=name + '.png')

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


def test_marker_fill_alpha(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_alpha_master') if master else 'marker_fill_alpha'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True, marker_fill_alpha=0.5,
             filename=name + '.png')

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


def test_marker_boxplot(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'boxplot_master') if master else 'boxplot'

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                box_fill_color=[0, 0, 1, 1, 2, 2], box_fill_alpha=0.3, box_edge_width=0,
                marker_edge_color=[0, 0, 1, 1, 2, 2], marker_type=['o', '+'],
                box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                filename=name + '.png')

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


def test_marker_boxplot2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'boxplot2_master') if master else 'boxplot2'

    # Make the plot
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                box_marker_edge_color=[0, 0, 1, 1, 2, 2], marker_size=10,
                box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                box_marker_edge_alpha=0.6, box_marker_fill_alpha=1, box_marker_fill_color=['#00FF00', '#FF0000'],
                box_marker_fill=True, box_marker_type=['o'],
                filename=name + '.png')

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


def test_marker_boxplot3(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'boxplot3_master') if master else 'boxplot3'

    # Make the plot
    fcp.set_theme('_test', verbose=True)
    df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                marker_size=10, box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
                box_marker_edge_alpha=0.6, box_marker_fill_alpha=1, box_marker_type=['+'], verbose=True,
                filename=name + '.png')
    fcp.set_theme('gray', verbose=True)

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


def test_hist(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'hist_master') if master else 'hist'

    # Make the plot
    df_hist = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.hist(df_hist, x='Value', show=SHOW, legend='Region', hist_fill_alpha=1, filename=name + '.png')

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


def test_hist_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'hist_color_master') if master else 'hist_color'

    # Make the plot
    df_hist = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
    fcp.hist(df_hist, x='Value', show=SHOW, legend='Region', hist_fill_alpha=1,
             colors=['#FF0000', '#00FF11'],
             filename=name + '.png')

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


def test_marker_type(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_type_master') if master else 'marker_type'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', 'd'],
             filename=name + '.png')

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


def test_marker_type_none(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_type_none_master') if master else 'marker_type_none'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', None, '+', '*', 'B', None],
             filename=name + '.png')

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


def test_marker_size(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_size_master') if master else 'marker_size'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2,
             filename=name + '.png')

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


def test_marker_size_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_size_legend_master') if master else 'marker_size_legend'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2, legend_marker_size=2,
             filename=name + '.png')

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


def test_marker_size_column(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_size_column_master') if master else 'marker_size_column'

    # Make the plot
    df['marker_size'] = df['Voltage'] * 10
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name + '.png', inline=False, marker_size='marker_size')

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


def test_fonts(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'fonts_master') if master else 'fonts'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             title='My Oh My This Is Way Too Long of a Title for This Plot What Were You Thinking!',
             title_font_style='italic', label_y_font_size=25, label_y_style='normal',
             label_x_font_weight='normal', tick_labels_major_font='fantasy', filename=name + '.png')

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


def test_theme_white(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'theme_white_master') if master else 'theme_white'

    # Make the plot
    fcp.set_theme('white')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name + '.png')
    fcp.set_theme('gray')  # return to default gray

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


def test_theme_white_fly(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'theme_white_fly_master') if master else 'theme_white_fly'

    # Make the plot
    fcp.set_theme('gray')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], theme='white',
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name + '.png')
    fcp.set_theme('gray')  # return to default gray

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


def test_theme_gray(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'theme_gray_master') if master else 'theme_gray'

    # Make the plot
    fcp.set_theme('gray')
    fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name + '.png')

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


if __name__ == '__main__':
    pass
