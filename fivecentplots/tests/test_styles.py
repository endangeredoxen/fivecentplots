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
MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', MPL, 'styles.py')

df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data.csv'))


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


def test_fill_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'fill_color_master') if master else 'fill_color'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],          filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', legend_fill_color='#FF0000', ax_fill_color='#FFFFFF',
             label_x_fill_color='#0000FF', label_y_fill_color='#FF00FF',
             tick_labels_major_fill_color='#AAFB05',
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


def test_edge_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'edge_color_master') if master else 'edge_color'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_edge_color='#00FF00', legend_edge_color='#FF0000', ax_edge_color='#FFFFFF',
             label_x_edge_color='#0000FF', label_y_edge_color='#FF00FF',
             tick_labels_major_edge_color='#AAFB05', tick_labels_major_edge_width=5,
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


def test_spines(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'spines_master') if master else 'spines'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_edge_color='#FF0000', spine_left=False, spine_right=False,
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


def test_alpha(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_master') if master else 'alpha'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fig_fill_color='#00FF00', fig_fill_alpha=0.5,
             legend_fill_color='#FF0000', legend_fill_alpha=0.52,
             ax_fill_color='#FFFFFF', ax_fill_alpha=0.7,
             label_x_fill_color='#0000FF', label_x_fill_alpha=0.2,
             label_y_fill_color='#FF00FF', label_y_fill_alpha=0.2,
             tick_labels_major_fill_color='#AAFB05', tick_labels_major_fill_alpha=0.45,
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


def test_alpha_marker(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_marker_master') if master else 'alpha_marker'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3,
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


def test_alpha_legend_marker(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'alpha_legend_marker_master') if master else 'alpha_legend_marker'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_alpha=0.3, legend_marker_alpha=0.3,
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


def test_line_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_master') if master else 'line_color'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_line_color_custom(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_custom_master') if master else 'line_color_custom'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=['#FF0000', '#FF7777', '#FFAAAA'],
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


def test_line_color_index(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_index_master') if master else 'line_color_index'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 3, 3, 6, 6],
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


def test_line_color_cmap(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_color_cmap_master') if master else 'line_color_cmap'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             cmap='inferno',
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


def test_line_style(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style_master') if master else 'line_style'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             lines_alpha=0.33, lines_style='--', lines_width=3,
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


def test_line_style2(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style2_master') if master else 'line_style2'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             line_alpha=0.33, line_style='--', line_width=3,
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


def test_line_style_by_line(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'line_style_by_line_master') if master else 'line_style_by_line'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], markers=False,          filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             colors=[0, 0, 1, 1, 2, 2],
             lines_width=[3, 1, 3, 1, 3, 1], lines_style=['--', '-'],
             lines_alpha=[0.6, 1],
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


def test_marker_edge(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_edge_master') if master else 'marker_edge'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'],
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


def test_marker_fill(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_master') if master else 'marker_fill'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_edge_color=['#555555'], marker_fill_color='#FFFFFF',
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


def test_marker_fill_default(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_default_master') if master else 'marker_fill_default'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True,
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


def test_marker_fill_alpha(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_fill_alpha_master') if master else 'marker_fill_alpha'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_fill=True, marker_fill_alpha=0.5,
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


def test_marker_boxplot(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'boxplot_master') if master else 'boxplot'

    # Make the plot
    df_box = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))
    fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], show=SHOW,
                box_fill_color=[0, 0, 1, 1, 2, 2], box_fill_alpha=0.3, box_edge_width=0,
                marker_edge_color=[0, 0, 1, 1, 2, 2], marker_type=['o', '+'],
                box_whisker_color=[0, 0, 1, 1, 2, 2], box_whisker_width=1, jitter=False,
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


def test_hist(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'hist_master') if master else 'hist'

    # Make the plot
    df_hist = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))
    fcp.hist(df=df_hist, x='Value', show=SHOW, legend='Region',
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


def test_hist_color(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'hist_color_master') if master else 'hist_color'

    # Make the plot
    df_hist = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))
    fcp.hist(df=df_hist, x='Value', show=SHOW, legend='Region', hist_fill_alpha=1,
             colors=['#FF0000', '#00FF11'],
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


def test_marker_type(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_type_master') if master else 'marker_type'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', 'd'],
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


def test_marker_type_none(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_type_none_master') if master else 'marker_type_none'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], \
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             markers=['o', None, '+', '*', 'B', None],
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


def test_marker_size(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_size_master') if master else 'marker_size'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2,
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


def test_marker_size_legend(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'marker_size_legend_master') if master else 'marker_size_legend'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             marker_size=2, legend_marker_size=2,
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


def test_fonts(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'fonts_master') if master else 'fonts'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             title='My Plot', title_font_style='italic',
             label_y_font_size=25, label_y_style='normal',
             label_x_font_weight='normal',
             tick_labels_major_font='fantasy',
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


def test_theme_white(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'theme_white_master') if master else 'theme_white'

    # Make the plot
    fcp.set_theme('white')
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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


def test_theme_gray(master=False, remove=True, show=False):

    name = osjoin(MASTER, 'theme_gray_master') if master else 'theme_gray'

    # Make the plot
    fcp.set_theme('gray')
    fcp.plot(df=df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'],
             filter='Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
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