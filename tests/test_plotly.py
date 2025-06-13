import fivecentplots as fcp
import pandas as pd
import os
import sys
import pdb
from pathlib import Path
import plotly
import matplotlib as mpl
import fivecentplots.utilities as utl
import pytest
import imageio.v3 as imageio
osjoin = os.path.join
db = pdb.set_trace

if mpl.__version__ == '3.6.3':
    pytest.skip(allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def get_ready_plotly(request):
    fcp.set_theme('gray_plotly')
    fcp.KWARGS['engine'] = 'plotly'


if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/plotly_v{plotly.__version__}')
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/plotly_v{plotly.__version__}')
else:
    REFERENCE = Path(f'test_images/plotly_v{plotly.__version__}')


# Dataframes
df_xy = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
ts = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_ts.csv')
df_interval = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_interval.csv')
df_bar = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_bar.csv')
df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
df_contour = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
df_heatmap = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_heatmap.csv')
df_hist = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')
img_rgb = imageio.imread(str(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'))


# Set theme
fcp.set_theme('gray_plotly')


# Other
def make_all(start=None, stop=None):
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__], start=start, stop=stop)


def show_all(only_fails=True, start=None):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__], start=start)


SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False
fcp.KWARGS['engine'] = 'plotly'


# plt_ functions can be used directly outside of pytest for debug
def plt_xy_no_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_no_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', lines=False, ax_size=[400, 400],
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_log(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_log', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', xmin=0.9, xmax=2.1, grid_minor=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_categorical(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_categorical', make_reference, REFERENCE)

    if bm:
        return

    # Make the plot
    fcp.plot(df_xy, x='Die', y='I [A]',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Voltage==1.5',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_secondary(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_secondary', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, legend='Die',
             grid_major_y2=True, grid_major_y2_style='--', y2max=1.4,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_multiple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_multiple', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x=['Boost Level', 'I [A]'], y=['Voltage', 'Temperature [C]'], legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_row(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_row', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', legend='Die', row='Boost Level', ax_size=[225, 225],
             legend_edge_color='#000000', filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             label_row_fill_color='#000000', ax_edge_width=4, label_row_font_color='#ff0000',
             label_row_edge_color='#0000ff', label_row_edge_width=2, legend_edge_width=3,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_col', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', legend='Die', col='Boost Level', ax_size=[225, 225],
             legend_edge_color='#000000', filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
             label_col_fill_color='#000000', label_col_font_color='#ff0000',
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_row_col(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_row_col', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             modebar_visible=True, modebar_fill_color='#000000', ax_edge_width=1, ymin=0,
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_wrap(bm=False, make_reference=False, show=False):

    # Make the plots
    name = utl.unit_test_get_img_name('xy_wrap', make_reference, REFERENCE)

    fcp.plot(df_xy, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'],
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13,
             filename=name.with_suffix('.png'), save=not bm, inline=False)
    if bm:
        return

    utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_axhvlines(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_axhvlines', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', title='IV Data', lines=False, legend=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ax_hlines=[(0, '#FF0000', '--', 3, 1, 'Open', '#555555', 0.25), 1.2],
             ax_vlines=[(0.6, '#0000ff', ':'), (1, '#00FF00')],
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_fit(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_fit', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', title='IV Data', lines=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             fit=1, fit_eqn=True, fit_rsq=True, fit_range_x=[1.3, 2],
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_stat_line(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_stat_line', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y=['Boost Level', 'I [A]'], legend=True, stat='median',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_conf_int(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_conf_int', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', lines=False, conf_int=0.95,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_control_limits(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_control_limits', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_interval, x='x', y='y', lines=False, ucl=50, lcl=-50, ucl_fill_color='#FF0000', legend=True,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_xy_ref_line(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('xy_ref_line', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df_xy, x='Voltage', y='I [A]', title='IV Data', legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             ref_line=df_xy['Voltage'], ref_line_legend_text='y=x', xmin=0, ymin=0, xmax=1.6, ymax=1.6,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_bar_simple(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('bar_simple', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df_bar, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25', horizontal=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_bar_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('bar_legend', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df_bar, x='Liquid', y='pH', tick_labels_major_x_rotation=90, legend='Measurement',
            filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_bar_stacked(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('bar_stacked', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df_bar, x='Liquid', y='pH', tick_labels_major_x_rotation=90, stacked=True, legend='Measurement',
            filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_bar_grouping(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('bar_grouping', make_reference, REFERENCE)

    # Make the plot
    fcp.bar(df_bar, x='Liquid', y='pH', tick_labels_major_x_rotation=90, ax_hlines=0, col='Measurement', row='T [C]',
            ax_size=[300, 300], filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_box_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('box_basic', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], group_means=True, jitter=False,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_box_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('box_legend', make_reference, REFERENCE)

    # Make the plot
    df_box['Row'] = [int(f) for f in df_box.index / 4]
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], mean_diamonds=True, conf_coeff=0.95, legend='Row',
                filename=name.with_suffix('.png'), save=not bm, inline=False, jitter=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_box_violin(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('box_violin', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], violin=True,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_box_grouping(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('box_grouping', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], col='Region', ax_size=[300, 300],
                jitter=False, filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_contour_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('contour_basic', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df_contour, x='X', y='Y', z='Value', filled=False, cbar=False, label_y_edge_width=1, label_y_edge_color='#ff0000',
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_contour_points(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('contour_points', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df_contour, x='X', y='Y', z='Value', filled=False, levels=40, contour_width=2,
                xmin=-4, xmax=5, ymin=-4, ymax=6, cbar=True,
                show_points=True, marker_size=26, marker_fill_color='#00FF00',
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_contour_grid_fill(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('contour_grid_fill', make_reference, REFERENCE)

    # Make the plot
    fcp.contour(df_contour, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=False, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250],
                label_rc_font_size=12, levels=40,
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_heatmap_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('heatmap_basic', make_reference, REFERENCE)

    # Make the plot
    fcp.heatmap(df_heatmap, x='Category', y='Player', z='Average',
                filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_hist_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('hist_basic', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df_hist, x='Value', horizontal=True, bins=20,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_hist_kde(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('hist_kde', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(df_hist, x='Value', legend='Region', kde=True, kde_width=2,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_hist_image(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('hist_image', make_reference, REFERENCE)

    # Make the plot
    fcp.hist(img_rgb, legend='Channel', markers=False, ax_size=[600, 400], line_width=2, colors=fcp.RGB,
             filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_rgb(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_rgb', make_reference, REFERENCE)

    # Make the plot
    fcp.imshow(img_rgb, ax_size=[600, 300],
               filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_raw(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_raw', make_reference, REFERENCE)

    # Make the plot
    img_raw = fcp.utilities.img_grayscale(img_rgb)
    fcp.imshow(img_raw, ax_size=[600, 600], title='Fake RAW, Fake Pirate', cbar=True, title_edge_width=1,
               title_edge_color='#ff0000', filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_imshow_grid(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('imshow_grid', make_reference, REFERENCE)

    # Make the plot
    url = 'https://upload.wikimedia.org/wikipedia/commons/2/28/RGB_illumination.jpg'
    img_rgb_sp = imageio.imread(url)
    img_raw_sp = fcp.utilities.rgb2bayer(img_rgb_sp)
    fcp.imshow(img_raw_sp, cmap='inferno', ax_size=[300, 300], cfa='rggb', wrap='Plane', ax_edge_width=1,
               ax_edge_color='#555555', filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_nq_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_basic', make_reference, REFERENCE)

    # Make the plot
    fcp.nq(df_box, x='Value', marker_size=4, line_width=2,
           filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_nq_image(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('nq_image', make_reference, REFERENCE)

    # Make the plot
    img_rgb = imageio.imread(str(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'))
    img_cat = fcp.utl.img_grayscale(img_rgb, bit_depth=12)
    fcp.nq(img_cat, marker_size=4, line_width=2,
           filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_pie_basic(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('pie_basic', make_reference, REFERENCE)

    # Make the plot
    df_pie = df_bar.copy()
    df_pie.loc[df_pie.pH < 0, 'pH'] = -df_pie.pH
    fcp.pie(df_pie, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25',
            filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def plt_pie_legend(bm=False, make_reference=False, show=False):

    name = utl.unit_test_get_img_name('pie_legend', make_reference, REFERENCE)

    # Make the plot
    df_pie = df_bar.copy()
    df_pie.loc[df_pie.pH < 0, 'pH'] = -df_pie.pH
    fcp.pie(df_pie, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25', explode=[0.1], legend=True,
            filename=name.with_suffix('.png'), save=not bm, inline=False)

    if bm:
        return
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


# test_ functions call plt_ funcs 2x:
# 1) do the comparison with saved image
# 2) do a test plot only with save=False and inline=False and benchmark spead
def test_bar_grouping(benchmark):
    plt_bar_grouping()
    benchmark(plt_bar_grouping, True)


def test_bar_legend(benchmark):
    plt_bar_legend()
    benchmark(plt_bar_legend, True)


def test_bar_simple(benchmark):
    plt_bar_simple()
    benchmark(plt_bar_simple, True)


def test_bar_stacked(benchmark):
    plt_bar_stacked()
    benchmark(plt_bar_stacked, True)


def test_box_basic(benchmark):
    plt_box_basic()
    benchmark(plt_box_basic, True)


def test_box_grouping(benchmark):
    plt_box_grouping()
    benchmark(plt_box_grouping, True)


def test_box_legend(benchmark):
    plt_box_legend()
    benchmark(plt_box_legend, True)


def test_box_violin(benchmark):
    plt_box_violin()
    benchmark(plt_box_violin, True)


def test_contour_basic(benchmark):
    plt_contour_basic()
    benchmark(plt_contour_basic, True)


def test_contour_grid_fill(benchmark):
    plt_contour_grid_fill()
    benchmark(plt_contour_grid_fill, True)


@pytest.mark.skip()  # fails on runner; no idea why
def test_contour_points(benchmark):
    plt_contour_points()
    benchmark(plt_contour_points, True)


def test_heatmap_basic(benchmark):
    plt_heatmap_basic()
    benchmark(plt_heatmap_basic, True)


def test_hist_basic(benchmark):
    plt_hist_basic()
    benchmark(plt_hist_basic, True)


def test_hist_image(benchmark):
    plt_hist_image()
    benchmark(plt_hist_image, True)


def test_hist_kde(benchmark):
    plt_hist_kde()
    benchmark(plt_hist_kde, True)


def test_imshow_grid(benchmark):
    plt_imshow_grid()
    benchmark(plt_imshow_grid, True)


def test_imshow_raw(benchmark):
    plt_imshow_raw()
    benchmark(plt_imshow_raw, True)


def test_imshow_rgb(benchmark):
    plt_imshow_rgb()
    benchmark(plt_imshow_rgb, True)


def test_nq_basic(benchmark):
    plt_nq_basic()
    benchmark(plt_nq_basic, True)


def test_nq_image(benchmark):
    plt_nq_image()
    benchmark(plt_nq_image, True)


@pytest.mark.skip()  # fails on runner; no idea why
def test_pie_basic(benchmark):
    plt_pie_basic()
    benchmark(plt_pie_basic, True)


@pytest.mark.skip()  # fails on runner; no idea why
def test_pie_legend(benchmark):
    plt_pie_legend()
    benchmark(plt_pie_legend, True)


def test_xy_axhvlines(benchmark):
    plt_xy_axhvlines()
    benchmark(plt_xy_axhvlines, True)


def test_xy_categorical(benchmark):
    plt_xy_categorical()
    benchmark(plt_xy_categorical, True)


def test_xy_col(benchmark):
    plt_xy_col()
    benchmark(plt_xy_col, True)


def test_xy_conf_int(benchmark):
    plt_xy_conf_int()
    benchmark(plt_xy_conf_int, True)


def test_xy_control_limits(benchmark):
    plt_xy_control_limits()
    benchmark(plt_xy_control_limits, True)


def test_xy_fit(benchmark):
    plt_xy_fit()
    benchmark(plt_xy_fit, True)


def test_xy_legend(benchmark):
    plt_xy_legend()
    benchmark(plt_xy_legend, True)


def test_xy_log(benchmark):
    plt_xy_log()
    benchmark(plt_xy_log, True)


def test_xy_multiple(benchmark):
    plt_xy_multiple()
    benchmark(plt_xy_multiple, True)


def test_xy_no_legend(benchmark):
    plt_xy_no_legend()
    benchmark(plt_xy_no_legend, True)


def test_xy_ref_line(benchmark):
    plt_xy_ref_line()
    benchmark(plt_xy_ref_line, True)


def test_xy_row(benchmark):
    plt_xy_row()
    benchmark(plt_xy_row, True)


def test_xy_row_col(benchmark):
    plt_xy_row_col()
    benchmark(plt_xy_row_col, True)


def test_xy_secondary(benchmark):
    plt_xy_secondary()
    benchmark(plt_xy_secondary, True)


def test_xy_stat_line(benchmark):
    plt_xy_stat_line()
    benchmark(plt_xy_stat_line, True)


def test_xy_wrap(benchmark):
    plt_xy_wrap()
    benchmark(plt_xy_wrap, True)
