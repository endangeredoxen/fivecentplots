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

test = 'misc'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
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


def test_text_box_single(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_single', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[120, 10],
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_text_box_single_style(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_single_style', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows\nbest response', text_position=[10, 340], text_font_size=20,
             text_edge_color='#FF0000', text_font_color='#00FF00', text_fill_color='#ffffff',
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_text_box_multiple(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_multiple', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text=['Die (-1,2) shows best response', '(c) 2019', 'Boom!'],
             text_position=[[10, 379], [10, 10], [320, 15]], text_font_color=['#000000', '#FF00FF'],
             text_font_size=[14, 8, 18], text_fill_color='#FFFFFF',
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_text_box_position_figure(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_position_figure', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[208, 85], text_coordinate='figure',
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_text_box_position_data(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_position_data', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[1.077, 0.00085], text_coordinate='data',
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_text_box_position_units(make_reference=False, remove=True, show=False):

    name = utl.unit_test_get_img_name('text_box_position_units', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             text='Die (-1,2) shows best response', text_position=[0.3, 0.025], text_units='relative',
             save=True, inline=False, filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)
