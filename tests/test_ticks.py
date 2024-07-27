import fivecentplots as fcp
import pandas as pd
import numpy as np
import os
import sys
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import matplotlib as mpl
import inspect
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')

test = 'ticks'
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
def make_all():
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__])
def show_all(only_fails=True):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__])
SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def test_grid_major(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_major_master') if master else 'grid_major'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_major_off(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_major_off_master') if master else 'grid_major_off'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             grid_major=False, ticks=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_major_off_y(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_major_off_y_master') if master else 'grid_major_off_y'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             grid_major_y=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_major_secondary(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_major_secondary_master') if master else 'grid_major_secondary'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             grid_major_y2=True, grid_major_y2_style='--', filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_minor(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_minor_master') if master else 'grid_minor'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             grid_minor=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_logit(master=False, remove=True, show=False):
    """This scale doesn't seem to work.  Punting for now"""
    name = osjoin(REFERENCE, 'grid_logit_master') if master else 'grid_logit'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die', ax_scale='logit',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_grid_symlog(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'grid_symlog_master') if master else 'grid_symlog'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die', ax_scale='symlog',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_ticks_minor(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'ticks_minor_master') if master else 'ticks_minor'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_minor=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_ticks_style(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'ticks_style_master') if master else 'ticks_style'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_major_direction='out', ticks_major_color='#aaaaaa', ticks_major_length=5, ticks_major_width=0.8,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_ticks_inc(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'ticks_inc_master') if master else 'ticks_inc'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_major_y_increment=0.2, filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_ticks_minor_number(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'ticks_minor_number_master') if master else 'ticks_minor_number'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_minor_x_number=5, ticks_minor_y_number=10, ticks_minor_y2_number=4,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_ticks_minor_number_log(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'ticks_minor_number_log_master') if master else 'ticks_minor_number_log'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_minor_x_number=5, ticks_minor_y_number=10, ticks_minor_y2_number=4, ax_scale='logy',
             ax2_scale='linear', filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_tight(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_labels_tight_master') if master else 'tick_labels_tight'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die', tick_labels_minor=True,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ax_size=[150, 150], filename=name.with_suffix('.png'), ticks_major_y_increment=0.05, ticks_major_x_increment=0.1)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_labels_log(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_labels_log_master') if master else 'tick_labels_log'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die', ax_scale='logy', ymin=0.8e-2, ymax=2e-2,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_major=False, filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_labels_log2(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_labels_log2_master') if master else 'tick_labels_log2'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die', ax_scale='logy', ymin=1.1e-2, ymax=2e-2,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ticks_major=False, filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_labels_minor(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_labels_minor_master') if master else 'tick_labels_minor'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             tick_labels_minor=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_cleanup(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_cleanup_master') if master else 'tick_cleanup'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             tick_labels_minor=True, ax_scale='logy', ax2_scale='lin', ticks_minor_x_number=5,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_cleanup_off(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_cleanup_off_master') if master else 'tick_cleanup_off'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             tick_labels_minor=True, ax_scale='logy', ax2_scale='lin', ticks_minor_x_number=5, tick_cleanup=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_tick_cleanup2(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'tick_cleanup2_master') if master else 'tick_cleanup2'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             tick_labels_minor=True, ax_scale='logy', ax2_scale='lin', ticks_minor_x_number=5,
             ax_size=[600, 400], tick_labels_minor_x_rotation=90,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_lin_sci(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'lin_sci_master') if master else 'lin_sci'

    # Make the plot
    x = np.linspace(1, 10, 10)
    y = np.linspace(1E-19, 1E-18, 10)
    fcp.plot(pd.DataFrame({'x': x, 'y': y}), x='x', y='y',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_lin_sci2(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'lin_sci2_master') if master else 'lin_sci2'

    # Make the plot
    x = np.linspace(1, 10, 10)
    y = np.linspace(1E18, 1E19, 10)
    fcp.plot(pd.DataFrame({'x': x, 'y': y}), x='x', y='y',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_lin_sci_off(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'lin_sci_off_master') if master else 'lin_sci_off'

    # Make the plot
    x = np.linspace(1, 10, 10)
    y = np.linspace(1E18, 1E19, 10)
    fcp.plot(pd.DataFrame({'x': x, 'y': y}), x='x', y='y', sci_y=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_log_sci(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'log_sci_master') if master else 'log_sci'

    # Make the plot
    fcp.plot(df, x='Voltage', y='Voltage', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ax_scale='logy', ymin=0.00001, ymax=100000000,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_log_sci2(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'log_sci2_master') if master else 'log_sci2'

    # Make the plot
    fcp.plot(df, x='Voltage', y='Voltage', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ax_scale='logy', ymin=0.00001, ymax=100000000, sci_y=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_log_exp(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'log_exp_master') if master else 'log_exp'

    # Make the plot
    fcp.plot(df, x='Voltage', y='Voltage', show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ax_scale='logy', ymin=0.00001, ymax=100000000, sci_y=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_sciz(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'sciz_master') if master else 'sciz'

    # Make the plot
    df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
    fcp.contour(df2, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=40, sci_z=True,
                filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_sciz_remove(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'sciz_remove_master') if master else 'sciz_remove'

    # Make the plot
    df2 = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_contour.csv')
    fcp.contour(df2, x='X', y='Y', z='Value', row='Batch', col='Experiment', filled=True,
                cbar=True, xmin=-3, xmax=3, ymin=-3, ymax=3, ax_size=[250, 250], show=SHOW,
                label_rc_font_size=12, levels=40, sci_z=True, tick_cleanup='remove',
                filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


fcp.set_theme('gray')


if __name__ == '__main__':
    pass
