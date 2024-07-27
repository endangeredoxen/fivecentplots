
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
mpl.use('agg')

test = 'ranges'
if Path('../tests/test_images').exists():
    REFERENCE = Path(f'../tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
elif Path('tests/test_images').exists():
    REFERENCE = Path(f'tests/test_images/mpl_v{mpl.__version__}') / f'{test}.py'
else:
    REFERENCE = Path(f'test_images/mpl_v{mpl.__version__}') / f'{test}.py'

# Sample data
df = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')
df_box = pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data_box.csv')


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


def test_default(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'default_master') if master else 'default'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)
             & (df['Boost Level'] == 0.2) & (df['Temperature [C]'] == 25)]
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'primary_master') if master else 'primary'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_qgroups(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'primary_qgroups_master') if master else 'primary_qgroups'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, groups='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmin='q1', xmax='q99', filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_no_scale(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'primary_no-auto-scale_master') if master else 'primary_no-auto-scale'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_explicit(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'primary_explicit_master') if master else 'primary_explicit'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'secondary_master') if master else 'secondary'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'secondary_limits_master') if master else 'secondary_limits'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmin=1.3,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits_no_scale(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'secondary_no-auto-scale_master') if master else 'secondary_no-auto-scale'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits_y(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'secondary_y-limit_master') if master else 'secondary_y-limit'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=1,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_multiple(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'multiple_master') if master else 'multiple'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_multiple_scaled(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'multiple_scaled_master') if master else 'multiple_scaled'

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=0.05,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_boxplot(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'boxplot_master') if master else 'boxplot'

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
                filename=name.with_suffix('.png'), jitter=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_boxplot_quantile(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'boxplot_quantile_master') if master else 'boxplot_quantile'

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW, ymax='95q',
                filename=name.with_suffix('.png'), jitter=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_boxplot_iqr(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'boxplot_iqr_master') if master else 'boxplot_iqr'

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
                ymin='1.5*iqr', ymax='1.5*iqr',
                filename=name.with_suffix('.png'), jitter=False)
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_master') if master else 'shared'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225],
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_false(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_false_master') if master else 'shared_false'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225], share_x=False, share_y=False,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_separate(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_separate_master') if master else 'shared_separate'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level',
             row='Temperature [C]', show=SHOW, ax_size=[225, 225],
             separate_ticks=True, separate_labels=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_rows(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_rows_master') if master else 'shared_rows'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level',
             row='Temperature [C]', show=SHOW, ax_size=[225, 225], share_row=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_cols(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_columns_master') if master else 'shared_columns'

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225], share_col=True,
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_no(master=False, remove=True, show=False):

    name = osjoin(REFERENCE, 'shared_no_master') if master else 'shared_no'

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=14,
             xmin=[0, 0.1, 0.2, 0.3, 0.4], ymax=[1, 2, 3, 4, 5, 6],
             filename=name.with_suffix('.png'))
    utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
