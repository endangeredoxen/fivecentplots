
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
def make_all(start=None, stop=None):
    utl.unit_test_make_all(REFERENCE, sys.modules[__name__], start=start, stop=stop)


def show_all(only_fails=True, start=None):
    utl.unit_test_show_all(only_fails, REFERENCE, sys.modules[__name__], start=start)


SHOW = False
fcp.KWARGS['save'] = True
fcp.KWARGS['inline'] = False


def test_boxplot(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
                filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_boxplot_iqr(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot_iqr', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
                ymin='1.5*iqr', ymax='1.5*iqr', filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_boxplot_quantile(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('boxplot_quantile', make_reference, REFERENCE)

    # Make the plot
    fcp.boxplot(df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW, ymax='q95',
                filename=name.with_suffix('.png'), jitter=False)
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_default(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('default', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)
             & (df['Boost Level'] == 0.2) & (df['Temperature [C]'] == 25)]
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_multiple(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('multiple', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_multiple_scaled(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('multiple_scaled', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=0.05,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('primary', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_qgroups(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('primary_qgroups', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', show=SHOW, groups='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmin='q1', xmax='q99', filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_no_scale(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('primary_no-auto-scale', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_primary_explicit(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('primary_explicit', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary_limits', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmin=1.3,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits_no_scale(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary_no-auto-scale', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmax=1.2, auto_scale=False,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_secondary_limits_y(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('secondary_y-limit', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=1,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_false(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared_false', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225], share_x=False, share_y=False,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_separate(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared_separate', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level',
             row='Temperature [C]', show=SHOW, ax_size=[225, 225],
             separate_ticks=True, separate_labels=True,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_rows(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared_rows', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level',
             row='Temperature [C]', show=SHOW, ax_size=[225, 225], share_row=True,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_cols(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared_columns', make_reference, REFERENCE)

    # Make the plot
    sub = df[(df.Substrate == 'Si') & (df['Target Wavelength'] == 450)].copy()
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             show=SHOW, ax_size=[225, 225], share_col=True,
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


def test_shared_no(make_reference=False, show=False):

    name = utl.unit_test_get_img_name('shared_no', make_reference, REFERENCE)

    # Make the plot
    fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',
             ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=14,
             xmin=[0, 0.1, 0.2, 0.3, 0.4], ymax=[1, 2, 3, 4, 5, 6],
             filename=name.with_suffix('.png'))
    return utl.unit_test_options(make_reference, show, name, REFERENCE)


if __name__ == '__main__':
    pass
