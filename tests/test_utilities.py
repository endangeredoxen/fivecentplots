import imageio.v3 as imageio
import fivecentplots as fcp
import numpy as np
import pandas as pd
import os
import pdb
from pathlib import Path
import fivecentplots.utilities as utl
import fivecentplots.data as data
import fivecentplots.engines.layout as layout
import matplotlib as mpl
import pytest
osjoin = os.path.join
db = pdb.set_trace
mpl.use('agg')


@pytest.fixture(scope='session')
def img_cat():
    return imageio.imread(Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png')


@pytest.fixture
def df(scope='session'):
    return pd.read_csv(Path(fcp.__file__).parent / 'test_data/fake_data.csv')


def test_ci(df):
    np.testing.assert_almost_equal(utl.ci(df['I [A]']), (0.1422958885037856, 0.18313061803216213))
    assert np.isnan(utl.ci(pd.Series())[0])


def test_dfkwarg(df):
    kwargs = utl.dfkwarg(df, {}, data.XY)
    assert 'df' in kwargs

    kwargs = utl.dfkwarg(np.zeros((5, 5)), {}, data.ImShow)
    assert 'df' in kwargs

    with pytest.raises(data.DataError):
        utl.dfkwarg(1, {}, data.XY)


def test_df_filter(df):
    df_ = df.copy()
    df_['hi (%)'] = 0
    df_.loc[0:20, 'hi (%)'] = 1

    assert len(utl.df_filter(df_.copy(), 'Substrate=="Si" & Target Wavelength==460 & Boost Level==0.2')) == 102
    assert len(utl.df_filter(df_.copy(), 'Substrate=="Si" & (Target Wavelength==460 | Boost Level==0.2)')) == 510
    assert len(utl.df_filter(df_.copy(), 'hi (%)==1')) == 21
    assert len(utl.df_filter(df_.copy(), 'Substrate not in ["Si", "GaAS"]')) == 918
    assert len(utl.df_filter(df_.copy(), 'Substrate in ["Si", "GaAs"]')) == 918
    assert len(utl.df_filter(df_.copy(), 'hi (%)==1', keep_filtered=True).dropna()) == 21
    assert 'hi (%)' not in utl.df_filter(df_.copy(), 'hi (%)==1', drop_cols=True).columns

    df_bad = utl.df_filter(df_.copy(), 'boom="yo"')
    pd.testing.assert_frame_equal(df_, df_bad)


def test_df_int_cols():
    test = pd.DataFrame(columns=range(0, 10), index=range(0, 10))
    test.loc[:, :] = 33
    test['say'] = 'it'
    test['aint'] = 'so'
    assert utl.df_int_cols(test) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert utl.df_int_cols(test, True) == ['say', 'aint']


def test_df_int_cols_convert():
    test = pd.DataFrame(columns=[str(f) for f in range(0, 10)], index=range(0, 10))
    test.loc[:, :] = 33
    test_ = utl.df_int_cols_convert(test, force=True)
    np.testing.assert_equal(test_.columns, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype='int64'))


def test_df_summary(df):
    df['two'] = 'turntables'
    summ = utl.df_summary(df, exclude=['Target Wavelength'])
    assert summ.Die.iloc[0] == 'Multiple'
    assert summ.two.iloc[0] == 'turntables'
    assert 'Target Wavelength' not in summ.columns

    summ = utl.df_summary(df, columns=['Substrate'], multiple=True)
    assert summ.values[0][0] == 'Si; InP'


def test_df_unique(df):
    df['two'] = 'turntables'
    df_ = utl.df_unique(df)
    assert df_['two'] == 'turntables'


def test_get_current_values(df):
    df['enemies'] = 'friends'
    assert utl.get_current_values(df, 'we hate our @enemies@') == 'we hate our friends'


def test_get_decimals():
    assert utl.get_decimals(1.3420001) == 3
    assert utl.get_decimals(1.3420001, max_places=2) == 1


def test_get_text_dimensions():
    dim = (259, 20)
    assert utl.get_text_dimensions('no alarms and no surprises', 'Deja Vu Sans', 12, 'normal', 'bold') == dim


def test_kwget():
    kwargs = {'ax_edge_color': '#000000'}
    fcpp = {'ax_edge_color': '#FF0000'}

    assert utl.kwget(kwargs, fcpp, 'ax_edge_color', '#00FF00') == '#000000'
    kwargs = {}
    assert utl.kwget(kwargs, fcpp, 'ax_edge_color', '#00FF00') == '#FF0000'
    fcpp = {}
    assert utl.kwget(kwargs, fcpp, 'ax_edge_color', '#00FF00') == '#00FF00'


def test_img_array_from_df():
    # Case 1: 2D RAW style
    df = pd.DataFrame({'Row': [0, 0, 0, 1, 1, 1, 2, 2, 2], 'Column': [0, 1, 2, 0, 1, 2, 0, 1, 2],
                      'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    shape = [3, 3]
    arr = utl.img_array_from_df(df, shape)
    np.testing.assert_array_equal(arr, np.arange(1, 10).reshape(3, 3))

    # Case 2: 3D RGB style
    df.columns = ['Row', 'Column', 'R']
    df['G'] = (2 * df['R']).astype(int)
    df['B'] = (2 * df['G']).astype(int)
    shape = [3, 3, 3]
    arr = utl.img_array_from_df(df, shape)
    test = np.ones((3, 3, 3))
    test[:, :, 0] = np.arange(1, 10).reshape(3, 3)
    test[:, :, 1] = (2 * np.arange(1, 10)).reshape(3, 3)
    test[:, :, 2] = (4 * np.arange(1, 10)).reshape(3, 3)
    np.testing.assert_array_equal(arr, test)

    # Case 3: 3D RGBA style
    df['A'] = 1
    shape = [3, 3, 4]
    arr = utl.img_array_from_df(df, shape)
    test = np.ones((3, 3, 4))
    test[:, :, 0] = np.arange(1, 10).reshape(3, 3)
    test[:, :, 1] = (2 * np.arange(1, 10)).reshape(3, 3)
    test[:, :, 2] = (4 * np.arange(1, 10)).reshape(3, 3)
    test[:, :, 3] = np.ones((3, 3))
    np.testing.assert_array_equal(arr, test)


def test_img_compare():
    img1 = Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'
    img2 = Path(fcp.__file__).parent / 'test_data/imshow_cat_pirate.png'
    img3 = Path(fcp.__file__).parent / 'test_data/hist_patch.png'

    assert not utl.img_compare(img1, img2)
    assert utl.img_compare(img1, img3)
    assert utl.img_compare(Path('hi'), Path('no'))
    assert utl.img_compare(img1, None)


def test_img_df_transform():
    # Case 1: already in the format
    df = pd.DataFrame({'Row': [0, 0, 0, 1, 1, 1, 2, 2, 2], 'Column': [0, 1, 2, 0, 1, 2, 0, 1, 2],
                      'Value': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
    df_groups, imgs = utl.img_df_transform(df)
    assert np.all(imgs[0].columns == ['Row', 'Column', 'Value'])
    np.testing.assert_array_equal(df_groups.values[0], np.array([3, 3, 1]))

    # Case 2: input is 2D DataFrame
    df = pd.DataFrame(columns=range(0, 5), index=range(0, 5), dtype=np.float64)
    df.loc[:, :] = 0.0
    df_groups, imgs = utl.img_df_transform(df)
    assert np.all(imgs[0].columns == ['Value', 'Row', 'Column'])
    np.testing.assert_array_equal(df_groups.values[0], np.array([5, 5, 1]))

    # Case 3: input is numpy array 2D
    df_groups, imgs = utl.img_df_transform(np.zeros((5, 5)))
    assert np.all(imgs[0].columns == ['Value', 'Row', 'Column'])
    np.testing.assert_array_equal(df_groups.values[0], np.array([5, 5, 1]))

    # Case 4: input is numpy array 3D RGB
    df_groups, imgs = utl.img_df_transform(np.ones((5, 5, 3)))
    assert np.all(imgs[0].columns == ['R', 'G', 'B', 'Row', 'Column'])
    np.testing.assert_array_equal(df_groups.values[0], np.array([5, 5, 3]))

    # Case 5: input is numpy array 3D RGBA
    df_groups, imgs = utl.img_df_transform(np.ones((5, 5, 4)))
    assert np.all(imgs[0].columns == ['R', 'G', 'B', 'A', 'Row', 'Column'])
    np.testing.assert_array_equal(df_groups.values[0], np.array([5, 5, 4]))

    # Case 6: wrong dtype
    with pytest.raises(TypeError):
        utl.img_df_transform('karma police')


def test_img_grayscale(img_cat):
    np.testing.assert_almost_equal(utl.img_grayscale_deprecated(img_cat).stack().mean(), 164.82078274505002)


def test_nq(img_cat):
    img_df = utl.img_grayscale_deprecated(img_cat)
    nq = utl.nq(img_df)
    np.testing.assert_almost_equal(nq.loc[nq.Sigma == 0]['Value'].values[0], img_df.stack().median())

    kwargs = {'sigma': 4, 'tail': 2, 'step_tail': 0.3, 'step_inner': 0.1, 'column': 'Boom'}
    nq = utl.nq(img_df, **kwargs)
    np.testing.assert_almost_equal(nq['Sigma'].min(), -4)
    np.testing.assert_almost_equal(nq['Sigma'].max(), 3.8)
    np.testing.assert_almost_equal(nq['Sigma'].diff().iloc[1], 0.3)
    np.testing.assert_almost_equal(nq['Sigma'].diff().iloc[41], 0.1)


def test_pie_wedge_labels():
    test = utl.pie_wedge_labels(np.array(['hi', 'this', 'is', 'loooooooooooooooooooooooong']),
                                np.array([0, 0.25, 0.5, 0.75, 1]), 0)
    assert test == (3, 4)

    test = utl.pie_wedge_labels(np.array(['hi', 'this', 'is', 'loooooooooooooooooooooooong']),
                                np.array([80, 0, 0.25, 0.5, 0.75]), 90)
    assert test == (0, 0)


def test_plot_num():
    assert utl.plot_num(0, 0, 2) == 1
    assert utl.plot_num(1, 0, 2) == 3
    assert utl.plot_num(0, 1, 2) == 2
    assert utl.plot_num(7, 3, 4) == 32


def test_rgb2bayer(img_cat):
    df_ = utl.rgb2bayer(img_cat)
    np.testing.assert_almost_equal(df_[::2, ::2].mean(), 175.746838)
    np.testing.assert_almost_equal(df_[1::2, ::2].mean(), 161.181442)
    np.testing.assert_almost_equal(df_[::2, 1::2].mean(), 161.2675)
    np.testing.assert_almost_equal(df_[1::2, 1::2].mean(), 154.865044)


def test_rectangle_overlap():
    r1 = [10, 10, (0, 0)]
    r2 = [5, 5, (2, 0)]
    r3 = [10, 10, (50, 50)]
    assert utl.rectangle_overlap(r1, r2) is True
    assert utl.rectangle_overlap(r1, r3) is False


def test_reload_defaults():
    fcp_params, colors, markers, rcParams = utl.reload_defaults('white.py')
    assert fcp_params['ax_fill_color'] == '#ffffff'
    fcp_params, colors, markers, rcParams = utl.reload_defaults(Path(fcp.__file__).parent / 'themes/white.py')
    assert fcp_params['ax_fill_color'] == '#ffffff'
    fcp_params, colors, markers, rcParams = utl.reload_defaults()
    assert fcp_params['ax_fill_color'] == '#eaeaea'


def test_repeated_list():
    with pytest.raises(ValueError):
        test = utl.RepeatedList([], 'fake_plastic_trees')
    with pytest.raises(ValueError):
        test = utl.RepeatedList(None, 'fake_plastic_trees')

    test = utl.RepeatedList([1, 2, 3], 'street_spirit', override={1: -1})
    assert test[0] == 1
    assert test[(0, 1)] == -1


def test_see():
    obj = layout.Element('boss')
    assert utl.see(obj).set_index('Attribute').loc['name'].iloc[0] == 'boss'


def test_set_save_filename(df):
    dd = data.XY(df=df, x='Voltage', y='I Set')
    ll = layout.BaseLayout(dd)
    assert utl.set_save_filename(df, 0, None, None, ll, {'save_ext': 'png'}) == 'I Set.png'
    assert utl.set_save_filename(df, 0, None, None, ll, {'filename': 'hi.png'}) == 'hi.png'
    assert utl.set_save_filename(df, 0, None, None, ll, {'filename': 'hi', 'save_ext': 'png'}) == 'hi.png'
    assert utl.set_save_filename(df, 0, None, None, ll, {'filename': Path('boss/hi.png')}) == 'boss/hi.png'
    assert utl.set_save_filename(df, 1, 'Boost Level', ['Boost Level'], ll, {'save_ext': 'png'}) == \
        'I Set where Boost Level=Boost Level.png'

    # row
    dd = data.XY(df=df, x='Voltage', y='I Set', row='Boost Level')
    ll = layout.BaseLayout(dd)
    assert utl.set_save_filename(df, 0, None, None, ll, {'save_ext': 'png'}) == 'I Set by Boost Level.png'

    # col
    dd = data.XY(df=df, x='Voltage', y='I Set', col='Boost Level')
    ll = layout.BaseLayout(dd)
    assert utl.set_save_filename(df, 0, None, None, ll, {'save_ext': 'png'}) == 'I Set by Boost Level.png'

    # wrap
    dd = data.XY(df=df, x='Voltage', y='I Set', wrap='Boost Level')
    ll = layout.BaseLayout(dd)
    assert utl.set_save_filename(df, 0, None, None, ll, {'save_ext': 'png'}) == 'I Set by Boost Level.png'

    # twinx
    kwargs = {'df': df, 'x': 'Voltage', 'y': ['I Set', 'I [A]'], 'twin_x': True, 'save_ext': 'png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, None, None, ll, {'save_ext': 'png'}) == 'I Set + I [A].png'

    # twiny
    kwargs = {'df': df, 'y': 'Voltage', 'x': ['I Set', 'I [A]'], 'twin_y': True, 'save_ext': 'png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, None, None, ll, kwargs) == 'Voltage vs I Set + I [A].png'

    # z
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'save_ext': 'png', 'z': 'Boost Level'}
    dd = data.Heatmap(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, None, None, ll, kwargs) == 'Boost Level vs Voltage vs I Set.png'

    # groups
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'groups': 'Die', 'save_ext': 'png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, None, None, ll, kwargs) == 'Voltage vs I Set by Die.png'

    # fig
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'fig_groups': 'Die', 'save_ext': 'png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, '(-1, 2)', ['Die'], ll, kwargs) == 'Voltage vs I Set where Die=(-1, 2).png'

    # fig2
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'fig_groups': 'Die', 'filename': 'hi.png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, '(-1, 2)', ['Die'], ll, kwargs) == 'hi where Die=(-1, 2).png'

    # fig3
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'fig_groups': ['Die', 'Boost Level'], 'save_ext': 'png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, ['(-1, 2)', '0.2'], ['Die', 'Boost Level'], ll, kwargs) == \
        'Voltage vs I Set where Die=(-1, 2) and where Boost Level=0.2.png'

    # fig4
    kwargs = {'df': df, 'y': 'Voltage', 'x': 'I Set', 'fig_groups': ['Die', 'Boost Level'], 'filename': 'hi.png'}
    dd = data.XY(**kwargs)
    ll = layout.BaseLayout(dd, **kwargs)
    kwargs.pop('df')
    assert utl.set_save_filename(df, 0, ['(-1, 2)', '0.2'], ['Die', 'Boost Level'], ll, kwargs) == \
        'hi where Die=(-1, 2) and where Boost Level=0.2.png'


def test_sigma(df):
    assert utl.sigma(df['I Set']) == 3


def test_split_color_planes(img_cat):
    # Case 1: DataFrame input, split to dictionary
    img = utl.img_grayscale_deprecated(img_cat)  # this is a DataFrame
    img_cp = utl.split_color_planes(img, cfa='grbg')
    np.testing.assert_almost_equal(img_cp['r'].stack().mean(), 164.84421923940002)
    np.testing.assert_almost_equal(img_cp['b'].stack().mean(), 164.79765575099998)

    # Case 2: DataFrame input, split to DataFrame
    img = utl.img_grayscale_deprecated(img_cat)  # this is a DataFrame
    img_cp = utl.split_color_planes(img, cfa='grbg', as_dict=False)
    red = img_cp.loc[img_cp.Plane == 'r']
    blue = img_cp.loc[img_cp.Plane == 'b']
    np.testing.assert_almost_equal(red[utl.df_int_cols(red)].stack().mean(), 164.84421923940002)
    np.testing.assert_almost_equal(blue[utl.df_int_cols(blue)].stack().mean(), 164.79765575099998)

    # Case 3: NDArray input, split to dict
    img = utl.img_grayscale_deprecated(img_cat, as_array=True)  # this is a np.array
    img_cp = utl.split_color_planes(img, cfa='grbg')
    np.testing.assert_almost_equal(img_cp['r'].mean(), 164.84421923940002)
    np.testing.assert_almost_equal(img_cp['b'].mean(), 164.79765575099998)

    # Case 4: NDArray input, split to DataFrame
    img = utl.img_grayscale_deprecated(img_cat, as_array=True)  # this is a np.array
    img_cp = utl.split_color_planes(img, cfa='grbg', as_dict=False)
    red = img_cp.loc[img_cp.Plane == 'r']
    blue = img_cp.loc[img_cp.Plane == 'b']
    np.testing.assert_almost_equal(red[utl.df_int_cols(red)].stack().mean(), 164.84421923940002)
    np.testing.assert_almost_equal(blue[utl.df_int_cols(blue)].stack().mean(), 164.79765575099998)


def test_timer():
    test = utl.Timer()
    assert test.get() is None
    test.start()
    test.get('hi')
    assert test.total > 0
    test.stop()

    test = utl.Timer(start=True, units='ms', print=True)
    test.get('hi', stop=True)
    assert test.total > 0
    test.get_total()

    with pytest.raises(ValueError):
        test = utl.Timer(start=True, units='gas')


def test_validate_list():
    assert utl.validate_list(None) is None
    assert utl.validate_list(('let', 'down')) == ['let', 'down']
    assert utl.validate_list(np.array(['let', 'down'])) == ['let', 'down']
    assert utl.validate_list('ok computer') == ['ok computer']
    assert utl.validate_list(['climbing', 'up', 'the', 'walls']) == ['climbing', 'up', 'the', 'walls']


def test_split_color_planes_error():
    with pytest.raises(utl.CFAError):
        utl.split_color_planes(np.zeros([2, 2]), cfa='hi')
