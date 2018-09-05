
import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb
import fivecentplots.utilities as utl
import inspect
osjoin = os.path.join
st = pdb.set_trace

MASTER = osjoin(os.path.dirname(fcp.__file__), 'tests', 'test_images', 'plot.py')

# Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data.csv'))
df_box = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data_box.csv'))


# Set theme
# fcp.set_theme('gray')
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


def test_default(master=False, remove=True):

    name = osjoin(MASTER, 'default_master') if master else 'default'

    # Make the plot
    sub = df[(df.Substrate=='Si')&(df['Target Wavelength']==450)&(df['Boost Level']==0.2)&(df['Temperature [C]']==25)]
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_default(master=False, remove=True):

    name = osjoin(MASTER, 'default_master') if master else 'default'

    # Make the plot
    sub = df[(df.Substrate=='Si')&(df['Target Wavelength']==450)&(df['Boost Level']==0.2)&(df['Temperature [C]']==25)]
    fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_primary(master=False, remove=True):

    name = osjoin(MASTER, 'primary_master') if master else 'primary'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_primary_no_scale(master=False, remove=True):

    name = osjoin(MASTER, 'primary_no-auto-scale_master') if master else 'primary_no-auto-scale'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_primary_explicit(master=False, remove=True):

    name = osjoin(MASTER, 'primary_explicit_master') if master else 'primary_explicit'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
             xmax=1.2, auto_scale=False,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_secondary(master=False, remove=True):

    name = osjoin(MASTER, 'secondary_master') if master else 'secondary'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_secondary_limits(master=False, remove=True):

    name = osjoin(MASTER, 'secondary_limits_master') if master else 'secondary_limits'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmin=1.3,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_secondary_limits_no_scale(master=False, remove=True):

    name = osjoin(MASTER, 'secondary_no-auto-scale_master') if master else 'secondary_no-auto-scale'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             xmax=1.2, auto_scale=False,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_secondary_limits_y(master=False, remove=True):

    name = osjoin(MASTER, 'secondary_y-limit_master') if master else 'secondary_y-limit'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=1,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_multiple(master=False, remove=True):

    name = osjoin(MASTER, 'multiple_master') if master else 'multiple_master'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_multiple_scaled(master=False, remove=True):

    name = osjoin(MASTER, 'multiple_scaled_master') if master else 'multiple_scaled'

    # Make the plot
    fcp.plot(df=df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=False, show=SHOW, legend='Die',
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"',
             ymin=0.05,
             filename=name + '.png', inline=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare


def test_boxplot(master=False, remove=True):

    name = osjoin(MASTER, 'boxplot_master') if master else 'boxplot'

    # Make the plot
    fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
                filename=name + '.png', inline=False, jitter=False)

    # Compare with master
    if master:
        return
    else:
        compare = utl.img_compare(name + '.png', osjoin(MASTER, name + '_master.png'))
        if remove:
            os.remove(name + '.png')

        assert not compare



# # Note: auto-scaling is not active for boxplots, contours, and heatmaps.

# # ## Statistical limits

# # <b><font color="blue" style="font-family:'Courier New'">fivecentplots </font></b> allows you to set axis limits based on some quantile percentage of the actual data or the inter-quartile range of the data.  This is most useful when working with boxplots that contain outliers which we do not want to skew y-axis range.

# # ### Quantiles

# # Quantile ranges are added to the standard min/max keywords as strings with the form: "&#60;quantile&gt;q".  <br>
# # Consider the plot below in which the boxplot for sample 2 has an outlier.  The default limit will cover the entire span of the data so the `ymax` value is above this outlier.

# # In[15]:


# fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW)


# # Obviously we could manually set a `ymax` value to exclude this outlier, but in the case of automated plot generation we likely do not know the outlier exists in advance.  Instead we can specify a 95% quantile limit to exclude tail points in the distribution. For boxplots, if the `range_lines` option is enabled, we can still visualize that there is an outlier in the data set that exceeds the y-axis range (see `here <boxplot.html#Range-lines>`_)

# # In[16]:


# fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW, ymax='95q')


# # ### Inter-quartile range

# # In some cases we may want to set a limit based on the inter-quartile range of the data set (i.e., the delta between the 25% and 75% quantiles).  This can also help to deal with outlier data.  The value supplied to the range keyword(s) is a string of the form: "&#60;factor&gt;*iqr", where "factor" is a float value to be multiplied to the inter-quartile range.

# # In[17]:


# fcp.boxplot(df=df_box, y='Value', groups=['Batch', 'Sample'], filter='Batch==101', show=SHOW,
#             ymin='1.5*iqr', ymax='1.5*iqr')


# # ## Axes sharing

# # Axes sharing applies when using `row`, `col`, or `wrap` grouping to split the plot into multiple subplots.  The boolean keywords of interest are:
# # <ul>
# # <li>`share_x`</li>
# # <li>`share_x2`</li>
# # <li>`share_y`</li>
# # <li>`share_y2`</li>
# # <li>`share_z`</li>
# # </ul>
# #

# # ### Shared axes

# # By default, gridded plots share axes ranges (and tick labels) for all axes.  Because axes are shared, the tick labels and axis labels only appear on the outermost subplots.

# # In[18]:


# sub = df[(df.Substrate=='Si') & (df['Target Wavelength']==450)].copy()
# fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',          show=SHOW, ax_size=[225, 225])


# # Sharing can be disabled by setting the `share_` keyword for one or more of the axes to `False`.  Notice that tick labels are added automatically and the spacing between plots is adjusted.

# # In[19]:


# sub = df[(df.Substrate=='Si') & (df['Target Wavelength']==450)].copy()
# fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',          show=SHOW, ax_size=[225, 225], share_x=False, share_y=False)


# # We can also force shared axes to display their own tick labels and/or axis labels using the keywords `separate_ticks` and `separate_labels`.

# # In[20]:


# sub = df[(df.Substrate=='Si') & (df['Target Wavelength']==450)].copy()
# fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',          show=SHOW, ax_size=[225, 225], separate_ticks=True, separate_labels=True)


# # <b>Note: for `wrap` plots based on column values, axis sharing is forced to `True` and cannot be overriden.</b>

# # ### Share rows

# # For `row` plots, we can opt to share both the x- and y-axis range uniquely across each row of subplots via the `share_row` keyword:

# # In[21]:


# sub = df[(df.Substrate=='Si') & (df['Target Wavelength']==450)].copy()
# fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',          show=SHOW, ax_size=[225, 225], share_row=True)


# # ### Share columns

# # Similarly for `cow` plots, we can opt to share the both the x- and y-axis range uniquely across each column of subplots via the `share_col` keyword:

# # In[22]:


# sub = df[(df.Substrate=='Si') & (df['Target Wavelength']==450)].copy()
# fcp.plot(df=sub, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]',          show=SHOW, ax_size=[225, 225], share_col=True)

