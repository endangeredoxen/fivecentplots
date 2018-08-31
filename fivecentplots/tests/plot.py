
import pytest
import fivecentplots as fcp
import pandas as pd
import numpy as np
import os, sys, pdb
osjoin = os.path.join
st = pdb.set_trace


# ### Sample data
df = pd.read_csv(osjoin(os.path.dirname(fcp.__file__), 'tests', 'fake_data.csv'))

# Set theme
#fcp.set_theme('gray')
#fcp.set_theme('white')


# ### Other
SHOW = False

def test_cell5():

    fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False,
             show=SHOW, filename='cell5.png', inline=False,
             filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')
    assert 4 == 5

# # ### Legend

# # Add a legend by the column "Die"

# # In[6]:


# fcp.plot(df, x='Voltage', y='I [A]', legend='Die', show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ### Log scale

# # Log-scaled axes can be enabled through the kwargs "ax_scale".  Valid options:
# # <ul>
# # <li>x-only: <font color="blue" style="font-family:'Courier New'"> logx | semilogx </font></li>
# # <li>y-only: <font color="blue" style="font-family:'Courier New'"> logy | semilogy </font></li>
# # <li>both: <font color="blue" style="font-family:'Courier New'"> loglog | log </font></li>
# # <li>symlog: <font color="blue" style="font-family:'Courier New'"> symlog </font> (doesn't work in matploblib 1.5.1)</li>
# # <li>logit: <font color="blue" style="font-family:'Courier New'"> logit </font></li>
# # </ul>

# # In[7]:


# fcp.plot(df, x='Voltage', y='I [A]', ax_scale='loglog', legend='Die', show=SHOW, xmin=0.9,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ## Secondary x|y plots

# # ### Shared x-axis (twin_x)

# # In[8]:


# fcp.plot(df, x='Voltage', y=['Voltage', 'I [A]'], twin_x=True, show=SHOW, legend='Die',
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"')


# # ### Shared y-axis (twin_y)

# # In[9]:


# fcp.plot(df, x=['Voltage', 'I [A]'], y='Voltage', legend='Die', twin_y=True, show=SHOW,          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25 & Die=="(-1,2)"')


# # ## Multiple x & y values

# # Instead of sharing (or twinning) one <i>independent</i> axis across a primary and secondary <i>dependent</i> axis, we can plot multiple columns of data on the same dependent axis.  In this case, all dependent values share the same limits on the plot.

# # ### Multiple y only

# # In[10]:


# fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], legend='Die', show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ### Multiple x only

# # In[11]:


# fcp.plot(df, x=['Boost Level', 'I [A]'], y='Voltage', legend='Die', show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ### Both

# # In[12]:


# fcp.plot(df, x=['Boost Level', 'I [A]'], y=['Voltage', 'Temperature [C]'], legend='Die', show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ## Grid plots

# # ### Row plot

# # Create a unique plot for each unique value of the "row" column in the specified DataFrame.  Plots are arranged row by row in a single column.  Each row contains a label indicating the unique value of the "row" column.

# # In[13]:


# fcp.plot(df, x='Voltage', y='I [A]', legend='Die', row='Boost Level', show=SHOW, ax_size=[225, 225],
#          filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25')


# # ### Column plot

# # Create a unique plot for each unique value of the "col" column in the specified DataFrame.  Plots are arranged in a single row, column by column.  Each column contains a label indicating the unique value of the "col" column.

# # In[14]:


# fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', show=SHOW, ax_size=[225, 225],
#          filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25')


# # ### Row x column grid

# # Remove "Temperature [C]" from the DataFrame subset and make a grid of plots for each unique combination of "row" and "col" values.  Both "row" and "col" labels are included.

# # In[15]:


# fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', row='Temperature [C]', show=SHOW,          ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13)


# # ### Wrap plot

# # A wrap plot is an alternate view of the row x column plot.  For wrap plots, the row and column labels are condensed into a  single label above each plot window and the spacing between plots is eliminated (by default; can be overriden).  The x and y sizes of the grid are determined by the square root of the total number of plot windows.  x and y axes ranges are shared.

# # In[16]:


# fcp.plot(df, x='Voltage', y='I [A]', legend='Die', wrap=['Temperature [C]', 'Boost Level'], show=SHOW,
#          ax_size=[225, 225], filter='Substrate=="Si" & Target Wavelength==450', label_rc_font_size=13)


# # ## Other options

# # ### Horizontal and vertical lines

# # We can add horizontal and vertical reference lines to a plot using one or more of the following keywords: `ax_hlines`, `ax_vlines`, `ax2_hlines`, `ax2_vlines` where "hlines" are horizontal, "vlines" are vertical, "ax" lines are applied to the primary axis, and "ax2" lines are applied to a secondary axis, if it exists.  The value of these keywords is:
# # <ul>
# # <li>a single float number</li>
# # <li>a list of float numbers</li>
# # <li>or a list of tuples, each consisting of the value (required), color (optional), line style (optional), line width (optional), and the line alpha (optional)</li>
# # </ul>

# # In[17]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          ax_hlines=[(0, '#FF0000', '--', 3), 1.2], ax_vlines=[0, (1, '#00FF00')])


# # ### Curve fitting

# # We can add a polynomial line of fit of some degree to the plot with the keyword `fit` with an option to display the fit equation and R^2 value:

# # In[18]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          fit=4, fit_eqn=True, fit_rsq=True, fit_font_size=9)


# # We can also constrain the region of interest for the fit by supplying a start and stop value to either the `fit_range_x` or `fit_range_y` keywords:

# # In[19]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          fit=1, fit_eqn=True, fit_rsq=True, fit_font_size=9, fit_range_x=[1.3, 2])


# # We can also add a line of fit for each item in a legend:

# # In[20]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW, legend='Die',
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          fit=1, fit_range_x=[1.3, 2], fit_width=2, fit_style='--')


# # ### Stat lines

# # Rather than drawing connecting lines between all points, you can draw a line through a representative statistic of the data, such as the "median" value of the points, using the `stat` keyword (any stats that can be applied to a pandas `groupby` object can be used).  For cases where the x-values are not identical for all data sets, use the keyword `stat_val` and specify an alternative x-axis to use for the statistical calculation (the actual plotted x-axis will be whatever is specified for the `x` keyword).
# #
# # First consider a plot of Voltage vs I [A].  Because the actual measured current values vary from the current set point value from measurement to measurement, the x-values for each data set are not identical.  The `stat` line thus computes the median value for each x value where there may only be a single data point.  The result is ugly and useful.

# # In[21]:


# fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          stat='median')


# # Instead, add the column name for the column containing the set point value of I [A] to the keyword `stat_val`:

# # In[22]:


# fcp.plot(df, x='I [A]', y='Voltage', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          stat='median', stat_val='I Set')


# # Stat lines also work with multiple DataFrame columns are plotted on a given axis.  For example, consider the following with two values on the y-axis:

# # In[23]:


# fcp.plot(df, x='Voltage', y=['Boost Level', 'I [A]'], show=SHOW, legend=True, stat='median',
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')


# # ### Confidence intervals

# # It is possible to display confidence intervals on a data set.  By default these are shown as partially transparent filled regions around a curve.  The actual confidence interval to display is set by the `conf_int` keyword.  Allowed values are any interval between 0 and 1 or the special value `range` which plots a region from the min to the max value of the data at a given point. The example below shows a 95% confidence interval:

# # In[24]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', lines=False, show=SHOW,
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          conf_int=0.95)


# # ### Reference line

# # We can add a reference line to the plot using the keyword `ref_line`.  The value of this keyword is: (1) an equation applied to an existing DataFrame column; or (2) a simple int or float.  This value is used to create a new column in the DataFrame that is plotted vs the value of `x`

# # #### y=x reference

# # In[25]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          ref_line=df['Voltage'], ref_line_text='y=x', xmin=0, ymin=0, xmax=1.6, ymax=1.6)


# # #### More complex calculation

# # Now let's use the fit equation calculated above and add the fit line as a reference line:

# # In[26]:


# fcp.plot(df, x='Voltage', y='I [A]', title='IV Data', show=SHOW, legend='Die',
#          filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25',
#          ref_line=1.555*df['Voltage']**4-3.451*df['Voltage']**3+2.347*df['Voltage']**2-0.496*df['Voltage']+0.014)

