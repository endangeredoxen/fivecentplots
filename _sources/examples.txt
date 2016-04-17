Example Library
======================

All plot examples here use the provided fake data set found `here
<https://raw.githubusercontent
.com/endangeredoxen/fivecentplots/master/fivecentplots/tests/fake_data.csv>`_.
It consists of some made up semiconductor IV data.

Setup
-----

.. code-block:: python

   import fivecentplots as fcp
   import pandas as pd

   df = pd.read_csv(r'fake_data.csv')


Example 1: Single IV curve grouped by die
-----------------------------------------
Assume we want to plot the IV curve for a single tested die.  Since this is
xy data we will use the ``fcp.plot`` command.  First, let's do this wrong
without any filtering:

.. code-block:: python

   fcp.plot(df, 'Voltage', 'I [A]', leg_groups='Die', show=True)



.. image:: _static/images/Ex1a.png

Notice that we do not have discrete IV curves but a continuum of data points
that connect back to the starting point each time.  This occurs because the
original DataFrame contains data for multiple IV curves at multiple sites
and conditions and our function
call does not account for this.  Next, we will add some filtering to get the
intended result:

.. code-block:: python

   filt = 'Substrate=="Si" & Target_Wavelength==450 & Boost_Level==0.2 & ' \
          'Temperature_C==25'
   fcp.plot(df, 'Voltage', 'I [A]', leg_groups='Die', show=True, filter=filt)



.. image:: _static/images/Ex1b.png

Much cleaner!

.. note:  Remember when filtering to replace spaces and brackets with
          underscores

Example 2: Facet grid by boost level and temperature
----------------------------------------------------

Next, we will group the IV data into multiple plots based on the boost level
and the temperature.  The various factors are separated by value into rows and
columns instead of overlaying all the plots on one axis:

.. code-block:: python

   fcp.plot(df, 'Voltage', 'I [A]', leg_groups='Die',
            row='Boost Level', col='Temperature [C]',
            filter='Substrate=="Si" & Target_Wavelength==450',
            xticks=4, show=True)


.. image:: _static/images/Ex2.png


Example 3: Facet grid by boost level and temperature (no axis sharing)
----------------------------------------------------------------------

In the previous example, the plot axis ranges of all plot windows are set by
the outlier device of die (-1,2) at 75°C and 0.2V boost.  This occurs
because axis sharing is enabled by default.  We can turn this off to get
better clarity of the plot windows without the outlier as follows:

.. code-block:: python

   fcp.plot(df, 'Voltage', 'I [A]', leg_groups='Die',
            row='Boost Level', col='Temperature [C]',
            filter='Substrate=="Si" & Target_Wavelength==450',
            sharex=False, sharey=False,
            xticks=4, show=True)


.. image:: _static/images/Ex3.png

Example 4: Multiple y parameters on same axis
---------------------------------------------

Often we need to display multiple plots that share the same y-axis.  In
this case, the legend will be updated to help us distinguish between the
different curves.  This can be done as follows:


.. code-block:: python

   filt = 'Substrate=="Si" & Target_Wavelength==450 & Boost_Level==0.2 & ' \
          'Temperature_C==25'
   fcp.plot(df, 'Voltage', ['I [A]', 'Voltage'], leg_groups='Die',
            filter=filt, ylabel='Values', show=True)


.. image:: _static/images/Ex4.png


Example 5: Multiple y parameters using a secondary y-axis (Broken)
------------------------------------------------------------------

As an alternative to Example 4, we can force the second y-value to plot on a
separate y-axis by "twinning" the x-axis as follows:


.. code-block:: python

   filt = 'Substrate=="Si" & Target_Wavelength==450 & Boost_Level==0.2 & ' \
          'Temperature_C==25'
   fcp.plot(df, 'Voltage', 'I [A]', y2='Voltage', twinx=True, leg_groups='Die',
            filter=filt, ylabel='I [A]', ylabel2='Voltage', show=True)


.. image:: _static/images/Ex5.png




