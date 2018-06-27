.. fivecentplots documentation master file, created by
   sphinx-quickstart on Sat Apr 16 09:57:41 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. role:: title
.. role:: tagline


.. image:: _static/images/index.png

|
|

:title:`fivecentplots`
======================

:tagline:`--Plot data bigly--`



Why another Python plotting library?
------------------------------------
There is no shortage of quality Python plotting libraries on the world wide
web (i.e., matplotlib, bokeh, seaborn, etc.).  Why add another?

With the development of **pandas**, statistical analysis in Python can
rival that of commercial software packages like JMP.  However, plotting those
analyses is often more difficult.  **fivecentplots** builds upon **pandas**
and **matplotlib** to make styling and formatting of plots easier and
more consistent.


What issues does fivecentplots address?
---------------------------------------

**1. Ease of Use**

   * All data come from pandas DataFrames eliminating direct management of
     numpy arrays

   * All style and format parameters can be invoked in a single function call
     via optional keyword arguments

   * Legend a plot or group multiple plots into rows and/or columns based on
     simply by DataFrame column names

   * Quickly filter a DataFrame prior to plotting using simple notation instead
     of complex boolean operators

|

**2. Repeatability**

   * Figure sizes expand and contract to accommodate plot elements rather
     than shrinking elements to fit in a certain figure dimension (standard
     practice in matplotlib).  Plots of the same size stay the same size
     regardless of tick marks or legend contents

   * All colors, sizes, marker themes, etc. can be specified from static set
     up files to preserve a custom look for all plots

|

**3. Automation**

   * In addition to explicit function calls, plots can be fully defined
     from ini-style config files making it easy create a batch of plots that are
     auto-generated from repeated calculations or data collection. These text
     files can be shared among and modified by multiple engineers and
     scientists without requiring a deep knowledge of Python syntax

|

**4. Extras (JMP-style plots that would be tedious to recreate every time in raw code)**

   * Boxplots with grouping labels:

      Although matplotlib has a boxplot function, it is not easy to group
      data into multiple, labeled subgroups.  Borrowing inspiration from the
      `variability chart in JMP <http://www.jmp.com/support/help/Variability_Gauge_Charts.shtml>`_,
      fivecentplots extends matplotlib's boxplot routine to make this possible.

   * Grouped facet grid plots:

      Like JMP's `grouped overlay plots <http://www.jmp.com/support/help/Additional_Examples_of_the_Overlay_Plot_Platform.shtml#192569>`_,
      take any multi-variate DataFrame and separate it into grids of plots,
      each represeting a unique combination of factors

|

Documentation
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   overview
   installation
   design
   defaults
   keyword
   plot.ipynb
   boxplot.ipynb
   modules


Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:: fivecentplots
 :members:
 :undoc-members:
