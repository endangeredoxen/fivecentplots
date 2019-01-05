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
**fivecentplots** exists to simplify plotting of **pandas** DataFrames.
While there is no shortage of quality Python plotting libraries on the
world wide web (i.e., matplotlib, bokeh, seaborn, etc.), the APIs can
be overly complicated and require multiple lines of code to make the plot
look just right.  This complexity is especially discouraging to new
or casual Python users and often results in them giving up on Python plotting
in favor of inferior yet simplar plotting tools like Excel.


What issues does fivecentplots address?
---------------------------------------

**1. Ease of Use**

   * Plots require *a single function call*.  All needed inputs from the data to plot
     to any style and/or formatting parameters are specified as optional keyword
     arguments in this single function call.

   * Data come from pandas DataFrames eliminating direct management of
     numpy arrays

   * Data can be quickly grouped into subsets and visualized.  This includes
     creating grids of subplots and legending by color and/or marker.  All of these
     options are invoked simply by specifying DataFrame column names as appropriate
     keyword arugments

|

**2. Repeatability**

   * Figure sizes expand and contract to accommodate plot elements rather
     than remaining fixed and shrinking the internal elements often in unexpected
     ways (as is the standard practice in matplotlib).  In **fivecentplots** the
     axes window size is king so plot areas of the same size stay the same size
     regardless of everything surrounding them.  Because element sizes are
     measured before the plot is generated and the figure size is not fixed,
     **fivecentplots** can adjust the layout to prevent elements from overlapping
     unexpectedly.

   * All colors, sizes, marker themes, etc. can be specified from input keywords
     or from a theme file

|

**3. Automation**

   * In addition to explicit function calls, plots can be fully defined
     from ini-style config files making it easy create a batch of plots that are
     auto-generated from repeated calculations or data collection. These text
     files can be shared among and modified by multiple engineers and
     scientists without requiring a deep knowledge of Python syntax

|

**4. JMP-style plots that would be tedious to create every time in raw code**

   With the development of **pandas**, statistical analysis in Python can rival or surpass that of
   popular commercial software packages like JMP. However, JMP has several plotting platforms
   that are very useful and would be tedious to create from scratch in Python.  **fivecentplot**
   handles all of this work so you don't have to, allowing you to create beautiful
   *and often more flexible* JMP-style plots with a single function call.  Examples include:

   * Boxplots with grouping labels (i.e., variability charts):

      While there are boxplot functions in Python plotting libraries, it is not easy to group
      data into multiple, *labeled* subgroups.  Borrowing inspiration from the
      `variability chart in JMP <http://www.jmp.com/support/help/Variability_Gauge_Charts.shtml>`_,
      **fivecentplots** makes this possible.

   * Grouped facet grid plots:

      Like JMP's `grouped overlay plots <http://www.jmp.com/support/help/Additional_Examples_of_the_Overlay_Plot_Platform.shtml#192569>`_,
      take any multi-variate DataFrame and separate it into grids of plots,
      each represeting a unique combination of factors


|

Current Options
---------------

The following plot types are supported in **fivecentplots**:

  1) `XY plots <plot.html>`_
  2) `box plots <boxplot.html>`_
  3) `contour plots <contour.html>`_
  4) `heatmaps <heatmap.html>`_
  5) `histograms <hist.html>`_
  6) `bar plots <barplot.html>`_

Follow the links to the example section to see the many options
available within each plot category.


Documentation
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Basics

   install
   layout
   keyword
   grouping.ipynb
   styles.ipynb
   ranges.ipynb
   ticks.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   plot.ipynb
   boxplot.ipynb
   contour.ipynb
   heatmap.ipynb
   hist.ipynb
   bar.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Other

   modules
   older

Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:: fivecentplots
 :members:
 :undoc-members:
