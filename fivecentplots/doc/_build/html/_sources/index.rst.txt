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
**fivecentplots** exists to simplify data visualization of **pandas** DataFrames.
While there is no shortage of quality Python plotting libraries on the
world wide web (i.e., matplotlib, bokeh, seaborn, etc.), the APIs can
be overly complicated and require multiple lines of code to get the plot
to look just right.  This complexity can discourage new Python users and
turn them back to inferior yet simplar data tools like Excel for plotting.


What issues does fivecentplots address?
---------------------------------------

**1. Ease of Use**

   * Data come from pandas DataFrames eliminating direct management of
     numpy arrays

   * Style and format parameters can be invoked in a single function call
     via optional keyword arguments

   * Data can be quickly grouped into subplots (row x column grid, wrapped
     by unique value, legend coloring) by simply specifying DataFrame column names

|

**2. Repeatability**

   * Figure sizes expand and contract to accommodate plot elements rather
     than shrinking elements like the axis in unexpected ways to fit
     labels, ticks, etc. (as is the standard
     practice in matplotlib).  In **fivecentplots** the axes window size
     is king so plots of the same size stay the same size
     regardless of everything surrounding them.

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

Documentation
^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1
   :caption: Basics

   overview
   install
   design
   keyword
   grouping.ipynb
   styles.ipynb
   ranges.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples

   plot.ipynb
   boxplot.ipynb
   contour.ipynb
   heatmap.ipynb

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
