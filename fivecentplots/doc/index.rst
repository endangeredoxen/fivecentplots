.. fivecentplots documentation master file, created by
   sphinx-quickstart on Sat Apr 16 09:57:41 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

fivecentplots
=============

.. image:: _static/images/logo2.png


*Making multi-factor & multi-axes Matplotlib manipulations more manageable*

Overview
^^^^^^^^^^^^^^^^^^

Why another plotting library?
-----------------------------
Given that there is no shortage of quality Python and non-Python plotting
software packages on the world wide web (matplotlib, bokeh, seaborn, JMP,
etc.), it is fair to ask why we need another.  The answer is simple:
*you may not; I do.*  **fivecentplots** exists to simplify the generation,
customization, and (most importantly to me) automation of certain types of
plots in Python.

**fivecenplots** is built on a framework of matplotlib and pandas.

What issues does fivecentplots solve?
-------------------------------------

1) Simplified syntax:

   Almost all input parameters are specified as optional keyword arguments of one
   function call (for each plot type). All data used for plotting is derived
   from pandas DataFrames so there is no direct management of numpy arrays.

|

2) Easy and repeatable look and feel:

   fivecentplots makes it easy to style plots with custom colors, sizes,
   marker themes, etc., and automate plot generation.  Unlike matplotlib
   which sets a size for the entire figure and adjusts elements to fit,
   fivecentplots starts with a fixed size for the plot or axis window and
   allows the figure size to grow as needed to accommodate the contents.  This
   promotes consistency from plot to plot and eliminates the need to invoke
   matplotlib routines like ``tight_layout`` which sometimes has unexpected
   results.

|

3) Simplified visualization of pandas DataFrames:

   fivecentplots is built to process multi-factor pandas DataFrames.
   Keyword arguments make it easy to legend a plot by another column or to
   group multiple plots in rows and/or columns based on other DataFrame
   columns.  It also accepts a conditional string to filter DataFrames
   before plotting.

|

4) Boxplots with grouping labels:

   Although matplotlib supports boxplot generation, it is not easy to group
   data into labeled, multi-factor subgroups.  Borrowing inspiration from the
   `variability chart in JMP <http://www.jmp.com/support/help/Variability_Gauge_Charts.shtml>`_,
   fivecentplots extends matplotlib's boxplot routine to make this possible.

5) Automation!

   In addition to explicit function calls, **fivecentplots** can pull plotting
   parameters from ini-style config files.  This makes it very easy to
   manage and organize plot parameters for repeated and automated activities.

   For example, consider the case of an test system in a lab or a
   production facility that repeatedly generates data that needs to be plotted
   for quick analysis.  Traditionally, you might write a script that contains
   multiple function calls to style and create each plot but this can be
   cumbersome to maintain especially for someone with limited coding
   experience.  With the ini-style config file option in **fivecentplots**
   you could create a much more readable file that is easy to reuse or
   modify by anyone.


Documentation
^^^^^^^^^^^^^


.. toctree::
   :maxdepth: 2

   overview
   design
   defaults
   keyword
   examples
   modules


Indices and tables
^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:: fivecentplots
 :members:
 :undoc-members:
