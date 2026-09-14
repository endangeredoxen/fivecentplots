API
===

**fivecentplots** is driven by keywords arguments or **kwargs**.  Kwargs for various plot types
and `Element` objects, along with some examples of their usage, can be found using the list in
the left sidebar.  The general usage pattern of kwargs is as follows:

.. code-block:: python

   fcp.valid_plot_type_name(pandas.DataFrame, keyword1=value1, keyword2=value2, ...)


.. note::

   While the majority of kwargs are optional, each plot type has a small list of *required* kwargs.
   These are usually related to specifying the columns of data that should be plotted.  For example,
   the `plot` method used for XY scatter plotting requires that the `x` and `y` column names be specified.


.. toctree::
   :maxdepth: 1
   :hidden:

   axes.rst
   bar.rst
   boxplot.rst
   cbar.rst
   contour.rst
   figure.rst
   gantt.rst
   gridlines.rst
   grouping.rst
   heatmap.rst
   hist.rst
   imshow.rst
   labels.rst
   legend.rst
   lines.rst
   markers.rst
   nq.rst
   options.rst
   pie.rst
   plot.rst
   tick_labels.rst
   ticks.rst
   title.rst
   ws.rst