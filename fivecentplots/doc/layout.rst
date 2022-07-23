Layout
======

All plots consist of a number of "elements" arranged in a specific
spatial "layout".  These "elements" define the look and feel of the plot and
include items like the axes area, text and tick labels, legends, etc.
In **fivecentplots**, these components are defined by an
``Element`` object that contains all of the attributes that define
the components style (i.e., color, font, size, etc).  The placement of these ``Element``
objects within the plot figure is handled by a ``Layout`` class that is defined
around an existing plotting engine in Python (such as `matplotlib` or `bokeh`).
Each engine when used in a stand-alone fashion requires its own unique set of commands
and parameters to generate a plot, but the ``Layout`` class in **fivecentplots** serves
to unify access to these specific parameters across any plotting engines.  This is
accomplished using a combination of a predefined theme file and optional keyword arguments
specified by the user.  This section will explain the important details of the
plot layout.  **fivecentplots** `kwargs <keyword.html>`_ will be covered in the next section.

|
**Example Plot:**

An simple x-y plot is shown below.  Example of ```Element`` objects found in this plot include:

   * `fig`: defines the area outside of the axes area where the plot is displayed
   * `axes`: defines the plot background color, size, axes scale, etc.
   * `label_x` and `label_y`: define the x and y label text, color, font size, etc
   * `tick_labels` and `tick_marks`: define the look of the ticks and labels around the axes
   * `legend`: defines the location, edge color, background color, and font size of the legend on the right side of the plot
   * `grid_major`: defines the color, width, and style of the major gridlines
   * `title`: defines the text format of the title
   * `lines`: defines the width, style, and color of the plot lines
   * `markers`: defines the marker edge color, fill color, and type

.. image:: _static/images/my_favorite_ever.png

|

.. note:: All of the ``Element`` parameters mentioned above have defaults that can be changed with a custom theme file or as kwargs in the plot function.


**Simple Layout Schematic:**

All of these elements are positioned by the ``Layout`` class as shown in the diagram below.
Within the ``Layout`` there are many user-adjustable options for the amount of whitespace separating all items.
Each of these whitespace parameters is illustrated in this drawing.


.. image:: _static/images/layout.png

|

.. note:: All of these whitespace parameters have defaults that can be changed with a custom theme file or as kwargs in the plot function.

**Grid Layout Schematic:**

With **fivecentplots** we can easily extend the simple single-axis plot and make a grid of subplots.
This plot option enables some additional ``Layout`` parameters illustrated below:

.. image:: _static/images/layout_grid.png



Engine
------
**fivecentplots** is not a new graphics library.  In fact, plots are created using existing Python plotting
packages.  These "engines" are wrapped by **fivecentplots** to provide a new, simpler API that allows complex
plotting using kwargs to invoke the more complicated API of the plotting package.  The most mature "engine"
supported by **fivecentplots** is ``matplotlib`` which is enabled by default, but limited support is available for ``bokeh``.  Additional
"engines" could easily be supported through the creation of a new ``Layout`` class (volunteers welcome!).  With
this approach of API unification through kwargs, **fivecentplots** enables the user to switch between plotting engines
with *the same plot command*.  The only kwarg that must change in the call is that of the `engine`.

For example, if you needed a high-quality plot for a paper and wanted to use **matplotlib**, you
could do the following:

.. code:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die',
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')

.. image:: _static/images/engine_mpl.png

But what if you needed to interactively manipulate the same data set, making a static image less attractive?  It may be more
convenient to plot via **bokeh**.  To switch engines, we simply add the keyword ``engine`` and
set to "bokeh":

.. code:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die', engine='bokeh',
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')

.. image:: _static/images/engine_bokeh.png

All with only **one** kwarg change!

.. note:: As of version 0.5.0, **bokeh** support is limited compared with **matplotlib**.  More
          development is needed.  Not all plot types are available at this time.

