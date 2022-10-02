Layout
======

All plots consist of a number of different "elements" arranged in a specific
spatial "layout".  In **fivecentplots**, each plot component (such as the
axes window, labels, legends, etc.) is defined as an
``Element`` class (or object) that contains all of the attributes that define
its style (i.e., color, font, size, etc).  The placement of these ``Element``
objects is handled by a ``Layout`` class.  ``Element`` objects are spaced
by either a user-defined or default whitespace.  The specifics
of the ``Layout`` class depend on the python plotting engine being used.
Currently, only ``matplotlib`` is available as a plotting engine, but additional
engines will be added in the future.

An example x-y plot and a schematic blueprint of its underlying layout
is shown below.  All ``Element`` attributes
and whitespace parameters can be controlled via keyword arguments in the plot
function call or from a predifined "theme" file that consists of dictionaries
of preferred values.

**Example Plot:**

.. image:: _static/images/my_favorite_ever.png


**Layout Schematic:**

.. image:: _static/images/layout.png

We can extend the simple single-axis plot and make a grid of subplots.
An example of the featured ``Element`` objects in this case is shown below:

**Layout Schematic with Grid:**

.. image:: _static/images/layout_grid.png

All keyword arguments for ``Element`` attributes and whitespace are defined
in the `next section <keyword.html>`_ of the guide.

Engine
------
In order to actually create a plot, **fivecentplots** must tap into a plotting "engine".
This engine is a standard Python plotting package like **matplotlib** or **bokeh**.
One of the key advantages of **fivecentplots** is the ability to switch between plotting engines
while using the same keyword-argument-based api.  The engine itself is toggled by a keyword.
**fivecentplots** uses **matplotlib** as its default engine.

For example, if you needed a high-quality plot for a paper and wanted to use **matplotlib**, you
could do the following:

.. code:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die',
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')

.. image:: _static/images/engine_mpl.png

Alternatively, what if you needed to interactively manipulate the same data set?  It may be more
convenient to plot via **bokeh**.  To switch engines, we simply add the keyword ``engine`` and
set to "bokeh":

.. code:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die', engine='bokeh',
            filter='Substrate=="Si" & Target Wavelength==450 & Boost Level==0.2 & Temperature [C]==25')

.. image:: _static/images/engine_bokeh.png

.. note:: As of version 0.3.0, **bokeh** support is limited compared with **matplotlib**.  More
          development is needed.

