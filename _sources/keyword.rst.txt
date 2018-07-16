Keyword Arguments
=================

All inputs to **fivecentplots** are treated as keyword arguments.  The only mandatory
inputs are:

+---------+------------------+------------------------------------------------------------------------+--------------------+
| Keyword | Data Type        | Description                                                            | Required?          |
+=========+==================+========================================================================+====================+
| df      | pandas DataFrame | DataFrame containing all data to be plotted including grouping columns | always             |
+---------+------------------+------------------------------------------------------------------------+--------------------+
| x       | str              | Column name of the x-axis data                                         | all except boxplot |
+---------+------------------+------------------------------------------------------------------------+--------------------+
| y       | str              | Column name of the y-axis data                                         | always             |
+---------+------------------+------------------------------------------------------------------------+--------------------+
| z       | str              | Column name of the z-axis data                                         | contour only       |
+---------+------------------+------------------------------------------------------------------------+--------------------+

All other keywords that can be used for grouping data, setting axis limits, and
styling the look of the plot have built-in defaults that can be overriden by explicitly
setting them in the plot function call or by saving them in a
`user theme file <themes.html>`_.

Data
----
All DataFrame releated operations in **fivecentplots** are handled using a special
``Data`` object.  This object takes input regarding process and stores data-related information


Elements
--------
Each item in a plot (axis, labels, legend, etc.) consists of an ``Element`` object
that contains various attributes (color, size, style, etc.) that
describe the element.  Depending on the element, these base attributes
can be:

    * overriden with new values
    * not used at all (i.e., a label has a ``fill_color`` but not a plot line ``width``)
    * joined with unique attributes that only pertain to the element at hand (i.e., an x-axis range can be shared across all subplots with the ``share_x`` keyword, but this attribute would be meaningless for a label)

Nomenclature
^^^^^^^^^^^^

Some attributes and their keyword arguements are unique to a given element.
However, many keywords inputs are similar in nature between many different
elements (like fill or edge color) and can be accessed using a standard nomenclature:

.. raw:: html

    <div class="admonition note">
    <p class="first admonition-title">Keyword Naming Scheme</p>
    <p class="last">&lt;element name&gt;_&lt;major|minor axis type (if any)&gt;_&lt;x|y|z axis (if any)&gt;_&lt;element attribute name&gt;</p>
    </div>


Consider the following examples:

    1) Change the x label font color to red:

        **label_x_font_color = '#FF0000'**

    2) Change the y-axis minor gridline width:

        **grid_major_y_width = 2**

    3) Change the axes edge color to black:

        **ax_edge_color = '#000000'**


Default attributes
^^^^^^^^^^^^^^^^^^

The base attributes of each ``Element`` are:

+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+
| Category | Keyword     | Description                                                                     | Default    | Example          |
+==========+=============+=================================================================================+============+==================+
| Fill     | fill_alpha  | fill color opacity [0 (fully transparent) - 1 (opaque)]                         | 1          | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | fill_color  | fill color (hex color code)                                                     | #ffffff    | https://test.org |
+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+
| Edges    | edge_alpha  | edge color opacity  [0 (fully transparent) - 1 (opaque)]                        | 1          | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | edge_width  | width in pixels of the element border                                           | 1          | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | edge_color  | edge color (hex color code)                                                     | #ffffff    | https://test.org |
+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+
| Fonts    | font        | font name for element text                                                      | sans-serif | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | font_color  | font color (hex color code)                                                     | #000000    | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | font_size   | font size in pixels                                                             | 14         | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | font_style  | font style ['normal', 'italic', 'oblique']                                      | normal     | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | font_weight | font weight ['light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'] | normal     | https://test.org |
+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+
| Lines    | alpha       | line opacity                                                                    | 1          | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | color       | line color (hex color code)                                                     | #000000    | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | style       | line style ['-', '--', '-.', ':']                                               | -          | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+------------+------------------+
|          | width       | line width in pixels                                                            | 1          | https://test.org |
+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+
| Other    | on          | toggle visibility of the element                                                | depends    | https://test.org |
+          +-------------+---------------------------------------------------------------------------------+            +                  +
|          | size        | [width, height] of the object                                                   |            |                  |
+          +-------------+---------------------------------------------------------------------------------+            +                  +
|          | text        | label text associated with the object                                           |            |                  |
+----------+-------------+---------------------------------------------------------------------------------+------------+------------------+

Specific Elements
-----------------
Listed below are the *additional* keyword arguments of specific elements (each element
also contains the base attributes of the ``Element`` class and can be used if applicable.

Axes
^^^^
.. image:: _static/images/element_axes.png

The ``axes`` element consists of the actual plotting window shown in yellow above.
Keywords for the primary ``axes`` object begin with the prefix "ax_".  Properties
of any optional secondary axes begin with the prefix

.. raw:: html

    <div class="admonition note">
    <p class="first admonition-title">Keyword Prefix</p>
    <p class="last"><b>Primary axis: ax_</b></p>
    <p class="last"><b>Secondary axis: ax2_</b></p>
    </div>

+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| Keyword       | Data Type    | Description                                                                             | Default    | Example                                         |
+===============+==============+=========================================================================================+============+=================================================+
| size          | list of ints | width, height of plot area                                                              | [400, 400] | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| ax_edge_color | str          | outer edge color of plot area                                                           | #aaaaaa    | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| ax_fill_color | str          | inner fill color of plot area                                                           | #eaeaea    | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+            +-------------------------------------------------+
| ax_scale      | str          | enable linear vs log scaling of axis                                                    |            | `log scale plot <plot.html#Log-scale>`_         |
+               +              +-----------------------------------------------------------------------------------------+            +                                                 +
|               |              | x-axis: ['linear', 'logx', 'semilogx']                                                  |            |                                                 |
+               +              +-----------------------------------------------------------------------------------------+            +                                                 +
|               |              | y-axis: ['linear', 'logx', 'semilogx']                                                  |            |                                                 |
+               +              +-----------------------------------------------------------------------------------------+            +                                                 +
|               |              | both: ['loglog', 'log']                                                                 |            |                                                 |
+               +              +-----------------------------------------------------------------------------------------+            +                                                 +
|               |              | other: ['symlog', 'logit']                                                              |            |                                                 |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_x       | boolean      | enable/disable primary x-axis range sharing across all subplots                         | True       | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_y       | boolean      | enable/disable primary y-axis range sharing across all subplots                         | True       | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_z       | boolean      | enable/disable primary z-axis range sharing across all subplots                         | True       | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_x2      | boolean      | enable/disable secondary x-axis range sharing across all subplots                       | True       | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_y2      | boolean      | enable/disable secondary y-axis range sharing across all subplots                       | True       | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_col     | boolean      | enable/disable axis range sharing for all subplots in a column of subplots              | False      | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| share_row     | boolean      | enable/disable axis range sharing for all subplots in a row of subplots                 | False      | None                                            |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| twin_x        | boolean      | enable/disable a secondary y-axis (x-axis is "twinned" or duplicated across two y-axes) | False      | `twin_x plot <plot.html#Shared-x-axis-twin_x>`_ |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+
| twin_y        | boolean      | enable/disable a secondary x-axis (y-axis is "twinned" or duplicated across two x-axes) | False      | `twin_y plot <plot.html#Shared-y-axis-twin_y>`_ |
+               +              +-----------------------------------------------------------------------------------------+            +                                                 +
|               |              | Note:  wrap plots cannot be used when ``twin_y == True``                                |            |                                                 |
+---------------+--------------+-----------------------------------------------------------------------------------------+------------+-------------------------------------------------+

Boxplots
^^^^^^^^
.. image:: _static/images/element_box.png

Boxplots have several unique ``Element`` objects that can be styled:

    * **Boxes:**  the actual boxes of the box plot (shown in white with a blue border above)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_</b></p>
         </div>

      +------------------+-----------+------------------------------------------------------+---------+-------------------------------------------+
      | Keyword          | Data Type | Description                                          | Default | Example                                   |
      +==================+===========+======================================================+=========+===========================================+
      | box_on           | boolean   | toggle box visibility                                | True    | `box plot <boxplot.html#multipe-groups>`_ |
      +------------------+-----------+------------------------------------------------------+---------+                                           +
      | box_edge_color   | str       | edge color of the boxes                              | #4b72b0 |                                           |
      +------------------+-----------+------------------------------------------------------+---------+                                           +
      | box_fill_color   | str       | fill color of the boxes                              | #ffffff |                                           |
      +------------------+-----------+------------------------------------------------------+---------+                                           +
      | box_median_color | str       | color of the median line within the boxes            | #ff7f0e |                                           |
      +------------------+-----------+------------------------------------------------------+---------+                                           +
      | notch            | boolean   | use a notched-style box instead of a rectangular box | False   |                                           |
      +------------------+-----------+------------------------------------------------------+---------+-------------------------------------------+

|

    * **Divider lines:**  optional vertical divider lines between groups

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_divider_</b></p>
         </div>

      These lines are styled using the `default <keyword.html#default-attributes>`_
      attributes for lines of the ``Element`` object.  They are enabled by default
      but can be turned off with ``box_divider=False``.  (Default ``zorder`` = 2)

      +----------------+-----------+--------------------------------------------+---------+
      | Keyword        | Data Type | Description                                | Default |
      +================+===========+============================================+=========+
      | box_divider_on | boolean   | toggle divider lines between groups on/off | True    |
      +----------------+-----------+--------------------------------------------+---------+
      | color          | str       | line color                                 | #bbbbbb |
      +----------------+-----------+--------------------------------------------+---------+
      | zorder         | int       | relative z-height of line in plot          | 2       |
      +----------------+-----------+--------------------------------------------+---------+

|

    * **Group labels:** labels directly under each box that indicate the unique group values
      of the given box (shown in yellow above).

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_group_label_</b></p>
         </div>


      These labels are styled using the `default <keyword.html#default-attributes>`_ label
      attributes of the ``Element`` object.

|

    * **Group titles:** labels to the right of the group labels that indicate the DataFrame
      column name of each grouping column (shown in salmon above).

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_group_title_</b></p>
         </div>


      These labels are styled using the `default <keyword.html#default-attributes>`_ label
      attributes of the ``Element`` object.

|

    * **Range lines:** optional lines within a single box that span the entire range of the
      data set.  These are useful for visualization of outlier points that may be
      outside of the selected ymin/ymax Range (Default ``zorder`` = 3)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_range_lines</b></p>
         </div>

      +--------------------+-----------+------------------------------------------------------------------------+---------+
      | Keyword            | Data Type | Description                                                            | Default |
      +====================+===========+========================================================================+=========+
      | box_range_lines_on | boolean   | toggle range lines on/off                                              | True    |
      +--------------------+-----------+------------------------------------------------------------------------+---------+
      | color              | str       | line color                                                             | #cccccc |
      +--------------------+-----------+------------------------------------------------------------------------+---------+
      | style              | str       | horizontal lines at the end of the range                               | -       |
      +--------------------+-----------+------------------------------------------------------------------------+---------+
      | style2             | str       | vertical lines connecting the horizontal lines at the end of the range | --      |
      +--------------------+-----------+------------------------------------------------------------------------+---------+
      | zorder             | int       | relative z-height of line in plot                                      | 3       |
      +--------------------+-----------+------------------------------------------------------------------------+---------+

|

    * **Stat lines:** optional connecting line between each box at some statistical
      value calculated from the data for a single box.  Options include any stat that
      can be computed via the ``groupby`` command on a pandas DataFrame (i.e., "mean",
      "median", "std", etc.) (Default ``zorder`` = 7 to be on top of the boxes)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>box_stat_line_</b></p>
         </div>

      +------------------+-----------+--------------------------------------------+---------+
      | Keyword          | Data Type | Description                                | Default |
      +==================+===========+============================================+=========+
      | box_stat_line_on | boolean   | toggle divider lines between groups on/off | True    |
      +------------------+-----------+--------------------------------------------+---------+
      | box_stat_line    | str       | set the statistic for the connecting line  | mean    |
      +------------------+-----------+--------------------------------------------+---------+
      | color            | str       | line color                                 | #666666 |
      +------------------+-----------+--------------------------------------------+---------+
      | zorder           | int       | relative z-height of line in plot          | 7       |
      +------------------+-----------+--------------------------------------------+---------+

Color Bar
^^^^^^^^^

Confidence Intervals
^^^^^^^^^^^^^^^^^^^^

Contour
^^^^^^^

Figure
^^^^^^

.. image:: _static/images/element_fig.png

.. note::

   To style the figure region in yellow for this example, the following keywords were
   used in the ``fcp.plot`` command:

   .. code-block:: python

      fig_edge_color='#000000', fig_fill_color='#fffd75', fig_fill_alpha=0.5

   Notice, we are just accessing default ``Element`` class attributes and prepending
   the keywords with the element name, ``fig_`` in this case.


The ``figure`` element is the full window in which other elements are rendered.
Most of the ``figure`` region is covered by these other elements but the visible
portion (shown in yellow above) can be styled.  ``figure`` elements can also
be subdivided into multiple subplots to display more data.  Unlike matplotlib,
size of the figure window is caluculated automatically by the sizes of the discrete
elements it contains.  It cannot be set directly.

.. raw:: html

    <div class="admonition note">
    <p class="first admonition-title">Keyword Prefix</p>
    <p class="last"><b>fig_</b></p>
    </div>

+---------+-----------+---------------------------+--------------------------+
| Keyword | Data Type | Description               | Default                  |
+=========+===========+===========================+==========================+
| dpi     | int       | dots per inch             |                          |
+---------+-----------+---------------------------+--------------------------+
| size    | list      | size of the figure window | calculated automatically |
+---------+-----------+---------------------------+--------------------------+

Gridlines
^^^^^^^^^

Labels
^^^^^^

.. image:: _static/images/element_label.png

.. image:: _static/images/element_label2.png

The following types of ``label`` elements can exist in a plot:

    * axis labels (colored in salmon above)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <ul>
            <li>Single axis:
                <ul>
                    <li><p class="last"><b>label_x_</b></p></li>
                    <li><p class="last"><b>label_y_</b></p></li>
                    <li><p class="last"><b>label_z_</b> (aka colorbar title)</p></li>
                </ul>
            </li>
            <li>All axes together:
                <ul>
                    <li><p class="last"><b>label_</b></p></li>
                </ul>
            </li>
        </ul>
        </div>

      The default text for axes labels is pulled from the corresponding
      DataFrame column names used to define these values (i.e., the ``label_x`` text
      will match the value of ``x``).  However, axes label text can be overriden by
      setting a value for the label such as ``label_x='New Name'``.


    * row, column, and wrap labels (colored in yellow above)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <ul>
            <li>Single grouping type:
                <ul>
                    <li><p class="last"><b>label_row_</b></p></li>
                    <li><p class="last"><b>label_col_</b></p></li>
                    <li><p class="last"><b>label_wrap_</b></p></li>
                </ul>
            </li>
            <li>All row/column labels together:
                <ul>
                    <li><p class="last"><b>rc_label_</b></p></li>
                </ul>
            </li>
        </ul>
        </div>

      The text for row and column labels is
      "<the DataFrame column name specified for ``row`` or ``col``> =
      <one of the unique values of that DataFrame column>".  The text for wrap labels
      will be a tuple of the unique values of the DataFrame columns specified for
      the ``wrap`` keyword. Unlike axes label text, row, column, or wrap label text
      cannot be overriden by the user but depends on the information in the DataFrame
      being plotted.


    * wrap titles: ``wrap_title`` (colored in cyan above)

      .. raw:: html

         <div class="admonition note">
         <p class="first admonition-title">Keyword Prefix</p>
         <p class="last"><b>wrap_title_</b></p>
         </div>

      The text for wrap titles is the column names specified for the ``wrap`` keyword.

All ``label`` elements are styled using the `default <keyword.html#default-attributes>`_
``Element`` class attributes for color and font.

Legend
^^^^^^

Line Fit
^^^^^^^^


Lines
^^^^^

Markers
^^^^^^^

Ticks
^^^^^

Title
^^^^^
.. image:: _static/images/element_title.png

The ``title`` element (shown above in yellow) adds a title to the top of the figure.
The title text is added directly via the ``title`` keyword:

.. code-block:: python

   title = 'IV Data'




Whitespace
----------

.. image:: _static/images/figure_design.png

.. image:: _static/images/figure_design_rc.png


Data
----


Other
-----
show

