Changelog
*********

0.5.4
=====
* Add compatibility for `matplotlib==3.7`
* Fix long figure title bug

0.5.3
=====
* For rc labels, add option to drop the row/column name in the label text with keyword `label_rc_names`, `label_row_names`, or `label_col_names` = `False`
* Remove deprecated kwarg in `np.histogram` call

0.5.2
=====
* Remove feature that restores mpl `rcParams` at end of plot as it can break interactive plotting results

0.5.1
=====
* Bug to fix bokeh plots
* Adjust save/show/inline behavior slightly.  ``return_filename`` and ``print_filename`` now only work when image is actually saved to disk

0.5.0
=====
* Major speed improvements achieved for **matplotlib** engine.  Actual time reduction depends on the plot type and the number of subplots, data sets, etc. Based on the plots included as unit tests:
    * ~30% increase for standard xy plots
    * ~40-50% increase for barplot, boxplots (sans advanced features like violins), contours, and heatmaps
* New plot types added:
    * gantt
    * imshow (preferred choice for display of image data over heatmap)
    * pie
* Significant cleanup of legacy code and refactoring of ``Data`` and ``Layout`` classes
* Full documentation of kwargs API
* Support added for:
    * upper- and lower-control limit area shading
    * marker size definition via a column in the DataFrame (allows emphasis of specific points)
    * cdf and pdf conversion with ``fcp.hist``
    * rolling mean in ``fcp.bar``
* Numerous bug fixes

0.4.3
=====
* fix heatmap but when DataFrame column names are of type ``object`` but are actually ``int``

0.4.2
=====
* handle change to pandas xlsx engine for >=v1.2
* remove unintentional overload of ``input``

0.4.1
=====
* update to ignore warnings with v3.3 of MPL

0.4.0
=====

* legend:

    * allow placing legend below plot (``legend_location=='below'`` or ``legend_location==11``)

    * fixes for figure grouping and multi-column plots

* heatmap:

    * font and font size fixes

* themes:

    * beautification of "white" theme

    * allow switching of themes via keyword at the plot call

    * allow overloading of default color list in theme file

* other:

    * fixed requirements so it will import after install in clean environment

    * deal with iterticks warning from ``matplotlib`` versions >= 3.1

    * added ``save_data`` keyword to dump a subset of the plotted data only from the original ``pandas.DataFrame``

    * file extension bug fix

    * filter improvements (allow "not in" list)

    * added option to disable alphabetical sorting of data and plot based on the order in which data appears in the original ``pandas.DataFrame``

0.3.0
=====

* old and deprecated