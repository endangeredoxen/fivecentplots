Changelog
*********

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

