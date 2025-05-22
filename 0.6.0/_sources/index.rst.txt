.. fivecentplots documentation master file, created by
   sphinx-quickstart on Sat Apr 16 09:57:41 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. role:: tagline

fivecentplots: a Python Plotting Analgesic
==========================================

:tagline:`Spend more time analyzing your data than analyzing how to plot your data`

.. raw:: html

   <div class="intro-container">
      <div class="intro-gallery">
         <img class="imgidx dark-light" id="imgidx0" alt="_images/intro_xy.png" src="_images/intro_xy.png">
         <img class="imgidx dark-light" id="imgidx1" alt="_images/intro_box.png" src="_images/intro_box.png">
         <img class="imgidx dark-light" id="imgidx2" alt="_images/intro_gantt.png" src="_images/intro_gantt.png">
         <img class="imgidx dark-light" id="imgidx3" alt="_images/intro_imshow.png" src="_images/intro_imshow.png">
      </div>

      <div class="intro-text" id="why-oh-why">
         <h2 style="width: 100%">Another Python plotting library, really??</h2>
         <p> There is no shortage of quality plotting libraries in Python.  Basic plots with default styling are
            often easy, but acheiving complex figures with subplots, legends, overlays, and custom styling can
            require a mastery of a daunting API and many lines of code.  This complexity is discouraging to the new/casual Python
            user and can lead them to abandon Python in favor of more comfortable, albeit inferior, plotting tools like Excel.
         </p>
      </div>
   </div>


**fivecentplots** exists to drastically simplify the API required to generate complex plots, specifically for data
within **pandas** DataFrames.

.. raw:: html

   <div class="section" id="advantages-of-fivecentplots">
   <h2>Advantages of fivecentplots</h2>
   <div class="flex-container">
      <div class="card">
         <div class="header">
            EASE OF USE
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li>Plots are generated from a <b>single function call</b></li>
                  <li>Plot contents and design <b>defined by optional keyword arguments</b> with defaults in a simple "theme" file</li>
                  <li><b>Simple data access</b> using DataFrame column names and text-based filtering</li>
               </ul>
            </p>
         </div>
      </div>

      <div class="card">
         <div class="header">
            THE POWER OF KWARGS
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li> Colors, sizes, marker styles, subplot grouping, you name it--<b>everything is defined by keyword arguments</b>
                  <li> Keyword names follow a <b>basic naming convention</b> (ex: <span class="code">legend_title_font_size</span> defines the font size of the legend title)
                  <li> <b>Behind the scenes</b>, keywords translated into the complex code / API of the underlying plotting package (like <span class="code">matplotlib</span>)</li>
               </ul>
            </p>
         </div>
      </div>

      <div class="card">
         <div class="header">
            <a href='https://www.jmp.com/en_us/home.html'>JMP</a>, <a href='https://www.youtube.com/watch?v=010KyIQjkTk'>JMP</a>
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li>pandas enables statistical analysis in Python comparable to that of commercial software packages
                      like JMP. However, JMP offers easy creation of many useful charts that are tedious to create in Python.  <b>fivencentplots solves this</b>,
                      making it easy to create:
                     <ul>
                        <li><a href='http://www.jmp.com/support/help/Variability_Gauge_Charts.shtml'>JMP-style variability gauge charts </a></li>
                        <li><a href='http://www.jmp.com/support/help/Additional_Examples_of_the_Overlay_Plot_Platform.shtml#192569'>Grouped overlay plots</a> </li>
                        <li><a href='https://www.jmp.com/support/help/en/16.2/index.shtml#page/jmp/example-of-a-normal-quantile-plot.shtml'>Normal quantile plots</a> </li>
                     </ul>
                  </li>
               </ul>
            </p>
         </div>
      </div>

      <div class="card">
         <div class="header">
            INTERCHANGEABLE PLOT ENGINES
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li>Need a high-quality static image in <span class="code">matplotlib</span> style?  No problem!  Prefer an
                      interactive web plot from <span class="code">plotly</span>? No problem!
                      <b>fivecentplots can wrap any plotting "engine"</b> (or package)</li>
                  <li>Most importantly, <b>fivecentplots maintains the same API</b> regardless of plotting library.
                      The same function call invoked for <span class="code">matplotlib</span> can be used for <span class="code">plotly</span>--no need to learn a new syntax and API</li>
               </ul>
            </p>
         </div>
      </div>

      <div class="card">
         <div class="header">
            AUTOMATION
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li>Automated bulk plotting is easy since all plot parameters can be accessed from dictionary-like structures (yaml, json, ini)</li>
                  <li>Useful for production environments with standardized measurement data so users do not have to manually create line-health plots</li>
               </ul>
            </p>
         </div>
      </div>


      <div class="card">
         <div class="header">
            SIZING
         </div>

         <div class="contain">
            <p>
               <ul>
                  <li>Most plotting libraries define size based on the overall figure,
                      leaving you to guess an appropriate size to properly fit your desired contents.
                      <b>fivecentplots shifts the sizing paradigm by automatically determining the figure size based on the internal contents</b></li>
                  <li>Axes areas are more consistent and contents are less likely to get squished to fit in the figure</li>
               </ul>
            </p>
         </div>
      </div>
   </div>
   </div>


|


.. raw:: html

   <hr>
   <h2>Example</h2>


Consider the following plot of some fake current vs voltage data contained in a dummy DataFrame, ``df``:

.. image:: _static/images/syntax.png
    :align: center

|

Using **fivecentplots**, we need a single function call with the some select keyword arguments:

.. code-block:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', ax_size=[225, 225], share_y=False,
            filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
            ref_line=df['Voltage'], ref_line_legend_text='y==x', ref_line_style='--',
            xmin=0, xmax=1.7, ymin=[0, 0, 0], ymax=[1.3, 1.7, 5.2])

Consider one possible approach to generate a very similar plot using pure **matplotlib** syntax:

.. code-block:: python

   import matplotlib.pylab as plt
   import matplotlib
   import natsort

   # Filter the DataFrame to get the subset of interest
   df_sub = df[(df.Substrate=="Si")&(df['Target Wavelength']==450)&(df['Temperature [C]']==25)]

   # Set some defaults
   markers = ['o', '+', 's']
   colors = ['#4b72b0', '#c34e52', '#54a767']
   ymax = [1.3, 1.7, 5.2]
   lines = []

   # Create the figure and axes
   f, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=[9.82, 3.46])

   # Plot the data and style the axes
   for iboost, boost in enumerate(df_sub['Boost Level'].unique()):
      df_boost = df_sub[df_sub['Boost Level']==boost]
      for idie, die in enumerate(natsort.natsorted(df_boost.Die.unique())):
         df_die = df_boost[df_boost.Die==die]
         axes[iboost].set_facecolor('#eaeaea')
         axes[iboost].grid(which='major', axis='both', linestyle='-', color='#ffffff', linewidth=1.3)
         lines += axes[iboost].plot(df_die['Voltage'], df_die['I [A]'], '-', color=colors[idie],
                                    marker=markers[idie], markeredgecolor=colors[idie], markerfacecolor='none',
                                    markeredgewidth=1.5, markersize=6)
         axes[iboost].set_axisbelow(True)
         axes[iboost].spines['bottom'].set_color('#aaaaaa')
         axes[iboost].spines['top'].set_color('#aaaaaa')
         axes[iboost].spines['right'].set_color('#aaaaaa')
         axes[iboost].spines['left'].set_color('#aaaaaa')
         if iboost==0:
               axes[iboost].set_ylabel('I [A]', fontsize=14, fontweight='bold', fontstyle='italic')
         axes[iboost].set_xlabel('Voltage', fontsize=14, fontweight='bold', fontstyle='italic')
      axes[iboost].set_xlim(left=0, right=1.6)
      axes[iboost].set_ylim(bottom=0, top=ymax[iboost])

      # Add the column labels
      rect = matplotlib.patches.Rectangle((0, 1.044), 1, 30/225, fill=True, transform=axes[iboost].transAxes,
                                          facecolor='#8c8c8c', edgecolor='#8c8c8c', clip_on=False)
      axes[iboost].add_patch(rect)
      text = 'Boost Level = {}'.format(boost)
      axes[iboost].text(0.5, 1.111, text, transform=axes[iboost].transAxes,
                        horizontalalignment='center', verticalalignment='center',
                        rotation=0, color='#ffffff', weight='bold', size=16)

      # Customize ticks
      axes[iboost].tick_params(axis='both', which='major', pad=5, colors='#ffffff',
                              labelsize=13, labelcolor='#000000', width=2.2)
      # Add reference line
      ref_line = df_die['Voltage']
      ref = axes[iboost].plot(df_die['Voltage'], ref_line, '-', color='#000000', linestyle='--')
      if iboost == 0 :
         lines = ref + lines

   # Style the figure
   f.set_facecolor('#ffffff')
   f.subplots_adjust(left=0.077, right=0.882, top=0.827, bottom=0.176, hspace=0.133, wspace=0.313)

   # Add legend
   leg = f.legend(lines[0:4], ['y==x'] + list(df_boost.Die.unique()), title='Die', numpoints=1,
                  bbox_to_anchor=(1, 0.85), prop={'size': 12})
   leg.get_frame().set_edgecolor('#ffffff')

   # Show the plot
   plt.show()

This example is obviously a bit contrived as you could simplify things by modifying `rc_params`
or eliminating some of the specific style elements used here, but the general idea should be
clear:  **fivecentplots** can reduce the barrier to generate complex plots.

.. raw:: html

   <br>

What if we wanted to do the same plot using code for **plotly** or **bokeh**?  Well, we'd need to learn an entirely
different API!  But with **fivecentplots** we can just change the kwarg defining the plotting engine
(``engine``) and we are all set:

.. code-block:: python

   fcp.plot(df, x='Voltage', y='I [A]', legend='Die', col='Boost Level', ax_size=[225, 225], share_y=False,
            filter='Substrate=="Si" & Target Wavelength==450 & Temperature [C]==25',
            ref_line=df['Voltage'], ref_line_legend_text='y==x', ref_line_style='--',
            xmin=0, xmax=1.7, ymin=[0, 0, 0], ymax=[1.3, 1.7, 5.2], engine='plotly')

.. raw:: html

   <div align="center"><iframe src="_static/plotly_index.html" style="height:375px;width:930px;border:none;overflow:hidden;"></iframe></div>


.. note:: As of v0.6 **matplotlib** has the richest available plotting feature set.  **plotly** supports most plot
         types and many of the same style features.  **bokeh** support is limited to only scatter plots at this
         time.



Refer to the topics on the sidebar for more details on plot types and options.


Contents
--------

* `Basics: <basics/index.html>`_ installation, basic figure construction, introduction to `Element` objects, data grouping, etc

.. toctree::
   :maxdepth: 2
   :hidden:

   basics/index


* `Plot types: <plot_types/index.html>`_ tutorials on various plot types supported by **fivecentplots**

.. toctree::
   :maxdepth: 2
   :hidden:

   plot_types/index

* `Engines: <engines/index.html>`_ description of how to use different plotting engines (matplotlib, bokeh, plotly) with **fivecentplots**

.. toctree::
   :maxdepth: 2
   :hidden:

   engines/index

* `Changelog <changelog.html>`_

.. toctree::
   :hidden:

   changelog

* `API: <api/index.html>`_ description of various keyword arguments supported by **fivecentplots**

.. toctree::
   :maxdepth: 2
   :hidden:

   api/index
