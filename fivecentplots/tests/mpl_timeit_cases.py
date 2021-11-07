import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib as mpl


def common(x, y, figsize=(5, 5)):

    plt.close('all')

    # make figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='#ffffff')

    # Set the axes colors
    ax.set_facecolor('#eaeaea')
    for f in ['bottom', 'top', 'right', 'left']:
        ax.spines[f].set_color('#aaaaaa')

    # Add and format gridlines
    ax.set_axisbelow(True)
    ax.xaxis.grid(b=True, which='major', color='#ffffff',
                  linestyle='-', linewidth=1)
    ax.yaxis.grid(b=True, which='major', color='#ffffff',
                  linestyle='-', linewidth=1)

    # Adjust tick marks
    ax.tick_params(axis='both',
                   which='major',
                   pad=4,
                   colors='#ffffff',
                   labelcolor='#000000',
                   labelsize=16,
                   top=False,
                   bottom=True,
                   right=False,
                   left=True,
                   length=6.2,
                   width=2.2,
                   direction='in',
                   zorder=100,
                   )
    ax.tick_params(axis='both',
                   which='minor',
                   top=False,
                   bottom=False,
                   left=False,
                   right=False)

    # Add axis labels
    ax.set_xlabel(x, fontsize=16, style='italic', weight='bold')
    ax.set_ylabel(y, fontsize=16, style='italic', weight='bold')

    return fig, ax


def test_plot(data):
    """
    fcp plot

    baseline mpl only:  60.1 ms ± 3.36 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

    fcp worst offenders:
        **Data obj: 7.576 [ms]
        dd.get_df_figure: 4.815 [ms]
        **layout class: 14.333 [ms]
        **ifig=None | make_figure: 170.611 [ms]
            --> 140ms is get_element_sizes
            --> 70ms is the render and post render
        ifig=None | ir=0 | ic=0 | plot: 2.166 [ms]
        ifig=None | ir=0 | ic=0 | set_axes_ranges: 1.286 [ms]
        ifig=None | ir=0 | ic=0 | set_axes_labels: 2.188 [ms]
        **ifig=None | ir=0 | ic=0 | set_axes_ticks: 28.781 [ms]
        ifig=None | set_figure_title: 1.627 [ms]
        ifig=None | return: 15.39 [ms] --> can't change just the show command, hits both pure mpl and fcp

    Total time: 251.014 [ms] --> 4x slower

    """

    x = 'Voltage'
    y = 'I [A]'
    data = data.loc[(data.Substrate=='Si')& \
                    (data['Target Wavelength']==450)& \
                    (data['Boost Level']==0.2)& \
                    (data['Temperature [C]']==25)
                   ]
    fig, ax = common(x, y, (6.5, 6.5))

    # plot the data
    ax.plot(data[x], data[y], 'o', color='#4b72b0', markerfacecolor='none',
            markeredgecolor='#4b72b0', markersize=9, markeredgewidth=1.5)

    # title
    fig.suptitle('IV Data', fontsize=20, weight='bold')
    ax.set_xlim((-0.08, 1.68))
    ax.set_ylim((-0.06275, 1.31775))

    # Clean up and display
    plt.tight_layout() # whatever auto adjust
    #plt.draw()
    #fig.set_size_inches((10, 10))
    plt.show(block=False)
    #plt.savefig('test.png')


def test_hist(data):
    """
    fcp hist worst offenders:

    ifig=None | make_figure: 544.925 [ms]
    Data obj: 249.443 [ms] - 233ms is switch_type --> dropped to 160
    ifig=None | ir=0 | ic=0 | set_axes_ticks: 112.095 [ms]
    layout class: 14.112 [ms]
    ifig=None | ir=0 | ic=0 | plot: 3.354 [ms]
    ifig=None | ir=0 | ic=0 | set_axes_labels: 2.687 [ms]
    ifig=None | ir=0 | ic=0 | set_axes_scale: 2.224 [ms]
    ifig=None | ir=0 | ic=0 | set_axes_grid_lines: 1.206 [ms]


    axes labels: 1.436 [ms]
    ticks: 1.55 [ms]
    legend: 3.546 [ms]
    box elements: 1.163 [ms]
    rc labels: 1.043 [ms]
    confidence: 0.276 [ms]
    text: 0.108 [ms]
    extras: 0.014 [ms]
    overrides: 0.005 [ms]


    """

    plt.close('all')

    # make figure
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#ffffff')

    # Set the axes colors
    ax.set_facecolor('#eaeaea')
    for f in ['bottom', 'top', 'right', 'left']:
        ax.spines[f].set_color('#aaaaaa')

    # Add and format gridlines
    ax.set_axisbelow(True)
    ax.xaxis.grid(b=True, which='major', color='#dddddd',
                  linestyle='-', linewidth=1)
    ax.yaxis.grid(b=True, which='major', color='#dddddd',
                  linestyle='-', linewidth=1)

    # plot the data and set scale
    hist = np.histogram(data, bins=int(data.max())-int(data.min()))
    ax.semilogy(hist[1][1:], hist[0], color='#4b72b0')

    # Adjust tick marks
    ax.tick_params(axis='both',
                   which='major',
                   pad=4,
                   colors='#ffffff',
                   labelcolor='#000000',
                   labelsize=12,
                   top=False,
                   bottom=True,
                   right=False,
                   left=True,
                   length=6.2,
                   width=2.2,
                   direction='in',
                   zorder=100,
                   )
    ax.tick_params(axis='both',
                   which='minor',
                   top=False,
                   bottom=False,
                   left=False,
                   right=False)

    # Add axis labels
    ax.set_xlabel('Value', fontsize=16, style='italic')
    ax.set_ylabel('Counts', fontsize=16, style='italic')

    plt.tight_layout() # whatever auto adjust
    plt.show(block=False)
    plt.savefig('test.png')
