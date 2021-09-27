import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import matplotlib as mpl

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
