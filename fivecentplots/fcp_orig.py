def plot_old(**kwargs):
    """ Main x-y plotting function

    This function wraps many variations of x-y plots from the matplotlib
    library.  At minimum, it requires a pandas DataFrame with at least two
    columns and two column names for the x and y axis.  Plots can be
    customized and enhanced by passing keyword arguments as defined below.
    Default values that must be defined in order to generate the plot are
    pulled from the fcp_params dictionary defined in defaults.py.

    Args:
        df (DataFrame): DataFrame containing data to plot
        x (str):        name of x column in df
        y (str|list):   name or list of names of y column(s) in df

    Keyword Args:
        see get_defaults for definitions

    Returns:
        layout (LayoutMPL obj):  contains all the spacing information used to
            construct the figure
    """


    mplp.close('all')

    # Init plot
    df, x, y, z, kw = init('plot', kwargs)
    kw['ptype'] = 'plot'
    if type(df) is bool and not df:
        return

    # Handle multiple y-values
    if kw['twinx']:
        kw['ws_row_label'] = fcp_params['fig_ax_ws']

    # Iterate over discrete figures
    for ifig, fig_item in enumerate(kw['fig_items']):

        # Make a data subset and filter
        df_fig = get_df_figure(df, fig_item, kw)

        # Set up the row grouping
        kw = get_rc_groupings(df_fig, kw)

        # Set up the legend grouping
        kw = get_legend_groupings(df_fig, y, kw)

        # Get tick label size
        kw = get_label_sizes(df_fig, x, y, kw)

        # Make the plot figure and axes
        layout, fig, axes, kw = make_fig_and_ax(kw)

        # Handle colormaps
        if kw['cmap']=='jmp_spectral':
            cmap = jmp_spectral
        elif kw['cmap'] is not None:
            cmap = mplp.get_cmap(kw['cmap'])

        # Make the plots by row and by column
        curves = []
        curve_dict = {}

        for ir in range(0, kw['nrow']):
            for ic in range(0, kw['ncol']):

                # Handle missing wrap plots
                if kw['wrap'] is not None:
                    if ir*kw['ncol'] + ic > len(kw['wrap'])-1:
                        axes[ir, ic].axis('off')
                        continue

                # Twinning
                if kw['twinx']:
                    ax2 = axes[ir, ic].twinx()
                else:
                    ax2 = None

                # Set colors
                axes[ir, ic] = set_axes_colors(axes[ir, ic], kw)
                if ax2 is not None:
                    ax2 = set_axes_colors(ax2, kw)

                # Style gridlines
                axes[ir, ic] = set_axes_grid_lines(axes[ir, ic], kw)
                if ax2 is not None:
                    ax2 = set_axes_grid_lines(ax2, kw, True)

                # Subset the data
                df_sub = get_rc_subset(df_fig, ir, ic, kw)

                # Set the axes scale
                plotter = set_axes_scale(axes[ir, ic], kw)

                # Apply any data transformations
                df_sub = set_data_transformation(df_sub, x, y, z, kw)

                # Add horizontal and vertical lines
                axes[ir, ic] = add_lines(axes[ir, ic], kw)

                # Legend grouping plots
                if kw['leg_groups'] is None and not kw['twinx']:
                    for iy, yy in enumerate(natsorted(y)):
                        if len(df[x].dropna()) == 0 or \
                                len(df[yy].dropna()) == 0:
                            continue
                        if len(df_sub[x].dropna()) == 0 or \
                                len(df_sub[yy].dropna()) == 0:
                            continue

                        # Define color and marker types
                        color = \
                            kw['line_color'] if kw['line_color'] is not None \
                                             else kw['colors'][iy]
                        marker = \
                            kw['marker_type'] if kw['marker_type'] is not None\
                                              else markers[iy]
                        # Plot
                        if kw['stat'] is None:
                            curve = add_curves(plotter,
                                                df_sub[x],
                                                df_sub[yy],
                                                color,
                                                marker,
                                                kw['points'],
                                                kw['lines'],
                                                markersize=kw['marker_size'],
                                                linestyle=kw['line_style'],
                                                linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve

                        else:
                            kw['stat_val'] = \
                                validate_columns(df_sub, kw['stat_val'])
                            if 'median' in kw['stat'].lower() \
                                    and kw['stat_val'] is not None:
                                df_stat = \
                                    df_sub.groupby(kw['stat_val']).median()
                            elif kw['stat_val'] is not None:
                                df_stat = \
                                    df_sub.groupby(kw['stat_val']).mean()
                            else:
                                print('Could not group data by stat_val '
                                      'columns.  Using full data set...')
                                df_stat = df_sub

                            if 'only' not in kw['stat'].lower():
                                # Plot the points for each data set
                                curve = add_curves(
                                             plotter,
                                             df_sub[x],
                                             df_sub[yy],
                                             color,
                                             marker,
                                             True,
                                             False,
                                             markersize=kw['marker_size'],
                                             linestyle='none',
                                             linewidth=0)
                                if curve is not None:
                                    curves += curve
                            # Plot the lines
                            if 'only' in kw['stat'].lower():
                                curve = add_curves(
                                    plotter,
                                    df_stat.reset_index()[x],
                                    df_stat[yy],
                                    color,
                                    marker,
                                    True,
                                    True,
                                    markersize=kw['marker_size'],
                                    linestyle=kw['line_style'],
                                    linewidth=kw['line_width'])
                                if curve is not None:
                                    curves += curve
                            else:
                                add_curves(
                                    plotter,
                                    df_stat.reset_index()[x],
                                    df_stat[yy],
                                    color,
                                    marker,
                                    False,
                                    True,
                                    markersize=kw['marker_size'],
                                    linestyle=kw['line_style'],
                                    linewidth=kw['line_width'])

                        if kw['line_fit'] is not None \
                                and kw['line_fit'] != False:
                            # Fit the polynomial
                            coeffs = np.polyfit(np.array(df_sub[x]),
                                                np.array(df_sub[yy]),
                                                kw['line_fit'])

                            # Calculate the fit line
                            xval = df_sub[x]
                            yval = np.polyval(coeffs, xval)

                            # Find r^2
                            ybar = df_sub[yy].sum()/len(df_sub[yy])
                            ssreg = np.sum((yval-ybar)**2)
                            sstot = np.sum((df_sub[y]-ybar)**2)
                            r_sq = ssreg/sstot

                            # Add fit line
                            xval = np.linspace(0.9*xval.min(),
                                               1.1*xval.max(), 10)
                            yval = np.polyval(coeffs, xval)
                            add_curves(plotter,
                                       xval,
                                       yval,
                                       'k',
                                       marker,
                                       False,
                                       True,
                                       linestyle='--')

                            # Add fit equation (need to update for more than
                            # 1D and formatting)
                            if coeffs[1] < 0:
                                sign = '-'
                            else:
                                sign = '+'
                            axes[ir,ic].text(
                                0.05, 0.9,
                                'y=%.4f * x %s %.4f\nR^2=%.5f' %
                                (coeffs[0], sign, abs(coeffs[1]), r_sq),
                                transform=axes[ir,ic].transAxes)

                    if kw['yline'] is not None:
                        if ir==0 and ic==0:
                            kw['leg_items'] += [kw['yline']]
                            curve = add_curves(plotter,
                                               df[x],
                                                df[kw['yline']],
                                               'k',
                                               None,
                                               False,
                                               True,
                                               linestyle='-')
                            if curve is not None:
                                curves += curve
                        else:
                            add_curves(plotter,
                                       df[x],
                                        df[kw['yline']],
                                       'k',
                                       None,
                                       False,
                                       True,
                                       linestyle='-')

                elif kw['leg_groups'] is None and kw['twinx']:

                    # Define color and marker types
                    color = \
                        kw['line_color'] if kw['line_color'] is not None \
                                         else kw['colors'][0:2]
                    marker = \
                        kw['marker_type'] if kw['marker_type'] is not None\
                                          else markers[0:2]
                    # Set the axes scale
                    if kw['ax_scale2'] == 'semilogy' or \
                       kw['ax_scale2'] == 'logy':
                        plotter2 = ax2.semilogy
                    else:
                        plotter2 = ax2.plot

                    if kw['stat'] is None:
                        curve = add_curves(plotter,
                                           df_sub[x],
                                           df_sub[y[0]],
                                           color[0],
                                           marker[0],
                                           True,
                                           markersize=kw['marker_size'],
                                           linestyle=kw['line_style'],
                                           linewidth=kw['line_width'])
                        if curve is not None:
                            curves += curve
                        curve = add_curves(plotter2,
                                           df_sub[x],
                                           df_sub[y[1]],
                                           color[1],
                                           marker[1],
                                           True,
                                           markersize=kw['marker_size'],
                                           linestyle=kw['line_style'],
                                           linewidth=kw['line_width'])
                        if curve is not None:
                            curves += curve
                    else:
                        kw['stat_val'] = \
                                validate_columns(df_sub, kw['stat_val'])
                        if 'median' in kw['stat'].lower()\
                                and kw['stat_val'] is not None:
                            df_stat = df_sub.groupby(kw['stat_val']).median()
                        elif kw['stat_val'] is not None:
                            df_stat = df_sub.groupby(kw['stat_val']).mean()
                        else:
                            print('Could not group data by stat_val '
                                  'columns.  Using full data set...')
                            df_stat = df_sub

                        # Plot the points for each data set
                        if 'only' not in kw['stat'].lower():
                            curve = add_curves(plotter,
                                               df_sub[x],
                                               df_sub[y[0]],
                                               color[0],
                                               marker[0],
                                               True,
                                               False,
                                               markersize=kw['marker_size'],
                                               linestyle='none',
                                               linewidth=0)
                            if curve is not None:
                                curves += curve
                            curve = add_curves(plotter2,
                                               df_sub[x],
                                               df_sub[y[1]],
                                               color[1],
                                               marker[1],
                                               True,
                                               False,
                                               markersize=kw['marker_size'],
                                               linestyle='none',
                                               linewidth=0)
                            if curve is not None:
                                curves += curve
                        # Plot the lines
                        if 'only' in kw['stat'].lower():
                            curve = add_curves(plotter,
                                               df_stat.reset_index()[x],
                                               df_stat[y[0]],
                                               color[0],
                                               marker[0],
                                               True,
                                               True,
                                               markersize=kw['marker_size'],
                                               linestyle=kw['line_style'],
                                               linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve
                            curve = add_curves(plotter,
                                               df_stat.reset_index()[x],
                                               df_stat[y[1]],
                                               color[1],
                                               marker[1],
                                               True,
                                               True,
                                               markersize=kw['marker_size'],
                                               linestyle=kw['line_style'],
                                               linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve
                        else:
                            add_curves(plotter,
                                       df_stat.reset_index()[x],
                                       df_stat[y[0]],
                                       color[0],
                                       marker[0],
                                       False,
                                       True,
                                       markersize=kw['marker_size'],
                                       linestyle=kw['line_style'],
                                       linewidth=kw['line_width'])
                            add_curves(plotter2,
                                       df_stat.reset_index()[x],
                                       df_stat[y[1]],
                                       color[1],
                                       marker[1],
                                       False,
                                       True,
                                       markersize=kw['marker_size'],
                                       linestyle=kw['line_style'],
                                       linewidth=kw['line_width'])

                elif kw['leg_groups'] and kw['twinx']:
                    # NEED TO ADJUST FOR LEG GROUPS FROM BELOW
                    # Define color and marker types
                    color = \
                        kw['line_color'] if kw['line_color'] is not None \
                                         else kw['colors'][0:2]
                    marker = \
                        kw['marker_type'] if kw['marker_type'] is not None\
                                          else markers[0:2]
                    # Set the axes scale
                    if kw['ax_scale2'] == 'semilogy' or \
                       kw['ax_scale2'] == 'logy':
                        plotter2 = ax2.semilogy
                    else:
                        plotter2 = ax2.plot

                    for ileg, leg_group in enumerate(kw['leg_items'][::2]):
                        group = leg_group.split(': ')[0]
                        subset = df_sub[kw['leg_groups']]==group
                        if kw['stat'] is None:
                            curve = add_curves(plotter,
                                               df_sub[x][subset],
                                               df_sub[y[0]][subset],
                                               kw['colors'][2*ileg],
                                               markers[2*ileg],
                                               True,
                                               markersize=kw['marker_size'],
                                               linestyle=kw['line_style'],
                                               linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve
                            curve = add_curves(plotter2,
                                               df_sub[x][subset],
                                               df_sub[y[1]][subset],
                                               kw['colors'][2*ileg+1],
                                               markers[2*ileg+1],
                                               True,
                                               markersize=kw['marker_size'],
                                               linestyle=kw['line_style'],
                                               linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve
                        else:
                            kw['stat_val'] = \
                                validate_columns(df_sub, kw['stat_val'])
                            if 'median' in kw['stat'].lower()\
                                    and kw['stat_val'] is not None:
                                df_stat = df_sub.groupby(kw['stat_val']).median()
                            elif kw['stat_val'] is not None:
                                df_stat = df_sub.groupby(kw['stat_val']).mean()
                            else:
                                print('Could not group data by stat_val '
                                      'columns.  Using full data set...')
                                df_stat = df_sub

                            # Plot the points for each data set
                            if 'only' not in kw['stat'].lower():
                                curve = add_curves(
                                                 plotter,
                                                 df_sub[x][subset],
                                                 df_sub[y[0]][subset],
                                                 kw['colors'][2*ileg],
                                                 markers[2*ileg],
                                                 True,
                                                 False,
                                                 markersize=kw['marker_size'],
                                                 linestyle='none',
                                                 linewidth=0)
                                if curve is not None:
                                    curves += curve
                                curve = add_curves(
                                                 plotter2,
                                                 df_sub[x][subset],
                                                 df_sub[y[1]][subset],
                                                 kw['colors'][2*ileg+1],
                                                 markers[2*ileg+1],
                                                 True,
                                                 False,
                                                 markersize=kw['marker_size'],
                                                 linestyle='none',
                                                 linewidth=0)
                                if curve is not None:
                                    curves += curve
                            # Plot the lines
                            if 'only' in kw['stat'].lower():
                                curve = add_curves(
                                             plotter,
                                             df_stat.reset_index()[x][subset],
                                             df_stat[y[0]][subset],
                                             kw['colors'][2*ileg],
                                             markers[2*ileg],
                                             True,
                                             True,
                                             markersize=kw['marker_size'],
                                             linestyle=kw['line_style'],
                                             linewidth=kw['line_width'])
                                if curve is not None:
                                    curves += curve
                                curve = add_curves(
                                             plotter,
                                             df_stat.reset_index()[x][subset],
                                             df_stat[y[1]][subset],
                                             kw['colors'][2*ileg+1],
                                             markers[2*ileg+1],
                                             True,
                                             True,
                                             markersize=kw['marker_size'],
                                             linestyle=kw['line_style'],
                                             linewidth=kw['line_width'])
                                if curve is not None:
                                    curves += curve
                            else:
                                add_curves(plotter,
                                           df_stat.reset_index()[x][subset],
                                           df_stat[y[0]][subset],
                                           kw['colors'][2*ileg],
                                           markers[2*ileg],
                                           False,
                                           True,
                                           markersize=kw['marker_size'],
                                           linestyle=kw['line_style'],
                                           linewidth=kw['line_width'])
                                add_curves(plotter2,
                                           df_stat.reset_index()[x][subset],
                                           df_stat[y[1]][subset],
                                           kw['colors'][2*ileg+1],
                                           markers[2*ileg+1],
                                           False,
                                           True,
                                           markersize=kw['marker_size'],
                                           linestyle=kw['line_style'],
                                           linewidth=kw['line_width'])

                        # Draw confidence intervals
                        conf_int(df_sub[subset], x, y, axes[ir, ic], color, kw)

                else:
                    for ileg, leg_group in enumerate(kw['leg_items']):

                        idx = kw['leg_items'].index(leg_group)

                        # Define color and marker types
                        if kw['cmap']:
                            color = cmap((ileg+1)/(len(kw['leg_items'])+1))
                        else:
                            color = kw['line_color'] \
                                if kw['line_color'] is not None \
                                else kw['colors'][idx]
                        marker = kw['marker_type'] \
                                 if kw['marker_type'] is not None \
                                 else markers[idx]

                        # Subset the data by legend group and plot
                        group = leg_group.split(': ')[0]
                        if len(y) > 1 and not kw['twinx']:
                            yy = leg_group.split(': ')[1]
                        else:
                            yy = y[0]
                        subset = df_sub[kw['leg_groups']]==group

                        if kw['stat'] is None:
                            curve = add_curves(plotter,
                                               df_sub[x][subset],
                                               df_sub[yy][subset],
                                               color,
                                               marker,
                                               kw['points'],
                                               kw['lines'],
                                               markersize=kw['marker_size'],
                                               linestyle=kw['line_style'],
                                               linewidth=kw['line_width'])
                            if curve is not None:
                                curves += curve
                                curve_dict[leg_group] = curve
                        else:
                            # Plot the points
                            if kw['points']:
                                curve = add_curves(
                                    plotter,
                                    df_sub[x][subset],
                                    df_sub[yy][subset],
                                    color,
                                    marker,
                                    True,
                                    False,
                                    markersize=kw['marker_size'],
                                    linestyle='none',
                                    linewidth=0)
                                if curve is not None:
                                    curves += curve
                                    curve_dict[leg_group] = curve

                            # Plot the lines
                            kw['stat_val'] = \
                                validate_columns(df_sub, kw['stat_val'])
                            if 'median' in kw['stat'].lower()\
                                    and kw['stat_val'] is not None:
                                df_stat = \
                                    df_sub[subset].groupby(kw['stat_val'])\
                                        .median().reset_index()
                            elif kw['stat'] is not None\
                                    and kw['stat_val'] is not None:
                                df_stat = \
                                    df_sub[subset].groupby(kw['stat_val'])\
                                        .mean().reset_index()
                            else:
                                print('Could not group data by stat_val '
                                      'columns.  Using full data set...')
                                df_stat = df_sub

                            if not kw['points']:
                                curve = add_curves(
                                    plotter,
                                    df_stat[x],
                                    df_stat[yy],
                                    color,
                                    marker,
                                    False,
                                    True,
                                    markersize=kw['marker_size'],
                                    linestyle=kw['line_style'],
                                    linewidth=kw['line_width'])
                                if curve is not None:
                                    curves += curve
                                    curve_dict[leg_group] = curve
                            else:
                                add_curves(
                                    plotter,
                                    df_stat.reset_index()[x],
                                    df_stat[yy],
                                    color,
                                    marker,
                                    False,
                                    True,
                                    markersize=kw['marker_size'],
                                    linestyle=kw['line_style'],
                                    linewidth=kw['line_width'])

                        if kw['line_fit'] is not None \
                                and kw['line_fit'] != False:
                            # Fit the polynomial
                            xval = df_sub[x][subset]
                            yval = df_sub[yy][subset]
                            coeffs = np.polyfit(np.array(xval),
                                                np.array(yval),
                                                kw['line_fit'])

                            # Calculate the fit line
                            yval = np.polyval(coeffs, xval)

                            # Add fit line
                            xval = np.linspace(0.9*xval.min(),
                                               1.1*xval.max(), 10)
                            yval = np.polyval(coeffs, xval)
                            add_curves(plotter,
                                       xval,
                                       yval,
                                       color,
                                       marker,
                                       False,
                                       True,
                                       linestyle='--')

                        # Draw confidence intervals
                        conf_int(df_sub[subset], x, yy, axes[ir, ic], color, kw)

                # Axis ranges
                axes[ir, ic], ax = set_axes_ranges(df_fig, df_sub, x, y,
                                                   axes[ir, ic], ax2, kw)

                # Add labels
                axes[ir, ic], ax2 = \
                    set_axes_labels(axes[ir, ic], ax2, ir, ic, kw)

                # Add row/column labels
                axes[ir, ic] = \
                    set_axes_rc_labels(axes[ir, ic], ir, ic, kw, layout)

                # Adjust the tick marks
                axes[ir, ic] = set_axes_ticks(axes[ir, ic], kw)
                if ax2 is not None:
                    ax2 = set_axes_ticks(ax2, kw, True)
                    if ic != kw['ncol'] - 1 and not kw['separate_labels']:
                        mplp.setp(ax2.get_yticklabels(), visible=False)

        # Make the leg/curve lists
        if len(curve_dict.keys()) > 0:
            kw['leg_items'] = natsorted(list(curve_dict.keys()))
            curves = [curve_dict[f][0] for f in kw['leg_items']]

        # Add the legend
        fig = add_legend(fig, curves, kw, layout)

        # Add a figure title
        set_figure_title(df_fig, axes[0,0], kw, layout)

        # Build the save filename
        filename = set_save_filename(df_fig, x, y, kw, ifig)

        # Save and optionally open
        save(fig, filename, kw)

        # Reset values for next loop
        kw['leg_items'] = []

    if not kw['inline']:
        mplp.close('all')

    else:
        return fig


