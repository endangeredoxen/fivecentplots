# fivecentplots — Agent Reference

Built from `fivecentplots` v0.6.1's **actual source** (`src/fivecentplots/kwargs/csv/*.csv`,
the same files the library itself uses to validate kwargs and generate its docstrings) rather
than scraped from rendered Sphinx docs. Cross-checked against `kwargs/kwargs_all.txt`, the
library's own flat list of every valid kwarg name.

## Quick start

```python
import fivecentplots as fcp
import pandas as pd

df = pd.DataFrame(...)  # your data

fcp.plot(df, x='time', y='value', legend='group')       # xy line/scatter
fcp.boxplot(df, y='value', groups=['batch', 'sample'])   # box plot
fcp.hist(df, x='value')                                  # histogram
fcp.bar(df, x='category', y='value')                     # bar chart
```

Every function follows the same pattern: `fcp.<plot_type>(df, **kwargs)`. `df` is a
`pandas.DataFrame` (or a numpy array for a couple of functions — see per-function notes below).
Style, grouping, and layout are controlled entirely through kwargs — there is no
object/builder API to learn.

Common kwargs across virtually every call:
- `title`, `filename`, `filepath` — labeling and output location
- `save=True/False`, `show=True/False` — write to disk vs. open a viewer
- `engine='mpl'|'bokeh'|'plotly'` — swap rendering backend without changing anything else
- `filter='ColA=="x" & ColB>5'` — pandas-query-style row filtering, done for you
- `row=`, `col=`, `wrap=` — facet into a subplot grid by column value(s)
- `ax_size=[width, height]` — axes (not figure) size in pixels

## How this doc is organized

1. **Per-function reference** — one section per `fcp.*` plot function: required kwargs,
   a minimal runnable example, and that function's own specific styling kwargs.
2. **Universal kwargs** — kwargs that work on *every* plot function but are **not**
   documented in any individual function's docstring (grouping, legend, titles, gridlines,
   ticks, whitespace, save/show options, etc.). This is the single largest source of
   false negatives if you rely on `inspect.getdoc()` alone.
3. **Known-valid-but-undocumented kwargs** — real kwarg families with zero recorded
   description (Gantt "today marker" and "workstream" features, Plotly toolbar controls).
4. **Naming conventions** — patterns you can generalize even for kwargs not explicitly listed.
5. **Gotchas** — known sharp edges worth knowing before you rely on this library's output.

## Naming conventions (generalize beyond this list)

fivecentplots' `kwargs_all.txt` validation list has **1,608** entries; the curated CSV docs
below explicitly describe about 550 of them. The remaining ~1,050 are almost entirely
predictable expansions of a documented base name, using these placeholder conventions:

| Placeholder | Expands to | Example |
|---|---|---|
| `_q` suffix (e.g. `label_q_font_size`) | `_x`, `_y`, `_x2`, `_y2` (sometimes `_z`) | `label_x_font_size`, `label_y2_font_size` |
| `[major\|minor]` | pick one | `ticks_major`, `ticks_minor` |
| `[x\|x2\|y\|y2\|z]` | pick one | `ticks_x`, `ticks_z` |
| `[ax\|ax2]` | pick one | `ax_hlines`, `ax2_hlines` |
| `[h\|v]` | pick one | `ax_hlines`, `ax_vlines` |
| trailing `_color`/`_alpha`/`_edge_color`/`_edge_alpha`/`_edge_width`/`_fill_color`/`_fill_alpha` | a family shared across most visual elements (boxes, bars, labels, titles, legend, gridlines...) | `box_fill_color`, `legend_edge_alpha`, `title_edge_width` |

If a kwarg you construct this way isn't actually supported, fivecentplots does **not**
raise an exception — `validate_kwargs()` emits a `warnings.warn(...)` and the plot still
renders, just without that kwarg applied. That's a safe way to probe for pattern-based
kwargs: try it, and check for a warning in stderr rather than a crash.

## Gotchas

- **Falsy group labels (`0`, `0.0`, `""`) render as `"True"` in the matplotlib engine
  (fixed after v0.6.1).** `engines/mpl.py`'s `add_label()` used `if not text_str:` to
  detect "no label override provided," which also matches legitimate falsy values.
  If you're on an affected version, cast falsy group columns to string first
  (`df[col] = df[col].astype(str)`) as a workaround.
- **A function's own docstring is not the full kwarg surface.** As documented above,
  ~250 "universal" kwargs (legend, titles, gridlines, ticks, whitespace, grouping, save
  options) apply to every plot type but appear nowhere in that plot type's docstring —
  they're documented once, separately. Don't conclude a kwarg is unsupported just because
  `inspect.getdoc(fcp.boxplot)` doesn't mention it.
- **Unsupported kwargs warn, they don't raise.** A typo'd or invalid kwarg produces a
  `UserWarning` via `validate_kwargs()`, not a `TypeError`. If a plot silently ignores a
  kwarg you passed, check for a warning rather than assuming it's a no-op by design.
- **`box_markers` kwargs are separate from `markers`.** Boxplot-specific marker styling
  (`box_markers.csv` in source) is a different set of kwargs than the generic `markers`
  family used by `fcp.plot()` — don't assume `marker_*` kwargs from the xy-plot docs
  automatically apply to boxplots.

---

# Per-function reference

## `fcp.plot()` — XY line/scatter plot

XY plot.

**Signature:** `fcp.plot(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str \| list | x-axis column name(s) [REQUIRED] |
| `y` | str \| list | y-axis column name(s) [REQUIRED] |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data.csv')
fcp.plot(df, x='Voltage', y='I [A]', legend=['Die', 'Substrate'], ax_size=[400, 300],
filter='Target Wavelength==450 & Temperature [C]==25 & Boost Level==0.2')
```

<details>
<summary>Plot-specific kwarg reference (52 kwargs, click to expand)</summary>

#### LINES

| kwarg | type | default | description |
|---|---|---|---|
| `cmap` | str | None | Color map name (overrides all other color parameters) |
| `lines` | boolean | True | Enable/disable plotting of lines |
| `line_alpha` | str\|list | 1 | Transparency value for the line(s) between 0-1 |
| `line_color` | str\|list | fcp.DEFAULT_COLORS | Hex color string or list of hex color strings for the plot lines |
| `line_style` | str\|list | '-' | Matplotlib string character for line style {'-'; '--'; '-.' ':'} |
| `line_width` | int\|list | 1 | Line width in pixels |

#### MARKERS

| kwarg | type | default | description |
|---|---|---|---|
| `markers` | boolean | True | Enable/disable data point markers |
| `marker_fill` | boolean | False | Enable/disable color fill in markers |
| `marker_edge_color` | str\|list | fcp.DEFAULT_COLORS | Hex color string for the marker edges |
| `marker_edge_width` | float | 1 | Marker edge line width in pixels |
| `marker_fill_color` | str\|list | fcp.DEFAULT_COLORS | Hex color string for the fill color of markers |
| `marker_jitter` / `jitter` | boolean | True | For boxplots add random noise on x-axis to show separation between markers |
| `marker_size` | float\|str | 6 | Size in pixels of the data point markers or a DataFrame column name with a custom marker size on each row |

#### AX_HLINES_VLINES

| kwarg | type | default | description |
|---|---|---|---|
| `[ax|ax2]_[h|v]lines` | float\|list of tuples and floats | None | Add horizontal / vertical lines to the plot; if only float value is provided add a solid black line with width=1 pixel at that value; if tuple add any one or more of the following in order: [1] float value or DataFrame column name [required]; [2] hex string for line color; [3] line style str; [4] line width in pixels; [5] line alpha transparency value from 0-1; [6] legend text  [added automatically if using a column name for value] |
| `[ax|ax2]_[h|v]lines_alpha` | float\|list of floats | 1 | Transparency value for the lines between 0-1; use a list to use different values for each subplot |
| `[ax|ax2]_[h|v]lines_by_plot` | bool | None | Add a line with a different value to each subplot when using row/col/wrap grouping |
| `[ax|ax2]_[h|v]lines_color` | str\|list of str | 1 | Transparency value for the lines between 0-1; use a list to use different values for each subplot |

#### CONTROL_LIMITS

| kwarg | type | default | description |
|---|---|---|---|
| `lcl` | float | None | Float value to start the lower control limit shading region |
| `ucl` | float | None | Float value to start the upper control limit shading region |
| `control_limit_side` | str | outside | Determines if shaded region is <= `lcl` and >= `ucl` {"outside"} or between the lcl and ucl {"inside"} |
| `lcl` / `ucl_edge_alpha` | float | 0.25 | Transparency value for the line starting the control limit shaded region between 0-1 |
| `lcl` / `ucl_edge_color` | str | fcp.DEFAULT_COLORS | Hex color string for the the line starting the control limit shaded region |
| `lcl` / `ucl_edge_style` | str | '-' | Line style for the line starting the control limit shaded region {‘-’, ‘--’, ‘-.’, ‘:’} |
| `lcl` / `ucl_edge_width` | float | 1 | Width of the line starting the control limit shaded region in pixels |
| `lcl` / `ucl_fill_alpha` | float | 0.20 | Transparency value for the control limit shaded region fill between 0-1 |
| `lcl` / `ucl_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string for the control limit shaded region fill |

#### CONFIDENCE_INTERVALS

| kwarg | type | default | description |
|---|---|---|---|
| `conf_int` | float | None | Interval with upper and lower bounds based on a single confidence value between 0-1 (typical=0.95) |
| `perc_int` | list of float | None | Interval with upper and lower bounds based on percentiles between 0-1 |
| `nq_int` | list of float | None | Interval with upper and lower bounds based on values of sigma (where the mean of a distribution is sigma=0) |
| `conf_int_` / `perc_int_` / `nq_int_edge_alpha` | float | 0.25 | Transparency value for the lines bounding the interval shaded region between 0-1 |
| `conf_int_` / `perc_int_` / `nq_int_edge_color` | str | fcp.DEFAULT_COLORS | Hex color string for the the lines bounding the interval shaded region |
| `conf_int_` / `perc_int_` / `nq_int_edge_style` | str | '-' | Line style for the lines bounding the interval shaded region {‘-’, ‘--’, ‘-.’, ‘:’} |
| `conf_int_` / `perc_int_` / `nq_int_edge_width` | float | 1 | Width of the lines bounding the interval shaded region in pixels |
| `conf_int_` / `perc_int_` / `nq_int_fill_alpha` | float | 0.20 | Transparency value for the interval shaded region fill between 0-1 |
| `conf_int_` / `perc_int_` / `nq_int_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string for the interval shaded region fill |

#### FIT

| kwarg | type | default | description |
|---|---|---|---|
| `fit` | int | None | Polynomial degree for the fit |
| `fit_color` | str | #000000 | Hex color string for the fit line |
| `fit_eqn` | boolean | False | Display the fit equation on the plot |
| `fit_font_size` | float | 12 | Font size of the fit eqn and rsq value |
| `fit_padding` | int | 10 | Padding in pixels from the top of the plot to the location of the fit eqn |
| `fit_range_x` | list | None | Compute the fit only over a given range of x-values |
| `fit_range_y` | list | None | Compute the fit only over a given range of y-values |
| `fit_rsq` | boolean | False | Display the rsq of the fit on the plot |

#### REFERENCE_LINES

| kwarg | type | default | description |
|---|---|---|---|
| `ref_line` | list\|pd.Series | None | The name of one or more columns in the DataFrame or a pandas Series with the same number of rows as the x column |
| `ref_line_alpha` | str\|list | 1 | Transparency value for the reference line(s) between 0-1 (use list if more than one ref_line plotted) |
| `ref_line_color` | str\|list | #000000 | Hex color string or list of hex color strings for the reference line (use list if more than one ref_line plotted) |
| `ref_line_legend_text` | str\|list | None | Custom string label(s) to add to a legend for the reference line data (use list if more than one ref_line plotted) |
| `ref_line_style` | str\|list | '-' | Matplotlib string character for reference line style {'-'; '--'; '-.' ':'} (use list if more than one ref_line plotted) |
| `ref_line_width` | int\|list | 1 | Reference line width in pixels (use list if more than one ref_line plotted) |

#### STAT_LINES

| kwarg | type | default | description |
|---|---|---|---|
| `stat` | str | None | Calculate a statistic on a data set (any stat value supported by `pandas.groupby` is valid {'mean', 'std', etc} |
| `stat_val` | str | None | Alternate column name used as a pseudo x-axis for the stat calculation for cases in which the plotted x-column values are not perfectly aligned |
| `stat_line_xxx` | various | None | Stat-line styling is controlled by the regular `line_xxx` values |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.boxplot()` — Box (variability) plot

Box plot modeled after the "Variability Chart" in JMP which Dummy function to return convenient, multi-level group labels automatically along the x-axis.

**Signature:** `fcp.boxplot(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `y` | str | y-axis column name contining the box plot data |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_box.csv')
fcp.boxplot(df, y='Value', groups=['Batch', 'Sample'])
```

<details>
<summary>Plot-specific kwarg reference (84 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `groups` | str\|list | None | Grouping columns for the box plot |
| `box_on` | bool | True | Toggle box visibility |
| `box_divider` | bool | True | Toggle box divider visibility |
| `box_divider_color` | str | #bbbbbb | Hex color string for the vertical line between groups |
| `box_divider_style` | str | - | Line style for the box divider lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_divider_width` | float | 1 | Width of the divider lines in pixels |
| `box_edge_color` | str | #aaaaaa | Hex color string for the edge of the box |
| `box_edge_width` | float | 0.5 | Width of the edge of the boxes in pixels |
| `box_fill_color` | str | #ffffff | Hex color string of the bar fill |
| `box_median_color` | str | #ff7f0e | Hex color string of the median line inside each box |
| `box_range_lines` | bool | True | Toggle the horizontal lines showing the min/max of the data range |
| `box_range_lines_color` | str | #cccccc | Hex color string for the box range lines |
| `box_range_lines_style` | str | -- | Line style for the box range lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_range_lines_width` | float | 1 | Width of the range lines in pixels |
| `box_whisker` | bool | True | Toggle range lines that extend from the box Q1/Q3 edges to the data min/max |
| `box_whisker_color` | str | #cccccc | Hex color string for the box whisker lines |
| `box_whisker_style` | str | - | Line style for the box whisker lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_whisker_width` | float | 0.5 | Width of the whisker lines in pixels |
| `box_width` | float | 0.5 [if violin on, 0.15] | Set the fractional width of the boxes between 0-1 |
| `notch` | bool | False | Use a notched-style box instead of a rectangular box |

#### MARKERS

| kwarg | type | default | description |
|---|---|---|---|
| `markers` / `box_markers` | bool | True | Toggle marker visibility |
| `marker_edge_alpha` / `box_marker_edge_alpha` | float | 1 | Transparency value for the marker edge between 0-1 |
| `marker_edge_color` / `box_marker_edge_color` | str | #c34e52 | Hex color string for the marker edge |
| `marker_edge_width` / `box_marker_edge_width` | float | 1.5 | Width of the marker edge in pixels |
| `marker_fill` / `box_marker_fill` | bool |  | Toggle marker fill on/off |
| `marker_fill_alpha` / `box_marker_fill_alpha` | float |  | Transparency value for the marker edge between 0-1 |
| `marker_fill_color` / `box_marker_fill_color` | str | True | Hex color string for the marker fill |
| `jitter` / `box_marker_jitter` | bool | `True for boxplots\|`False for all other plot types | Add a random offset or jitter to the points around their x-value (useful for box plots) |
| `marker_size` / `box_marker_size` | float | 7 | Size of the markers in pixels |
| `marker_type` / `box_marker_type` | str\|list |  | Marker characters |

#### GROUPING_TEXT

| kwarg | type | default | description |
|---|---|---|---|
| `box_group_label_fill_alpha` | float | 1 | Transparency value for group label fill between 0-1 |
| `box_group_label_fill_color` | str | #ffffff | Hex color string for the group label background color |
| `box_group_label_edge_alpha` | float | 1 | Transparency value for group label line edge between 0-1 |
| `box_group_label_edge_width` | float | 1 | Width of the edge of the line around the group labels in pixels |
| `box_group_label_edge_color` | str | #ffffff | Hex color string for the group label rectangle edge |
| `box_group_label_font` | str | Sans-serif | Font name for box group label |
| `box_group_label_font_color` | str | #000000 | Hex color string for group label font |
| `box_group_label_font_size` | float | 12 | Font size for group label text in pixels |
| `box_group_label_font_style` | str | 'normal’ | Font style for the group label text {'normal', 'italic', 'oblique'} |
| `box_group_label_font_weight` | str | 'normal’ | Font weight for the group label text {'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'} |
| `box_group_title_fill_alpha` | float | 1 | Transparency value for group title fill between 0-1 |
| `box_group_title_fill_color` | str | #ffffff | Hex color string for the group title background color |
| `box_group_title_edge_alpha` | float | 1 | Transparency value for group title line edge between 0-1 |
| `box_group_title_edge_width` | float | 1 | Width of the edge of the line around the group titles in pixels |
| `box_group_title_edge_color` | str | #ffffff | Hex color string for the group title rectangle edge |
| `box_group_title_font` | str | Sans-serif | Font name for box group title |
| `box_group_title_font_color` | str | #000000 | Hex color string for group title font |
| `box_group_title_font_size` | float | 13 | Font size for group title text in pixels |
| `box_group_title_font_style` | str | 'normal’ | Font style for the group title text {'normal', 'italic', 'oblique'} |
| `box_group_title_font_weight` | str | 'normal’ | Font weight for the group title text {'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'} |

#### STAT_LINES

| kwarg | type | default | description |
|---|---|---|---|
| `box_grand_mean` | bool | False | Toggle visibility of a line showing the mean of all data on the plot |
| `box_grand_mean_color` / `grand_mean_color` | str | #555555 | Hex color string for the grand mean line |
| `box_grand_mean_style` / `grand_mean_style` | str | '-’ | Line style for the box grand mean lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_grand_mean_width` / `grand_mean_width` | float | 1 | Width of the grand mean line in pixels |
| `box_grand_median` | bool | False | Toggle visibility of a line showing the median of all data on the plot |
| `box_grand_median_color` / `grand_median_color` | str | #0000ff | Hex color string for the grand median line |
| `box_grand_median_style` / `grand_median_style` | str | '-’ | Line style for the box grand median lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_grand_median_width` / `grand_median_width` | float | 1 | Width of the grand median line in pixels |
| `box_group_mean` | bool | False | Toggle visibility of a line showing the mean of each data group on the plot |
| `box_group_mean_color` / `group_mean_color` | str | #555555 | Hex color string for the group mean line |
| `box_group_means_style` / `group_mean_style` | str | '-’ | Line style for the box group mean lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_group_means_width` / `group_mean_width` | float | 1 | Width of the group mean line in pixels |
| `box_stat_line` | str | mean | Set the statistic for the connecting line {‘mean’, ‘median’, ‘std’, ‘qXX’ [qunatile where XX is a number between 0-100]} |
| `box_stat_line_on` | bool | True | Toggle visibility of the stat line between boxes |
| `box_stat_line_color` | str | #666666 | Hex color string for the stat line |
| `box_stat_line_width` | float | 1 | Width of the stat line in pixels |

#### DIAMONDS

| kwarg | type | default | description |
|---|---|---|---|
| `box_mean_diamonds` / `mean_diamonds` | bool | False | Toggle visibility of a diamond overlay on the box showing the group mean and a confidence interval |
| `box_mean_diamonds_alpha` / `mean_diamonds_alpha` | float | 1 | Transparency value for the diamonds between 0-1 |
| `box_mean_diamonds_edge_color` / `mean_diamonds_edge_color` | str | #FF0000 | Hex color string for the edges of the diamond |
| `box_mean_diamonds_edge_style` / `mean_diamonds_edge_style` | str | '-’ | Line style for the diamonds lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `box_mean_diamonds_edge_width` / `mean_diamonds_edge_width` | float | 0.7 | Width of the diamond lines in pixels |
| `box_mean_diamonds_fill_color` / `mean_diamonds_fill_color` | str | None | Hex color string for the fill of the diamond |
| `box_mean_diamonds_width` / `mean_diamonds_width` | float | 0.8 | Set the fractional width of the diamonds between 0-1 |
| `conf_coeff` | float | 0.95 | Confidence interval from 0 to 1 |

#### VIOLINS

| kwarg | type | default | description |
|---|---|---|---|
| `box_violin` / `violin` | bool | False | Toggle visibility of violin plot showing the distribution of box plot data |
| `violin_box_color` | str | #555555 | Hex color string for the fill of an optional box overlay on the violin |
| `violin_box_on` | bool | True | Toggle visibility of a box over the violin plot |
| `violin_edge_color` | str | #aaaaaa | Hex color string for the edge of the violins |
| `violin_fill_alpha` | float | 0.5 | Transparency value for the violin plots between 0-1 |
| `violin_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string for the fill of the violins |
| `violin_markers` | bool | False | Toggle visibility of data point markers on the violin plots |
| `violin_median_color` | str | #ffffff | Hex color string for the median point in each violin |
| `violin_median_marker` | str | 'o’ | Marker type for the median point in each violin |
| `violin_median_size` | int | 2 | Size of the median point marker in each violin |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.bar()` — Bar chart (vertical/horizontal, stacked)

Bar chart.

**Signature:** `fcp.bar(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str | x-axis column name |
| `y` | str | y-axis column name |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_bar.csv')
fcp.bar(df, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25',
tick_labels_major_x_rotation=90)
```

<details>
<summary>Plot-specific kwarg reference (14 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `bar_align` / `align` | str | ‘center’ | If ‘center’ aligns center of bar to x-axis value; if ‘edge’ aligns the left edge of the bar to the x-axis value |
| `bar_color_by` / `color_by` | str\|None | 'bar' | Color each bar differently based on a grouping criterion |
| `bar_edge_color` | str | fcp.DEFAULT_COLORS | Hex color string for the edge of the bar |
| `bar_edge_width` | float | 0 | Width of the edge of the bar in pixels |
| `bar_error_bars` / `error_bars` | bool | False | Display error bars on each bar |
| `bar_error_color` / `error_color` | str | #555555 | Hex color string of the error bar lines |
| `bar_fill_alpha` | float | 0.75 | Transparency value for the bars between 0-1 |
| `bar_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string of the bar fill |
| `bar_horizontal` / `horizontal` | bool | False | Display bars horizontally |
| `bar_rolling` / `bar_rolling_mean` / `rolling` / `rolling_mean` | int | No default | Rolling mean window size [enables this curve] |
| `bar_stacked` / `stacked` | bool | False | Stack bars of a given group |
| `bar_width` | float | 0.8 | Set the fractional width of the bars between 0-1; for stacked barplots the width corresponds to the height of the bars |
| `rolling_mean_line_color` | str | fcp.DEFAULT_COLORS | Hex color string for the rolling mean line |
| `rolling_mean_line_width` | int | 2 | Width for the rolling mean line in pixels |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.contour()` — Contour plot

Contour plot module.

**Signature:** `fcp.contour(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str | x-axis column name |
| `y` | str | y-axis column name |
| `z` | str | z-axis column name |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_contour.csv')
fcp.contour(cc, x='X', y='Y', z='Value', cbar=True, cbar_size=40, xmin=-4, xmax=2, ymin=-4, ymax=2)
```

<details>
<summary>Plot-specific kwarg reference (8 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `contour_width` | float | 1 | Width of the contour lines |
| `cmap` | str | inferno | Name of a color map |
| `filled` | bool | True | Color area between contour lines |
| `levels` | int | 20 | Number of contour lines/levels to draw |
| `interp` | str | 'cubic’ | Scipy interpolate.griddata method to make Z points {‘linear’, ‘nearest’, ‘cubic’} |
| `show_points` | bool | False | Show points on top of the contour plot |

#### COLOR_BAR

| kwarg | type | default | description |
|---|---|---|---|
| `cbar` | bool | False | Toggle colorbar on/off for contour and heatmap plots |
| `size` | int | 30 | cbar width [height will match the height of the axes] |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.gantt()` — Gantt chart

Gantt chart plotting function.  This plot is built off of a horizontal    implementation of `fcp.bar`.

**Signature:** `fcp.gantt(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | list | two x-axis column names containing Datetime values - 1) the start time for each item in the Gantt chart - 2) the stop time for each item in the Gantt chart |
| `y` | str | y-axis column name |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_gantt.csv')
fcp.gantt(df, x=['Start', 'Stop'], y='Task', ax_size=[600, 400])
```

<details>
<summary>Plot-specific kwarg reference (12 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `auto_expand` | bool | True | Auto-expand the horizontal axes size to fit the Gantt bar labels |
| `gantt_bar_labels` / `bar_labels` | bool | False | Display data column labels to the right of Gantt bars |
| `gantt_color_by` / `color_by` | str\|None | 'bar' | Color each bar differently based on a grouping criterion |
| `gantt_height` / `bar_height` | float | 0.9 | Set the fractional height of the Gantt bars between 0-1 |
| `gantt_edge_color` | str | fcp.DEFAULT_COLORS | Hex color string for the edge of the Gantt bars |
| `gantt_edge_width` | float | 0 | Width of the edge of the Gantt bars in pixels |
| `gantt_fill_alpha` | int | 0.75 | Transparency value for the Gantt bars between 0-1 |
| `gantt_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string of the Gantt bar fill |
| `gantt_label_x` | str | '’ | By default, x-axis labels are disabled for this plot type |
| `gantt_tick_labels_x_rotation` / `tick_labels_x_rotation` | int | 90 | Gantt-specific version of the this kwarg to ensure rotations are not applied globably to all plots from a theme file |
| `sort` | str | 'descending’ | Sort order for the Gantt bars {‘ascending’, ‘descending’} |
| `us_holidays` | bool | True | Skip US holidays based on the Federal calendar |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.heatmap()` — Heatmap

Heatmap plot.

**Signature:** `fcp.heatmap(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str | x-axis column name |
| `y` | str | y-axis column name |
| `z` | str | z-axis column name |

**Minimal example:**
```python
# Categorical heatmap:
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_heatmap.csv')
fcp.heatmap(df, x='Category', y='Player', z='Average')
```

<details>
<summary>Plot-specific kwarg reference (10 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `cell_size` | int | 60 | Width of a heatmap cell in pixels |
| `cmap` | bool | inferno | Name of a color map to apply to the plot |
| `data_labels` | bool | False | Toggle visibility of value text labels on the heatmap cells |
| `heatmap_edge_width` | float | 0 | Width of the edges of the heat map cells |
| `heatmap_font_color` | str | #ffffff | Hex color string for the value label text |
| `heatmap_font_size` | int | 12 | Font size of the value label text |
| `heatmap_interp` / `interp` | str | 'none’ | imshow interpolation scheme [see matplotlib docs for more details] |
| `heatmap_rounding` | int | None | Number of digits to round heatmap data labels |

#### COLOR_BAR

| kwarg | type | default | description |
|---|---|---|---|
| `cbar` | bool | False | Toggle colorbar on/off for contour and heatmap plots |
| `size` | int | 30 | cbar width [height will match the height of the axes] |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.hist()` — Histogram

Histogram plot.

**Signature:** `fcp.hist(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str | x-axis column name (i.e., the "value" column from which "counts" are calculated) |

**Minimal example:**
```python
# Simple histogram:
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_box.csv')
fcp.hist(df, x='Value')
```

<details>
<summary>Plot-specific kwarg reference (15 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `bars` | bool | True unless 2D image then False | Toggle between bars or a line plot for the counts (True=bars enabled, False=use line) |
| `cdf` | bool | False | Convert the histogram into a cumulative distribution plot |
| `cfa` | str | None | Color-filter array pattern that is used to split data from a Bayer image into separate color planes |
| `hist_align` | str | mid | If "mid" aligns center of histogram bar to x-axis value; if "left" aligns the left edge of the histogram bar to the x-axis value {"left"; "mid"; "right"} |
| `hist_bins` / `bins` | int | 20 | Number of histogram bins to use; when plotting the histogram of a raw image file the number of bins is automatically adjusted to enable one bin per DN code |
| `hist_cumulative` / `cumulative` | bool | False | From matplotlib: If True then a histogram is computed where each bin gives the counts in that bin plus all bins for smaller values; if -1 direction of accumulation is reversed |
| `hist_edge_color` | str | fcp.DEFAULT_COLORS | Hex color string for the edge of the histogram bar |
| `hist_edge_width` | float | 0 | Width of the edge of the histogram bar in pixels |
| `hist_fill_alpha` | int | 0.5 | Transparency value for the histogram bars between 0-1 |
| `hist_fill_color` | str | fcp.DEFAULT_COLORS | Hex color string of the histogram bar fill |
| `hist_horizontal` / `horizontal` | bool | False | Enable a horizontal histogram plot [default is vertical] |
| `hist_kde` / `kde` | bool | False | Toggle visibility of a kernel-density estimator curve over the histogram bars |
| `hist_normalize` / `normalize` | bool | False | Sets the "density" parameter for matplotlib-based plots; from matplotlib: if True draw and return a probability density: each bin will display each bin"s raw count divided by the total number of counts and the bin width so that the area under the histogram integrates to 1; automatically enabled if kde=True |
| `hist_rwidth` | float\|None | None | From matplotlib: the relative width of the bars as a fraction of the bin width; None means auto-calculation |
| `pdf` | bool | False | Convert the histogram into a probability density function plot |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.imshow()` — Image display

Image show plotting function.

**Signature:** `fcp.imshow(df, **kwargs)`

**Minimal example:**
```python
# Basic:
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
import imageio.v3 as imageio
# Read an image from the world-wide web
url = 'https://imagesvc.meredithcorp.io/v3/mm/image?q=85&c=sc&rect=0%2C214%2C2000%2C1214&'         >>>       + 'poi=%5B920%2C546%5D&w=2000&h=1000&url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads'         >>>       + '%2Fsites%2F47%2F2020%2F10%2F07%2Fcat-in-pirate-costume-380541532-2000.jpg'
imgr = imageio.imread(url)
# Convert to grayscale
img = fcp.utilities.img_grayscale(imgr)
fcp.imshow(img, ax_size=[600, 600])
```

<details>
<summary>Plot-specific kwarg reference (5 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `cfa` | str | None | Color-filter array pattern that is used to split data from a Bayer image into separate color planes |
| `cmap` | bool | gray | Name of a color map to apply to the plot |
| `imshow_interp` / `interp` | str | 'none’ | imshow interpolation scheme [see matplotlib docs for more details] |
| `stretch` | float\|list | None | Calculate "stretch" times the standard deviation above and below the mean to set new z-limits. Can be a single value used as +/- limits or a two-value list for the lower/upper multiplier values |
| `wh_ratio` | float | 1.0 | Aspect ratio of the image (width/height) |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.nq()` — Normal quantile plot

Plot the normal quantiles of a data set.

**Signature:** `fcp.nq(df, **kwargs)`

**Minimal example:**
```python
# "Normal" distribution:
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
import numpy as np
# Make a normal distribution from noise
img = np.ones([1000, 1000]) * 2**12 / 2
img += np.random.normal(-0.025*img.mean(), 0.025*img.mean(), img.shape)
fcp.nq(img, marker_size=4, line_width=2)
```

<details>
<summary>Plot-specific kwarg reference (5 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `x` | str | None | x-axis column name (if using a 1D dataset) |

#### CALCULATION

| kwarg | type | default | description |
|---|---|---|---|
| `sigma` | float | Auto-calculated based on the dataset using "fcp.utilities.sigma" | Maximum sigma value to use for the calculation; range will be +/- this value |
| `step_inner` | float | 0.5 | Delta between sigma values outside of the tail (around sigma=0) |
| `step_tail` | float | 0.2 | Delta between sigma values in the tails (all value >= and <= to keyword "tail") |
| `tail` | float | 3 | Sigma value that represents the start of the tail of the distribution |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*


## `fcp.pie()` — Pie chart

Pie chart

**Signature:** `fcp.pie(df, **kwargs)`

**Required kwargs:**

| kwarg | type | description |
|---|---|---|
| `x` | str | x-axis column name with categorical data |
| `y` | str | y-axis column name with values |

**Minimal example:**
```python
import fivecentplots as fcp
from pathlib import Path
import pandas as pd
df = fcp.get_test_data('fake_data_bar.csv')
df.loc[df.pH < 0, 'pH'] = -df.pH
fcp.pie(df, x='Liquid', y='pH', filter='Measurement=="A" & T [C]==25')
```

<details>
<summary>Plot-specific kwarg reference (20 kwargs, click to expand)</summary>

#### BASIC

| kwarg | type | default | description |
|---|---|---|---|
| `pie_colors` / `colors` | str\|list | fcp.DEFAULT_COLORS | Wedge fill colors |
| `pie_counter_clock` / `counter_clock` | bool | False | Places wedges in a counter-clockwise fashion |
| `pie_edge_color` / `edge_color` | str | #ffffff | Hex color string for the edge of the pie wedges |
| `pie_edge_style` / `edge_style` | str | '-’ | Line style for the wedge edge lines {‘-’, ‘--’, ‘-.’, ‘:’} |
| `pie_edge_width` / `edge_width` | float | 1 | Width of the wedge edge lines in pixels |
| `pie_explode` / `explode` | list of float | None | Emphasize one or more wedges by offsetting it from the center of the pie by some amount |
| `pie_font_color` / `font_color` | str | #444444 | Font color for the wedge labels |
| `pie_font_size` / `font_size` | float | 11 | Font size for the wedge labels |
| `pie_font_weight` / `font_weight` | str | 'normal' | Font weight for the wedge labels {'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'} |
| `pie_fill_alpha` / `fill_alpha` | float | 0.85 | Transparency value for the bars between 0-1 |
| `pie_label_distance` / `label_distance` | float | 1.1 | Distance from the center of the pie to the category labels |
| `pie_percents` / `percents` | bool | False | Label each pie wedge with the percentage for that category |
| `pie_percents_distance` / `percents_distance` | float | 0.6 | Distance from center [0] to edge [pie_radius] at which percentage labels are placed |
| `pie_percents_font_color` / `percents_font_color` | str | #444444 | Font color for the percentage labels |
| `pie_percents_font_size` / `percents_font_size` | float | 11 | Font size for the percentage labels |
| `pie_percents_font_weight` / `percents_font_weight` | str | 'normal' | Font weight for the percentage labels {'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'} |
| `pie_radius` / `radius` | float | 1 | Sets the radius of the pie chart |
| `pie_rotate_labels` / `rotate_labels` | bool | False | Rotate the pie labels to align with the bisection line from center of the pie through the wedge |
| `pie_shadow` / `shadow` | bool | False | Add a shadow to give a 3D appearance to the pie chart |
| `pie_start_angle` / `start_angle` | float | 90 | The angle at which the first wedge starts with [3 o'clock = 0; 12 o'clock =90; etc] |

</details>

*Every plot type above also accepts the "Universal kwargs" below (grouping, legend, titles, gridlines, ticks, whitespace, save/show options, etc.) — those are shared across all functions and documented once, not repeated per-function.*
# Universal kwargs (apply to every `fcp.*` plot function)

These are **not** repeated in each function-specific docstring above, but they work on `plot`, `boxplot`, `bar`, `hist`, `contour`, `heatmap`, `gantt`, `pie`, `nq`, and `imshow` alike. This is the single biggest gap between what an LLM would guess from `inspect.getdoc()` alone and what actually works — these ~250 kwargs live only in the library's internal `kwargs/csv/*.csv` source files, not in any function's own docstring.

## Grouping (row/col/wrap subplots)

| kwarg | type | default | description |
|---|---|---|---|
| `col` | str | None | [1] name of DataFrame column for grouping into columns of subplots based on each unique value; or [2] `col="x"` with multiple values defined for "x" creates columns of subplots for each x-value |
| `row` | str | None | [1] name of DataFrame column for grouping into rows of subplots based on each unique value; or [2] `row="y"` with multiple values defined for "y" creates rows of subplots for each y-value |
| `wrap` | str\|list | None | [1] name or list of names of DataFrame column(s) for grouping into a grid of subplots; [2] `wrap="x"` with multiple values defined for "x" creates a grid of subplots for each x-value; or [3] `wrap="y"` with multiple values defined for "y" creates a grid of subplots for each y-value |
| `groups` | str | None | for xy plot = name of DataFrame column that can be used to separate the data into unique groups so plot lines do not circle back on themselves |
| `groups` | str\|list | None | for boxplot = name or list of names of DataFrame column(s) used to split the data into separate boxes |

## Titles

| kwarg | type | default | description |
|---|---|---|---|
| `title` | str | None | Figure title text |
| `title_font` | str | sans-serif | Font for the figure title |
| `title_font_color` | str | #000000 | Hex color string for font color of the figure title |
| `title_font_size` | str | 18 | Font size for the figure title |
| `title_font_style` | str | italic | Font style {'normal'\|'italic'\|'oblique'} for the figure title |
| `title_font_weight` | str | bold | Font weight {a numeric value in range 0-1000\|'ultralight'\|'light'\|'normal'\|'regular'\|'book'\|'medium'\|'roman'\|'semibold'\|'demibold'\|'demi'\|'bold'\|'heavy'\|'extra bold'\|'black'} for the figure title |
| `title_edge_alpha` | float | 1 | Transparency value for the title area edge between 0-1 |
| `title_edge_color` | str | #ffffff | Hex color string for the title area edge |
| `title_edge_width` | float | 1 | Width of the border edge of a title area in pixels |
| `title_fill_alpha` | float | 1 | Transparency value for the title area background fill between 0-1 |
| `title_fill_color` | str | #ffffff | Hex color string for the title area edge |

## Axis labels

| kwarg | type | default | description |
|---|---|---|---|
| `label_padding` | float | 2 | Percent padding around the label text for the background object behind the text |
| `label_q` | str | DataFrame column name | Custom text for a specific axes label [where q = x, y, x2, y2] |
| `label_q_font` | str | sans-serif | Font for a specific axes label [where q = x, y, x2, y2] |
| `label_q_font_color` | str | #000000 | Hex color string for font color of a specific axes label [where q = x, y, x2, y2] |
| `label_q_font_size` | str | 14 | Font size for a specific axes label [where q = x, y, x2, y2] |
| `label_q_font_style` | str | italic | Font style {'normal'\|'italic'\|'oblique'} for a specific axes label [where q = x, y, x2, y2] |
| `label_q_font_weight` | str | bold | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} for a specific axes label [where q = x, y, x2, y2] |
| `label_q_edge_alpha` | float | 1 | Transparency value for the label edge between 0-1 [where q = x, y, x2, y2] |
| `label_q_edge_color` | str | #ffffff | Hex color string for the label edge [where q = x, y, x2, y2] |
| `label_q_edge_width` | float | 1 | Width of the border edge of a label in pixels [where q = x, y, x2, y2] |
| `label_q_fill_alpha` | float | 1 | Transparency value for the label background fill between 0-1 [where q = x, y, x2, y2] |
| `label_q_fill_color` | str | #ffffff | Hex color string for the label edge [where q = x, y, x2, y2] |

## Row/Col/Wrap labels

| kwarg | type | default | description |
|---|---|---|---|
| `label_q_font` | str | sans-serif | Font for a specific axes label [where q = rc, col, row, wrap; rc changes all] |
| `label_q_font_color` | str | #ffffff | Hex color string for font color of a specific axes label [where q = rc, col, row, wrap; rc changes all] |
| `label_q_font_size` | str | 16 | Font size for a specific axes label [where q = rc, col, row, wrap; rc changes all] |
| `label_q_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for a specific axes label [where q = rc, col, row, wrap; rc changes all] |
| `label_q_font_weight` | str | bold | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} for a specific axes label [where q = rc, col, row, wrap; rc changes all] |
| `label_q_edge_alpha` | float | 1 | Transparency value for the label edge between 0-1 [where q = rc, col, row, wrap; rc changes all] |
| `label_q_edge_color` | str | #8c8c8c | Hex color string for the label edge [where q = rc, col, row, wrap; rc changes all] |
| `label_q_edge_width` | float | 0 | Width of the border edge of a label in pixels [where q = rc, col, row, wrap; rc changes all] |
| `label_q_fill_alpha` | float | 1 | Transparency value for the label background fill between 0-1 [where q = rc, col, row, wrap; rc changes all] |
| `label_q_fill_color` | str | #8c8c8c | Hex color string for the label edge [where q = rc, col, row, wrap; rc changes all] |
| `label_q_names` | boolean | False | Toggle including the DataFrame column names in the row or column labels [where q = rc, col, row; rc changes all] |
| `label_q_size` | str | label_wrap_font | Label background rectangle height for an col/wrap label OR width for row label,30,None title_wrap_font,str,Font for the wrap title bar text" |
| `title_wrap_font_color` | str | label_wrap_font_color | Hex color string for the wrap title bar text |
| `title_wrap_font_size` | str | 16 | Font size for the wrap title bar text |
| `title_wrap_font_style` | str | label_wrap_font_style | Font style {'normal'\|'italic'\|'oblique'} for the wrap title bar text |
| `title_wrap_font_weight` | str | label_wrap_font_weight | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} for the wrap title bar text |
| `title_wrap_edge_alpha` | float | label_rc_ | Transparency value for the wrap title bar edge between 0-1 |
| `title_wrap_edge_color` | str | #5f5f5f | Hex color string for the wrap title bar edge |
| `title_wrap_edge_width` | float | label_wrap_edge_width | Width of the wrap title bar edge in pixels |
| `title_wrap_fill_alpha` | float | label_wrap_fill_alpha | Transparency value for the wrap title bar background fill between 0-1 |
| `title_wrap_fill_color` | str | #5f5f5f | Hex color string for the wrap title bar |
| `title_wrap_size` | str | label_wrap_size | Label background rectangle height for the wrap title bar |

## Legend

| kwarg | type | default | description |
|---|---|---|---|
| `legend_on` | boolean | True [if legend enabled] | Enable/disable visibility of legend that has been created using the `legend` kwarg |
| `legend_edge_color` | str | #ffffff | Hex color string for the legend border |
| `legend_edge_width` | float | 1 | Width of the legend border in pixels |
| `legend_font_size` | float | 12 | Font size of the legend text |
| `legend_location` | int | 0 | Position of the legend {0 = `outside`; 1 = `upper right`; 2 = upper left`; 3 = `lower left`; 4 = `lower right`; 5 = `right`; 6 = `center left`; 7 = `center right`; 8 = `lower center`; 9 = `upper center`; 10 = `center`; 11 = `below`} |
| `legend_marker_alpha` | float | 1 | Transparency value for legend markers between 0-1 |
| `legend_marker_size` | float | 7 | Marker size in the legend in pixels |
| `legend_points` | int | 1 | Number of points in the legend region for each entry [to enable multiple markers as in matplotlib] |
| `legend_title` | str | None | Custom title for the legend region [default is the column name used for the legend grouping] |

## Axes styling

| kwarg | type | default | description |
|---|---|---|---|
| `ax_size` | list of int \| str | [400, 400] | Axes size [width, height]; note this is not the size of the entire figure but just the axes area; for boxplots can enter 'auto' to auto-scale the width |
| `ax_edge_alpha` | str | 1 | Transparency value for axes edge between 0-1 |
| `ax_edge_color` | str | #aaaaaa | Hex color string for the border edge of the axes region |
| `ax_edge_bottom` | boolean | True | Enable/disable the bottom axes edge (or spine) |
| `ax_edge_left` | boolean | True | Enable/disable the left axes edge (or spine) |
| `ax_edge_right` | boolean | True | Enable/disable the right axes edge (or spine) |
| `ax_edge_top` | boolean | True | Enable/disable the top axes edge (or spine) |
| `ax_edge_width` | float | 1 | Width of the axes border in pixels |
| `ax_fill_alpha` | str | 1 | Transparency value for axes fill between 0-1 |
| `ax_fill_color` | str | #eaeaea | Hex color string for the fill color of the axes region |
| `ax_scale` / `ax2_scale` | str | 'linear' | Set the scale type of the axes {'linear'; 'logx'; 'semilogx'; 'logy'; 'semilogy'; 'loglog'; 'log'; 'symlog'; 'logit'} |
| `share_x` | boolean | True | Share the x-axis range across grouped plots with multiple axes |
| `share_x2` | boolean | True | Share the secondary x-axis range across grouped plots with multiple axes |
| `share_y` | boolean | True | Share the y-axis range across grouped plots with multiple axes |
| `share_y2` | boolean | True | Share the secondary y-axis range across grouped plots with multiple axes |
| `share_col` | boolean | True | Share the x and y axis ranges of subplots in the same column when grouping |
| `share_row` | boolean | True | Share the x and y axis ranges of subplots in the same row when grouping |
| `twin_x` | boolean | False | Add a secondary y-axis by "twinning" the x-axis |
| `twin_y` | boolean | False | Add a secondary x-axis by "twinning" the y-axis |

## Gridlines

| kwarg | type | default | description |
|---|---|---|---|
| `grid_major` | boolean | True | Enable/disable major x-axis and y-axis gridlines |
| `grid_major_x` | boolean | True | Enable/disable major x-axis gridlines |
| `grid_major_x2` | boolean | True | Enable/disable secondary-axis major x-axis gridlines |
| `grid_major_y` | boolean | True | Enable/disable major y-axis gridlines |
| `grid_major_y2` | boolean | True | Enable/disable secondary-axis major y-axis gridlines |
| `grid_major_alpha` | str | 1 | Transparency value for major gridlines between 0-1 |
| `grid_major_x_alpha` | str | 1 | Transparency value for major x-axis gridlines between 0-1 |
| `grid_major_x2_alpha` | str | 1 | Transparency value for secondary-axis major x-axis gridlines between 0-1 |
| `grid_major_y_alpha` | str | 1 | Transparency value for major y-axis gridlines between 0-1 |
| `grid_major_y2_alpha` | str | 1 | Transparency value for secondary-axis major y-axis gridlines between 0-1 |
| `grid_major_color` | str | #ffffff | Hex-axis color string for x-axis and y-axis major gridlines |
| `grid_major_x_color` | str | #ffffff | Hex-axis color string for x-axis major gridlines |
| `grid_major_x2_color` | str | #ffffff | Hex-axis color string for secondary-axis x-axis major gridlines |
| `grid_major_y_color` | str | #ffffff | Hex-axis color string for y-axis major gridlines |
| `grid_major_y2_color` | str | #ffffff | Hex-axis color string for secondary-axis y-axis major gridlines |
| `grid_major_width` | float | 1.3 | Major gridline width in pixels (float ok) |
| `grid_major_x_width` | float | 1.3 | Major x-axis gridline width in pixels (float ok) |
| `grid_major_x2_width` | float | 1.3 | Major secondary x-axis gridline width in pixels (float ok) |
| `grid_major_y_width` | float | 1.3 | Major y-axis gridline width in pixels (float ok) |
| `grid_major_y2_width` | float | 1.3 | Major secondary y-axis gridline width in pixels (float ok) |
| `grid_minor` | boolean | True | Enable/disable minor x-axis and y-axis gridlines |
| `grid_minor_x` | boolean | True | Enable/disable minor x-axis gridlines |
| `grid_minor_x2` | boolean | True | Enable/disable secondary-axis minor x-axis gridlines |
| `grid_minor_y` | boolean | True | Enable/disable minor y-axis gridlines |
| `grid_minor_y2` | boolean | True | Enable/disable secondary-axis minor y-axis gridlines |
| `grid_minor_alpha` | str | 1 | Transparency value for minor gridlines between 0-1 |
| `grid_minor_x_alpha` | str | 1 | Transparency value for minor x-axis gridlines between 0-1 |
| `grid_minor_x2_alpha` | str | 1 | Transparency value for secondary-axis minor x-axis gridlines between 0-1 |
| `grid_minor_y_alpha` | str | 1 | Transparency value for minor y-axis gridlines between 0-1 |
| `grid_minor_y2_alpha` | str | 1 | Transparency value for secondary-axis minor y-axis gridlines between 0-1 |
| `grid_minor_color` | str | #ffffff | Hex-axis color string for x-axis and y-axis minor gridlines |
| `grid_minor_x_color` | str | #ffffff | Hex-axis color string for x-axis minor gridlines |
| `grid_minor_x2_color` | str | #ffffff | Hex-axis color string for secondary-axis x-axis minor gridlines |
| `grid_minor_y_color` | str | #ffffff | Hex-axis color string for y-axis minor gridlines |
| `grid_minor_y2_color` | str | #ffffff | Hex-axis color string for secondary-axis y-axis minor gridlines |
| `grid_minor_width` | float | 0.5 | Minor gridline width in pixels (float ok) |
| `grid_minor_x_width` | float | 0.5 | Minor x-axis gridline width in pixels (float ok) |
| `grid_minor_x2_width` | float | 0.5 | Minor secondary x-axis gridline width in pixels (float ok) |
| `grid_minor_y_width` | float | 0.5 | Minor y-axis gridline width in pixels (float ok) |
| `grid_minor_y2_width` | float | 0.5 | Minor secondary y-axis gridline width in pixels (float ok) |
| `tick_cleanup` | str\|boolean | shrink | Set the tick cleanup style when dealing with overlaping tick labels {False -> ignore \| "shrink" -> change the font \| "remove" -> delete one of the overlapping labels} |

## Ticks

| kwarg | type | default | description |
|---|---|---|---|
| `ticks_[major|minor]` | boolean | True | Enable/disable major or minor tick marks; if no axis label is provided, setting is applied to all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]` | boolean | True | Enable/disable major or minor tick marks for a specific axis |
| `ticks_[x|x2|y|y2|z]` | boolean | True | Enable/disable major and minor tick marks for a specific axis |
| `ticks_[major|minor]_direction` | str | 'in' | Point tick marks 'in' or 'out' from the axes area for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_direction` | str | 'in' | Point tick marks 'in' or 'out' from the axes area for a specific axis |
| `ticks_[x|x2|y|y2|z]_direction` | str | 'in' | Point major and minor tick marks 'in' or 'out' from the axes area for a specific axis |
| `ticks_[major|minor]_increment` | float | None | Specify the spacing of major or minor tick marks for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_increment` | float | None | Specify the spacing of major or minor tick marks for a specific axis |
| `ticks_[x|x2|y|y2|z]_increment` | float | None | Specify the major and minor spacing of major or minor tick marks for a specific axis |
| `ticks_[major|minor]_length` | float | 6.2 | Specify the length of the major or minor tick marks in pixels for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_length` | float | 6.2 | Specify the length of the major or minor tick marks in pixels for a specific axis |
| `ticks_[x|x2|y|y2|z]_length` | float | 6.2 | Specify the length of the major and minor tick marks in pixels for a specific axis |
| `ticks_[major|minor]_width` | float | 2.2 | Specify the width of the major or minor tick marks in pixels for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_width` | float | 2.2 | Specify the width of the major or minor tick marks in pixels for a specific axis |
| `ticks_[x|x2|y|y2|z]_width` | float | 2.2 | Specify the width of the major and minor tick marks in pixels for a specific axis |
| `ticks_[major|minor]_alpha` | str | 1 | Transparency-axis value for major or minor tick marks between 0-1 for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_alpha` | str | 1 | Transparency-axis value for major or minor tick lines between 0-1 for a specific axis |
| `ticks_[x|x2|y|y2|z]_alpha` | str | 1 | Transparency-axis value for major and minor tick lines between 0-1 for a specific axis |
| `ticks_[major|minor]_color` | str | #ffffff | Hex-axis color string for x-axis and y-axis major or minor tick marks for all axes |
| `ticks_[major|minor]_[x|x2|y|y2|z]_color` | str | #ffffff | Hex-axis color string for x-axis major or minor tick marks for a specific axis |
| `ticks_[x|x2|y|y2|z]_color` | str | #ffffff | Hex-axis color string for x-axis major and minor tick marks for a specific axis |
| `ticks_[major|minor]_width` | float | 1.3 | major or minor tickline width in pixels for all axes (float ok) |
| `ticks_[major|minor]_[x|x2|y|y2|z]_width` | float | 1.3 | major or minor x-axis tickline width in pixels for a specific axis (float ok) |
| `ticks_[x|x2|y|y2|z]_width` | float | 1.3 | Major and minor x-axis tickline width in pixels for a specific axis (float ok) |
| `auto_tick_threshold` | list[float] | [1e-6,1e6] | Threshold levels to determine if ticks should use scientific notation |

## Tick labels

| kwarg | type | default | description |
|---|---|---|---|
| `tick_labels` | boolean | True | Enable/disable all tick labels |
| `tick_labels_[major|minor]` | boolean | True | Enable/disable all major tick labels |
| `tick_labels_[major|minor]_q` | boolean | True | Enable/disable major tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_minor` | boolean | False | Enable/disable all minor tick labels |
| `tick_labels_minor_q` | boolean | False | Enable/disable minor tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_font` | str | sans-serif | Font for all tick labels |
| `tick_labels_[major|minor]_font` | str | sans-serif | Font for all major tick labels |
| `tick_labels_[major|minor]_q_font` | str | sans-serif | Font for major tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_minor_font` | str | sans-serif | Font for all minor tick labels |
| `tick_labels_minor_q_font` | str | sans-serif | Font for minor tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_font_color` | str | #000000 | Hex color string for font color of all tick labels |
| `tick_labels_[major|minor]_font_color` | str | #000000 | Hex color string for font color of all major tick labels |
| `tick_labels_[major|minor]_q_font_color` | str | #000000 | Hex color string for font color of major tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_minor_font_color` | str | #000000 | Hex color string for font color of all minor tick labels |
| `tick_labels_minor_q_font_color` | str | #000000 | Hex color string for font color of minor tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_font_size` | str | 13 | Font size for all tick labels |
| `tick_labels_[major|minor]_font_size` | str | 13 | Font size for all major tick labels |
| `tick_labels_[major|minor]_q_font_size` | str | 13 | Font size for major tick labels of a specific axes label [where q = x, y, x2, y2] |
| `tick_labels_minor_font_size` | str | 10 | Font size of a specific axes label [where q = x, y, x2, y2] |
| `tick_labels_minor_q_font_size` | str | 10 | Font size for all minor tick labels |
| `tick_labels_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for all tick labels |
| `tick_labels_[major|minor]_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for all major tick labels |
| `tick_labels_[major|minor]_q_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for major tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_minor_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for all minor tick labels |
| `tick_labels_minor_q_font_style` | str | normal | Font style {'normal'\|'italic'\|'oblique'} for minor tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_font_weight` | str | normal | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} of all tick labels |
| `tick_labels_[major|minor]_font_weight` | str | normal | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} of all major tick labels |
| `tick_labels_[major|minor]_q_font_weight` | str | normal | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} of major tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_minor_font_weight` | str | normal | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} of all minor tick labels |
| `tick_labels_minor_q_font_weight` | str | normal | Font weight {'normal'\|'bold'\|'heavy'\|'light'\|'ultrabold'\|'ultralight'} of minor tick labels of a specific axes [where q = x, y, x2, y2] |
| `tick_labels_edge_alpha` | float | 0 | Transparency value for all tick labels background edges between 0-1 |
| `tick_labels_[major|minor]_edge_alpha` | float | 0 | Transparency value for all major tick labels background edges between 0-1 |
| `tick_labels_[major|minor]_q_edge_alpha` | float | 0 | Transparency value for major tick labels background edges of a specific axes between 0-1 [where q = x, y, x2, y2] |
| `tick_labels_minor_edge_alpha` | float | 0 | Transparency value for all minor tick labels background edges between 0-1 |
| `tick_labels_minor_q_edge_alpha` | float | 0 | Transparency value for the label edge of a specific axes between 0-1 [where q = x, y, x2, y2] |
| `tick_labels_edge_color` | str | #ffffff | Hex color string for all tick label background edges |
| `tick_labels_[major|minor]_edge_color` | str | #ffffff | Hex color string for all major tick label background edges |
| `tick_labels_[major|minor]_q_edge_color` | str | #ffffff | Hex color string for major tick label background edges of a specific axis [where q = x, y, x2, y2] |
| `tick_labels_minor_edge_color` | str | #ffffff | Hex color string for all minor tick label background edges |
| `tick_labels_minor_q_edge_color` | str | #ffffff | Hex color string for minor tick label background edges of a specific axis [where q = x, y, x2, y2] |
| `tick_labels_edge_width` | float | 0 | Width of the border edge of all tick labels in pixels |
| `tick_labels_[major|minor]_edge_width` | float | 0 | Width of the border edge of all major tick labels in pixels |
| `tick_labels_[major|minor]_q_edge_width` | float | 0 | Width of the border edge of all major tick labels of a specific axis in pixels [where q = x, y, x2, y2] |
| `tick_labels_minor_edge_width` | float | 0 | Width of the border edge of all minor tick labels in pixels |
| `tick_labels_minor_q_edge_width` | float | 0 | Width of the border edge of all minor tick labels of a specific axis in pixels [where q = x, y, x2, y2] |
| `tick_labels_fill_alpha` | float | 1 | Transparency value for the background fill of all tick labels between 0-1 |
| `tick_labels_[major|minor]_fill_alpha` | float | 1 | Transparency value for the background fill of all major tick labels between 0-1 |
| `tick_labels_[major|minor]_q_fill_alpha` | float | 1 | Transparency value for the background fill of all major tick labels of a specific axis between 0-1 [where q = x, y, x2, y2] |
| `tick_labels_minor_fill_alpha` | float | 1 | Transparency value for the background fill of all minor tick labels between 0-1 |
| `tick_labels_minor_q_fill_alpha` | float | 1 | Transparency value for the background fill of all minor tick labels of a specific axis between 0-1 [where q = x, y, x2, y2] |
| `tick_labels_fill_color` | str | #ffffff | Hex color string for all tick label background edges |
| `tick_labels_[major|minor]_fill_color` | str | #ffffff | Hex color string for all major tick label background edges |
| `tick_labels_[major|minor]_q_fill_color` | str | #ffffff | Hex color string for major tick label background edges of a specific axis [where q = x, y, x2, y2] |
| `tick_labels_minor_fill_color` | str | #ffffff | Hex color string for all minor tick label background edges |
| `tick_labels_minor_q_fill_color` | str | #ffffff | Hex color string for minor tick label background edges of a specific axes [where q = x, y, x2, y2] |
| `sci_[x|x2|y|y2|z]` | bool | False | Enable/disable scientific notation for tick labels |

## Figure

| kwarg | type | default | description |
|---|---|---|---|
| `dpi` | int | 100 | Dots per square inch resolution for the figure |
| `fig_edge_alpha` | str | 1 | Transparency value for figure edge between 0-1 |
| `fig_edge_color` | str | #aaaaaa | Hex color string for the border edge of the figure region |
| `fig_edge_width` | float | 3 | Width of the figure border in pixels |
| `fig_fill_alpha` | str | 1 | Transparency value for figure fill between 0-1 |
| `fig_fill_color` | str | #eaeaea | Hex color string for the fill color of the figure region |

## Whitespace / margins

| kwarg | type | default | description |
|---|---|---|---|
| `ws_ax_box_title` | int | 10 | White space between axes edge and box plot titles |
| `ws_ax_cbar` | int | 10 | White space from right axes edge to left side of cbar |
| `ws_ax_leg` | int | 5 | White space from right edge of axes to left edge of legend [if present] |
| `ws_ax_fig` | int | 10 | White space right edge of axes to right edge of figure [if no legend present] |
| `ws_ax_label_xs` | int | 5 | Extra white space between axes and label when using separate labels |
| `ws_col` | int | 30 | White space between column subplots [ignored if tick labels or axes labels are present and wider than this] |
| `ws_fig_label` | int | 10 | White space between top of figure and x2 label [if present] |
| `ws_label_col` | int | ws_label_rc | White space from axes and col labels |
| `ws_label_rc` | int | 10 | White space between axes and row & col labels |
| `ws_label_row` | int | ws_label_rc | White space from axes to row labels |
| `ws_leg_fig` | int | 10 | White space from right of legend in position 0 and figure right edge |
| `ws_fig_ax` | int | 10 | White space from left figure edge to axes left edge |
| `ws_fig_title` | int | 10 | White space from top of figure to top of title [if present] |
| `ws_label_fig` | int | ws_fig_label | White space from bottom of x label to bottom of figure |
| `ws_label_tick` | int | 10 | White space from edge of axes label to edge of tick labels |
| `ws_row` | int | 30 | White space between row subplots [ignored if tick labels or axes labels are present and wider than this] |
| `ws_tick_minimum` | int | 10 | Minimum width for tick labels |
| `ws_ticks_ax` | int | 5 | White space from tick labels to edge of axes |
| `ws_title_ax` | int | 10 | White space bottom of title to top of axes |

## Data selection / scaling

| kwarg | type | default | description |
|---|---|---|---|
| `auto_scale` | bool | True | Auto-scale the plot ranges |
| `ax_limit_padding` | float | 0.05 | Padding in percentage added to min/max ranges based on data range (i.e., xmin = xmin - (xmax - xmin) * ax_limit_padding) |

## Save / show / engine options

| kwarg | type | default | description |
|---|---|---|---|
| `DEFAULT_COLORS` | list | None | Default color scheme used for lines and markers (from ``colors.py``); alias COLORS also can be used for brevity |
| `engine` | str | 'mpl' | Specify the plotting engine {'mpl', 'bokeh'} |
| `filename` | str | Automatic name based on conditions with extention '.png' | Name of the saved image (with or without path and/or extension) |
| `filepath` | str | current directory | Name of the directory to save images (convenient if you want to use the default naming but save in a different directory |
| `HIST` | dict | None | Shortcut of useful kwargs to format ``hist`` plots {'ax_scale': 'logy', 'markers': False, 'line_width': 2, 'preset': 'HIST'} |
| `hold` | bool | False | For interactive plotting with ``matplotlib``, keeps the previous plots enabled when creating a new plot with ``fcp``; otherwise, previous plots are closed with each new ``fcp`` plot |
| `inline` | boolean | True | Flag to display the rendered plot in the native plotting viewer or jupyter notebook (convenient to disable if doing automated batch plotting) |
| `print_filename` | boolean | False | Print the output filename, if the plot is saved |
| `return_filename` | boolean | False | Return the output filename, if the plot is saved |
| `RCCG` | list | None | Color scheme for Bayer RCCG channel data so lines and markers match CFA type |
| `RGB` | list | None | Color scheme for Bayer RGB channel data so lines and markers match color channnels |
| `RGGB` | list | None | Color scheme for Bayer RGGB channel data so lines and markers match CFA type |
| `save` | boolean | False | Save the plot to disk |
| `save_data` | boolean | False | Save the `DataFrame` subset that is created and used by a given plot |
| `save_ext` | str | depends on plotting engine {'mpl': '.png', 'bokeh': '.html'} | Set the file extension of saved plots to determine the format |
| `show` | str | False | Show the "saved" plot image file using the default image viewer of the host PC.  Setting as "True" forces the image to be saved to disk |
| `theme` | str | None | Select a theme file for the current plot only |
| `timer` | boolean | False | Debug feature to get a time log for each step in the plotting process |

## Colorbar (contour & heatmap only)

| kwarg | type | default | description |
|---|---|---|---|
| `cbar` | bool | False | Toggle colorbar on/off for contour and heatmap plots |
| `size` | int | 30 | cbar width [height will match the height of the axes] |
# Known-valid but undocumented kwarg families

These kwarg names appear in the library's own validation list (`fivecentplots/kwargs/kwargs_all.txt`, used to warn on typos) but have **no description, type, or default recorded anywhere** in the source docs. They are real and will not raise "unsupported kwarg" warnings, but their exact behavior must be inferred from naming convention or confirmed by testing / reading `engines/mpl.py` source directly:

- **gantt() — "today marker" line on Gantt charts (`gantt_today`, `gantt_today_color`, `gantt_today_style`, ...)**
  - `gantt_today`, `gantt_today_alpha`, `gantt_today_color`, `gantt_today_coordinate`, `gantt_today_edge_alpha`, `gantt_today_edge_color`, `gantt_today_edge_width`, `gantt_today_fill_alpha`, `gantt_today_fill_color`, `gantt_today_font`, `gantt_today_font_color`, `gantt_today_font_size`, `gantt_today_font_style`, `gantt_today_font_weight`, `gantt_today_padding`, `gantt_today_rotation`, `gantt_today_style`, `gantt_today_text`, `gantt_today_units`, `gantt_today_zorder`

- **gantt() — workstream grouping brackets on Gantt charts (`gantt_workstreams`, `gantt_workstreams_title`, `gantt_workstreams_label_*`, ...)**
  - `gantt_workstreams`, `gantt_workstreams_alpha`, `gantt_workstreams_brackets`, `gantt_workstreams_color`, `gantt_workstreams_edge_alpha`, `gantt_workstreams_edge_color`, `gantt_workstreams_edge_width`, `gantt_workstreams_fill_alpha`, `gantt_workstreams_fill_color`, `gantt_workstreams_font`, `gantt_workstreams_font_color`, `gantt_workstreams_font_size`, `gantt_workstreams_font_style`, `gantt_workstreams_font_weight`, `gantt_workstreams_highlight_row`, `gantt_workstreams_label_align`, `gantt_workstreams_label_edge_style`, `gantt_workstreams_label_font_size`, `gantt_workstreams_label_font_style`, `gantt_workstreams_label_font_weight`, `gantt_workstreams_label_padding`, `gantt_workstreams_label_size`, `gantt_workstreams_location`, `gantt_workstreams_order`, `gantt_workstreams_rotation`, `gantt_workstreams_style`, `gantt_workstreams_title`, `gantt_workstreams_title_align`, `gantt_workstreams_title_edge_color`, `gantt_workstreams_title_edge_style`, `gantt_workstreams_title_edge_width`, `gantt_workstreams_title_fill_alpha`, `gantt_workstreams_title_fill_color`, `gantt_workstreams_title_font_color`, `gantt_workstreams_title_font_size`, `gantt_workstreams_title_font_style`, `gantt_workstreams_title_font_weight`, `gantt_workstreams_title_padding`, `gantt_workstreams_title_rotation`, `gantt_workstreams_title_size`, `gantt_workstreams_width`, `gantt_workstreams_zorder`

- **Plotly engine only — toolbar visibility/position (`toolbar_location`, `toolbar_active_zoom`, ...)**
  - `toolbar`, `toolbar_active_zoom`, `toolbar_edge_width`, `toolbar_location`, `toolbar_rotation`, `toolbar_sticky`, `toolbar_style`, `toolbar_tools`, `toolbar_width`
