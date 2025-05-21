""" Keyword docstrings """
import pandas as pd
import fivecentplots.utilities as utl
import os
import sys
import pdb
import textwrap
import re
import warnings
from itertools import product
from pathlib import Path
with open(Path(__file__).parents[1] / r'version.txt', 'r') as fid:
    __version__ = fid.readlines()[0].replace('\n', '')
db = pdb.set_trace
osjoin = os.path.join
cur_dir = Path(__file__)
sys.path = [str(cur_dir.parents[1])] + sys.path
try:
    from colors import DEFAULT_COLORS
except ModuleNotFoundError:
    from .colors import DEFAULT_COLORS


def check_undefined_kwargs():
    """
    Check for missing kwargs in the docstring csv files
    """
    # Read the csv docstring files
    files = []
    csv_files = os.listdir(cur_dir.parents[1] / 'kwargs/csv')
    for ff in csv_files:
        files += [pd.read_csv(cur_dir.parents[1] / 'kwargs/csv' / ff)]
    df = pd.concat(files, ignore_index=True)
    _defined = list(df['Keyword'].values)
    defined = []
    for dd in _defined:
        if 'conf_int_|perc_int_|nq_int_' in dd:
            # Special case
            suffix = '_'.join(dd.split('_')[-2:])
            defined += [f'{x.replace(suffix, "")}{suffix}' for x in dd.split('|')]
        elif '[' in dd and ']' in dd:
            # Find the substrings enclosed in square brackets
            brackets = re.findall(r"\[(.*?)\]", dd)

            # Find the options separated by '|'
            options_list = [section.split('|') for section in brackets]

            # Generate all combinations of the options
            combinations = list(product(*options_list))

            # Replace bracket sections with their combinations
            results = []
            for combo in combinations:
                result = dd
                for i, option in enumerate(combo):
                    # Replace the i-th bracket section with the selected option
                    result = result.replace(f'[{brackets[i]}]', option, 1)
                results += [result]
            defined += results
        elif '|' in dd:
            defined += [f for f in dd.split('|')]
        else:
            defined += [dd]

    # Read the list of allowed kwargs
    with open(cur_dir.parents[0] / 'kwargs_all.txt', 'r') as input:
        kwargs = [line.rstrip('\n') for line in input.readlines()]

    # Ignore
    ignore = ['alpha', 'ax', 'box', 'bar', ]

    # Find undefined kwargs
    undefined = [f for f in kwargs if f not in defined]
    undefined = [f for f in undefined if f not in ignore]

    return undefined


def get_all_allowed_kwargs() -> list:
    with open(Path(__file__).parents[0] / 'kwargs_all.txt', 'r') as input:
        kwargs = input.readlines()
    return [f.replace('\n', '') for f in kwargs]


def get_all_allowed_kwargs_parse(path: Path, write: bool = False) -> list:
    """Brute force through all files to look for supported kwargs.

    Args:
        path: top-level directory
        write: save the kwargs_all list

    Returns:
        list of kwarg names found
    """
    # TODO separate by engine

    # f-string replacements
    axs = ['x', 'y', 'z', 'x2', 'y2']
    control_limits = ['ucl', 'lcl']
    intervals = ['perc_int', 'nq_int', 'conf_int']
    axlines = ['ax_hlines', 'ax_vlines', 'ax2_hlines', 'ax2_vlines']
    color_params = ['fill_alpha', 'fill_color', 'edge_alpha', 'edge_color', 'color']
    other = ['marker_type']
    expanded_elements = {'label': ['label_x', 'label_x2', 'label_y', 'label_y2', 'label_z'],
                         'tick_labels_major': ['tick_labels_major_x', 'tick_labels_major_x2', 'tick_labels_major_y',
                                               'tick_labels_major_y2', 'tick_labels_major_z'],
                         'tick_labels_minor': ['tick_labels_minor_x', 'tick_labels_minor_x2', 'tick_labels_minor_y',
                                               'tick_labels_minor_y2', 'tick_labels_minor_z'],
                         'ticks_major': ['ticks_major_x', 'ticks_major_x2', 'ticks_major_y',
                                         'ticks_major_y2', 'ticks_major_z'],
                         'ticks_minor': ['ticks_minor_x', 'ticks_minor_x2', 'ticks_minor_y',
                                         'ticks_minor_y2', 'ticks_minor_z'],
                         }
    exclude = ['prop', 'on', 'kwargs', 'axline', 'self', 'fcpp', 'utl.kwargs']

    # Get files
    py_files = utl.get_nested_files(path, '.py', ['.pyc', '.png'])

    # regex to find method calls in python files
    func_regex = r'(\w+)\(((?:[^()]*\([^()]*\))*[^()]*)\)'
    bracket_regex = r',(?![^\(\[]*[\]\)])'
    element_regex = r"Element\('([^']+)'"  # get all the Element class names

    kwargs_list = ['df', 'imgs'] + axs + [f'{f}min' for f in axs] + [f'{f}max' for f in axs]
    names_list = []

    for py in py_files:
        with open(py, 'r') as input:
            contents = input.read()
            funcs = re.findall(func_regex, contents)

            # Step one:  find all calls to utl.kwget
            found_kwget = [f[1] for f in funcs if 'kwget' in f[0]]
            for ff in funcs:
                if 'kwget' not in ff[1]:
                    continue
                elif len(ff[1].split('kwget')) > 2:
                    for f in ff[1].split('kwget'):
                        if f.startswith('(kwargs'):
                            found_kwget += [f.replace(')', '').replace('(', '')]
                else:
                    found_kwget += [ff[1].replace('kwget(', '').replace(')', '')]

            for fk in found_kwget:
                # skip the actual kwget function
                if fk == 'dict1: dict, dict2: dict, vals: [str, list], default: [list, dict]':
                    continue

                vals = re.split(bracket_regex, fk)
                if vals[0].startswith('kwargs, self.fcpp') and len(vals[0].split(',')) > 2:
                    kwargs_list += [vals[0].split(',')[2].strip().replace("'", '')]
                    continue
                elif len(vals) < 4:
                    continue
                kwargs = vals[2].replace('[', '').replace(']', '').split(',')
                for kwarg in kwargs:
                    kw = kwarg.strip()

                    # Apply f-string replacements
                    if "f'{name}" in kw:
                        # Do these later
                        if kw not in names_list and kw:
                            names_list += [kw]
                        break
                    elif '{lab}' in kw:
                        for ax in axs:
                            kwargs_list += [kw.replace('{lab}', ax).replace("f'", "").replace("'", '')]
                    elif '{ax}' in kw:
                        for ax in axs:
                            kwargs_list += [kw.replace('{ax}', ax).replace("f'", "").replace("'", '')]
                    elif '{axline}' in kw:
                        for axline in axlines:
                            kwargs_list += [kw.replace('{axline}', axline).replace("f'", "").replace("'", '')]
                    elif '{interval}' in kw:
                        for interval in intervals:
                            kwargs_list += [kw.replace('{interval}', interval).replace("f'", "").replace("'", '')]
                    elif '{cl}' in kw:
                        for cl in control_limits:
                            kwargs_list += [kw.replace('{cl}', cl).replace("f'", "").replace("'", '')]
                    elif '{cp}' in kw:
                        for cp in color_params:
                            kwargs_list += [kw.replace('{cp}', cp).replace("f'", "").replace("'", '')]
                    else:
                        # no f-string, replace extra quotes
                        kwargs_list += [kw.replace("'", '')]

            # Step 2: find all calls to kwargs.get
            found_kwargs_get = [f[1] for f in funcs if f[0] == 'get']
            found_kwargs_get += [f[1].split('.get')[1] for f in funcs if '.get(' in f[1]]

            for fkg in found_kwargs_get:
                kws = fkg.split(',')
                if any(ele in kws[0] for ele in [' ', '.', '[', ']', '{', '}']) \
                        or not any(ele in kws[0] for ele in ['"', "'"]):
                    continue
                else:
                    kwargs_list += [kws[0].replace("'", '').replace('"', '').replace('(', '').replace(')', '').strip()]

            # Step 3: get all Element names and populate kwargs
            names = []
            # found_element = [f[1] for f in funcs if 'Element' in f[0] and f[1] not in ['Element', 'DF_Element']]
            found_element = re.findall(element_regex, contents)
            found_element = [f.replace('.', '_') for f in found_element]

            for k, v in expanded_elements.items():
                # some elements are created programatically and aren't directly detected
                if k in found_element:
                    index = found_element.index(k)
                    found_element[index:index + 1] = v

            for fe in found_element:
                names += [fe.split(',')[0].replace("f'", '')]
            for name in names:
                if '{ax}' in name:
                    for ax in axs:
                        name_ = name.replace('{ax}', ax).replace("'", '')
                        kwargs_list += [name_]
                        kwargs_list += [f.replace("f'{name}", name_).replace("'", '').strip() for f in names_list]
                elif name == "'label'":
                    # special case axes
                    for ax in axs:
                        name_ = name.replace("'", '')
                        kwargs_list += [f.replace("f'{name}", f'{name_}_{ax}').replace("'", '').strip()
                                        for f in names_list]
                elif name == 'axline':
                    # special case axlines
                    for axline in axlines:
                        kwargs_list += [f.replace("f'{name}", axline).replace("'", '').strip() for f in names_list]
                else:
                    kwargs_list += [name.replace("'", '').strip()]
                    kwargs_list += [f.replace("f'{name}", name).replace("'", '').strip() for f in names_list]

    # special cases ticks
    kwargs_list += [f.replace('_major', '') for f in kwargs_list if 'tick_labels_major' in f]
    kwargs_list += [f.replace('_major', '') for f in kwargs_list if 'ticks_major' in f]
    kwargs_list += [f.replace('_minor', '') for f in kwargs_list if 'ticks_minor' in f]

    # special case lines
    kwargs_list += [f.replace('lines_', 'line_') for f in kwargs_list if f.startswith('lines_')]

    # manual extras that just don't work
    kwargs_list += other

    # excludes
    kwargs_list = [f for f in kwargs_list if f not in exclude]

    # unused element defaults (although all elements support certain kwargs, some are never used)
    no_alphas = [
        'ax', 'bar_alpha', 'box_group_label', 'box_group_title', 'cbar', 'contour', 'fig', 'fills', 'fit',
        'gantt_milestone_text', 'heatmap_alpha', 'hist', 'imshow', 'interval', 'legend_alpha', 'label_rc',
        'label_wrap', 'title_wrap', 'text', 'title', 'label_col', 'label_row',
        'tick_labels_major_x', 'tick_labels_major_x2', 'tick_labels_major_y', 'tick_labels_major_y2',
        'tick_labels_major_z', 'tick_labels_minor_x', 'tick_labels_minor_x2', 'tick_labels_minor_y',
        'tick_labels_minor_y2', 'tick_labels_minor_z',
        'violin', 'gantt_alpha', 'label_x', 'label_x2', 'label_y', 'label_y2',
        'markers_alpha'
    ]
    no_fonts = [
        'ax', 'bar', 'box_grand_mean', 'box_grand_median', 'box_group_means', 'box_mean_diamonds',
        'box_divider', 'box_range_lines', 'box_stat_line', 'box_whisker', 'cbar', 'contour', 'fig', 'fills',
        'grid', 'hist', 'kde', 'imshow', 'interval', 'line', 'lines', 'markers', 'pie',
        'ticks', 'violin', 'rolling_mean',
    ]
    no_labels = [
        'ax', 'box_grand_mean', 'box_grand_median', 'box_group_means', 'box_mean_diamonds',
        'box_divider', 'box_range_lines', 'box_stat_line', 'box_whisker', 'cbar', 'contour', 'fig', 'fills',
        'fit', 'gantt_milestone_text', 'heatmap_label', 'hist', 'kde', 'imshow', 'interval', 'legend',
        'markers', 'pie', 'ref_line', 'ticks', 'rolling_mean', 'violin',
    ]
    no_edges = [
        'box_grand_mean', 'box_grand_median', 'box_group_means', 'box_divider', 'box_range_lines', 'box_stat_line',
        'box_whisker', 'cbar', 'contour', 'fills', 'fit', 'grid', 'kde', 'imshow', 'line', 'lines', 'ref_line',
        'rolling_mean',
    ]
    no_fills = [
        'box_grand_mean', 'box_grand_median', 'box_group_means', 'box_divider', 'box_range_lines', 'box_stat_line',
        'box_whisker', 'cbar', 'contour', 'fills', 'fit', 'grid', 'heatmap', 'kde', 'imshow', 'line', 'lines',
        'ref_line', 'rolling_mean',
    ]
    no_styles = [
        'ax', 'bar', 'box_mean_diamonds', 'box_group_label', 'box_group_title', 'cbar', 'contour',
        'fills', 'gantt_workstreams_title', 'hist', 'imshow', 'interval', 'legend', 'markers',
        'pie', 'label_rc', 'label_wrap', 'title_wrap', 'text', 'title', 'label_col', 'label_row',
        'ticks', 'violin', 'gantt_bar_labels',
        'label_x', 'label_x2', 'label_y', 'label_y2',
        'tick_labels_major_x', 'tick_labels_major_x2', 'tick_labels_major_y', 'tick_labels_major_y2',
        'tick_labels_major_z', 'tick_labels_minor_x', 'tick_labels_minor_x2', 'tick_labels_minor_y',
        'tick_labels_minor_y2', 'tick_labels_minor_z',

    ]
    no_widths = [
        'ax', 'box_group_label', 'box_group_title', 'cbar', 'fills', 'hist', 'imshow', 'interval',
        'legend', 'label_rc', 'label_wrap', 'title_wrap', 'text', 'title', 'label_col', 'label_row',
        'ticks', 'violin',
        'label_x', 'label_x2', 'label_y', 'label_y2', 'markers_width',
        'tick_labels_major_x', 'tick_labels_major_x2', 'tick_labels_major_y', 'tick_labels_major_y2',
        'tick_labels_major_z', 'tick_labels_minor_x', 'tick_labels_minor_x2', 'tick_labels_minor_y',
        'tick_labels_minor_y2', 'tick_labels_minor_z',
    ]
    no_std_color = [
        'ax', 'bar', 'box_group_label', 'box_group_title', 'cbar', 'contour', 'fig', 'fills',
        'gantt_labels', 'gantt_milestone_text', 'hist', 'imshow', 'interval', 'legend', 'markers', 'label_rc',
        'label_wrap', 'title_wrap', 'text', 'title', 'label_col', 'label_row', 'label_x', 'label_x2', 'label_y',
        'label_y2', 'ticks',
        'tick_labels_major_x', 'tick_labels_major_x2', 'tick_labels_major_y', 'tick_labels_major_y2',
        'tick_labels_major_z', 'tick_labels_minor_x', 'tick_labels_minor_x2', 'tick_labels_minor_y',
        'tick_labels_minor_y2', 'tick_labels_minor_z',
    ]
    no_rotations = [
        'ax', 'bar', 'box_grand_mean', 'box_grand_median', 'box_group_means', 'box_mean_diamonds',
        'box_divider', 'box_range_lines', 'box_stat_line', 'box_whisker', 'cbar', 'contour', 'fig', 'fills',
        'fit', 'grid', 'heatmap', 'hist', 'kde', 'imshow', 'interval', 'legend', 'line', 'lines',
        'markers', 'pie', 'ref_line', 'violin', 'rolling_mean',

    ]
    no_zorders = [
        'bar', 'cbar', 'contour', 'fig', 'fills',  'heatmap', 'hist', 'imshow',
        'legend', 'pie', 'rolling_mean', 'toolbars'
    ]
    new_kwargs = []
    for kw in kwargs_list:
        if any(f in kw for f in no_alphas) and '_alpha' in kw and \
                '_edge_alpha' not in kw and \
                '_fill_alpha' not in kw:
            pass
        elif any(f in kw for f in no_fonts) and '_font' in kw:
            pass
        elif any(f in kw for f in no_labels) and '_label' in kw and 'ws' not in kw:
            pass
        elif any(f in kw for f in no_edges) and '_edge_' in kw:
            pass
        elif any(f in kw for f in no_fills) and '_fill_' in kw:
            pass
        elif any(f in kw for f in no_styles) and '_style' in kw and \
                '_edge_style' not in kw and \
                '_font_style' not in kw:
            pass
        elif any(f in kw for f in no_widths) and '_edge' not in kw and '_width' in kw:
            pass
        elif any(f in kw for f in no_std_color) and '_color' in kw and \
                '_edge_color' not in kw and \
                '_fill_color' not in kw and \
                '_font_color' not in kw:
            pass
        elif any(f in kw for f in no_rotations) and '_rotation' in kw:
            pass
        elif any(f in kw for f in no_zorders) and '_zorder' in kw:
            pass
        else:
            new_kwargs += [kw]
    kwargs_list = new_kwargs

    # special case: reversible named kwargs
    names = ['title_wrap', 'label_wrap', 'label_row', 'label_row']
    for name in names:
        inverted = name.split('_')
        vals = [f.replace(name, f'{inverted[1]}_{inverted[0]}') for f in kwargs_list if name.startswith(f)]
        kwargs_list += vals

    # comment lines
    kwargs_list = [f for f in kwargs_list if not f.startswith('#')]

    # dots
    kwargs_list = [f.replace('.', '_') for f in kwargs_list]

    # delete duplicates and sort
    kwargs_list = sorted(list(set(kwargs_list)))

    if write:
        with open(Path('kwargs_all.txt'), 'w') as output:
            output.write(''.join(f"{row}\n" for row in kwargs_list))

    return kwargs_list


def make_docstrings():
    """Parse the keyword arg lists from csv files."""
    url = f'https://endangeredoxen.github.io/fivecentplots/{__version__}/'
    path = Path(cur_dir).parents[1] / 'kwargs/csv'
    files = os.listdir(path)
    kw = {}
    for ff in files:
        k = ff.split('.')[0]
        try:
            kw[k] = pd.read_csv(path / ff)
        except pd.errors.ParserError as e:
            print(e)
            print(f'fivecentplots could not read {path / ff}; import failed')
            raise SystemExit()
        kw[k] = kw[k].replace('`', '', regex=True)
        kw[k] = kw[k].sort_values(by='Keyword')
        kw[k]['Keyword'] = kw[k]['Keyword'].apply(lambda x: str(x).split(':')[-1])
        if 'Example' in kw[k].columns:
            kw[k]['Example'] = kw[k]['Example'].apply(lambda x: f'{url}{x.split("<")[-1].split(">")[0]}'
                                                      if '.html' in str(x) else x)
            kw[k]['Example'] = kw[k]['Example'].replace('None', '')
        else:
            kw[k]['Example'] = ''
        nans = kw[k][kw[k]['Keyword'] == 'nan']
        if len(nans) > 0:
            kw[k] = kw[k].dropna()
            for irow, row in nans.iterrows():
                row = row.dropna()
                idx = kw[k].index[kw[k].index < irow][-1]
                for col in row.index:
                    if col != 'Keyword':
                        kw[k].loc[idx, col] += ' | ' + row[col]

    return kw


def kw_header(val, indent=' ' * 8):
    """
    Indent header names
    """

    return f'{indent}{val}:\n'


def kw_print(kw, width=120):
    """
    Print friendly version of kw dicts
    """

    indent = ' ' * 8
    kwstr = ''

    for irow, row in kw.iterrows():
        kw = row['Keyword'].split(':')[-1]
        if 'No default' in str(row['Default']):
            default = '. No default'
        else:
            default = '. Defaults to %s' % row['Default']
        if row['Example'] == '':
            line = kw + ' (%s)' % row['Data Type'] + ': ' + \
                str(row['Description']) + default + '.'
        else:
            line = kw + ' (%s)' % row['Data Type'] + ': ' + \
                str(row['Description']) + default + '. Example: %s' % row['Example']

        kwstr += textwrap.fill(line, width, initial_indent=indent,
                               subsequent_indent=indent + '  ')
        kwstr += '\n'

    kwstr = kwstr.replace('`', '')

    return kwstr


def html_param(argx: list) -> list:
    """Convert the docstring for keyword parameters into a markdown friendly format

    Args:
        argx: list of parameters

    Returns:
        cleaned up version
    """
    # case of none
    if argx[0] == 'None':
        return [f'<div style="padding-left: 30px">{argx[0]}</div>']

    # format the arg names
    argx = [f.replace('(', '(<i><font color="#0c9c6e">').replace('): ', '</font></i>):<br>') for f in argx]
    argx = [f.replace("'", '"') for f in argx]

    # clean up web link
    argx = [f.replace('Example: None', '') for f in argx]
    argx = [f.replace('Example:', '') for f in argx]
    for iargx, aaa in enumerate(argx):
        if 'https' in aaa:
            aaa = aaa.lstrip().split('https')
            if len(aaa) == 1:
                argx[iargx] = f'<a href=https{aaa[0]}>See example</a>'
            elif aaa[0] == '':
                aaa.pop(0)
                argx[iargx] = f'<a href=https{aaa[0]}>See example</a>'
            else:
                argx[iargx - 1] += f' {aaa[0]}'
                argx[iargx] = f'<a href=https{aaa[1]}>See example</a>'

    # bold kewyords
    argx = [f'<b>{f.lstrip()}' if '):<br>' in f else f for f in argx]
    argx = [f.replace(' (<i>', '</b> (<i>') for f in argx]

    # check next line and clean
    for i in range(0, len(argx) - 1):
        if '<b>' not in argx[i + 1] and argx[i + 1] != '' and 'https' not in argx[i + 1] \
                and ':</font></i>' not in argx[i + 1]:
            argx[i] += f' {argx[i + 1].lstrip()}'
            argx[i + 1] = ''

    # add html spaces
    argx = [' '.join(f.split()) for f in argx]

    # remove empties and None
    argx = [f for f in argx if f != '']
    argx = [f.replace('. None', '') for f in argx]

    # add color patches
    for iargx, aaa in enumerate(argx):
        has_hex = re.findall(r' #(?:[0-9a-fA-F]{1,2}){3}', aaa)
        if 'Defaults' in aaa and len(has_hex) > 0:
            idx = aaa.find(has_hex[0].strip())  # this could break in some cases
            if has_hex[0].strip().lower() == '#ffffff':
                border = '; border: 1px solid #cccccc;'
            else:
                border = ''
            argx[iargx] = aaa[0:idx + 7] \
                + f' <span id="rectangle" style="height: 12px; width: 12px; background-color:{has_hex[0].strip()};' \
                + f'{border}display:inline-block"></span>' \
                + aaa[idx + 7:]

        elif 'fcp.DEFAULT_COLORS' in aaa:
            color_str = ''
            for color in DEFAULT_COLORS[0:10]:
                color_str += f'<span id="rectangle" style="height: 12px; width: 12px; background-color:{color};' \
                    + 'display:inline-block"></span>'
            idx = aaa.find('fcp.DEFAULT_COLORS')
            argx[iargx] = aaa[:idx + 18] \
                + f' {color_str}' \
                + aaa[idx + 18:]

    # add section divs
    arg0 = argx[0]
    argx = '<br>'.join(argx).split('<b>')[1:]
    for iargx, aaa in enumerate(argx):
        aaa = aaa.split(':<br>')
        aaa[1] = f'<div style="padding-left: 30px">{aaa[1]}'
        aaa[-1] += '</div>'
        aaa = [f.replace('<br>', ' ') for f in aaa]
        aaa = [f.replace('<a href', '<br><a href') for f in aaa]
        aaa = ':<br>'.join(aaa)
        argx[iargx] = f'<div style="padding-left: 30px"><b>{aaa}</div>'
    if not arg0.split(':')[0] in argx[0]:
        argx = [arg0] + argx

    # fix category labels
    idx = [i for (i, f) in enumerate(argx) if '#cc00ff' in f]
    for ii in idx:
        if ii == 0:
            argx[ii + 1] = f'<div style="padding-left: 30px">{argx[ii]}</div>{argx[ii + 1]}'
        else:
            argx[ii + 1] = argx[ii].replace('<font color="#cc00ff"', f'</div><br><font color="#cc00ff"') + argx[ii + 1]  # noqa
        argx[ii] = ''
    argx = [f for f in argx if f != '']

    return argx


def markdown(docstring: str) -> str:
    """Really **amazing** way to stupidly create markdown docstrings to
    paste into jupyter notebooks for docs.

    Args:
        docstring: the actual docstring

    Returns:
        markdown version
    """
    br = ['']

    # split docstring into a list
    doclist = docstring.split('\n')

    # strip left-indents
    doclist = [f.lstrip() for f in doclist]

    # find the list index of each parameter
    param_idx = [i for i, s in enumerate(doclist) if ')' in s and '): ' in s]

    # find the keyword type section indices
    try:
        arg = doclist.index('Args:')
    except:  # noqa
        arg = 0
    try:
        rkw = doclist.index('Required Keyword Args:')
    except:  # noqa
        rkw = 0
    okw = doclist.index('Optional Keyword Args:')

    # find subheadings
    sub = []
    for pi in param_idx:
        if '):' in doclist[pi - 1] \
                or 'Example:' in doclist[pi - 1] \
                or 'https:' in doclist[pi - 1] \
                or doclist[pi - 1] == 'Args:' \
                or doclist[pi - 1] == 'Required Keyword Args:' \
                or doclist[pi - 1] == 'Optional Keyword Args:':
            continue
        if ':' in doclist[pi - 1] and len(doclist[pi - 1].split(' ')) < 4:
            sub += [pi - 1]
    for ss in sub:
        doclist[ss] = f'<font color="#cc00ff"><i>{doclist[ss]}</font></i>'

    # rebuild in markdown-friendly code
    func = ['<p style="line-height:30px"><b><font color="#999999" '
            + f'style="font-family:Arial; font-size:24px">fivecentplots.{doclist[0]}</font></b><br>']
    if arg != 0:
        func_desc = [f'<b><i>{" ".join(doclist[1: arg - 1])}</i></b><br>']
    else:
        func_desc = [f'<b><i>{" ".join(doclist[1: okw - 1])}</i></b><br>']

    if arg != 0:
        h_arg = [f'<br><b>{doclist[arg]}</b><br>']
        argv = html_param(doclist[arg + 1: rkw])
    else:
        h_arg = []
        argv = []

    if arg != 0:
        h_rkw = [f'<br><b>{doclist[rkw]}</b><br>']
        rkwv = html_param(doclist[rkw + 1: okw])
    else:
        h_rkw = []
        rkwv = []

    h_okw = [f'<br><b>{doclist[okw]}</b><br>']
    okwv = html_param(doclist[okw + 1:])

    output = ''.join(func + func_desc + br + h_arg + argv + br
                     + h_rkw + rkwv + br + h_okw + okwv) + '</p>'

    return output


def validate_kwargs(kwargs):
    allowed = get_all_allowed_kwargs()
    invalid = []
    for k, v in kwargs.items():
        if k not in allowed:
            invalid += [k]

    if len(invalid) > 0:
        msg = 'The following kwargs are not supported: \n    - ' + "\n    - ".join(invalid)
        warnings.warn(msg, utl.CustomWarning)


if __name__ == '__main__':

    kw = make_docstrings()

    bar = kw_print(kw['bar'])

    boxplot = \
        kw_header('BASIC',  indent=' ' * 8) + \
        kw_print(kw['box']) + \
        kw_header('GROUPING_TEXT') + \
        kw_print(kw['box_label']) + \
        kw_header('STAT_LINES') + \
        kw_print(kw['box_stat']) + \
        kw_header('DIAMONDS') + \
        kw_print(kw['box_diamond']) + \
        kw_header('VIOLINS') + \
        kw_print(kw['box_violin'])

    contour = \
        kw_header('BASIC',  indent=' ' * 8) + \
        kw_print(kw['contour']) + \
        kw_header('COLOR_BAR',  indent=' ' * 8) + \
        kw_print(kw['cbar'])

    gantt = kw_print(kw['gantt'])

    heatmap = \
        kw_header('BASIC',  indent=' ' * 8) + \
        kw_print(kw['heatmap']) + \
        kw_header('COLOR_BAR',  indent=' ' * 8) + \
        kw_print(kw['cbar'])

    hist = kw_print(kw['hist'])

    imshow = kw_print(kw['imshow'])

    pie = kw_print(kw['pie'])

    nq = \
        kw_header('BASIC',  indent=' ' * 8) + \
        kw_print(kw['nq']) + \
        kw_header('CALCULATION',  indent=' ' * 8) + \
        kw_print(kw['nq_calc'])

    plot = \
        kw_header('LINES',  indent=' ' * 8) + \
        kw_print(kw['lines']) + \
        kw_header('MARKERS',  indent=' ' * 8) + \
        kw_print(kw['markers']) + \
        kw_header('AX_[H|V]LINES',  indent=' ' * 8) + \
        kw_print(kw['ax_lines']) + \
        kw_header('CONTROL_LIMITS',  indent=' ' * 8) + \
        kw_print(kw['control_limits']) + \
        kw_header('CONFIDENCE_INTERVALS',  indent=' ' * 8) + \
        kw_print(kw['conf_int']) + \
        kw_header('FIT',  indent=' ' * 8) + \
        kw_print(kw['fit']) + \
        kw_header('REFERENCE_LINES',  indent=' ' * 8) + \
        kw_print(kw['ref_line']) + \
        kw_header('STAT_LINES',  indent=' ' * 8) + \
        kw_print(kw['stat_line'])

    axes = kw_print(kw['axes'])

    cbar = kw_print(kw['cbar'])

    figure = kw_print(kw['figure'])

    gridlines = kw_print(kw['gridlines'])

    grouping = kw_print(kw['grouping'])

    labels = \
        kw_header('AXES_LABELS',  indent=' ' * 8) + \
        kw_print(kw['labels']) + \
        kw_header('RC_LABELS',  indent=' ' * 8) + \
        kw_print(kw['labels_rc'])

    legend = kw_print(kw['legend'])

    lines = kw_print(kw['lines'])

    options = kw_print(kw['options'])

    markers = kw_print(kw['markers'])

    ticks = kw_print(kw['ticks'])

    tick_labels = kw_print(kw['tick_labels'])

    titles = kw_print(kw['titles'])

    ws = kw_print(kw['ws'])

    docs = ['bar', 'boxplot', 'contour', 'gantt', 'heatmap', 'hist', 'imshow', 'pie', 'nq',
            'plot', 'axes', 'cbar', 'figure', 'gridlines', 'labels', 'legend', 'lines',
            'markers', 'ticks', 'tick_labels', 'ws', 'grouping', 'titles', 'options']
    for doc in docs:
        with open(Path('docstrings', f'{doc}.txt'), 'w') as output:
            output.write(globals()[doc])
