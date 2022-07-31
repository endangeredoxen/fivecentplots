""" Keyword docstrings """
import pandas as pd
import os
import sys
import pdb
import textwrap
import re
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

    return '%s%s:\n' % (indent, val)


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


if __name__ == '__main__':
    docs = ['bar', 'boxplot', 'contour', 'gantt', 'heatmap', 'hist', 'imshow', 'pie', 'nq',
            'plot', 'axes', 'cbar', 'figure', 'gridlines', 'labels', 'legend', 'lines',
            'markers', 'ticks', 'tick_labels', 'ws', 'grouping', 'titles', 'options']
    for doc in docs:
        with open(Path('docstrings', f'{doc}.txt'), 'w') as output:
            output.write(globals()[doc])
