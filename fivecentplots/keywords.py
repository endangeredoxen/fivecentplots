""" Keyword docstrings """
import pandas as pd
import os
import pdb
import textwrap
with open(os.path.join(os.path.dirname(__file__), r'version.txt'), 'r') as fid:
    __version__ = fid.readlines()[0].replace('\n', '')
from distutils.version import LooseVersion
db = pdb.set_trace
osjoin = os.path.join
cur_dir = os.path.dirname(__file__)


def make_docstrings():
    """Parse the keywords in the excel doc."""
    url = f'https://endangeredoxen.github.io/fivecentplots/{__version__}/'
    if LooseVersion(pd.__version__) >= LooseVersion('1.2'):
        kw = pd.read_excel(osjoin(cur_dir, 'keywords.xlsx'), engine='openpyxl', sheet_name=None)
    else:
        kw = pd.read_excel(osjoin(cur_dir, 'keywords.xlsx'), sheet_name=None)

    for k, v in kw.items():
        kw[k] = kw[k].replace('`', '', regex=True)
        kw[k]['Keyword'] = kw[k]['Keyword'].apply(lambda x: str(x).split(':')[-1])
        if 'Example' in kw[k].columns:
            kw[k]['Example'] = kw[k]['Example'].apply(lambda x: f'{url}{x.split("<")[-1].split(">")[0]}'
                                                      if '.html' in str(x) else x)
        else:
            kw[k]['Example'] = 'None'
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


def kw_header(val, indent='       '):
    """
    Indent header names
    """

    return '%s%s\n    ' % (indent, val)


def kw_print(kw, width=100):
    """
    Print friendly version of kw dicts
    """

    indent = '        '
    kwstr = ''

    for irow, row in kw.iterrows():
        kw = row['Keyword'].split(':')[-1]
        if 'No default' in str(row['Default']):
            default = '. No default'
        else:
            default = '. Defaults to %s' % row['Default']
        line = kw + ' (%s)' % row['Data Type'] + ': ' + \
            row['Description'] + default + '. Example: %s' % row['Example']

        if irow == 0:
            kwstr += textwrap.fill(line, width, initial_indent='    ',
                                   subsequent_indent=indent + '  ')
        else:
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
    space = '&nbsp;'
    indent = space * 4
    indentr = '    '

    # style the dtype and move subsequent to next line
    argx = [f.replace('(', '(<i><font color="#0c9c6e">').replace('): ', f'</font></i>):<br>{indentr * 2}')
            for f in argx]
    # textwrap for alignment
    argx = [textwrap.fill(f, 250, initial_indent=indentr, subsequent_indent=indentr * 2)
            for f in argx]

    # clean up example
    argx = [f.replace('Example: None', '') for f in argx]
    argx = [f.replace('Example:', '') for f in argx]
    argx = [f'{indentr * 2}<a href={f.lstrip()}>See example</a>' if 'https' in f else f for f in argx]

    # # bold kewyords
    argx = [f'{indentr}<b>{f.lstrip()}' if '):<br>' in f else f for f in argx]
    argx = [f.replace(' (<i>', '</b> (<i>') for f in argx]

    # check next line and clean
    for i in range(0, len(argx) - 1):
        if f'{indentr}<b>' not in argx[i + 1] and argx[i + 1] != '' and 'https' not in argx[i + 1] \
                and ':</font></i>' not in argx[i + 1]:
            argx[i] += f' {argx[i + 1].lstrip()}'
            argx[i + 1] = ''

    # add html spaces
    argx = [f.replace(f'{indentr}', f'{indent}') for f in argx]

    # remove empties and None
    argx = [f for f in argx if f != '']
    argx = [f.replace('. None', '') for f in argx]

    #db()
    # <span id="rectangle" style="height: 12px; width: 12px; background-color:blue; display:inline-block"></span>

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
    arg = doclist.index('Args:')
    rkw = doclist.index('Required Keyword Args:')
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
            + f'style="font-family:Arial; font-size:24px">fivecentplots.{doclist[0]}</font></b>']
    func_desc = [f'<b><i>{" ".join(doclist[1: arg - 1])}</i></b>']

    h_arg = [f'<b>{doclist[arg]}</b>']
    argv = html_param(doclist[arg + 1: rkw])

    h_rkw = [f'<b>{doclist[rkw]}</b>']
    rkwv = html_param(doclist[rkw + 1: okw])

    h_okw = [f'<b>{doclist[okw]}</b>']
    okwv = html_param(doclist[okw + 1:])

    return '<br>'.join(func + func_desc + br + h_arg + argv + br
                       + h_rkw + rkwv + br + h_okw + okwv) + '</p>'
