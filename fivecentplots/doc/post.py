from pathlib import Path
import os, sys
import pdb
import re
import json
import shutil
cur_dir = Path(__file__).resolve().parent
db = pdb.set_trace
sys.path = [str(cur_dir.parent)] + sys.path
try:
    from colors import DEFAULT_COLORS
except ModuleNotFoundError:
    from .colors import DEFAULT_COLORS


# clean up ipynb doc files post build script
print('Cleaning ipynb files...')

# get the names of the ipynb files
cur_dir = Path(__file__).parent.absolute()
ipynb = [Path(f).stem + '.html' for f in os.listdir(cur_dir) if '.ipynb' in f and 'check' not in f]
regex = r'\[[0-9][0-9]?\]:'

for nb in ipynb:
    with open(cur_dir / '_build/html' / nb, 'r') as input:
        # remove in/out numbers
        ff = input.read()

    # replace the in/out numbers
    match = re.findall(regex, ff)
    for mm in match:
        ff = ff.replace(mm, '')

    # # fix the docstring
    # docstring0 = ff.index('</h1>') + 5
    # try:
    #     docstring1 = ff.index('<div class="section" id="Setup">')
    #     nn = open(cur_dir / nb.replace('.html', '.ipynb'))
    #     nn = json.load(nn)
    #     try:
    #         cell = nn['cells'][1]['source'][0]
    #         ff = ff[0:docstring0] + cell + '<br>' + ff[docstring1:]
    #     except IndexError:
    #         pass
    # except ValueError:
    #     pass

    with open(cur_dir / '_build/html' / nb, 'w') as output:
        output.write(ff)

# COPY SPECIAL IMAGES
print('Copying special images...')
images = {'_build/html/_images':
          ['_static/images/index.png']
}

for k, v in images.items():
    for f in v:
        shutil.copy(f, k)

# FIX API PAGES
print('Adjusting api pages...')
keys = ['LINES', 'MARKERS', 'CONTROL_LIMITS', 'CONFIDENCE_INTERVALS', 'FIT',
        'REFERENCE_LINES', 'STAT_LINES', 'AXES_LABELS', 'RC_LABELS', 'CALCULATION', 'COLOR_BAR',
        'GROUPING_TEXT', 'DIAMONDS', 'VIOLINS', 'BASIC',
        'AX_</strong><strong>[</strong><strong>H</strong><strong>|</strong><strong>V</strong><strong>]</strong><strong>LINES']
color_str = ''
for color in DEFAULT_COLORS[0:12]:
    color_str += f'<span id="rectangle" style="height: 12px; width: 12px; background-color:{color};' \
                    + 'display:inline-block"></span>'
api_filepath = Path(__file__).parent / '_build' / 'html' / 'api'
files = os.listdir(api_filepath)
for ff in files:
    # read the html files
    with open(api_filepath / ff, 'r') as input:
        html = input.read()

    # check for previously modified files (must be fresh)
    if color_str in html or '<hr style' in html or '<span id="rectangle" style="height' in html:
        continue

    # replace kwarg grouping headers
    for key in keys:
        kk = f'<li><p><strong>{key}</strong> – </p></li>'
        idx = html.find(kk)
        if idx == -1:
            continue
        html = html.replace(kk, '<hr style="margin-top:10px; margin-bottom:10px">')

    # format required
    required = [m.span() for m in re.finditer('\[REQUIRED\]', html)]
    for rr in reversed(required):
        html = html[0:rr[0]] \
               + '<span style="color:#ff0000; display:inline; font-weight:bold">[REQUIRED]</span>' \
               + html[rr[1]:]

    # find color strings
    has_hex = [m.span() for m in re.finditer(r' #(?:[0-9a-fA-F]{1,2}){3}', html)]
    for hh in reversed(has_hex):
        val = html[hh[0]: hh[1]]
        if '#ffffff' in val:
            border = '; border: 1px solid #cccccc;'
        else:
            border = '; '
        html = html[0:hh[1]] \
            + f' <span id="rectangle" style="height: 12px; width: 12px; background-color:{val.strip()}' \
            + f'{border} display:inline-block"></span>' \
            + html[hh[1]:]

    # example formatting
    examples = [m.span() for m in re.finditer('Example:', html)]
    for ex in reversed(examples):
        tag_start = html[ex[0]:].find('>https://endangeredoxen.github.io') + ex[0] + 1
        tag_stop = tag_start + html[tag_start:].find('</a>')
        html = html[0:tag_start] + 'More details' + html[tag_stop:]
        html = html[0:ex[0]] + html[ex[1] + 1:]

    # change default colors
    if 'Defaults to fcp.DEFAULT_COLORS' in html and color_str not in html:
        html = html.replace('Defaults to fcp.DEFAULT_COLORS.', f'<br>Defaults to fcp.DEFAULT_COLORS {color_str}.')

    # write back the file
    with open(api_filepath / ff, 'w') as output:
        output.write(html)

