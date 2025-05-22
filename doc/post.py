from pathlib import Path
import os, sys
import pdb
import re
import json
import shutil
import fivecentplots as fcp
cur_dir = Path(__file__).resolve().parent
db = pdb.set_trace
sys.path = [str(cur_dir.parent / 'src' / 'fivecentplots')] + sys.path
try:
    from colors import DEFAULT_COLORS
except ModuleNotFoundError:
    from .colors import DEFAULT_COLORS


# MANUAL FILE CHANGES
html_files = list(Path(cur_dir / '_build/html').rglob('*.html'))
for hf in html_files:
    # read the html files
    with open(hf, 'r') as input:
        html = input.read()

    # check for previously modified files (must be fresh)
    idx = html.find('<div class="navbar-item"><ul class="navbar-icon-links"')
    if idx == -1:
        continue

    # add version number to navbar
    html = html[:idx] + f'<div class="version">v{fcp.__version__}</div>' + html[idx:]

    # write back the file
    with open(hf, 'w') as output:
        output.write(html)

# CLEAN UP IPYNB FILES
print('Cleaning ipynb files...')
# get the names of the ipynb files
cur_dir = Path(__file__).parent.absolute()
subfolders = [f for f in cur_dir.iterdir() if f.is_dir()]
ipynb = []
for sub in subfolders:
    ipynb += [sub.parent / '_build/html' / sub.stem / (Path(f).stem + '.html')
              for f in os.listdir(sub) if '.ipynb' in f and 'check' not in f]
regex = r'\[[0-9][0-9]?\]:'

for nb in ipynb:
    with open(nb, 'r') as input:
        # remove in/out numbers
        ff = input.read()

    # replace the in/out numbers
    match = re.findall(regex, ff)
    for mm in match:
        ff = ff.replace(mm, '')

    # fix the API link
    api = ff.find('See the full API')
    api_p = ff.find('<p>See the full API <')
    api_href = -1 if api_p > -1 else ff.find('>See the full API <')
    if api > -1 and api_href == -1:
        stop = ff[api:].find('>') + api + 1
        url = '<a' + ff[api:stop].split('<a')[1] + 'See the full API </a>'
        ff = ff[:api] + url + ff[ff[api:].find('/a>') + api + 3:]

    # write output
    with open(cur_dir / '_build/html' / nb, 'w') as output:
        output.write(ff)

# COPY SPECIAL IMAGES
print('Copying special images...')
images = {'_build/html/_images':
          ['_static/images/intro_xy.png',
           '_static/images/intro_box.png',
           '_static/images/intro_gantt.png',
           '_static/images/intro_imshow.png',
          ]
}

for k, v in images.items():
    for f in v:
        shutil.copy(f, k)

# FIX API PAGES
print('Adjusting api pages...')
keys = ['LINES', 'MARKERS', 'CONTROL_LIMITS', 'CONFIDENCE_INTERVALS', 'FIT',
        'REFERENCE_LINES', 'STAT_LINES', 'AXES_LABELS', 'RC_LABELS', 'CALCULATION', 'COLOR_BAR',
        'GROUPING_TEXT', 'DIAMONDS', 'VIOLINS', 'BASIC', 'REQUIRED', 'ROLLING MEAN',
        'AX_</strong><strong>[</strong><strong>H</strong><strong>|</strong><strong>V</strong><strong>]</strong><strong>LINES']  # noqa
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
    offset = 20
    hr = f'<hr style="margin-left:-{offset}px; margin-top:20px; margin-bottom:10px">'
    for key in keys:
        kk = f'<li><p><strong>{key}</strong></p></li>'
        idx = html.find(kk)
        if idx == -1:
            continue
        kk_new = f'<p style="color:{DEFAULT_COLORS[0]}; margin-left:-{offset}px"><strong>{key}</strong>: </p>'
        if key != 'REQUIRED':
            html = html.replace(kk, f'{hr}{kk_new}')
        else:
            html = html.replace(kk, f'{kk_new}')

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

    # examples: update block quote local style
    html = html.replace('<blockquote>', '<blockquote style="border-left: 0px">')

    # write back the file
    with open(api_filepath / ff, 'w') as output:
        output.write(html)

# FIX INDEX TITLE
with open(Path(__file__).parent / '_build' / 'html' / 'index.html', 'r') as input:
    html = input.read()
html = html.replace('&lt;no title&gt; &mdash; ', '')
with open(Path(__file__).parent / '_build' / 'html' / 'index.html', 'w') as output:
    output.write(html)
