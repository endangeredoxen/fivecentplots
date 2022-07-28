from pathlib import Path
import os
import pdb
import re
import json
import shutil
db = pdb.set_trace

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

    # fix the docstring
    docstring0 = ff.index('</h1>') + 5
    try:
        docstring1 = ff.index('<div class="section" id="Setup">')
        nn = open(cur_dir / nb.replace('.html', '.ipynb'))
        nn = json.load(nn)
        try:
            cell = nn['cells'][1]['source'][0]
            ff = ff[0:docstring0] + cell + '<br>' + ff[docstring1:]
        except IndexError:
            pass
    except ValueError:
        pass

    with open(cur_dir / '_build/html' / nb, 'w') as output:
        output.write(ff)

# test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'tests', 'notebooks')
# nbs = [f for f in os.listdir(test_dir) if '.ipynb' in f]
# nbs = [f for f in nbs if 'checkpoint' not in f]
# for nb in nbs:
#     if os.path.exists(os.path.join(cur_dir, nb)):
#         os.remove(os.path.join(cur_dir, nb))
#     shutil.copyfile(os.path.join(test_dir, nb), os.path.join(cur_dir, nb))


# COPY SPECIAL IMAGES
print('Copying special images...')
images = {'_build/html/_images':
          ['_static/images/index.png']
}

for k, v in images.items():
    for f in v:
        shutil.copy(f, k)


# FIX API PAGES