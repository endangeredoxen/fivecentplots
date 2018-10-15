### Keyword docstrings
import pandas as pd
import os
import pdb
st = pdb.set_trace
osjoin = os.path.join
cur_dir = os.path.dirname(__file__)
PATH = osjoin(cur_dir, r'doc\_static\docstrings')

# kw_markers = \
#     {'markers (bool or list of str or int)': 'if bool: enable/disable markers; if list of str: set the marker type list to new characters; if list of int: select specific markers from the default list by index\n',
#      'marker_fill (bool)': 'fill markers; default = False\n',
#      'marker_edge_color (hex color str)': 'marker edge color; ex: "#00FF00"\n',
#      'marker_fill_color (hex color str)': 'marker fill color (only if marker_fill==True); ex: "#00FF00"\n',
#      'marker_size (int)': 'marker size in pixels\n',
#      'jitter | marker_jitter (bool)': 'add a random point to the marker position\n'
#     }

kw_markers = pd.read_csv(osjoin(PATH, 'markers.csv'))