# only matplotlib is required; all other plotting libs are optional
from . import mpl
try:
    from . import bokeh
except:
    pass