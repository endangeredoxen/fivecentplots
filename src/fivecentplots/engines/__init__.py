# only matplotlib is required; all other plotting libs are optional
from . import mpl  # noqa
try:  # noqa
    from . import bokeh  # noqa
except:  # noqa
    pass  # noqa