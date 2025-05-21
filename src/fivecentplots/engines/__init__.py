# only matplotlib is required; all other plotting libs are optional
from . import mpl  # noqa

# bokeh
try:  # noqa
    from . import bokeh  # noqa
except:  # noqa
    pass  # noqa

# plotly
try:  # noqa
    from . import plotly  # noqa
except:  # noqa
    pass  # noqa

# add new engines here
