# only matplotlib is required; all other plotting libs are optional
from fivecentplots.engines import mpl  # noqa

# bokeh
try:  # noqa
    from fivecentplots.engines import bokeh  # noqa
except:  # noqa
    pass  # noqa

# plotly
try:  # noqa
    from fivecentplots.engines import plotly  # noqa
except:  # noqa
    pass  # noqa

# add new engines here