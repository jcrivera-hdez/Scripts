from .version import __version__
from . import plot_style
from .base import data_module_base
from .data_complex import data_complex
from .data_IQ import data_IQ
from .functions import *
from .fit_functions import *
from .data_grid import data_grid

# Init bokeh
from bokeh.resources import INLINE
from bokeh.io import output_notebook
output_notebook(INLINE, hide_banner=True)

# Print version number
print('DataModule v'+__version__)

# Set default plot style
plot_style.set()
