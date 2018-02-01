import sympy as sp
import numpy as np
from lbmpy.scenarios import *
from lbmpy.creationfunctions import *
from pystencils import makeSlice, showCode
from lbmpy.boundaries import *
from lbmpy.postprocessing import *
from lbmpy.lbstep import LatticeBoltzmannStep

try:
    from IPython import get_ipython
    ipython = get_ipython()

    from pystencils.jupytersetup import *
    from lbmpy.parameterization import ScalingWidget
    import lbmpy.plot2d as plt
except ImportError:
    pass
