import sympy as sp
import numpy as np
from lbmpy.scenarios import *
from lbmpy.creationfunctions import *
from pystencils import makeSlice, show_code
from lbmpy.boundaries import *
from lbmpy.postprocessing import *
from lbmpy.lbstep import LatticeBoltzmannStep
from lbmpy.geometry import *
from lbmpy.parameterization import ScalingWidget
import lbmpy.plot2d as plt
from pystencils.jupytersetup import *