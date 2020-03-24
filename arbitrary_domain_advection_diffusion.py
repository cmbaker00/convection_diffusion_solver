from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer, ConvectionTerm
from fipy.tools import numerix
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
import numpy as np
from functools import lru_cache
import time
from scipy.interpolate import interp2d

