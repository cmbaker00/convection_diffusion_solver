# https://www.ctcms.nist.gov/fipy/examples/diffusion/generated/examples.diffusion.circle.html

import matplotlib.pyplot as plt
cellSize = 0.05
radius = 1.

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer
from fipy.tools import numerix

mesh = Gmsh2D('''
cellSize = %(cellSize)g;
radius = %(radius)g;
Point(1) = {0, 0, 0, cellSize};
Point(2) = {-radius*2, 0, 0, cellSize};
Point(3) = {0, radius, 0, cellSize};
Point(4) = {radius, 0, 0, cellSize};
Point(5) = {0, -radius, 0, cellSize};
Ellipse(6) = {2, 1, 3};
Ellipse(7) = {3, 1, 4};
Ellipse(8) = {4, 1, 5};
Ellipse(9) = {5, 1, 2};
Line Loop(10) = {6, 7, 8, 9};
Point(12) = {.5, 0, 0, cellSize};
Point(13) = {0, .5, 0, cellSize};
Point(14) = {-.1, -.1, 0, cellSize};
Spline(15) = {12,13,14,12};
Line Loop(16) = {15};
Plane Surface(11) = {10, 16};
''' % locals())  # doctest: +GMSH



#
# phi = CellVariable(name = "solution variable",
#                    mesh = mesh,
#                    value = 0.) # doctest: +GMSH
#

# eq = TransientTerm() == DiffusionTerm(coeff=D)
X, Y = mesh.faceCenters  # doctest: +GMSH

if __name__ == "__main__":
    plt.scatter(X,Y)
    plt.show()