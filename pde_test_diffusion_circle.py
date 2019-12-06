# https://www.ctcms.nist.gov/fipy/examples/diffusion/generated/examples.diffusion.circle.html

cellSize = 0.05
radius = 1.

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer
from fipy.tools import numerix
mesh = Gmsh2D('''
cellSize = %(cellSize)g;
radius = %(radius)g;
Point(1) = {0, 0, 0, cellSize};
Point(2) = {-radius, 0, 0, cellSize};
Point(3) = {0, radius, 0, cellSize};
Point(4) = {radius, 0, 0, cellSize};
Point(5) = {0, -radius, 0, cellSize};
Circle(6) = {2, 1, 3};
Circle(7) = {3, 1, 4};
Circle(8) = {4, 1, 5};
Circle(9) = {5, 1, 2};
Line Loop(10) = {6, 7, 8, 9};
Plane Surface(11) = {10};
''' % locals())  # doctest: +GMSH

phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = 0.) # doctest: +GMSH

viewer = None
from builtins import input
if __name__ == '__main__':
    try:
        viewer = Viewer(vars=phi, datamin=-1, datamax=1.)
        viewer.plotMesh()
        # input("Irregular circular mesh. Press <return> to proceed...") # doctest: +GMSH
    except:
        print("Unable to create a viewer for an irregular mesh (try Matplotlib2DViewer or MayaviViewer)")

    D = 1.
    eq = TransientTerm() == DiffusionTerm(coeff=D)
    X, Y = mesh.faceCenters  # doctest: +GMSH
    phi.constrain(X, mesh.exteriorFaces)  # doctest: +GMSH
    timeStepDuration = 10 * 0.9 * cellSize ** 2 / (2 * D)
    steps = 10
    from builtins import range
    for step in range(steps):
        eq.solve(var=phi, dt = timeStepDuration)  # doctest: +GMSH
        if viewer is not None:
            viewer.plot()  # doctest: +GMSH