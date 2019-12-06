# https://www.ctcms.nist.gov/fipy/examples/diffusion/generated/examples.diffusion.circle.html

cellSize = 0.02
radius = 1.

from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer, ConvectionTerm
from fipy.tools import numerix
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
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


phi = CellVariable(name = "solution variable",
                   mesh = mesh,
                   value = .1) # doctest: +GMSH


viewer = None
from builtins import input
if __name__ == '__main__':
    try:
        viewer = Viewer(vars=phi, datamin=-1, datamax=1.)
        viewer.plotMesh()
        # input("Irregular circular mesh. Press <return> to proceed...") # doctest: +GMSH
    except:
        print("Unable to create a viewer for an irregular mesh (try Matplotlib2DViewer or MayaviViewer)")

    D = 0.02
    eq = TransientTerm() == ConvectionTerm(coeff = (1,1)) + DiffusionTerm(coeff=D) - ImplicitSourceTerm(coeff = 1)
    # eq = TransientTerm() == DiffusionTerm(coeff=D)
    X, Y = mesh.faceCenters  # doctest: +GMSH
    mask = ((X > 0.4) & (Y > 0.65))
    print(X.shape)
    # phi.constrain(X, mesh.exteriorFaces)  # doctest: +GMSH
    # phi.faceGrad.constrain(0, mesh.exteriorFaces)  # doctest: +GMSH
    phi.faceGrad.constrain(0* mesh.faceNormals, mesh.exteriorFaces)
    phi.faceGrad.constrain(2* mesh.faceNormals, mesh.exteriorFaces & mask)
    timeStepDuration = 10 * 0.9 * cellSize ** 2 / (2 * D)
    steps = 100
    from builtins import range
    for step in range(steps):
        eq.solve(var=phi, dt = timeStepDuration)  # doctest: +GMSH
        # if viewer is not None:
        #     viewer.plot()  # doctest: +GMSH