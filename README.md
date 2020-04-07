## **Documentation**

This code was written to solve a linear convection-diffusion equation with spatio-temporally varying convection in 2D using the finite volume method.

#### **Requirements**

##### gmsh


The software requires `gmsh` (http://gmsh.info/) to run. 

_Windows_

Download `gmsh` and place the executable in the root directory (i.e. the same folder that contains `arbitrary_domain_convection_diffusion.py`)

_MacOS_

_Linux_

##### Python

This was written in python 3.7. Modules used areinclude: fipy, numpy, functools,
 time, csv, itertools, copy, scipy and matplotlib.
 
######  Note: `fipy` requires the module `futures`, which needs to be installed via pip before installing `fipy`.
 
 #### Using the solver
 
 The file `example.py` provides an example of how to run the code.
 
 The key input is the 'model input file', which defines all of the model parameters. See `example/model_inputs.csv` for the required layout.
 
 The other three input files are for the domain(/region), for the x convection and the y convection.
 
 The domain file is a list of x, y coordinates in order that must form a closed loop. The first set of x and y's define a boundary shape. Further sets of coordinates can be specified, with an empty row between each. These subsequent regions define 'holes' or 'islands' within the domain.
 
 The convection files define the convection at x, y coordinates through time. These do not need to match the region x and y's, and the code will interpolate between them where required. These do not need to be ordered. The x and y rows, and the time column must be identical between the 'x' convection file and the 'y' convection file.
 
 Model results are saved in a file <name>_results.csv, where <name> is the name specified in the model parameters file. 
 
 The results and the data can be saved in any directory, provided the directory is set in the model parameters file.
  