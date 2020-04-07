from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer, ConvectionTerm
from fipy.tools import numerix
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
import numpy as np
from functools import lru_cache
import time
import csv
import itertools
import copy
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import warnings


class PDEObject:
    def __init__(self, parameter_file):
        self.parameter = self.create_parameter_class()  # Creates a class to store parameters

        self.parameter.parameter_file = parameter_file  # Stores the name of the file containing parameters

        # Load in parameters, region and convection data.
        self.load_parameter_file()  # Loads parameters, stored in self.parameter
        self.region_coordinates = self.load_region()
        self.convection_data = self.load_convection()

        # Do an early check that the results directory exists

        try:
            np.savetxt(
                "{path}{name}.csv".format(
                    path=self.parameter.results_path,
                    name=self.parameter.simulation_name
                ),
                ['Test'],
                delimiter=',',
                fmt='%s')
        except FileNotFoundError:
            raise FileNotFoundError(
                "The results directory has not been created, please create: {}".format(self.parameter.results_path))

        # Create important PDE variables
        self.mesh = self.create_mesh()  # Create the mesh using gmsh
        self.baseline_convection = None  # Initialise a place to store the convection variable
        self.solution_variable = self.define_solution_variable()  # Define the Fipy solution variable
        self.pde_equation = self.define_ode(self.parameter.simulation_start_time)  # Define the Fipy PDE.

        # Create an array that defines solution save times
        self.parameter.save_times = np.arange(start=self.parameter.simulation_start_time,
                                              stop=self.parameter.simulation_duration,
                                              step=self.parameter.save_frequency)

        # Initiate an array to store the solution
        self.output_as_lists_of_lists = self.create_output_as_lists_of_lists()

    # Defines an empty class that is used to store variables
    @staticmethod
    def create_parameter_class():
        class Parameter:
            pass
        return Parameter

    # Load in parameters
    def load_parameter_file(self):
        with open(self.parameter.parameter_file) as csvfile:  # Load file
            try:
                while True:  # Loop through the file
                    param_reader = csv.reader(csvfile)

                    # Loop through the rows, with different commands, depending on the heading
                    current_row = next(param_reader)

                    # Parameters stored under the 'names' heading
                    if 'Names' in current_row[0]:
                        current_row = next(param_reader)
                        self.parameter.simulation_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.results_path = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.data_path = current_row[1]

                    # Parameters stored under the 'attributes' heading
                    if 'Attributes' in current_row[0]:
                        current_row = next(param_reader)
                        try:
                            self.parameter.simulation_start_time = float(current_row[1])
                        except ValueError:
                            raise ValueError("Simulation start time must be a number")

                        current_row = next(param_reader)
                        try:
                            self.parameter.save_frequency = float(current_row[1])
                        except ValueError:
                            raise ValueError("Save frequency must be a number")

                        current_row = next(param_reader)
                        try:
                            self.parameter.simulation_duration = float(current_row[1])
                        except ValueError:
                            raise ValueError("Simulation duration must be a number")

                        current_row = next(param_reader)
                        try:
                            self.parameter.source_end_time = float(current_row[1])
                        except ValueError:
                            raise ValueError("Source end time must be a number")

                        current_row = next(param_reader)
                        try:
                            self.parameter.cellSize = float(current_row[1])
                        except ValueError:
                            raise ValueError("Mesh sizing must be a number")

                        current_row = next(param_reader)
                        try:
                            self.parameter.time_step = float(current_row[1])
                        except ValueError:
                            raise ValueError("Time step sizing must be a number")

                        current_row = next(param_reader)
                        self.parameter.plotting = True if current_row[1] == 'TRUE' else False

                    # Parameters stored under the 'file names' heading
                    if 'File names' in current_row[0]:
                        current_row = next(param_reader)
                        self.parameter.region_file_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.convection_x_file_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.convection_y_file_name = current_row[1]

                    # Parameters stored under the 'parameters' heading
                    if 'Parameters' in current_row[0]:
                        current_row = next(param_reader)
                        self.parameter.Diffusivity = float(current_row[1])
                        current_row = next(param_reader)
                        self.parameter.Decay = current_row[1]

                    if 'Initial condition' in current_row[0]:
                        self.parameter.IC_value = float(current_row[1])
                        current_row = next(param_reader)
                        ic_xmin = float(current_row[1])
                        current_row = next(param_reader)
                        ic_xmax = float(current_row[1])
                        current_row = next(param_reader)
                        ic_ymin = float(current_row[1])
                        current_row = next(param_reader)
                        ic_ymax = float(current_row[1])
                        ic_region = {'xmin': ic_xmin, 'xmax': ic_xmax, 'ymin': ic_ymin, 'ymax': ic_ymax}

                        self.parameter.IC_region = ic_region

                    if 'Internal source ' in current_row[0]:
                        self.parameter.internal_source_value = float(current_row[1])
                        current_row = next(param_reader)
                        internal_source_xmin = float(current_row[1])
                        current_row = next(param_reader)
                        internal_source_xmax = float(current_row[1])
                        current_row = next(param_reader)
                        internal_source_ymin = float(current_row[1])
                        current_row = next(param_reader)
                        internal_source_ymax = float(current_row[1])

                        internal_source_region = self.create_parameter_class()
                        internal_source_region.xmin = internal_source_xmin
                        internal_source_region.xmax = internal_source_xmax
                        internal_source_region.ymin = internal_source_ymin
                        internal_source_region.ymax = internal_source_ymax

                        self.parameter.internal_source_region = internal_source_region

                    if 'Boundary source' in current_row[0]:
                        self.parameter.boundary_source_value = float(current_row[1])
                        current_row = next(param_reader)
                        boundary_source_xmin = float(current_row[1])
                        current_row = next(param_reader)
                        boundary_source_xmax = float(current_row[1])
                        current_row = next(param_reader)
                        boundary_source_ymin = float(current_row[1])
                        current_row = next(param_reader)
                        boundary_source_ymax = float(current_row[1])

                        boundary_source_region = self.create_parameter_class()
                        boundary_source_region.xmin = boundary_source_xmin
                        boundary_source_region.xmax = boundary_source_xmax
                        boundary_source_region.ymin = boundary_source_ymin
                        boundary_source_region.ymax = boundary_source_ymax

                        self.parameter.boundary_source_region = boundary_source_region

            except StopIteration:  # Once iteration over the rows completes:
                # Checks for problems with input data.
                # Currently, boundary source and internal source do not work. Raise exception if these are not all 0.

                # Boundary error:
                if boundary_source_xmin != 0 or boundary_source_xmax != 0 or boundary_source_ymin != 0 or \
                        boundary_source_ymax != 0:
                    raise ValueError("Boundary region must be all set to 0 as feature not yet complete")
                if self.parameter.boundary_source_value != 0:
                    raise ValueError("Boundary source must be set to 0 as feature not yet complete")

                # Internal error:
                if internal_source_xmin != 0 or internal_source_xmax != 0 or internal_source_ymin != 0 or \
                        internal_source_ymax != 0:
                    raise ValueError("Internal region must be all set to 0 as feature not yet complete")
                if self.parameter.boundary_source_value != 0:
                    raise ValueError("Internal source must be set to 0 as feature not yet complete")
                return None

    # Load in region from csv file.
    def load_region(self):
        fname = "{path}{fname}".format(path=self.parameter.data_path, fname=self.parameter.region_file_name)
        with open(fname) as csvfile:
            csv_reader = csv.reader(csvfile)
            coordinate_list = []  # List to store the list of (sub) region lists
            current_list = []
            try:
                while True:  # Loop through every row
                    c_line = next(csv_reader)  # Get next row
                    if c_line[1] is 'y':  # Checks if 'y' is in the 2nd column -> skip that row.
                        c_line = next(csv_reader)
                    if c_line[0] is '':  # If there is a gap in, meaning a break to start a new internal region
                        coordinate_list.append(current_list)  # Append the current list to the list of regions
                        current_list = []  # Start a new list
                        c_line = next(csv_reader)

                    current_list.append([float(c_line[0]), float(c_line[1])])

            except StopIteration:
                coordinate_list.append(current_list)  # Once the end of the csv is reached, add the current list
                return coordinate_list

    # A function to convert an array of strings (which are all numbers) to an array of numbers
    @staticmethod
    def string_array_to_float(input_array):
        if type(input_array[0]) is str:  # If it's a 1D array, just convert to float using map.
            return list(map(float, input_array))
        else:
            if type(input_array[0][0]) is str:  # IF it's a 2D array, loop through the sublists, using map on each.
                return [list(map(float, sublist)) for sublist in input_array]
            else:
                # If the input is something else, raise exception.
                raise Exception('No support for type {} in list to float conversion'.format(type(input_array[0][0])))

    # Function that can load in the x or y convection data
    def load_single_convection(self, direction):
        # Load appropriate file
        if direction is 'x':
            fname = "{path}{fname}".format(path=self.parameter.data_path, fname=self.parameter.convection_x_file_name)
        else:
            if direction is 'y':
                fname = "{path}{fname}".format(path=self.parameter.data_path, fname=self.parameter.convection_y_file_name)
            else:
                raise ValueError("{} is not a valid direction option".format(direction))

        # Open file
        with open(fname) as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            # The list of timesteps is the first row of the file
            time_array = next(csv_reader)[2:]
            try:
                # Initialise lists to store the xy coordinates at the convectin value at teach
                xy_vals = []
                conv_vals = []
                while True:
                    line = next(csv_reader)
                    xy_vals.append(line[:2])  # The coordinates are the first two values
                    conv_vals.append(line[2:])  # The convection values are the rest of the line, one for each time step
            except StopIteration:
                return time_array, xy_vals, conv_vals

    def load_convection(self):
        x_time, x_xy_vals, x_conv_vals = self.load_single_convection('x')
        y_time, y_xy_vals, y_conv_vals = self.load_single_convection('y')

        # All the data is loaded in as strings by default, run string_array_to_float to switch them all to floats
        x_time = self.string_array_to_float(x_time)
        y_time = self.string_array_to_float(y_time)

        # The convection values also need to be switched, so it's a list of lists where each sublist is every convection
        # value at a time point for all coordinates. (Rather than a list of all time points for a coordiante.)
        x_conv_vals = list(np.array(self.string_array_to_float(x_conv_vals)).transpose())
        y_conv_vals = list(np.array(self.string_array_to_float(y_conv_vals)).transpose())

        x_xy_vals = self.string_array_to_float(x_xy_vals)
        y_xy_vals = self.string_array_to_float(y_xy_vals)

        # The xy values also need to be transposed
        x_xy_vals = np.array(x_xy_vals).transpose()
        y_xy_vals = np.array(y_xy_vals).transpose()

        # Check that x, y and time are consistant across the two files
        try:
            if (x_xy_vals != y_xy_vals).all():
                raise ValueError("Input advection x and y values must be the same between files")
            if x_time != y_time:
                raise ValueError("Input times must be the same across x and y files")
        except ValueError:
            raise ImportError("There is a problem with the convection input files: {} and {}".format(
                self.parameter.convection_x_file_name, self.parameter.convection_y_file_name
                )
            )

        return {'convection_time': x_time, 'convection_coordinates': x_xy_vals,
                'convection_x': x_conv_vals, 'convection_y': y_conv_vals}

    def create_mesh(self): # TODO save and re-load mesh!
        print("Creating mesh")
        cellSize = self.parameter.cellSize
        radius = 0.1  # Not actually important without curved surfaces
        splines_flag = False  # Splines caused issues with sharp boundaries

        # get region data
        coordinates = self.region_coordinates

        # setup the base string
        g_str = '''cellSize = %(cellSize)g;
                        '''

        # Loop through coordinates and create the gmsh input file

        count = itertools.count(1)  # counter to label the points
        line_loop_list = []  # list to keep track of the line loops
        if splines_flag:
            for point_list in coordinates:  # loop through each of the sublists of points
                current_point_list = []  # keep track of the points in the current list

                for points in point_list:  # loop through the point coordinates
                    current_point = next(count)  # update point number
                    current_point_list.append(current_point)  # add point number to current list
                    g_str = g_str + 'Point(' + str(current_point) + ') = {' + str(points[0]) + ', ' + \
                        str(points[1]) + ', 0, cellSize};\n'  # add the point to the string

                spline_number = next(count)  # label current spline
                g_str = g_str + 'Spline(' + str(spline_number) + ') = {'  # add spline number

                # loop through the points and add them to the current spline
                for point in current_point_list:
                    g_str = g_str + str(point) + ','
                g_str = g_str + str(current_point_list[0]) + '};\n'  # add the first point again to complete the loop

                # add the line loop
                line_loop_number = next(count)
                line_loop_list.append(line_loop_number)
                g_str = g_str + 'Line Loop (' + str(line_loop_number) + ') = {' + str(spline_number) + '};\n'

            surface_number = next(count)  # define surface number
            g_str = g_str + 'Plane Surface(' + str(surface_number) + ') = {'  # add surface number to string

            # loop through the line loops and add to the string (up to the final one, which doesn't need a comma).
            for line_loop in line_loop_list[:-1]:
                g_str = g_str + str(line_loop) + ','
            g_str = g_str + str(line_loop_list[-1]) + '};\n'  # add the final line loop and close brackets
        else:
            for point_list in coordinates:  # loop through each of the sublists of points
                current_point_list = []  # keep track of the points in the current list

                for points in point_list:  # loop through the point coordinates
                    current_point = next(count)  # update point number
                    current_point_list.append(current_point)  # add point number to current list
                    g_str = g_str + 'Point(' + str(current_point) + ') = {' + str(points[0]) + ', ' + \
                        str(points[1]) + ', 0, cellSize};\n'  # add the point to the string

                current_line_list = []
                for point1, point2 in zip(current_point_list, current_point_list[1:] + [current_point_list[0]]):
                    current_line = next(count)
                    current_line_list.append(current_line)
                    g_str = g_str + 'Line(' + str(current_line) + ') = {' + str(point1) + ',' + str(point2) + '};\n'

                # add the line loop
                line_loop_number = next(count)
                line_loop_list.append(line_loop_number)
                g_str = g_str + 'Line Loop (' + str(line_loop_number) + ') = {'
                for line_number in current_line_list[:-1]:
                    g_str = g_str + str(line_number) + ','
                g_str = g_str + str(current_line_list[-1]) + '};\n'

            surface_number = next(count)  # define surface number
            g_str = g_str + 'Plane Surface(' + str(surface_number) + ') = {'  # add surface number to string

            # loop through the line loops and add to the string (up to the final one, which doesn't need a comma).
            for line_loop in line_loop_list[:-1]:
                g_str = g_str + str(line_loop) + ','
            g_str = g_str + str(line_loop_list[-1]) + '};\n'  # add the final line loop and close brackets

        mesh = Gmsh2D(g_str % locals())  # define mesh

        x = mesh.cellCenters.value[0]
        print("Mesh created with {x} points".format(x=len(x)))
        return mesh

    def define_solution_variable(self, existing_solution=None, boundary_source = None):
        if existing_solution is None:
            initial_condition_value = self.parameter.IC_value
            ic_region = self.parameter.IC_region
            ic_array = copy.deepcopy(self.mesh.cellCenters[0])

            xmesh = self.mesh.cellCenters[0]
            ymesh = self.mesh.cellCenters[1]

            ic_array[xmesh < ic_region['xmin']] = 0
            ic_array[xmesh >= ic_region['xmin']] = initial_condition_value
            ic_array[xmesh > ic_region['xmax']] = 0
            ic_array[ymesh < ic_region['ymin']] = 0
            ic_array[ymesh > ic_region['ymax']] = 0

            ic_array.mesh = self.mesh

        else:
            ic_array = existing_solution

        # Create solution variable
        phi = CellVariable(name="solution variable",
                           mesh=self.mesh,
                           value=ic_array)

        # Edit to add boundary condition if requried.
        if boundary_source is not 'no flux' or boundary_source is not None:
            x, y = self.mesh.faceCenters

            boundary_source_value = self.parameter.boundary_source_value
            boundary_source_region = self.parameter.boundary_source_region

            boundary_source_mask = (
                (x > boundary_source_region.xmin) &
                (x < boundary_source_region.xmax) &
                (y > boundary_source_region.ymin) &
                (y < boundary_source_region.ymax)
            )

            # phi.faceGrad.constrain(0*self.mesh.faceNormals, self.mesh.exteriorFaces)
            phi.faceGrad.constrain(boundary_source_value*self.mesh.faceNormals,
                                   self.mesh.exteriorFaces & boundary_source_mask)

        return phi

    # Scatter plot of mesh cell centres
    def plot_mesh(self):
        x = self.mesh.cellCenters.value[0]
        y = self.mesh.cellCenters.value[1]
        plt.scatter(x, y)
        plt.show()

    # Check whether the convection data changed between t0 and t1
    def detect_change_in_convection_data(self, t0, t1):
        time_array = self.convection_data['convection_time']
        return [t1 >= t for t in time_array] != [t0 >= t for t in time_array]

    # Get the convection data for the current time
    def get_convection_x_and_y(self, current_time):

        # Find the index corresponding to the current time - essentially round the
        # current time down to the last time data
        time_array = self.convection_data['convection_time']
        if current_time > max(time_array):
            time_index = -1
        else:
            time_index = int([current_time >= t for t in time_array].index(False) - 1)

        x_coords, y_coords = self.convection_data['convection_coordinates']

        # Get an array of the convection data at the current time
        x_convection = np.array(self.convection_data['convection_x'][time_index])
        y_convection = np.array(self.convection_data['convection_y'][time_index])

        # Create an interpolation function
        x_function_convection = interp2d(x_coords, y_coords, x_convection)
        y_function_convection = interp2d(x_coords, y_coords, y_convection)

        # Get mesh coordinates
        x_mesh, y_mesh = self.mesh.x, self.mesh.y

        # Loop through the mesh coordiantes to store the convection value for each.
        x_convection_values = [x_function_convection(x, y)[0] for x, y in zip(x_mesh, y_mesh)]
        y_convection_values = [y_function_convection(x, y)[0] for x, y in zip(x_mesh, y_mesh)]

        return x_convection_values, y_convection_values

    def define_convection_variable(self, current_time):
        x_convection, y_convection = self.get_convection_x_and_y(current_time)
        if self.baseline_convection is None:
            convection_variable = CellVariable(mesh=self.mesh, rank=1) #  Only define the variable from scratch once
        else:
            convection_variable = self.baseline_convection
        convection_variable.setValue((x_convection, y_convection))
        return convection_variable

    def define_ode(self, current_time):

        x, y = self.mesh.faceCenters

        # Internal source specificatio - currently no functional
        internal_source_value = self.parameter.internal_source_value
        internal_source_region = self.parameter.internal_source_region

        internal_source_mask = (
            (x > internal_source_region.xmin) &
            (x < internal_source_region.xmax) &
            (y > internal_source_region.ymin) &
            (y < internal_source_region.ymax)
        )

        # Get convection data
        convection = self.define_convection_variable(current_time)

        eq = TransientTerm() == - ConvectionTerm(coeff=convection) \
            + DiffusionTerm(coeff=self.parameter.Diffusivity)\
            - ImplicitSourceTerm(coeff=self.parameter.Decay)\
            # + ImplicitSourceTerm(coeff=internal_source_value*internal_source_mask)  # Internal source not working

        return eq

    # Check if the solution needs to be saved
    def detect_save_time(self, t0, t1):
        save_times = self.parameter.save_times
        return not all((save_times < t0) == (save_times < t1))

    # Initialises the save array wtih the x, y coordinates
    def create_output_as_lists_of_lists(self):
        data = [self.solution_variable.mesh.faceCenters[0], self.solution_variable.mesh.faceCenters[1]]
        return data

    # Appends the current time and state to the output to save
    def add_current_state_to_save_file(self, time):
        self.output_as_lists_of_lists.append([time, self.solution_variable.faceValue])

    def run_pde(self):
        t0 = self.parameter.simulation_start_time  # Start time for simulation
        current_time = t0  # Variable to store the current time
        t_step = self.parameter.time_step

        pde_equation = self.pde_equation
        sol_variable = self.solution_variable

        if self.parameter.plotting:
            viewer = Viewer(vars=sol_variable, datamin=0, datamax=1.)
            viewer.plotMesh()

        # source_flag = True  # currently not implemented

        while current_time < t0 + self.parameter.simulation_duration:
            ### This code was to allow boundary flux to change, but non-flux boundaries are not working
            # if source_flag and current_time > self.parameter.source_end_time:
            #     sol_variable = self.define_solution_variable(sol_variable, boundary_source='no flux')
                # sol_variable.faceGrad.constrain(0 * self.mesh.faceNormals, self.mesh.exteriorFaces)

            pde_equation.solve(var=sol_variable, dt=t_step)  # solve one time step

            # Increment time
            previous_time = current_time
            current_time += t_step


            # Check for change in convection data
            if self.detect_change_in_convection_data(previous_time, current_time):
                pde_equation = self.define_ode(current_time)

            # Check if the solution should be saved
            if self.detect_save_time(previous_time, current_time):
                self.add_current_state_to_save_file(current_time)

            if self.parameter.plotting:
                viewer.plot()
            print('Current time step: {}, finish time: {}'.format(current_time, self.parameter.simulation_duration))

            # If the final time step goes beyond the duration, do a shorter finishing step.
            if current_time > t0 + self.parameter.simulation_duration:
                t_step = current_time - (t0 + self.parameter.simulation_duration)
                pde_equation.solve(var=sol_variable, dt=t_step)  # solve one time step

        # At completion, save the final state
        self.add_current_state_to_save_file(current_time)

    def write_output_to_file(self):
        x = self.output_as_lists_of_lists[0]
        y = self.output_as_lists_of_lists[1]
        data = self.output_as_lists_of_lists[2:]

        coords = ['x', 'y']
        times = [r[0] for r in data]  # Time is the first entry in each list
        rows = [[r[1].numericValue[t] for r in data] for t in range(len(x))]
        data_output = [coords+times]
        for x_val, y_val, row in zip(x, y, rows):
            new_row = [[x_val, y_val] + row]
            data_output += new_row

        np.savetxt(
            "{path}{name_results}.csv".format(
                path=self.parameter.results_path,
                name=self.parameter.simulation_name
            ),
            data_output,
            delimiter=',',
            fmt='%s')


if __name__ == "__main__":
    test = PDEObject('model_parameters_test.csv')
    test.run_pde()
    test.write_output_to_file()