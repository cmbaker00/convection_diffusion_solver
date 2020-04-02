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


class PDEObject:
    def __init__(self, parameter_file):
        self.parameter = self.create_parameter_class()

        self.parameter_file = parameter_file
        self.load_parameter_file()
        self.parameter.cellSize = 0.1

        self.region_coordinates = self.load_region()
        self.convection_data = self.load_convection()

        self.mesh = self.create_mesh()
        self.solution_variable = self.define_solution_variable()

        self.pde_equation = self.define_ode(self.parameter.simulation_start_time)

        # self.plot_mesh()

    def create_parameter_class(self):
        class Parameter:
            def __init__(self):
                pass
        return Parameter


    def load_parameter_file(self):
        attribute_dictionary = {}
        with open(self.parameter_file) as csvfile:
            try:
                while True:
                    param_reader = csv.reader(csvfile)
                    current_row = next(param_reader)
                    if 'Names' in current_row[0]:
                        current_row = next(param_reader)
                        self.parameter.simulation_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.results_path = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.data_path = current_row[1]

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

                    if 'File names' in current_row[0]:
                        current_row = next(param_reader)
                        self.parameter.region_file_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.convection_x_file_name = current_row[1]
                        current_row = next(param_reader)
                        self.parameter.convection_y_file_name = current_row[1]

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

            except StopIteration:
                return attribute_dictionary

    def load_region(self):
        fname = self.parameter.region_file_name
        with open(fname) as csvfile:
            csv_reader = csv.reader(csvfile)
            coordinate_list = []
            current_list = []
            try:
                while True:
                    c_line = next(csv_reader)
                    if c_line[1] is 'y':
                        c_line = next(csv_reader)
                    if c_line[0] is '':
                        coordinate_list.append(current_list)
                        current_list = []
                        c_line = next(csv_reader)

                    current_list.append([float(c_line[0]), float(c_line[1])])

            except StopIteration:
                coordinate_list.append(current_list)
                return coordinate_list

    @staticmethod
    def string_array_to_float(input_array):
        if type(input_array[0]) is str:
            return list(map(float, input_array))
        else:
            if type(input_array[0][0]) is str:
                return [list(map(float, sublist)) for sublist in input_array]
            else:
                raise Exception('No support for type {} in list to float conversion'.format(type(input_array[0][0])))

    def load_convection(self):
        x_time, x_xy_vals, x_conv_vals = self.load_single_convection('x')
        y_time, y_xy_vals, y_conv_vals = self.load_single_convection('y')

        x_time = self.string_array_to_float(x_time)
        y_time = self.string_array_to_float(y_time)

        x_conv_vals = list(np.array(self.string_array_to_float(x_conv_vals)).transpose())
        y_conv_vals = list(np.array(self.string_array_to_float(y_conv_vals)).transpose())

        x_xy_vals = self.string_array_to_float(x_xy_vals)
        y_xy_vals = self.string_array_to_float(y_xy_vals)

        x_xy_vals = np.array(x_xy_vals).transpose()
        y_xy_vals = np.array(y_xy_vals).transpose()

        if (x_xy_vals != y_xy_vals).all():
            raise ValueError("Input advection x and y values must be the same between files")
        if x_time != y_time:
            raise ValueError("Input times must be the same across x and y files")

        return {'convection_time': x_time, 'convection_coordinates': x_xy_vals,
                'convection_x': x_conv_vals, 'convection_y': y_conv_vals}

    def load_single_convection(self, direction):
        if direction is 'x':
            fname = self.parameter.convection_x_file_name
        else:
            fname = self.parameter.convection_y_file_name
        with open(fname) as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            time_array = next(csv_reader)[2:]
            try:
                xy_vals = []
                conv_vals = []
                while True:
                    line = next(csv_reader)
                    xy_vals.append(line[:2])
                    conv_vals.append(line[2:])
            except StopIteration:
                return time_array, xy_vals, conv_vals

    def create_mesh(self):
        print("Creating mesh")

        # Todo - decide where these two parameters need to be defined.
        cellSize = self.parameter.cellSize
        radius = 0.1
        splines_flag = False

        # get region data
        coordinates = self.region_coordinates

        # setup the base string
        g_str = '''cellSize = %(cellSize)g;
                    radius = %(radius)g;
                        '''

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

    def define_solution_variable(self, existing_solution=None):
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

        phi = CellVariable(name="solution variable",
                           mesh=self.mesh,
                           value=ic_array)

        x, y = self.mesh.faceCenters

        boundary_source_value = self.parameter.boundary_source_value
        boundary_source_region = self.parameter.boundary_source_region

        boundary_source_mask = (
            (x > boundary_source_region.xmin) &
            (x < boundary_source_region.xmax) &
            (y > boundary_source_region.ymin) &
            (y < boundary_source_region.ymax)
        )

        phi.faceGrad.constrain(0*self.mesh.faceNormals, self.mesh.exteriorFaces)
        phi.faceGrad.constrain(boundary_source_value*self.mesh.faceNormals,
                               self.mesh.exteriorFaces & boundary_source_mask)

        return phi

    def plot_mesh(self):
        x = self.mesh.cellCenters.value[0]
        y = self.mesh.cellCenters.value[1]
        plt.scatter(x, y)
        plt.show()

    def detect_change_in_convectin_data(self, t0, t1):
        time_array = self.convection_data['convection_time']
        return [t1 >= t for t in time_array] != [t0 >= t for t in time_array]

    def get_convection_x_and_y(self, current_time):
        time_array = self.convection_data['convection_time']
        if current_time > max(time_array):
            time_index = -1
        else:
            time_index = int([current_time >= t for t in time_array].index(False) - 1)

        x_coords, y_coords = self.convection_data['convection_coordinates']

        x_convection = np.array(self.convection_data['convection_x'][time_index])
        y_convection = np.array(self.convection_data['convection_y'][time_index])

        x_function_convection = interp2d(x_coords, y_coords, x_convection)
        y_function_convection = interp2d(x_coords, y_coords, y_convection)

        x_mesh, y_mesh = self.mesh.x, self.mesh.y

        x_convection_values = [x_function_convection(x, y)[0] for x, y in zip(x_mesh, y_mesh)]
        y_convection_values = [y_function_convection(x, y)[0] for x, y in zip(x_mesh, y_mesh)]

        return x_convection_values, y_convection_values

    def define_convection_variable(self, current_time):
        x_convection, y_convection = self.get_convection_x_and_y(current_time)
        convection_variable = CellVariable(mesh=self.mesh, rank=1)
        convection_variable.setValue((x_convection, y_convection))
        return convection_variable

    def define_ode(self, current_time):

        x, y = self.mesh.faceCenters

        internal_source_value = self.parameter.internal_source_value
        internal_source_region = self.parameter.internal_source_region

        internal_source_mask = (
            (x > internal_source_region.xmin) &
            (x < internal_source_region.xmax) &
            (y > internal_source_region.ymin) &
            (y < internal_source_region.ymax)
        )

        convection = self.define_convection_variable(current_time)

        eq = TransientTerm() == - ConvectionTerm(coeff=convection) \
            + DiffusionTerm(coeff=self.parameter.Diffusivity)\
            - ImplicitSourceTerm(coeff=self.parameter.Decay)\
            # + ImplicitSourceTerm(coeff=internal_source_value*internal_source_mask) #Todo work out why this doesn't work

        return eq

    def run_ode_test(self):
        # TODO: add support for multiple time periods
        t0 = self.parameter.simulation_start_time
        current_time = t0
        t_step = .1
        steps = 100
        pde_equation = self.pde_equation
        sol_variable = self.solution_variable

        viewer = Viewer(vars=sol_variable, datamin=-1, datamax=1.)
        viewer.plotMesh()

        source_flag = True

        while current_time < t0 + self.parameter.simulation_duration:
            if source_flag and current_time > self.parameter.source_end_time:
                sol_variable.faceGrad.constrain(0 * self.mesh.faceNormals, self.mesh.exteriorFaces)
            pde_equation.solve(var=sol_variable, dt=t_step)

            current_time += t_step
            if self.detect_change_in_convectin_data(current_time - t_step, current_time):
                pde_equation = self.define_ode(current_time)


            viewer.plot()
            print(current_time)


if __name__ == "__main__":
    test = PDEObject('model_parameters_test.csv')
    test.run_ode_test()