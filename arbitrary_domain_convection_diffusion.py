from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer, ConvectionTerm
from fipy.tools import numerix
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
import numpy as np
from functools import lru_cache
import time
import csv
import itertools
from scipy.interpolate import interp2d


class PDE_object:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.parameter_dictionary = self.load_parameter_file()

        self.region_coordinates = self.load_region()
        self.convection_data = self.load_convection()

        self.create_mesh()

    def load_parameter_file(self):
        attribute_dictionary = {}
        with open(self.parameter_file) as csvfile:
            try:
                while True:
                    param_reader = csv.reader(csvfile)
                    current_row = next(param_reader)
                    if 'Names' in current_row[0]:
                        current_row = next(param_reader)
                        attribute_dictionary['simulation_name'] = current_row[1]
                        current_row = next(param_reader)
                        attribute_dictionary['results_path'] = current_row[1]
                        current_row = next(param_reader)
                        attribute_dictionary['data_path'] = current_row[1]

                    if 'Attributes' in current_row[0]:
                        current_row = next(param_reader)
                        simulation_time = current_row[1]
                        if abs(float(simulation_time) - int(simulation_time)) > .01:
                            raise ValueError("Simulation time must be an integer")
                        attribute_dictionary['simulation_time'] = int(current_row[1])
                        current_row = next(param_reader)
                        try:
                            attribute_dictionary['save_frequency'] = float(current_row[1])
                        except ValueError:
                            raise ValueError("Save frequency must be a number")

                    if 'File names' in current_row[0]:
                        current_row = next(param_reader)
                        attribute_dictionary['region_file_name'] = current_row[1]
                        current_row = next(param_reader)
                        attribute_dictionary['convection_x_file_name'] = current_row[1]
                        current_row = next(param_reader)
                        attribute_dictionary['convection_y_file_name'] = current_row[1]

                    if 'Parameters' in current_row[0]:
                        current_row = next(param_reader)
                        attribute_dictionary['Diffusivity'] = current_row[1]
                        current_row = next(param_reader)
                        attribute_dictionary['Decay'] = current_row[1]

                    if 'Initial condition' in current_row[0]:
                        attribute_dictionary['IC_value'] = current_row[1]
                        current_row = next(param_reader)
                        ic_xmin = current_row[1]
                        current_row = next(param_reader)
                        ic_xmax = current_row[1]
                        current_row = next(param_reader)
                        ic_ymin = current_row[1]
                        current_row = next(param_reader)
                        ic_ymax = current_row[1]
                        ic_region = {'xmin': ic_xmin, 'xmax': ic_xmax, 'ymin': ic_ymin, 'ymax': ic_ymax}

                        attribute_dictionary['IC_region'] = ic_region

                    if 'Internal source ' in current_row[0]:
                        attribute_dictionary['internal_source_value'] = current_row[1]
                        current_row = next(param_reader)
                        internal_source_xmin = current_row[1]
                        current_row = next(param_reader)
                        internal_source_xmax = current_row[1]
                        current_row = next(param_reader)
                        internal_source_ymin = current_row[1]
                        current_row = next(param_reader)
                        internal_source_ymax = current_row[1]
                        internal_source_region = {'xmin': internal_source_xmin, 'xmax': internal_source_xmax,
                                                  'ymin': internal_source_ymin, 'ymax': internal_source_ymax}

                        attribute_dictionary['internal_source_region'] = internal_source_region

                    if 'Boundary source' in current_row[0]:
                        attribute_dictionary['boundary_source_value'] = current_row[1]
                        current_row = next(param_reader)
                        boundary_source_xmin = current_row[1]
                        current_row = next(param_reader)
                        boundary_source_xmax = current_row[1]
                        current_row = next(param_reader)
                        boundary_source_ymin = current_row[1]
                        current_row = next(param_reader)
                        boundary_source_ymax = current_row[1]
                        boundary_source_region = {'xmin': boundary_source_xmin, 'xmax': boundary_source_xmax,
                                                  'ymin': boundary_source_ymin, 'ymax': boundary_source_ymax}

                        attribute_dictionary['boundary_source_region'] = boundary_source_region

            except StopIteration:
                return attribute_dictionary

    def load_region(self):
        fname = self.parameter_dictionary['region_file_name']
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

    def load_convection(self):
        x_time, x_xy_vals, x_conv_vals = self.load_single_convection('x')
        y_time, y_xy_vals, y_conv_vals = self.load_single_convection('y')

        if x_xy_vals != y_xy_vals:
            raise ValueError("Input advection x and y values must be the same between files")
        if x_time != y_time:
            raise ValueError("Input times must be the same across x and y files")

        return {'convection_time': x_time, 'convection_coordinates': x_xy_vals,
                'convection_x': x_conv_vals, 'convection_y': y_conv_vals}

    def load_single_convection(self, direction):
        if direction is 'x':
            fname = self.parameter_dictionary['convection_x_file_name']
        else:
            fname = self.parameter_dictionary['convection_y_file_name']
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
        #Todo - decide where these two parameters need to be defined.
        cellSize = 0.05
        radius = 1.

        # get region data
        coordinates = self.region_coordinates

        # setup the base string
        g_str = '''cellSize = %(cellSize)g;
                    radius = %(radius)g;
                        '''
        count = itertools.count(1)  # counter to label the points
        line_loop_list = []  # list to keep track of the line loops
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

        mesh = Gmsh2D(g_str % locals())  # define mesh
        return mesh


if __name__ == "__main__":
    test = PDE_object('model_parameters_test.csv')
