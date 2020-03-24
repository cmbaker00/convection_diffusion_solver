from fipy import CellVariable, Gmsh2D, TransientTerm, DiffusionTerm, Viewer, TSVViewer, ConvectionTerm
from fipy.tools import numerix
from fipy.terms.implicitSourceTerm import ImplicitSourceTerm
import numpy as np
from functools import lru_cache
import time
import csv
from scipy.interpolate import interp2d


class PDE_object:
    def __init__(self, parameter_file):
        self.parameter_file = parameter_file
        self.parameter_dictionary = self.load_parameter_file()

        self.region_coordinates = self.load_region()

        # self.domain_file = domain_file
        # self.domain_file = convection_file_x
        # self.domain_file = convection_file_y

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


if __name__ == "__main__":
    test = PDE_object('model_parameters_test.csv')
