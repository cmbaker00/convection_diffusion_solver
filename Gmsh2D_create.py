from fipy import Gmsh2D, CellVariable, Viewer
from itertools import count
import matplotlib.pyplot as plt



def create_line_loop(x_vals, y_vals, start_number=1):
    if len(x_vals) != len(y_vals):
        raise TypeError('Length of x_vals and y_vals does not match')
    output_string = 'cellSize = %(cellSize)g;\nradius = %(radius)g;\n'
    c1 = count(start_number)
    for x, y, n in zip(x_vals, y_vals, c1):
        c_string = 'Point({0}) = {1}{2}, {3}, 0, cellSize{4};\n'.format(n, '{', x, y, '}')
        output_string += c_string
    c2 = count(start_number)
    spline_string = ''
    for x, n in zip(x_vals, c2):
        spline_string += '{},'.format(n)
    spline_string += '{}'.format(start_number)
    spline_loop_number = next(c1)
    output_string += 'Spline({0}) = {1}{2}{3};\n'.format(spline_loop_number, '{', spline_string, '}')
    line_loop_number = next(c1)
    output_string += 'Line Loop({0}) = {1}{2}{3};\n'.format(line_loop_number, '{', spline_loop_number, '}')
    return output_string, line_loop_number


def create_plane_surface(line_loops, line_loops_numbers, start_number):
    output_string = ''.join(line_loops)
    output_string += 'Plane Surface({0}) = {1}{2}{3};'.format(start_number, '{',
                                                              str(line_loops_numbers)[1:-1], '}')
    return output_string


def define_surface(x_list, y_list):
    if not isinstance(x_list, list) or not isinstance(y_list, list):
        raise TypeError('x_list and y_list must be lists of number or lists of lists of numbers')
    if not isinstance(x_list[0], list):
        x_list = [x_list]
        if isinstance(y_list[0], list):
            raise TypeError('x_list and y_list must be the same type and shape')
        y_list = [y_list]
    if len(x_list) != len(y_list):
        raise TypeError('x_list and y_list must have the same length')
    index_number = 1
    line_loop_strings = []
    line_loop_numbers_array = []
    for x, y in zip(x_list, y_list):
        new_string, new_index = create_line_loop(x, y, index_number)
        line_loop_strings.append(new_string)
        line_loop_numbers_array.append(new_index)
        index_number = new_index + 1
    return create_plane_surface(line_loop_strings, line_loop_numbers_array, index_number)


if __name__ == "__main__":

    cellSize = 0.01
    radius = .001


    # t = define_surface([0, 0, 0, 1], [0, .9, 1, 0])

    xv = [
        [0, 0, 0, 0, 0, 0, .1, .15, .2, .3, 1],
        [.4, .5, .45],
        [.2, .3, .25]
          ]

    yv = [
        [0, .6, .7, .8, .9, 1, 1, 1, 1, .9, 0],
        [.1, .1, .3],
        [.1, .1, .6]
    ]

    t = define_surface(xv, yv)


    # t = define_surface([0, 0, 0, 1], [0, .9, 1, 0])
    # t = create_plane_surface(t, [6], 7)
    print(t)

    mesh = Gmsh2D('''
    {}
    '''.format(t) % locals())
    #
    X, Y = mesh.faceCenters
    plt.scatter(X,Y)
    for x, y in zip(xv,yv):
        plt.scatter(x, y)
    plt.show()