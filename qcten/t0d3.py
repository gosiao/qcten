import sys
import numpy as np
import scipy.linalg as la
import math
import pandas as pd
from .global_data import *
from .common import *
from pprint import pprint


class t0d3():

    """
    This class holds settings and operations
    on tensors of rank 0 (scalars) in 3D space.

    @author:       Gosia Olejniczak
    @contact:      gosia.olejniczak@gmail.com
    """


    def __init__(self, cli_options, output_options, input_data):

        # general setup
        self.input_options = cli_options # FIXME - move this out
        self.calc_options  = output_options
        self.input_data    = input_data
        self.flog          = self.input_options['flog']

        # global data structures 
        self.t0d3          = {}
        self.t0d3_points   = []

        # column names defined by the user:
        self.colnames_inp = []
        self.colnames_out = []
        # column names used in this class:
        self.colnames_qcten = {}

        self.all_fun_t0d3 = global_data.all_fun_t0d3

        # working data
        self.work_data = pd.DataFrame()

        # data columns that will be written to output(s)
        self.data_cols_to_export = {}

        # variables to be saved to the output:
        self.data_to_export= {}
        self.t0d3_cols     = []

        # grid spacing
        self.dx = 0
        self.dy = 0
        self.dz = 0

        # grid dimensions
        self.dim_x = 0
        self.dim_y = 0
        self.dim_z = 0
        self.dim_cube = 0

        self.projection_axis = {}


    def run(self, verbose=False):

        """
        main routine
        """

        # 1. verify input data
        self.verify_data_for_calcs(verbose=verbose)

        # 1. assign the data specified by a user to names used in qcten
        self.assign_t0d3_input_names(verbose=verbose)

        # 2. get the data (into pandas dataframe)
        self.get_t0d3_data_points(verbose=verbose)

        # 3. get grid information

        # 4. prepare the data for output
        self.assign_t0d3_output_names()

        # do the calculations
        if self.input_options['calc_from_tensor_0order_3d'] is not None:

            # TODO: here starts a loop over points, can be expensive!

            for arg in self.input_options['calc_from_tensor_0order_3d']:

                if (arg == 'gradient'):
                    self.get_t0d3_gradient()


    def verify_data_for_calcs(self, verbose=False):
        """
        verify whether all data needed for the type of calculations exists;
        take care of missing data, exceptions, etc.
        """

        if self.input_options['grid'] is None:
            msg = 'ERROR: check --grid in your input'
            sys.exit(msg)
        else:
            pass
            # fixme: get grid data

        if self.input_options['calc_from_tensor_0order_3d'] is None:

            msg = 'WARNING: Nothing to calculate from the vector field. ' \
                + 'Check --calc_from_tensor_0order_3d in your input'
            sys.exit(msg)

            for arg in self.input_options['calc_from_tensor_0order_3d']:
                if arg not in self.all_fun_t0d3:
                    msg = 'ERROR: requested function not in the list of available functions ' \
                        + 'Check --calc_from_tensor_0order_3d in your input. ' \
                        + 'Available functions: ', self.all_fun_t0d3
                    sys.exit(msg)


    def get_grid_info(self):

        """
        if the grid is regular, then calculate grid spacing
        """

        # TODO: add a flag for it
        self.dim_x = len(np.unique([p['x'] for p in self.t0d3_points]))
        self.dim_y = len(np.unique([p['y'] for p in self.t0d3_points]))
        self.dim_z = len(np.unique([p['z'] for p in self.t0d3_points]))
        self.dim_cube = self.dim_x * self.dim_y * self.dim_z


    def assign_t0d3_output_names(self, verbose=False):

        """
        prepare the data for output(s)
        """

        cols_available_for_outputs = self.all_fun_t0d3 + self.colnames_inp 

        for v in self.calc_options:
            if v.file_path is not None:
                data_cols = []
                for icol, col in enumerate(v.file_column_names):
                    if ':' in col:
                        old_col = col.strip().split(':')[0]
                    else:
                        old_col = col

                    if old_col in cols_available_for_outputs:
                        data_cols.append(old_col.strip())
                    else:
                        msg = 'ERROR: column {} not available for output, ' \
                            + 'check --fout'.format(col)

                self.colnames_out=data_cols
        print("TU: colnames_out ", self.colnames_out)





    def get_t0d3_gradient(self):

        '''
        
        find the gradient of t0d3 vector
        - either get it from file
        - or calculate it numerically

        '''

        if not self.input_options['use_grad_from_file']:

            # calculate gradient numerically
            # ------------------------------

            # calculate grid spacing
            #self.find_spacing_uniform_grid()

            # calculate gradient numerically:
            print('TUTU ', self.t0d3_points)
            grad_s = self.gradient(self.selected_vector_element(self.t0d3['s']))

        else:
            # assign grad_s to data read from file

            s_x  = np.array([ p['ds_dx'] for p in self.t0d3_points ], dtype=np.float64)
            s_y  = np.array([ p['ds_dy'] for p in self.t0d3_points ], dtype=np.float64)
            s_z  = np.array([ p['ds_dz'] for p in self.t0d3_points ], dtype=np.float64)

            grad_s = [s_x, s_y, s_z]



        # here starts an expensive loop over points:
        # todo: move it outside

        for i, d in enumerate(self.t0d3_points):

            # 1. first, store the gradient and the curl as global variables:

            self.t0d3_points[i]['ds_dx'] = grad_s[0][i]  # ds/dx
            self.t0d3_points[i]['ds_dy'] = grad_s[1][i]  # ds/dy
            self.t0d3_points[i]['ds_dz'] = grad_s[2][i]  # ds/dz
        pass

    def assign_t0d3_input_names(self, verbose=False):

        """

        assign user-specified data names to names used in qcten:


        1. grid
        -------
        assign user-specified names for grid coordinates 
        (with "--grid=["coorx, coory, coorz]")
        to names of grid coordinates used in qcten: "x", "y", "z";

        NOTE: grid points are read in the following order from the input data file:

            x, y, z


        2. vector field
        ---------------
        assign user-specified names for vector components 
        (with "--form_tensor_0order_3d=["vec_x, vec_y, vec_z]")
        to names of vector components used in qcten: "vx", "vy", "vz";

        NOTE: vector components are read in the following order from the input data file:

            vx, vy, vz


        3. the gradient of the vector field components
        ----------------------------------------------
        assign user-specified names for components of the gradient of the vector
        (with "--form_grad_tensor_0order_3d=["vec_x/dx, vec_x/dy, vec_x/dz, vec_y/dx, ...]")
        to names of components of the gradient of the vector used in qcten:
        "dvx_dx", "dvx_dy", "dvx_dz", "dvy_dx", "dvy_dy", "dvy_dz", "dvz_dx", "dvz_dy", "dvz_dz"

        NOTE: components of the gradient of the vector are read in the following order from the input data file:

            dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz


        """


        # grid
        args = [arg.strip().strip('[').strip(']') for arg in self.input_options['grid'].split(',')]
        self.colnames_qcten['x'] = args[0]
        self.colnames_qcten['y'] = args[1]
        self.colnames_qcten['z'] = args[2]
        self.colnames_inp = args

        if verbose:
            msg = 'grid columns are assigned: ' \
                + 'x={}, y={}, z={}, '.format(self.colnames_qcten['x'],
                                              self.colnames_qcten['y'],
                                              self.colnames_qcten['z'])
            print(msg)


        # t0d3
        #args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_tensor_0order_3d']]
        args = [self.input_options['form_tensor_0order_3d'].strip('[').strip(']')]
        self.colnames_qcten['s'] = args[0]
        self.colnames_inp.extend(args)

        print("TU: colnames_qcten ", self.colnames_qcten)
        print("TU: colnames_inp ", self.colnames_inp)



    def get_t0d3_data_points(self, verbose=False):

        """

        read input data into a "self.work_data" dataframe;
        to proceed, we read only these columns which are needed for the computation, i.e.:

        * columns corresponding to grid: self.t0d3['x'], ... 
        * columns corresponding to t0d3: self.t0d3['s'], ... 
        * (if needed) columns corresponding to grad(t0d3): self.t0d3['ds_dx'], ... 

        """

        cols = {v: k for k, v in self.colnames_qcten.items() if v is not None}
        self.work_data = self.input_data.rename(columns=cols)
        self.work_data = self.work_data[cols.values()]

        print('TU: self.work_data ')
        pprint(self.work_data)
        if verbose:
            print('working input data in t0d3: ')
            pprint(self.work_data)


    def find_spacing_uniform_grid(self):

        '''
        find spacing between grid points in x, y, z directions
        we assume a regular grid
        '''

        print('TU: find_spacing_uniform_grid :', self.t0d3_points)
        x0 = self.t0d3_points[0]['x']
        y0 = self.t0d3_points[0]['y']
        z0 = self.t0d3_points[0]['z']

        dx = abs(x0)
        dy = abs(y0)
        dz = abs(z0)

        for p in self.t0d3_points[1:]:
            x = p['x']
            y = p['y']
            z = p['z']
            if (abs(x - x0) < dx) and (abs(x - x0) > 0):
                dx = abs(x - x0)
            if (abs(y - y0) < dy) and (abs(y - y0) > 0):
                dy = abs(y - y0)
            if (abs(z - z0) < dz) and (abs(z - z0) > 0):
                dz = abs(z - z0)

        self.dx = dx
        self.dy = dy
        self.dz = dz

        print('find_spacing_uniform_grid :', x0, y0, z0, self.dx, self.dy, self.dz)
        with open(self.flog, 'a') as f:
            f.write('Grid spacing: dx, dy, dz = {}, {}, {}\n'.format(dx, dy, dz))


    def find_data_in_point_plusminus(self, d, f):

        result = {}

        x0 = d['x']
        y0 = d['y']
        z0 = d['z']

        reltol=1e-09
        abstol=1e-07

        for p in self.t0d3_points:
            x = p['x']
            y = p['y']
            z = p['z']

            # +- dx/dy/dz 
            if math.isclose(x, x0 + self.dx, rel_tol=reltol, abs_tol=abstol) and \
               math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
               math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['x_plus'] = p[f]

            elif math.isclose(x, x0 - self.dx, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['x_minus'] = p[f]

            elif math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0 + self.dy, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['y_plus'] = p[f]

            elif math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0 - self.dy, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['y_minus'] = p[f]

            elif math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0 + self.dz, rel_tol=reltol, abs_tol=abstol):
                result['z_plus'] = p[f]

            elif math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0 - self.dz, rel_tol=reltol, abs_tol=abstol):
                result['z_minus'] = p[f]


        return result


    def find_data_on_border(self, d, f, which_border):

        result = {}

        x0 = d['x']
        y0 = d['y']
        z0 = d['z']

        reltol=1e-09
        abstol=1e-07


        for p in self.t0d3_points:
            x = p['x']
            y = p['y']
            z = p['z']

            # +- dx/dy/dz 
            if not which_border[1] and \
               math.isclose(x, x0 + self.dx, rel_tol=reltol, abs_tol=abstol) and \
               math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
               math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['x_plus'] = p[f]

            if not which_border[0] and \
                 math.isclose(x, x0 - self.dx, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['x_minus'] = p[f]

            if not which_border[3] and \
                 math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0 + self.dy, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['y_plus'] = p[f]

            if not which_border[2] and \
                 math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0 - self.dy, rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0,           rel_tol=reltol, abs_tol=abstol):
                result['y_minus'] = p[f]

            if not which_border[5] and \
                 math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0 + self.dz, rel_tol=reltol, abs_tol=abstol):
                result['z_plus'] = p[f]

            if not which_border[4] and \
                 math.isclose(x, x0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(y, y0,           rel_tol=reltol, abs_tol=abstol) and \
                 math.isclose(z, z0 - self.dz, rel_tol=reltol, abs_tol=abstol):
                result['z_minus'] = p[f]


        return result


    def gradient_from_finite_elements_value_in_point(self, d, f):

        '''
        we calculate the gradient of d[f] in point d['x'], d['y'], d['z']

        if this grid point is 'inside' the cube, then we use the central differences formula:

        else, if the grid point is on the boundary, we use the first differences formula:

        '''

        x = d['x']
        y = d['y']
        z = d['z']
        s = d[f]

        if (x == min(p['x'] for p in self.t0d3_points)) or (x == max(p['x'] for p in self.t0d3_points)) or \
           (y == min(p['y'] for p in self.t0d3_points)) or (y == max(p['y'] for p in self.t0d3_points)) or \
           (z == min(p['z'] for p in self.t0d3_points)) or (z == max(p['z'] for p in self.t0d3_points)):
            #print('point on the border')

            which_border = [x == min(p['x'] for p in self.t0d3_points),
                            x == max(p['x'] for p in self.t0d3_points),
                            y == min(p['y'] for p in self.t0d3_points),
                            y == max(p['y'] for p in self.t0d3_points),
                            z == min(p['z'] for p in self.t0d3_points),
                            z == max(p['z'] for p in self.t0d3_points)]

            data = self.find_data_on_border(d, f, which_border)
            if which_border[0]:
                grad_x = (data['x_plus'] - s)/self.dx
            elif which_border[1]:
                grad_x = (s - data['x_minus'])/self.dx
            else:
                grad_x = (data['x_plus'] - data['x_minus'])/(2.0*self.dx)

            if which_border[2]:
                grad_y = (data['y_plus'] - s)/self.dy
            elif which_border[3]:
                grad_y = (s - data['y_minus'])/self.dy
            else:
                grad_y = (data['y_plus'] - data['y_minus'])/(2.0*self.dy)

            if which_border[4]:
                grad_z = (data['z_plus'] - s)/self.dz
            elif which_border[5]:
                grad_z = (s - data['z_minus'])/self.dz
            else:
                grad_z = (data['z_plus'] - data['z_minus'])/(2.0*self.dz)

        else:
            data = self.find_data_in_point_plusminus(d, f)
            grad_x = (data['x_plus'] - data['x_minus'])/(2.0*self.dx)
            grad_y = (data['y_plus'] - data['y_minus'])/(2.0*self.dy)
            grad_z = (data['z_plus'] - data['z_minus'])/(2.0*self.dz)

        result = [grad_x, grad_y, grad_z]

        return result


    def gradient_from_finite_elements(self, f):
        '''
        assuming a regular grid
        '''

        print('calculating a gradient of ', f)
        self.find_spacing_uniform_grid()

        for i, p in enumerate(self.t0d3_points):
            grad_f = self.gradient_from_finite_elements_value_in_point(p, f)
            self.t0d3_points[i]['grad_x'] = grad_f[0]
            self.t0d3_points[i]['grad_y'] = grad_f[1]
            self.t0d3_points[i]['grad_z'] = grad_f[2]

        return grad_f


    def test_grad_numpy(self, f):

        '''
        be careful, this works OK if f has at most quadratix dependence on r
        otherwise the approximation is too hars (see jupyter notebook in test_gradient)
        TODO: requires more testing
        '''

        f_values = np.array([p[f] for p in self.t0d3_points], dtype=np.float64)
        f_array  = f_values.reshape((self.dim_x, self.dim_y, self.dim_z))

        grad_f   = np.gradient(f_array, self.dx, self.dy, self.dz, edge_order=2)
        grad_f_x = grad_f[0].reshape((self.dim_cube))
        grad_f_y = grad_f[1].reshape((self.dim_cube))
        grad_f_z = grad_f[2].reshape((self.dim_cube))


        return [grad_f_x, grad_f_y, grad_f_z]


    def selected_vector_element(self, f):
        '''
        vector elements are read into self.t0d3['vx'], self.t0d3['vy'], self.t0d3['vz']
        but in the input the user asks to calculate the rortex of a vector element
        which can have an arbitrary name (given in the first row of data file);

        here we identify to which one of self.t0d3['vx'], self.t0d3['vy'], self.t0d3['vz']
        this element corresponds
        '''

        vector_elements         = dict(zip(self.t0d3.values(), self.t0d3.keys()))
        selected_vector_element = vector_elements[f]

        return selected_vector_element



    def gradient(self, f):
        '''
        calculate the gradient of f
        f is a selected element of a vector
        '''

        if self.input_options['calc_from_tensor_0order_3d_calc_grad'] == 'numpy':
            grad = self.test_grad_numpy(f)
        elif self.input_options['calc_from_tensor_0order_3d_calc_grad'] == 'finite_elements':
            grad = self.gradient_from_finite_elements(f)
        else:
            print('warning: wrong choice of calc_from_tensor_0order_3d_calc_grad')
            grad=None

        return grad

