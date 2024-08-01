import sys
import numpy as np
import scipy.linalg as la
import math
import pandas as pd
from .global_data import *
from .common import *
from pprint import pprint


class t2d3():

    """
    This class holds settings and operations
    on tensors of rank 2 in 3D space.

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
        self.t2d3          = {}
        self.t2d3_points   = []

        # column names defined by the user:
        self.colnames_inp = []
        self.colnames_out = []
        # column names used in this class:
        self.colnames_qcten = {}

        self.all_fun_t2d3 = global_data.all_fun_t2d3
        self.fun_t2d3_req_grad = global_data.fun_t2d3_req_grad

        # working data
        self.work_data = pd.DataFrame()

        # data columns that will be written to output(s)
        self.data_cols_to_export = {}

        # variables to be saved to the output:
        self.data_to_export= {}
        self.t2d3_cols     = []

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

        #print("ERROR: operations on t2d3 not available in this version")
        #sys.exit()

        # 1. verify input data
        self.verify_data_for_calcs(verbose=verbose)

        # 1. assign the data specified by a user to names used in qcten
        self.assign_t2d3_input_names(verbose=verbose)

        # 2. get the data (into pandas dataframe)
        self.get_t2d3_data_points(verbose=verbose)

        # 3. get grid information

        # 4. prepare the data for output
        self.assign_t2d3_output_names()

        # work
        if self.input_options['calc_from_tensor_2order_3d'] is not None:

            for arg in self.input_options['calc_from_tensor_2order_3d']:

                if (arg == 'tensor_inv1'):
                    self.tensor_inv1()

                if (arg == 'tensor_inv2'):
                    self.tensor_inv2()

                if (arg == 'tensor_inv3'):
                    self.tensor_inv3()

                if (arg == 'trace'):
                    self.trace()

                if (arg == 'isotropic'):
                    self.isotropic()

                if (arg == 'deviator'):
                    self.deviator()

                if (arg == 'antisymmetric'):
                    self.antisymmetric()

                if (arg == 'deviator_anisotropy'):
                    self.deviator_anisotropy()

                if (arg == 'rortex_tensor_combined'):
                    self.rortex_tensor_combined()

                if (arg == 'omega_rortex_tensor_combined'):
                    self.omega_rortex_tensor_combined()


        if 'calc_from_tensor_2order_3d_fragments' in self.input_options and self.input_options['calc_from_tensor_2order_3d_fragments'] is not None:
            args = self.input_options['calc_from_tensor_2order_3d_fragments'].split(':')
            if (args[0]  == 'gradient'):
                self.gradient(args[1])

        # prepare output
        self.prepare_output()


    def prepare_output(self):
        pass


    def assign_t2d3_output_names(self, verbose=False):

        """
        prepare the data for output(s)
        """

        cols_available_for_outputs = self.all_fun_t2d3 + self.colnames_inp 

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

        if self.input_options['calc_from_tensor_2order_3d'] is None:

            msg = 'WARNING: Nothing to calculate from the vector field. ' \
                + 'Check --calc_from_tensor_2order_3d in your input'
            sys.exit(msg)

            for arg in self.input_options['calc_from_tensor_2order_3d']:
                if arg not in self.all_fun_t2d3:
                    msg = 'ERROR: requested function not in the list of available functions ' \
                        + 'Check --calc_from_tensor_2order_3d in your input. ' \
                        + 'Available functions: ', self.all_fun_t2d3
                    sys.exit(msg)

        else:
            for arg in self.input_options['calc_from_tensor_2order_3d']:
                if (arg in self.fun_t2d3_req_grad):
                    if self.input_options['use_grad_from_file']:
                        print('Gradient of t2d3 is read from file')
                    else:
                        print('Gradient of t2d3 is calculated')
                        # TODO: calc gradient here!


    def assign_t2d3_input_names(self, verbose=False):

        """

        assign user-specified data names to names used in qcten:


        1. grid
        -------
        assign user-specified names for grid coordinates 
        (with "--grid=["coorx, coory, coorz]")
        to names of grid coordinates used in qcten: "x", "y", "z";

        NOTE: grid points are read in the following order from the input data file:

            x, y, z

        2. tensor field
        ---------------
        assign user-specified names for tensor components 
        (with "--form_tensor_2order_3d=["t_xx, t_xy, ...]")
        to names of tensor components used in qcten: "xx", "xy", ...

        NOTE: tensor components are read in the following order from the input data file:

            xx, xy, xz, yx, yy, yz, zx, zy, zz


        3. the gradient of the tensor field components
        ----------------------------------------------
        assign user-specified names for components of the gradient of the tensor
        (with "--form_grad_tensor_2order_3d=["t_xx/dx, t_xx/dy, t_xx/dz, t_xy/dx, ...]")
        to names of components of the gradient of the tensor used in qcten:
        "dxx_dx", "dxx_dy", "dxx_dz", "dxy_dx", "dxy_dy", ...

        NOTE: components of the gradient of the tensor are read in the following order from the input data file:

            dxx_dx, dxx_dy, dxx_dz, dxy_dx, dxy_dy, dxy_dz, ...

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


        # t2d3
        args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_tensor_2order_3d'].split(',')]

        self.colnames_qcten['xx'] = args[0]
        self.colnames_qcten['xy'] = args[1]
        self.colnames_qcten['xz'] = args[2]
        self.colnames_qcten['yx'] = args[3]
        self.colnames_qcten['yy'] = args[4]
        self.colnames_qcten['yz'] = args[5]
        self.colnames_qcten['zx'] = args[6]
        self.colnames_qcten['zy'] = args[7]
        self.colnames_qcten['zz'] = args[8]
        self.colnames_inp.extend(args)

        if verbose:
            msg = 'vector columns are assigned: ' \
                + 'vx={}, vy={}, vz={}, '.format(self.colnames_qcten['xx'],
                                                 self.colnames_qcten['xy'],
                                                 self.colnames_qcten['xz'],
                                                 self.colnames_qcten['yx'],
                                                 self.colnames_qcten['yy'],
                                                 self.colnames_qcten['yz'],
                                                 self.colnames_qcten['zx'],
                                                 self.colnames_qcten['zy'],
                                                 self.colnames_qcten['zz'])
            print(msg)

        # grad(t1d3)
        if (self.input_options['form_grad_tensor_2order_3d'] is not None) and (self.input_options['use_grad_from_file']):

            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_grad_tensor_2order_3d'].split(',')]

            self.colnames_qcten['dxx_dx'] = args[0]
            self.colnames_qcten['dxx_dy'] = args[1]
            self.colnames_qcten['dxx_dz'] = args[2]

            self.colnames_qcten['dxy_dx'] = args[3]
            self.colnames_qcten['dxy_dy'] = args[4]
            self.colnames_qcten['dxy_dz'] = args[5]

            self.colnames_qcten['dxz_dx'] = args[6]
            self.colnames_qcten['dxz_dy'] = args[7]
            self.colnames_qcten['dxz_dz'] = args[8]

            self.colnames_qcten['dyx_dx'] = args[9]
            self.colnames_qcten['dyx_dy'] = args[10]
            self.colnames_qcten['dyx_dz'] = args[11]

            self.colnames_qcten['dyy_dx'] = args[12]
            self.colnames_qcten['dyy_dy'] = args[13]
            self.colnames_qcten['dyy_dz'] = args[14]

            self.colnames_qcten['dyz_dx'] = args[15]
            self.colnames_qcten['dyz_dy'] = args[16]
            self.colnames_qcten['dyz_dz'] = args[17]

            self.colnames_qcten['dzx_dx'] = args[18]
            self.colnames_qcten['dzx_dy'] = args[19]
            self.colnames_qcten['dzx_dz'] = args[20]

            self.colnames_qcten['dzy_dx'] = args[21]
            self.colnames_qcten['dzy_dy'] = args[22]
            self.colnames_qcten['dzy_dz'] = args[23]

            self.colnames_qcten['dzz_dx'] = args[24]
            self.colnames_qcten['dzz_dy'] = args[25]
            self.colnames_qcten['dzz_dz'] = args[26]

            self.colnames_inp.extend(args)

            #if verbose:
            #    msg = 'grad(vector) columns are assigned: ' \
            #        + ' dvx_dx='+self.colnames_qcten['dvx_dx'] \
            #        + ' dvx_dy='+self.colnames_qcten['dvx_dy'] \
            #        + ' dvx_dz='+self.colnames_qcten['dvx_dz'] \
            #        + ' dvy_dx='+self.colnames_qcten['dvy_dx'] \
            #        + ' dvy_dy='+self.colnames_qcten['dvy_dy'] \
            #        + ' dvy_dz='+self.colnames_qcten['dvy_dz'] \
            #        + ' dvz_dx='+self.colnames_qcten['dvz_dx'] \
            #        + ' dvz_dy='+self.colnames_qcten['dvz_dy'] \
            #        + ' dvz_dz='+self.colnames_qcten['dvz_dz']
            #    print(msg)


    def get_t2d3_data_points(self, verbose=False):

        """

        read input data into a "self.work_data" dataframe;
        to proceed, we read only these columns which are needed for the computation, i.e.:

        * columns corresponding to grid: self.t2d3['x'], ... 
        * columns corresponding to t1d3: self.t2d3['xx'], ... 
        * (if needed) columns corresponding to grad(t2d3): self.t2d3['dxx_dx'], ... 

        """

        cols = {v: k for k, v in self.colnames_qcten.items() if v is not None}
        self.work_data = self.input_data.rename(columns=cols)
        self.work_data = self.work_data[cols.values()]

        if verbose:
            print('working input data in t2d3: ')
            pprint(self.work_data)




    def trace(self):

        '''
        calculate the trace of the second-order tensor

        For tensor T:

            trace = T['xx'] +T['yy'] + T['zz']

        type of output data: scalar
        '''

        data = self.work_data[['xx','xy','xz','yx','yy','yz','zx','zy','zz']].rename(columns={
                               'xx':'t11','xy':'t12','xz':'t13','yx':'t21','yy':'t22','yz':'t23','zx':'t31','zy':'t32','zz':'t33'})

        trace = trace_of_t2d3(data)                       

        data = pd.concat([self.work_data, trace], axis=1)
        self.work_data = data


    def isotropic(self):

        '''
        calculate the isotropic part of the second-order tensor

        For tensor T:

            isotropic = (T['xx'] +T['yy'] + T['zz'])/3.0

        type of output data: scalar
        '''

        for i, d in enumerate(self.t2d3_points):

            trace = d['xx'] + d['yy'] + d['zz']
            isotropic = trace/3.0
            self.t2d3_points[i]['isotropic'] = isotropic

        # add to data to be wriiten to the output file:
        self.t2d3_cols.append('isotropic')


    def deviator(self):

        '''
        calculate the 'deviator' of the second-order tensor;
        deviator = symmetric traceless anisotropic part of T

        For tensor T, the elements of the deviator D are calculated as:

            D_ij = S_ij - isotropic(S)*delta_ij

            where:
                S_ij = 0.5*(T_ij + T_ji)
                isotropic(S) = isotropic(T)

        type of output data: second-order tensor

        '''


        self.isotropic()

        for i, d in enumerate(self.t2d3_points):

            result = {}
            cols   = []

            for a in ['x', 'y', 'z']:
                for b in ['x', 'y', 'z']:

                    e1 = a+b
                    e2 = b+a
                    s  = (d[e1] + d[e2])/2.0
                    if (a == b):
                        result[e1] = s - self.t2d3_points[i]['isotropic']
                    else:
                        result[e1] = s

                    self.t2d3_points[i]['deviator'+'_'+e1] = result[e1]
                    cols.append('deviator'+'_'+e1)

        # add to data to be wriiten to the output file:
        for col in cols:
            self.t2d3_cols.append(col)


    def antisymmetric(self):

        '''
        calculate the antisymmetric part of the second-order tensor;

        For tensor T, the elements of the antisymmetric part of this tensor are calculated as:

            A_ij = 0.5(T_ij - T_ji)

        type of output data: second-order tensor

        '''

        for i, d in enumerate(self.t2d3_points):

            result = {}
            cols   = []

            for a in ['x', 'y', 'z']:
                for b in ['x', 'y', 'z']:

                    e1 = a+b
                    e2 = b+a
                    t  = (d[e1] - d[e2])/2.0
                    result[e1] = t

                    self.t2d3_points[i]['antisymmetric'+'_'+e1] = result[e1]
                    cols.append('antisymmetric'+'_'+e1)

        # add to data to be wriiten to the output file:
        for col in cols:
            self.t2d3_cols.append(col)



    def deviator_anisotropy(self):

        '''
        calculate the anisotropy of the 'deviator' of the second-order tensor;


        For tensor T, the elements of the deviator D are calculated as:

            D_ij = S_ij - isotropic(S)*delta_ij

            where:
                S_ij = 0.5*(T_ij + T_ji)
                isotropic(S) = isotropic(T)

            Its anisotropy (AD) can be calculated in terms of T elements:

            (AD)^2 = [ (T[xx] - T[yy])^2 + (T[yy] - T[zz])^2 + (T[zz] - T[xx])^2 ]/3.0
                   + [ (T[xy] + T[yx])^2 + (T[yz] + T[zy])^2 + (T[zx] + T[xz])^2 ]/2.0

        type of output data: scalar


        note: we save:
            * deviator_anisotropy_squared = (AD)^2 from eq. above
            * deviator_anisotropy = AD = sqrt((AD)^2)
        '''

        self.isotropic()

        for i, d in enumerate(self.t2d3_points):

            result = {}

            a1 = d['xx'] - d['yy']
            a2 = d['yy'] - d['zz']
            a3 = d['zz'] - d['xx']

            b1 = d['xy'] + d['yx']
            b2 = d['yz'] + d['zy']
            b3 = d['zx'] + d['xz']

            result = (a1**2 + a2**2 + a3**2)/3.0 \
                   + (b1**2 + b2**2 + b3**2)/2.0

            self.t2d3_points[i]['deviator_anisotropy_squared'] = result
            self.t2d3_points[i]['deviator_anisotropy']         = np.sqrt(result)

        # add to data to be wriiten to the output file:
        self.t2d3_cols.append('deviator_anisotropy_squared')
        self.t2d3_cols.append('deviator_anisotropy')


    def gradient(self, f):
        '''
        calculate gradient of f
        f is an 'original' element name
        '''

        tensor_elements         = dict(zip(self.t2d3.values(), self.t2d3.keys()))
        selected_tensor_element = tensor_elements[f]

        self.gradient_from_finite_elements(selected_tensor_element)

        #self.t2d3_points[i]['gradient'] = result
        #print('tensor element: ', selected_tensor_element)



    def gradient_from_finite_elements(self, f):
        '''
        gradient
        '''
        pass


    def rortex_tensor_combined(self):
        '''
        '''
        pass

    def test_eigen(self, a, eigenvalues, eigenvectors):
        for i in range(3):
            v     = eigenvectors[:, i].reshape(3,1)
            left  = a @ v
            right = eigenvalues[i]*v
            compare_OK = np.allclose(left, right, atol=1e-8)
            if not compare_OK:
                with open(self.flog, 'a') as f:
                    f.write('WARNING: problems with eigendecomposition!\n')


    def tensor_eigendecomposition(self):
        '''
        Here we do the eigendecomposition of the second-order tensor.
        Notes:
            * this is done in every point on a grid, can be expensive!
            * we use scipy.linalg package, TODO: 
                * compare with other python packages, esp. in terms of timing
                * better test checking whether the imaginary part of an
                  eigenvalue is 0 or close to 0
                * is the test_eigen necessary/sufficient?
                * check all values used as tolerance to compare numbers

        '''

        for i, d in enumerate(self.t2d3_points):

            a = np.array([[d['xx'], d['xy'], d['xz']], \
                          [d['yx'], d['yy'], d['yz']], \
                          [d['zx'], d['zy'], d['zz']]])

            eigenvalues, eigenvectors = la.eig(a)

            # first, second and third column on "eigenvectors" 
            # corresponds to first, second and third eigenvector, respectively 
            # the corresponding eigenvalues are eigenvalues[0], eigenvalues[1], eigenvalues[2]
            ev1 = eigenvectors[:, 0]
            ev2 = eigenvectors[:, 1]
            ev3 = eigenvectors[:, 2]

            eigenvalues = [e.real if (e.imag==0.0) else e for e in eigenvalues]
            #eigenvalues = [e.real if math.isclose(e.imag, 0.0, abs_tol=1e-15) else e for e in eigenvalues]
            # add verbose
            #for e in eigenvalues:
            #    if np.iscomplex(e) and math.isclose(e.imag, 0.0, abs_tol=1e-15):
            #        with open(self.flog, 'a') as f:
            #            f.write('WARNING: small imaginary part of eigenvalue: {}\n'.format(e))

            self.test_eigen(a, eigenvalues, eigenvectors)
            #number_complex_eigenvalues = 0
            #for i, e in enumerate(eigenvalues):
            #    if isinstance(e, complex):
            #        number_complex_eigenvalues += 1
            #if (number_complex_eigenvalues) > 0:
            #    print('we have complex eigenvalues in point ', i, number_complex_eigenvalues)

            self.t2d3_points[i]['eigenvalue1']  = eigenvalues[0]
            self.t2d3_points[i]['eigenvalue2']  = eigenvalues[1]
            self.t2d3_points[i]['eigenvalue3']  = eigenvalues[2]
            self.t2d3_points[i]['eigenvector1'] = ev1
            self.t2d3_points[i]['eigenvector2'] = ev2
            self.t2d3_points[i]['eigenvector3'] = ev3

        if self.input_options['fout_select'] == 'all':
            self.t2d3_cols.append('eigenvalue1')
            self.t2d3_cols.append('eigenvalue2')
            self.t2d3_cols.append('eigenvalue3')
            self.t2d3_cols.append('eigenvector1')
            self.t2d3_cols.append('eigenvector2')
            self.t2d3_cols.append('eigenvector3')


    def tensor_inv1(self):
        '''
        Here we calculate the first principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I1 = tr(T) = l1 + l2 + l3

        '''

        self.tensor_eigendecomposition()

        for i, d in enumerate(self.t2d3_points):
            result = d['eigenvalue1'] + d['eigenvalue2'] + d['eigenvalue3']
            self.t2d3_points[i]['tensor_inv1'] = result

        #self.t2d3_cols.append('tensor_inv1')


    def tensor_inv2(self):
        '''
        Here we calculate the second principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I2 = 0.5*{ [tr(T)]^2 - tr(T^2) } 
               = l1*l2 + l2*l3 + l1*l3

        '''

        self.tensor_eigendecomposition()

        for i, d in enumerate(self.t2d3_points):
            result = d['eigenvalue1']*d['eigenvalue2'] + d['eigenvalue2']*d['eigenvalue3'] + d['eigenvalue1']*d['eigenvalue3']
            self.t2d3_points[i]['tensor_inv2'] = result

        #self.t2d3_cols.append('tensor_inv2')


    def tensor_inv3(self):
        '''
        Here we calculate the third principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I3 = det(T) = l1*l2*l3

        '''

        self.tensor_eigendecomposition()

        for i, d in enumerate(self.t2d3_points):
            result = d['eigenvalue1']*d['eigenvalue2']*d['eigenvalue3']
            self.t2d3_points[i]['tensor_inv3'] = result

        #self.t2d3_cols.append('tensor_inv3')



    def tensor_frobenius_norm(self, tensor):
        '''
        Here we calculate the tensor Frobenius norm

        For tensor T, whose elements are t_ij

            F = sqrt(sum_{ij} (t_{ij})**2)

        '''

        sum_t2 = 0
        for t in tensor:
            sum_t2 += t**2

        result = np.sqrt(sum_t2)

        return result


    def omega_rortex_tensor_combined(self):
        print('omega_rortex_tensor_combined: inprep')

    #    for i, d in enumerate(self.t2d3_points):

    #        t1 = [d['xx'], d['xy'], d['xz'],
    #              d['yx'], d['yy'], d['yz'],
    #              d['zx'], d['zy'], d['zz']]

    #        norm_t1 = self.tensor_frobenius_norm(t1)

    #    t2 = 
    #    norm_t2 = self.tensor_frobenius_norm(t2)
    #    for t in tensor:
    #        sum_t2 += t**2

    #    result = np.sqrt(sum_t2)

    #    return result


