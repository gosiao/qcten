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

        # input data and general setup
        self.input_options   = cli_options # FIXME - move this out
        self.output_options  = output_options
        self.data            = input_data  # dataframe to work on
        self.flog            = self.input_options['flog']

        # column names defined by the user:
        self.colnames_inp = []
        self.colnames_out = []


    def run(self, verbose=False):

        """
        main routine
        """

        for arg in self.input_options['calc_from_tensor_2order_3d']:

            if (arg == 'trace'):
                self.trace(verbose)

            if (arg == 'isotropic'):
                self.isotropic(verbose)

            if (arg == 'deviator'):
                self.deviator(verbose)

            if (arg == 'antisymmetric'):
                self.antisymmetric(verbose)

            if (arg == 'deviator_anisotropy'):
                self.deviator_anisotropy(verbose)


                #if (arg == 'tensor_inv1'):
                #    self.tensor_inv1()

                #if (arg == 'tensor_inv2'):
                #    self.tensor_inv2()

                #if (arg == 'tensor_inv3'):
                #    self.tensor_inv3()


                #if (arg == 'rortex_tensor_combined'):
                #    self.rortex_tensor_combined()

                #if (arg == 'omega_rortex_tensor_combined'):
                #    self.omega_rortex_tensor_combined()



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
        # ====
        for col in global_data.cols_to_use['grid']:
            self.colnames_qcten[col] = self.input_data[col]

        # data
        # ====

        # tensor
        for col in global_data.cols_to_use['t2d3']:
            self.colnames_qcten[col] = self.input_data[col]

        # tensor gradient
        if (self.input_options['form_grad_tensor_2order_3d'] is not None) and (self.input_options['use_grad_from_file']):
            for col in global_data.grad_cols_to_use['t2d3']:
                self.colnames_qcten[col] = self.input_data[col]

        if verbose:
            print('grid and data columns are assigned: ' + str(x) for x in self.colnames_qcten[col])


    def get_t2d3_data_points(self, verbose=False):

        """

        read input data into a "self.data" dataframe;
        to proceed, we read only these columns which are needed for the computation, i.e.:

        * columns corresponding to grid: self.t2d3['x'], ... 
        * columns corresponding to t1d3: self.t2d3['xx'], ... 
        * (if needed) columns corresponding to grad(t2d3): self.t2d3['dxx_dx'], ... 

        """

        cols = {v: k for k, v in self.colnames_qcten.items() if v is not None}
        self.data = self.input_data.rename(columns=cols)
        self.data = self.data[cols.values()]

        if verbose:
            print('working input data in t2d3: ')
            pprint(self.data)




    def trace(self, verbose):

        '''
        calculate the trace of the second-order tensor

        For tensor T:

            trace = T['xx'] +T['yy'] + T['zz']

        type of output data: scalar
        '''

        trace = trace_of_t2d3(self.data)                       
        data = pd.concat([self.data, trace], axis=1)
        self.data = data

        print('Output from trace:')
        pprint(self.data)
        if verbose:
            print('Output from trace:')
            pprint(self.data)


    def isotropic(self, verbose):

        '''
        calculate the isotropic part of the second-order tensor

        For tensor T:

            isotropic = (T['xx'] +T['yy'] + T['zz'])/3.0

        type of output data: scalar
        '''

        self.trace(verbose)
        isotropic = self.data['trace']/3.0
        self.data['isotropic'] = isotropic

        print('Output from isotropic:')
        pprint(self.data)
        if verbose:
            print('Output from isotropic:')
            pprint(self.data)



    def deviator(self, verbose):

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

        self.isotropic(verbose)

        for a in ['1', '2', '3']:
            for b in ['1', '2', '3']:
                l1 = a+b
                l2 = b+a
                self.data['deviator_'+l1] = (self.data['t'+l1] + self.data['t'+l2])/2.0

                if (a == b):
                    self.data['deviator_'+l1] = self.data['deviator_'+l1] - self.data['isotropic']

        print('Output from deviator:')
        pprint(self.data)
        if verbose:
            print('Output from deviator:')
            pprint(self.data)


    def antisymmetric(self, verbose):

        '''
        calculate the antisymmetric part of the second-order tensor;

        For tensor T, the elements of the antisymmetric part of this tensor are calculated as:

            A_ij = 0.5(T_ij - T_ji)

        type of output data: second-order tensor

        '''

        for a in ['1', '2', '3']:
            for b in ['1', '2', '3']:
                l1 = a+b
                l2 = b+a
                self.data['antisymmetric_'+l1] = (self.data['t'+l1] - self.data['t'+l2])/2.0

        print('Output from antisymmetric:')
        pprint(self.data)
        if verbose:
            print('Output from antisymmetric:')
            pprint(self.data)



    def deviator_anisotropy(self, verbose):

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

        self.isotropic(verbose)

        a1 = self.data['t11'] - self.data['t22']
        a2 = self.data['t22'] - self.data['t33']
        a3 = self.data['t33'] - self.data['t11']

        b1 = self.data['t12'] + self.data['t21']
        b2 = self.data['t23'] + self.data['t32']
        b3 = self.data['t31'] + self.data['t13']

        result = (a1**2 + a2**2 + a3**2)/3.0 \
               + (b1**2 + b2**2 + b3**2)/2.0

        self.data['deviator_anisotropy_squared'] = result
        self.data['deviator_anisotropy']         = np.sqrt(result)


        print('Output from deviator anisotropy:')
        pprint(self.data)
        if verbose:
            print('Output from deviator anisotropy:')
            pprint(self.data)



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


