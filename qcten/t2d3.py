import numpy as np
import scipy.linalg as la
import math


class t2d3():

    def __init__(self, input_options, grid, input_data):

        # general setup
        self.input_options = input_options
        self.grid    = grid
        self.input_data    = input_data
        self.flog          = input_options['flog']

        # global data structures 
        self.t2d3          = {}
        self.t2d3_points   = []

        # variables to be saved to the output:
        self.data_to_export= []
        self.t2d3_cols     = []

        self.tensor_2order_3d_is_assigned = False


    def run(self):

        # prepare
        self.assign_tensor_2order_3d()
        self.assign_output_for_tensor_2order_3d()
        self.get_tensor_2order_3d_data_points()

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

                #if (arg == 'omega_rortex_tensor_combined'):
                #    self.omega_rortex_tensor_combined()



        if 'calc_from_tensor_2order_3d_fragments' in self.input_options and self.input_options['calc_from_tensor_2order_3d_fragments'] is not None:
            args = self.input_options['calc_from_tensor_2order_3d_fragments'].split(':')
            if (args[0]  == 'gradient'):
                self.gradient(args[1])

        # prepare output
        self.prepare_output()


    def prepare_output(self):
        pass

    def assign_output_for_tensor_2order_3d(self):
        output=[]
        if self.input_options['data_out'] is not None:
            self.data_to_export = [arg.strip().strip('[').strip(']') for arg in self.input_options['data_out'].split(',')]
            #for data in self.t2d3_cols:
            #    if data in self.data_to_export:


    def assign_tensor_2order_3d(self):


        '''

        1. tensor field
        ---------------
        tensor components are read in the following order from the input data file:
        (this is the order of elements read with the --form_tensor_2order_3d)

            [grid_x, grid_y, grid_z], xx, xy, xz, yx, yy, yz, zx, zy, zz

            where xx, xy, ...  are tensor elements

        here we assign the user-chosen names of tensor elements to 'xx', 'xy', etc.


        2. the gradient of the vector field components
        ----------------------------------------------
        if the gradient data is available, then assign the user-chosen names to:

            [grid_x, grid_y, grid_z],  dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz


        3. the gradient of the tensor field components
        ----------------------------------------------
        TODO

        TODO: refactor!!! 

        NOTE: self.t2d3 does not store data, only the names of variables

        '''

        if ('form_tensor_2order_3d' in self.input_options) and (self.input_options['form_tensor_2order_3d'] is not None):
            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_tensor_2order_3d'].split(',')]
            
            self.t2d3['xx'] = args[0]
            self.t2d3['xy'] = args[1]
            self.t2d3['xz'] = args[2]
            self.t2d3['yx'] = args[3]
            self.t2d3['yy'] = args[4]
            self.t2d3['yz'] = args[5]
            self.t2d3['zx'] = args[6]
            self.t2d3['zy'] = args[7]
            self.t2d3['zz'] = args[8]
            self.tensor_2order_3d_is_assigned = True


        #if (self.input_options['form_grad_vector_3d'] is not None) and (self.input_options['use_grad_from_file']):
        if ('form_grad_vector_3d' in self.input_options) and (self.input_options['form_grad_vector_3d'] is not None) and (self.input_options['use_grad_from_file']):

            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_grad_vector_3d'].split(',')]

            self.t1d3['dvx_dx'] = args[0]
            self.t1d3['dvx_dy'] = args[1]
            self.t1d3['dvx_dz'] = args[2]

            self.t1d3['dvy_dx'] = args[3]
            self.t1d3['dvy_dy'] = args[4]
            self.t1d3['dvy_dz'] = args[5]

            self.t1d3['dvz_dx'] = args[6]
            self.t1d3['dvz_dy'] = args[7]
            self.t1d3['dvz_dz'] = args[8]


    def get_tensor_2order_3d_data_points(self):

        if not self.tensor_2order_3d_is_assigned:
            sys.exit('TENSOR NOT ASSIGNED')

        for i, r in self.input_data.iterrows():

            d = {}

            d['grid_x']  = r[self.grid['grid_x']]
            d['grid_y']  = r[self.grid['grid_y']]
            d['grid_z']  = r[self.grid['grid_z']]

            d['xx'] = r[self.t2d3['xx']]
            d['xy'] = r[self.t2d3['xy']]
            d['xz'] = r[self.t2d3['xz']]
            d['yx'] = r[self.t2d3['yx']]
            d['yy'] = r[self.t2d3['yy']]
            d['yz'] = r[self.t2d3['yz']]
            d['zx'] = r[self.t2d3['zx']]
            d['zy'] = r[self.t2d3['zy']]
            d['zz'] = r[self.t2d3['zz']]

            if ('form_grad_vector_3d' in self.input_options) and (self.input_options['form_grad_vector_3d'] is not None) and (self.input_options['use_grad_from_file']):

                d['dvx_dx']  = r[self.t1d3['dvx_dx']]
                d['dvx_dy']  = r[self.t1d3['dvx_dy']]
                d['dvx_dz']  = r[self.t1d3['dvx_dz']]

                d['dvy_dx']  = r[self.t1d3['dvy_dx']]
                d['dvy_dy']  = r[self.t1d3['dvy_dy']]
                d['dvy_dz']  = r[self.t1d3['dvy_dz']]

                d['dvz_dx']  = r[self.t1d3['dvz_dx']]
                d['dvz_dy']  = r[self.t1d3['dvz_dy']]
                d['dvz_dz']  = r[self.t1d3['dvz_dz']]

            self.t2d3_points.append(d)

        # decide what data will be written to the output file, if the user did not specify that:
        if self.input_options['data_out'] is None:

            self.t2d3_cols.append('grid_x')
            self.t2d3_cols.append('grid_y')
            self.t2d3_cols.append('grid_z')
            
            if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
            
                self.t2d3_cols.append('xx')
                self.t2d3_cols.append('xy')
                self.t2d3_cols.append('xz')
                self.t2d3_cols.append('yx')
                self.t2d3_cols.append('yy')
                self.t2d3_cols.append('yz')
                self.t2d3_cols.append('zx')
                self.t2d3_cols.append('zy')
                self.t2d3_cols.append('zz')
            
            if self.input_options['fout_select'] == 'all':
            
                self.t2d3_cols.append('dvx_dx')
                self.t2d3_cols.append('dvx_dy')
                self.t2d3_cols.append('dvx_dz')
                self.t2d3_cols.append('dvy_dx')
                self.t2d3_cols.append('dvy_dy')
                self.t2d3_cols.append('dvy_dz')
                self.t2d3_cols.append('dvz_dx')
                self.t2d3_cols.append('dvz_dy')
                self.t2d3_cols.append('dvz_dz')

        else: # if self.input_options['data_out'] is None:
            for col in self.data_to_export:
                self.t2d3_cols.append(col)


    def trace(self):

        '''
        calculate the trace of the second-order tensor

        For tensor T:

            trace = T['xx'] +T['yy'] + T['zz']

        type of output data: scalar
        '''

        for i, d in enumerate(self.t2d3_points):

            trace = d['xx'] + d['yy'] + d['zz']
            self.t2d3_points[i]['trace'] = trace

        # add to data to be wriiten to the output file:
        self.t2d3_cols.append('trace')


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


    #def omega_rortex_tensor_combined(self):
    #    '''

    #    '''

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


