import sys
import numpy as np
import math
import pandas as pd
from .global_data import *
from .common import *
from pprint import pprint


class t1d3():

    """
    This class holds settings and operations
    on tensors of rank 1 (vectors) in 3D space.

    @author:       Gosia Olejniczak
    @contact:      gosia.olejniczak@gmail.com
    """


    def __init__(self, cli_options, output_options, input_data):

        # input data and general setup
        self.input_options   = cli_options # FIXME - move this out
        self.output_options  = output_options
        self.data            = input_data  # dataframe to work on
        self.flog            = self.input_options['flog']

        # global data structures
        self.t1d3          = {}

        # column names defined by the user:
        self.colnames_inp = []
        self.colnames_out = []


    def run(self, verbose=False):

        """
        main routine
        """

        for arg in self.input_options['calc_from_tensor_1order_3d']:

            if (arg == 'rortex' or arg == 'omega_rortex'):
                self.rortex_and_shear(verbose)

            if (arg == 'norm'):
                self.norm(verbose)

            if (arg == 'mean'):
                self.mean(verbose)

            if (arg == 'vorticity'):
                self.vorticity(verbose)

            if (arg == 'omega'):
                self.omega(verbose)
#
#
#            if (arg == 'curlv_cdot_axis'):
#                self.curlv_cdot_axis()
#
#            if (arg == 'rortex_cdot_axis'):
#                self.rortex_cdot_axis()
#
#        if self.input_options['projection_axis'] is not None:
#            self.project_v_on_projection_axis()



    def assign_t1d3_output_names(self, verbose=False):

        """
        prepare the data for output(s)
        """

        cols_available_for_outputs = global_data.all_fun_t1d3 + self.colnames_inp 

        for v in self.output_options:
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


    #        # the order of elements:
    #        # (is the same as in Xu, Phys.Fluids 31, 095102 (2019), which we follow here)
    #        #
    #        #  xx  xy  xz       dvx/dx  dvx/dy  dvx/dz 
    #        #  yx  yy  yz  ->   dvy/dx  dvy/dy  dvy/dz
    #        #  zx  zy  zz       dvz/dx  dvz/dy  dvz/dz
    #        #
    #        full_grad_tensor = np.array([[grad_vecx[0][i], grad_vecx[1][i], grad_vecx[2][i]],
    #                                     [grad_vecy[0][i], grad_vecy[1][i], grad_vecy[2][i]],
    #                                     [grad_vecz[0][i], grad_vecz[1][i], grad_vecz[2][i]]],
    #                                     dtype=np.float64)





    def selected_vector_element(self, f):
        '''
        vector elements are read into self.t1d3['vx'], self.t1d3['vy'], self.t1d3['vz']
        but in the input the user asks to calculate the rortex of a vector element
        which can have an arbitrary name (given in the first row of data file);

        here we identify to which one of self.t1d3['vx'], self.t1d3['vy'], self.t1d3['vz']
        this element corresponds
        '''

        vector_elements         = dict(zip(self.t1d3.values(), self.t1d3.keys()))
        selected_vector_element = vector_elements[f]

        return selected_vector_element


    def rortex_in_point(self, m, w, projection_axis, verbose):

        if verbose:
            print('rortex_in_point: entering m ' , type(m))
            print('number_complex_eigenvalues = ', m['number_complex_eigenvalues'], type(m['number_complex_eigenvalues']))
            print('real_eigval_ind = ', m['real_eigval_ind'])
            print('eig_pair_0 = ', m['eig_pair_0'], type(m['eig_pair_0']))
            print('eig_pair_1 = ', m['eig_pair_1'], type(m['eig_pair_1']))
            print('eig_pair_2 = ', m['eig_pair_2'], type(m['eig_pair_2']))

        thr_same_numbers = 10**(-10)
        res = {}

        # 1. first, work on points, in which the number of complex eigenvalues == 2

        if m['number_complex_eigenvalues'] == 2:

            # find vectors corresponding to complex and real eigenvalues,
            # and rename variables as in Xu et al. Phys Fluids 31, 095102 (2019)

            eig_val_complex = []
            for i in range(3):
                eig_val = m['eig_pair_'+str(i)][0]
                eig_vec = m['eig_pair_'+str(i)][1]
                if isinstance(eig_val, float):
                    lambda_r = eig_val
                    eig_vec_real = [e.real for e in eig_vec]
                else:
                    eig_val_complex.append(eig_val)
            if abs(eig_val_complex[0].real - eig_val_complex[1].real) < thr_same_numbers:
                lambda_ci = abs(eig_val_complex[0].imag)
                lambda_cr =     eig_val_complex[0].real
            else:
                print('ERROR')

            if verbose:
                print('lambda_r = ', lambda_r)
                print('lambda_ci= ', lambda_ci)
                print('lambda_cr= ', lambda_cr)

            # calculate the normalized real eigenvector corresponding to the real eigenvalue:

            eigvec_real_magn = np.sqrt(eig_vec_real[0]**2 + eig_vec_real[1]**2 + eig_vec_real[2]**2)
            eigvec_real_normalized = [e/eigvec_real_magn for e in eig_vec_real]
            if verbose:
                print('eigvec_real_magn = ', eigvec_real_magn)
                print('eigvec_real_normalized = ', eigvec_real_normalized)

            # calculate rortex vector
            # these are eqs. 33 and 34 in Xu et al. Phys Fluids 31, 095102 (2019)

            # step 1: eq. 30 in Xu et al. Phys Fluids 31, 095102 (2019)
            omega_cdot_r = w[0]*eigvec_real_normalized[0]  \
                         + w[1]*eigvec_real_normalized[1]  \
                         + w[2]*eigvec_real_normalized[2]
            sign_changed =  False
            if omega_cdot_r < 0.0:
                sign_changed = True
                factor = -1.0
            else:
                factor = 1.0
            omega_cdot_r = factor * omega_cdot_r

            # step 2: eq. 33
            val = omega_cdot_r**2 - 4*(lambda_ci**2)
            if (val < 0.0):
                raise Exception('WARNING: omega_cdot_r**2 - 4*(lambda_ci**2) < 0 and equals {}'.format(val))

            rortex_magnitude = omega_cdot_r - np.sqrt(val)

            # step 3: eq. 34

            rortex_vector = [factor * rortex_magnitude * e for e in eigvec_real_normalized]

            if projection_axis is not None:
                rortex_cdot_axis = rortex_vector[0]*projection_axis[0] \
                                 + rortex_vector[1]*projection_axis[1] \
                                 + rortex_vector[2]*projection_axis[2]
            else:
                rortex_cdot_axis = None

            # rortex in tensor form (eq. 3 in Xu et al. Phys Fluids 31, 095102 (2019)):
            phi = 0.5*rortex_magnitude
            rortex_tensor = [0.0 for _ in range(9)]
            rortex_tensor[1] = rortex_tensor[3] = -phi
            rortex_tensor = {}
            rortex_tensor['t11'] = 0.0
            rortex_tensor['t12'] = -phi
            rortex_tensor['t13'] = 0.0
            rortex_tensor['t21'] = phi
            rortex_tensor['t22'] = 0.0
            rortex_tensor['t23'] = 0.0
            rortex_tensor['t31'] = 0.0
            rortex_tensor['t32'] = 0.0
            rortex_tensor['t33'] = 0.0

            res['rortex_magnitude'] = rortex_magnitude
            res['rortex_cdot_axis'] = rortex_cdot_axis
            for i in range(3):
                l1=str(i+1)
                res['rortex_vector_'+l1] = rortex_vector[i]
                for j in range(3):
                    l2=str(i+1)+str(j+1)
                    res['rortex_tensor_'+l2] = rortex_tensor["t"+l2]

            # calculate 'omega_rortex'
            # using Eq. 36 from Xu et al. Phys Fluids 31, 095102 (2019)
            omega_rortex = omega_cdot_r**2 / (2*(omega_cdot_r**2 - 2*(lambda_ci**2) + 2*(lambda_cr**2) + lambda_r**2))
            res['omega_rortex'] = omega_rortex


        # 2. then, work on the remaining points
        else:
            res['rortex_magnitude'] = np.nan
            res['rortex_cdot_axis'] = np.nan
            res['omega_rortex'] = np.nan
            for i in range(3):
                label=str(i+1)
                res['rortex_vector_'+label] = np.nan
                for j in range(3):
                    l2=str(i+1)+str(j+1)
                    res['rortex_tensor_'+l2] = np.nan
        return res


    def rortex_and_shear(self, verbose):

        '''
        algorithm implemented here is from Xu et al. Phys Fluids 31, 095102 (2019):
        * rortex is calculated as in steps (1)-(4) from sec. II.C therein
        * shear is calculated as the 'gradient of the vector field - rortex'

        rortex and shear are in general presented as second-order tensors,
        here:
        * rortex is calculated in its vector form (see Xu et al. Phys Fluids 31, 095102 (2019))
        * the elements of a shear tensor are calculated explicitly
        (see eq. 2 and 4 in Xu et al. Phys Fluids 31, 095102 (2019))

        we need the gradient of velocity vector field:

        * the order of elements on the gradient tensor (after Xu, Phys.Fluids 31, 095102 (2019)):
        
         xx  xy  xz       dvx/dx  dvx/dy  dvx/dz 
         yx  yy  yz  ->   dvy/dx  dvy/dy  dvy/dz
         zx  zy  zz       dvz/dx  dvz/dy  dvz/dz
        
        gosia TODO - this needs testing
        '''
        self.vorticity(verbose)

        # 1. construct the gradient tensor
        tmp = [x for x in global_data.grad_cols_to_use['t1d3'] if x not in self.data.columns]
        if tmp:
            self.data['t1_dx'], self.data['t1_dy'], self.data['t1_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't1') 
            self.data['t2_dx'], self.data['t2_dy'], self.data['t2_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't2') 
            self.data['t3_dx'], self.data['t3_dy'], self.data['t3_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't3') 


        full_grad_tensor = self.data[['t1_dx', 't1_dy', 't1_dz',
                                      't2_dx', 't2_dy', 't2_dz',
                                      't3_dx', 't3_dy', 't3_dz']].rename(columns={
                                      't1_dx':'t11', 't1_dy':'t12', 't1_dz':'t13',
                                      't2_dx':'t21', 't2_dy':'t22', 't2_dz':'t23',
                                      't3_dx':'t31', 't3_dy':'t32', 't3_dz':'t33'})

        print("full_grad_tensor")
        pprint(full_grad_tensor)

        # 2. do the eigendecomposition of the gradient tensor (pointwise):
        full_grad_tensor["t2d3_as_npndarray"] = full_grad_tensor.apply(lambda row: row.to_numpy().reshape((3,3)), axis=1)
        tmp = full_grad_tensor.apply(lambda x: tensor_eigendecomposition(x["t2d3_as_npndarray"]), axis=1)
        tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)
        tmp = pd.concat([tmp, self.data[['curlv_x', 'curlv_y', 'curlv_z']]], axis=1)

        if self.input_options['projection_axis'] is not None:
            paxis = [self.data['projection_axis_x'], self.data['projection_axis_y'], self.data['projection_axis_z']]
        else:
            paxis = None
        tmp2 = tmp.apply(lambda x: self.rortex_in_point(x, w = [x['curlv_x'], x['curlv_y'], x['curlv_z']], projection_axis=paxis, verbose=verbose), axis=1)

        tmp2 = pd.DataFrame(tmp2.to_list(), index=tmp2.index)
        res = pd.concat([tmp2, self.data], axis=1)
        #print('RES: ',  type(res), res.shape, res.columns, res.index)
        #print('DATA: ', type(self.data), self.data.shape, self.data.columns, self.data.index)
        for col in res.columns:
            if col not in self.data.columns:
                self.data[col] = res[col]
        print(self.data.columns)
        #pprint(self.data)
        
        #pprint(res)
        #pd.set_option('display.max_rows', None)
        #pd.set_option('display.max_columns', None)
        #print(res)

        # 4. save results
        #self.data = pd.concat([self.data, tmp2], axis=1)

        #pd.set_option('display.max_rows', None)
        #pd.set_option('display.max_columns', None)
        #print(self.data)


        ## 3. calculate rortex and shear



    #def shear_in_point(self, point_index, point_data, full_grad_tensor):

    #    '''
    #    shear is calculated from Eq. (2) in Xu et al. Phys Fluids 31, 095102 (2019);
    #    shear is a tensor calculated as:
    #    'velocity gradient tensor - rortex tensor'

    #    however, the velocity gradient tensor needs to be expressed in Shur form;
    #    here we will construct the Shur form of the velocity gradient tensor
    #    from eq. 24 in that paper
    #    '''

    #    if (point_data['number_complex_eigenvalues'] == 2):
    #        #full_grad_tensor_shur = construct_shur_form_gradient_tensor(full_grad_tensor)
    #        phi = 0.5*self.t1d3_points[point_index]['rortex_magnitude']
    #        self.t1d3_points[point_index]['shear_tensor_xx'] = full_grad_tensor[0,0]
    #        self.t1d3_points[point_index]['shear_tensor_xy'] = 0.0
    #        self.t1d3_points[point_index]['shear_tensor_xz'] = 0.0
    #        self.t1d3_points[point_index]['shear_tensor_yx'] = full_grad_tensor[1,0] - phi
    #        self.t1d3_points[point_index]['shear_tensor_yy'] = full_grad_tensor[1,1]
    #        self.t1d3_points[point_index]['shear_tensor_yz'] = 0.0
    #        self.t1d3_points[point_index]['shear_tensor_zx'] = full_grad_tensor[2,0]
    #        self.t1d3_points[point_index]['shear_tensor_zy'] = full_grad_tensor[2,1]
    #        self.t1d3_points[point_index]['shear_tensor_zz'] = full_grad_tensor[2,2]
    #    else:
    #        self.t1d3_points[point_index]['shear_tensor_xx'] = None
    #        self.t1d3_points[point_index]['shear_tensor_xy'] = None
    #        self.t1d3_points[point_index]['shear_tensor_xz'] = None
    #        self.t1d3_points[point_index]['shear_tensor_yx'] = None
    #        self.t1d3_points[point_index]['shear_tensor_yy'] = None
    #        self.t1d3_points[point_index]['shear_tensor_yz'] = None
    #        self.t1d3_points[point_index]['shear_tensor_zx'] = None
    #        self.t1d3_points[point_index]['shear_tensor_zy'] = None
    #        self.t1d3_points[point_index]['shear_tensor_zz'] = None

    #    if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
    #        self.t1d3_cols.append('shear_tensor_xx')
    #        self.t1d3_cols.append('shear_tensor_xy')
    #        self.t1d3_cols.append('shear_tensor_xz')
    #        self.t1d3_cols.append('shear_tensor_yx')
    #        self.t1d3_cols.append('shear_tensor_yy')
    #        self.t1d3_cols.append('shear_tensor_yz')
    #        self.t1d3_cols.append('shear_tensor_zx')
    #        self.t1d3_cols.append('shear_tensor_zy')
    #        self.t1d3_cols.append('shear_tensor_zz')



    def construct_shur_form_gradient_tensor(t):
        alpha=0

    def v1_cdot_v2(self, v1, v2):
        '''
        scalar product of two vectors
        '''

        v1_cdot_v2  = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

        return v1_cdot_v2


    #def project_v_on_projection_axis(self):

    #    args = [arg.strip().strip('[').strip(']') for arg in self.input_options['projection_axis'].split(',')]
    #    self.projection_axis['x'] = int(args[0])
    #    self.projection_axis['y'] = int(args[1])
    #    self.projection_axis['z'] = int(args[2])


    #    for i, d in enumerate(self.t1d3_points):
    #        v_cdot_axis = self.t1d3_points[i]['vx']*self.projection_axis['x'] \
    #                    + self.t1d3_points[i]['vy']*self.projection_axis['y'] \
    #                    + self.t1d3_points[i]['vz']*self.projection_axis['z']

    #        self.t1d3_points[i]['v_cdot_axis'] = v_cdot_axis

    #    #self.t1d3_cols.append('v_cdot_axis')



    def norm(self, verbose):
        """
        calculate the L2 norm of a 3d vector

        For vector v, whose elements are v_i:

        norm = sqrt(sum_{i} (v_{i})**2)

        """

        norm = norm_of_t1d3(self.data)

        data = pd.concat([self.data, norm], axis=1)
        self.data = data

        print('Output from norm:')
        pprint(self.data)
        if verbose:
            print('Output from norm:')
            pprint(self.data)



    def mean(self, verbose):
        """
        calculate the mean of vector elements

        For vector v, whose elements are v_i:

        mean = 1/3 (sum_{i} v_{i})

        """

        mean = get_mean_of_t1d3(self.data)

        data = pd.concat([self.data, mean], axis=1)
        self.data = data

        if verbose:
            print('Output from mean:')
            pprint(self.data)



    def vorticity(self, verbose):

        '''
        calculate the curl of the vector:
        
        For vector v = (vx, vy, vz):

        w = \nabla \times v

        wx = d(vz)/dy - d(vy)/dz
        wy = d(vx)/dz - d(vz)/dx
        wz = d(vy)/dx - d(vx)/dy

        '''

        tmp = [x for x in global_data.grad_cols_to_use['t1d3'] if x not in self.data.columns]
        if tmp:
            self.data['t1_dx'], self.data['t1_dy'], self.data['t1_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't1') 
            self.data['t2_dx'], self.data['t2_dy'], self.data['t2_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't2') 
            self.data['t3_dx'], self.data['t3_dy'], self.data['t3_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't3') 

        curlv_x = self.data['t3_dy'] - self.data['t2_dz']
        curlv_y = self.data['t1_dz'] - self.data['t3_dx']
        curlv_z = self.data['t2_dx'] - self.data['t1_dy']

        curlv_magnitude = np.sqrt(curlv_x**2 + curlv_y**2 + curlv_z**2)

        self.data['curlv_x'] = curlv_x
        self.data['curlv_y'] = curlv_y
        self.data['curlv_z'] = curlv_z
        self.data['curlv_magnitude'] = curlv_magnitude

        if self.input_options['projection_axis'] is not None:
            curlv_cdot_axis = self.data['curlv_x']*self.data['projection_axis_x'] \
                            + self.data['curlv_y']*self.data['projection_axis_y'] \
                            + self.data['curlv_z']*self.data['projection_axis_z']

            self.data['curlv_cdot_axis'] = curlv_cdot_axis





    def omega(self, verbose):

        """
        calculate omega ("Omega vortex identification method")

        based on Eq. 12 in Liu et. al, Journal of Hydrodynamics, 31, 205, 2019 (DOI: https://link.springer.com/article/10.1007/s42241-019-0022-4)

        calculations need the gradient of a vector field: nabla(v)

        then:
        1. calculate the symmetric (S) and antisymmetric (A) parts of nabla(v)
        2. calculate the Frobenius norms of these parts, squared, |S|^2 and |A|^2, respectively
        3. calculate omega = |A|^2 / (|A|^2 + |S|^2)

        """

        tmp = [x for x in global_data.grad_cols_to_use['t1d3'] if x not in self.data.columns]
        if tmp:
            self.data['t1_dx'], self.data['t1_dy'], self.data['t1_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't1') 
            self.data['t2_dx'], self.data['t2_dy'], self.data['t2_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't2') 
            self.data['t3_dx'], self.data['t3_dy'], self.data['t3_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't3') 


        full_grad_tensor = self.data[['t1_dx', 't1_dy', 't1_dz',
                                      't2_dx', 't2_dy', 't2_dz',
                                      't3_dx', 't3_dy', 't3_dz']].rename(columns={
                                      't1_dx':'t11', 't1_dy':'t12', 't1_dz':'t13',
                                      't2_dx':'t21', 't2_dy':'t22', 't2_dz':'t23',
                                      't3_dx':'t31', 't3_dy':'t32', 't3_dz':'t33'})

        sym_part = get_sym_part_of_t2d3(full_grad_tensor)
        antisym_part  = get_antisym_part_of_t2d3(full_grad_tensor)

        sym_norm     = frobenius_norm_squared_t2d3(sym_part)
        antisym_norm = frobenius_norm_squared_t2d3(antisym_part)
        omega = antisym_norm/(antisym_norm + sym_norm)

        self.data = pd.concat((self.data, omega.rename('omega')), axis=1)

        if verbose:
            print('Output from omega:')
            pprint(self.data)


