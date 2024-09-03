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

            if (arg == 'invariant1'):
                self.invariant1(verbose)

            if (arg == 'invariant2'):
                self.invariant2(verbose)

            if (arg == 'invariant3'):
                self.invariant3(verbose)


                #if (arg == 'rortex_tensor_combined'):
                #    self.rortex_tensor_combined()

                #if (arg == 'omega_rortex_tensor_combined'):
                #    self.omega_rortex_tensor_combined()




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





    def rortex_tensor_combined(self):
        '''
        '''
        pass


    def invariants_in_point(self, m, verbose):

        if verbose:
            print('invariants_in_point: entering m ' , type(m))
            print('number_complex_eigenvalues = ', m['number_complex_eigenvalues'], type(m['number_complex_eigenvalues']))
            print('real_eigval_ind = ', m['real_eigval_ind'])
            print('eig_pair_0 = ', m['eig_pair_0'], type(m['eig_pair_0']))
            print('eig_pair_1 = ', m['eig_pair_1'], type(m['eig_pair_1']))
            print('eig_pair_2 = ', m['eig_pair_2'], type(m['eig_pair_2']))

        res = {}

        inv1 = m['eig_pair_0'][0] + m['eig_pair_1'][0] + m['eig_pair_2'][0]
        res['inv1'] = complex_to_real(inv1)
        #res['inv1'] = inv1

        inv3 = m['eig_pair_0'][0] * m['eig_pair_1'][0] * m['eig_pair_2'][0]
        #res['inv3'] = complex_to_real(inv3)
        #FIXME
        res['inv3'] = np.nan

        inv2 = m['eig_pair_0'][0] * m['eig_pair_1'][0] \
             + m['eig_pair_0'][1] * m['eig_pair_2'][0] \
             + m['eig_pair_2'][0] * m['eig_pair_2'][0]
        #res['inv2'] = complex_to_real(inv2)
        #FIXME
        res['inv2'] = np.nan

        return res


    
    def invariant1(self, verbose):
        '''
        Here we calculate the first principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I1 = tr(T) = l1 + l2 + l3

        '''

        data = self.data[global_data.cols_to_use["t2d3"]]
        data["t2d3_as_npndarray"] = data.apply(lambda row: row.to_numpy().reshape((3,3)), axis=1)
        tmp = data.apply(lambda x: tensor_eigendecomposition(x["t2d3_as_npndarray"]), axis=1)
        tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)
        tmp2 = tmp.apply(lambda x: self.invariants_in_point(x, verbose=verbose), axis=1)
        tmp2 = pd.DataFrame(tmp2.to_list(), index=tmp2.index)

        self.data['t2d3_invariant1'] = tmp2['inv1']

        print('Output from invariant1:')
        pprint(self.data)
        if verbose:
            print('Output from invariant1:')
            pprint(self.data)


    def invariant2(self, verbose):
        '''
        Here we calculate the second principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I2 = 0.5*{ [tr(T)]^2 - tr(T^2) } 
               = l1*l2 + l2*l3 + l1*l3

        '''

        data = self.data[global_data.cols_to_use["t2d3"]]
        data["t2d3_as_npndarray"] = data.apply(lambda row: row.to_numpy().reshape((3,3)), axis=1)
        tmp = data.apply(lambda x: tensor_eigendecomposition(x["t2d3_as_npndarray"]), axis=1)
        tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)
        tmp2 = tmp.apply(lambda x: self.invariants_in_point(x, verbose=verbose), axis=1)
        tmp2 = pd.DataFrame(tmp2.to_list(), index=tmp2.index)

        self.data['t2d3_invariant2'] = tmp2['inv2']

        print('Output from invariant2:')
        pprint(self.data)
        if verbose:
            print('Output from invariant2:')
            pprint(self.data)


    def invariant3(self, verbose):
        '''
        Here we calculate the third principal invariant of the second-order tensor.

        For tensor T, whose eigenvalues are l1, l2, l3:

            I3 = det(T) = l1*l2*l3

        '''

        data = self.data[global_data.cols_to_use["t2d3"]]
        data["t2d3_as_npndarray"] = data.apply(lambda row: row.to_numpy().reshape((3,3)), axis=1)
        tmp = data.apply(lambda x: tensor_eigendecomposition(x["t2d3_as_npndarray"]), axis=1)
        tmp = pd.DataFrame(tmp.tolist(), index=tmp.index)
        tmp2 = tmp.apply(lambda x: self.invariants_in_point(x, verbose=verbose), axis=1)
        tmp2 = pd.DataFrame(tmp2.to_list(), index=tmp2.index)

        self.data['t2d3_invariant3'] = tmp2['inv3']

        print('Output from invariant3:')
        pprint(self.data)
        if verbose:
            print('Output from invariant3:')
            pprint(self.data)



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


