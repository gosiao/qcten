import sys
import numpy as np
import scipy.linalg as la
import math


class t0d3():

    """
    This class holds settings and operations
    on tensors of rank 0 (scalars) in 3D space.

    @author:       Gosia Olejniczak
    @contact:      gosia.olejniczak@gmail.com
    """


    def __init__(self, input_options, grid, input_data):

        # general setup
        self.input_options = input_options
        self.grid    = grid
        self.input_data    = input_data
        self.flog          = input_options['flog']

        # global data structures 
        self.t0d3          = {}
        self.t0d3_points   = []

        # variables to be saved to the output:
        self.data_to_export= []
        self.t0d3_cols     = []

        self.tensor_0order_3d_is_assigned = False


    def run(self):

        print("ERROR: operations on t0d3 not available in this version")
        sys.exit()


        # prepare
        self.assign_tensor_0order_3d()
        self.assign_output_for_tensor_0order_3d()
        self.get_tensor_0order_3d_data_points()

        # work
        if self.input_options['calc_from_tensor_0order_3d'] is not None:

            for arg in self.input_options['calc_from_tensor_0order_3d']:

                if (arg == 'gradient'):
                    self.gradient()

        # prepare output
        self.prepare_output()


    def prepare_output(self):
        pass

    def assign_output_for_tensor_0order_3d(self):
        output=[]
        if self.input_options['data_out'] is not None:
            self.data_to_export = [arg.strip().strip('[').strip(']') for arg in self.input_options['data_out'].split(',')]
            #for data in self.t0d3_cols:
            #    if data in self.data_to_export:


    def assign_tensor_0order_3d(self):

        if ('form_tensor_0order_3d' in self.input_options) and (self.input_options['form_tensor_0order_3d'] is not None):
            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_tensor_0order_3d'].split(',')]
            
            self.t0d3['s'] = args[0]
            self.tensor_0order_3d_is_assigned = True

        if ('form_grad_0tensor_3d' in self.input_options) and (self.input_options['form_grad_0tensor_3d'] is not None) and (self.input_options['use_grad_from_file']):

            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_grad_0tensor_3d'].split(',')]

            self.t1d3['ds_dx'] = args[0]
            self.t1d3['ds_dy'] = args[1]
            self.t1d3['ds_dz'] = args[2]


    def get_tensor_0order_3d_data_points(self):

        if not self.tensor_0order_3d_is_assigned:
            sys.exit('TENSOR NOT ASSIGNED')

        for i, r in self.input_data.iterrows():

            d = {}

            d['grid_x']  = r[self.grid['grid_x']]
            d['grid_y']  = r[self.grid['grid_y']]
            d['grid_z']  = r[self.grid['grid_z']]

            d['s'] = r[self.t0d3['s']]

            if ('form_grad_tensor_1order_3d' in self.input_options) and (self.input_options['form_grad_tensor_1order_3d'] is not None) and (self.input_options['use_grad_from_file']):

                d['dvx_dx']  = r[self.t1d3['dvx_dx']]
                d['dvx_dy']  = r[self.t1d3['dvx_dy']]
                d['dvx_dz']  = r[self.t1d3['dvx_dz']]

                d['dvy_dx']  = r[self.t1d3['dvy_dx']]
                d['dvy_dy']  = r[self.t1d3['dvy_dy']]
                d['dvy_dz']  = r[self.t1d3['dvy_dz']]

                d['dvz_dx']  = r[self.t1d3['dvz_dx']]
                d['dvz_dy']  = r[self.t1d3['dvz_dy']]
                d['dvz_dz']  = r[self.t1d3['dvz_dz']]

            self.t0d3_points.append(d)

        # decide what data will be written to the output file, if the user did not specify that:
        if self.input_options['data_out'] is None:

            self.t0d3_cols.append('grid_x')
            self.t0d3_cols.append('grid_y')
            self.t0d3_cols.append('grid_z')
            
            if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
            
                self.t0d3_cols.append('s')
            
            if self.input_options['fout_select'] == 'all':
            
                self.t0d3_cols.append('s')

        else: # if self.input_options['data_out'] is None:
            for col in self.data_to_export:
                self.t0d3_cols.append(col)




    def gradient(self, f):
        '''
        calculate gradient of f
        f is an 'original' element name
        FIXME testthis
        '''

        tensor_elements         = dict(zip(self.t0d3.values(), self.t0d3.keys()))
        selected_tensor_element = tensor_elements[f]

        self.gradient_from_finite_elements(selected_tensor_element)

        #self.t0d3_points[i]['gradient'] = result
        #print('tensor element: ', selected_tensor_element)




