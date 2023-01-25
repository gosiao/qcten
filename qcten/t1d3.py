import sys
import numpy as np
import scipy.linalg as la
import math
import pandas as pd


class t1d3():

    def __init__(self, cli_options, output_options, input_data):

        # general setup
        self.input_options = cli_options #you should not need this here!
        self.calc_options  = output_options
        self.input_data    = input_data
        self.flog          = self.input_options['flog']

        # global data structures
        self.t1d3          = {}
        self.t1d3_points   = []
        self.work_data = pd.DataFrame()

        # variables to be saved to the output:
        self.data_to_export= {}
        self.t1d3_cols     = []

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

    def run(self):

        # 1. assign the names of data columns corresponding to t1d3 (a vector),
        #    and to its gradient (if available);
        #
        #    this sets:
        #        self.t1d3['vx'], self.t1d3['vy'], self.t1d3['vz']
        #        and, if available: 
        #        self.t1d3['dvx_dx'], ..., self.t1d3['dvz_dz']
        #
        self.assign_t1d3_input_names()

        # 2. get the input data (pointwise) the names of data columns for what will be calculated for this vector
        #    this sets:
        #        self.t1d3_cols and self.data_to_export
        # self.assign_output_for_vector_3d()
        # - start to fill the data
        # - this sets self.t1d3_cols
        self.get_t1d3_data_points()

        # prepare output
        self.assign_output_for_vector_3d()



        if self.input_options['calc_from_tensor_1order_3d'] is not None:

            # TODO: here starts a loop over points, can be expensive!

            for arg in self.input_options['calc_from_tensor_1order_3d']:

                if (arg == 'rortex' or arg == 'omega_rortex'):
                    #self.get_t1d3_gradient()
                    self.rortex_and_shear()

                if (arg == 'norm'):
                    self.norm()

                if (arg == 'vorticity'):
                    #self.get_t1d3_gradient()
                    self.vorticity()

                if (arg == 'omega'):
                    #self.get_t1d3_gradient()
                    self.omega()

                if (arg == 'mean'):
                    self.mean()

        if self.input_options['projection_axis'] is not None:
            self.project_v_on_projection_axis()

    def assign_output_for_vector_3d(self):
        #
        if self.calc_options is not None:
            for k, v in self.calc_options.items():
                #print('output from assign_output_for_vector_3d: ', k, v)
                #omega_vorticity_z.csv {'file_type': 'csv', 'file_path': PosixPath('/home/gosia/devel/qcten.NEW/qcten/tests/t1d3_omega/omega_vorticity_z.csv'), 'file_column_names': ['x', 'y', 'z', 'omega:omega_bz', 'curlv_cdot_axis:bz_wz'], 'file_column_old_names': [None, None, None, 'omega', 'curlv_cdot_axis'], 'file_column_new_names': [None, None, None, 'omega_bz', 'bz_wz'], 'file_column_separator': ',', 'file_skiprow': None}
                #output from assign_output_for_vector_3d:  omega_vorticity_z.vti {'file_type': 'vti', 'file_path': PosixPath('/home/gosia/devel/qcten.NEW/qcten/tests/t1d3_omega/omega_vorticity_z.vti'), 'file_column_names': ['x', 'y', 'z', 'omega:omega_bz', 'curlv_cdot_axis:bz_wz'], 'file_column_old_names': [None, None, None, 'omega', 'curlv_cdot_axis'], 'file_column_new_names': [None, None, None, 'omega_bz', 'bz_wz'], 'file_column_separator': ',', 'file_skiprow': None}
            #    self.data_to_export[k] = [arg.strip().strip('[').strip(']') for arg in v['file_column_new_names'].split(',')]
                if v['file_path'] is not None:
                    fout = v['file_path']
                    fout.parent.mkdir(parents=True, exist_ok=True)
                    df = pd.DataFrame()
                    requested_cols = []
                    data_cols = []
                    for icol, col in enumerate(v['file_column_names']):
                        if ':' in col:
                            old_col = col.strip().split(':')[0]
                            new_col = col.strip().split(':')[1]
                            data_cols.append(old_col.strip())
                            requested_cols.append(new_col.strip())
                        else:
                            data_cols.append(col.strip())
                            requested_cols.append(col.strip())

                    print('MATCHCOL: ', data_cols)
                    print('MATCHCOL: ', requested_cols)

                    self.t1d3_cols = [col.replace('x', 'x').replace('y', 'y').replace('z', 'z') for col in self.t1d3_cols]

                    if all(col in self.t1d3_cols for col in data_cols):
                        print('ALLIN')
                    else:
                        for col in self.t1d3_cols:
                            if not col in data_cols:
                                print('MISSING: ', col)
                                sys.exit()

                    self.t1d3_cols = [col.replace('x', 'x').replace('y', 'y').replace('z', 'z') for col in data_cols]
#                            
                    #if not df.empty:
                    #    df = df.astype(np.float64)
                    #    df.to_csv(fout, index=False)


    def get_t1d3_gradient(self):

        '''
        
        find the gradient of t1d3 vector
        - either get it from file
        - or calculate it numerically

        '''

        if not self.input_options['use_grad_from_file']:

            # calculate gradient numerically
            # ------------------------------

            # calculate grid spacing
            self.find_spacing_uniform_grid()

            # calculate gradient numerically:
            grad_vecx = self.gradient(self.selected_vector_element(self.t1d3['vx']))
            grad_vecy = self.gradient(self.selected_vector_element(self.t1d3['vy']))
            grad_vecz = self.gradient(self.selected_vector_element(self.t1d3['vz']))

            # calculate vorticity if not available 
            # TODO: add a possibility to read it from file (since DIRAC calculates curl_j)
            curlv_x = grad_vecz[1] - grad_vecy[2]
            curlv_y = grad_vecx[2] - grad_vecz[0]
            curlv_z = grad_vecy[0] - grad_vecx[1]
            curlv_magn = np.sqrt(curlv_x*curlv_x + curlv_y*curlv_y + curlv_z*curlv_z)

        else:
            # assign grad_vecx, grad_vecy, grad_vecz to data read from file

            gx_x  = np.array([ p['dvx_dx'] for p in self.t1d3_points ], dtype=np.float64)
            gx_y  = np.array([ p['dvx_dy'] for p in self.t1d3_points ], dtype=np.float64)
            gx_z  = np.array([ p['dvx_dz'] for p in self.t1d3_points ], dtype=np.float64)

            gy_x  = np.array([ p['dvy_dx'] for p in self.t1d3_points ], dtype=np.float64)
            gy_y  = np.array([ p['dvy_dy'] for p in self.t1d3_points ], dtype=np.float64)
            gy_z  = np.array([ p['dvy_dz'] for p in self.t1d3_points ], dtype=np.float64)

            gz_x  = np.array([ p['dvz_dx'] for p in self.t1d3_points ], dtype=np.float64)
            gz_y  = np.array([ p['dvz_dy'] for p in self.t1d3_points ], dtype=np.float64)
            gz_z  = np.array([ p['dvz_dz'] for p in self.t1d3_points ], dtype=np.float64)

            grad_vecx = [gx_x, gx_y, gx_z]
            grad_vecy = [gy_x, gy_y, gy_z]
            grad_vecz = [gz_x, gz_y, gz_z]

            # calculate vorticity if not available:
            # TODO: add a possibility to read it from file (since DIRAC calculates curl_j)
            curlv_x = grad_vecz[1] - grad_vecy[2]
            curlv_y = grad_vecx[2] - grad_vecz[0]
            curlv_z = grad_vecy[0] - grad_vecx[1]
            curlv_magn = np.sqrt(curlv_x*curlv_x + curlv_y*curlv_y + curlv_z*curlv_z)




        # here starts an expensive loop over points:
        # todo: move it outside

        for i, d in enumerate(self.t1d3_points):

            # 1. first, store the gradient and the curl as global variables:

            self.t1d3_points[i]['dvx_dx'] = grad_vecx[0][i]  # dvx/dx
            self.t1d3_points[i]['dvx_dy'] = grad_vecx[1][i]  # dvx/dy
            self.t1d3_points[i]['dvx_dz'] = grad_vecx[2][i]  # dvx/dz

            self.t1d3_points[i]['dvy_dx'] = grad_vecy[0][i]  # dvy/dx
            self.t1d3_points[i]['dvy_dy'] = grad_vecy[1][i]  # dvy/dy
            self.t1d3_points[i]['dvy_dz'] = grad_vecy[2][i]  # dvy/dz

            self.t1d3_points[i]['dvz_dx'] = grad_vecz[0][i]  # dvz/dx
            self.t1d3_points[i]['dvz_dy'] = grad_vecz[1][i]  # dvz/dy
            self.t1d3_points[i]['dvz_dz'] = grad_vecz[2][i]  # dvz/dz

            # the order of elements:
            # (is the same as in Xu, Phys.Fluids 31, 095102 (2019), which we follow here)
            #
            #  xx  xy  xz       dvx/dx  dvx/dy  dvx/dz 
            #  yx  yy  yz  ->   dvy/dx  dvy/dy  dvy/dz
            #  zx  zy  zz       dvz/dx  dvz/dy  dvz/dz
            #
            full_grad_tensor = np.array([[grad_vecx[0][i], grad_vecx[1][i], grad_vecx[2][i]],
                                         [grad_vecy[0][i], grad_vecy[1][i], grad_vecy[2][i]],
                                         [grad_vecz[0][i], grad_vecz[1][i], grad_vecz[2][i]]],
                                         dtype=np.float64)

            self.t1d3_points[i]['curlv_x']    = curlv_x[i]
            self.t1d3_points[i]['curlv_y']    = curlv_y[i]
            self.t1d3_points[i]['curlv_z']    = curlv_z[i]
            self.t1d3_points[i]['curlv_magn'] = curlv_magn[i]


        pass

    def assign_t1d3_input_names(self, verbose=False):

        '''

        1. grid
        -------
        grid points are read in the following order from the input data file:

            x, y, z

        here we assign the user-chosen names of grid coordinates to 'x', 'y' and 'z'


        2. vector field
        ---------------
        vector components are read in the following order from the input data file:
        (this is the order of elements read with the --form_tensor_1order_3d keyword)

            [x, y, z], vx, vy, vz

            where vx, vy, vz are vector elements

        here we assign the user-chosen names of vector elements to 'vx', 'vy' and 'vz'


        3. the gradient of the vector field components
        ----------------------------------------------
        if the gradient data is available, then assign the user-chosen names to:

            [x, y, z],  dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz


        '''


        # columns with grid data
        args = [arg.strip().strip('[').strip(']') for arg in self.input_options['grid'].split(',')]
        self.t1d3['x'] = args[0]
        self.t1d3['y'] = args[1]
        self.t1d3['z'] = args[2]

        # columns with t1d3 data
        args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_tensor_1order_3d'].split(',')]

        self.t1d3['vx'] = args[0]
        self.t1d3['vy'] = args[1]
        self.t1d3['vz'] = args[2]

        if verbose:
            print('vector columns are assigned: vx={}, vy={}, vz={}'.format(self.t1d3['vx'],self.t1d3['vy'],self.t1d3['vz']))

        # columns with grad(t1d3) data
        if (self.input_options['form_grad_tensor_1order_3d'] is not None) and (self.input_options['use_grad_from_file']):

            args = [arg.strip().strip('[').strip(']') for arg in self.input_options['form_grad_tensor_1order_3d'].split(',')]

            # note the order of tensor elements!
            # now adapted to the order in which 'get_gj' tensor elements are exported from DIRAC in visual
            # FIXME: change the input so that you can write explicitly in the input which element is which
            # for instance --txx=... --txy=... etc.

            self.t1d3['dvx_dx'] = args[0]
            self.t1d3['dvx_dy'] = args[1]
            self.t1d3['dvx_dz'] = args[2]

            self.t1d3['dvy_dx'] = args[3]
            self.t1d3['dvy_dy'] = args[4]
            self.t1d3['dvy_dz'] = args[5]

            self.t1d3['dvz_dx'] = args[6]
            self.t1d3['dvz_dy'] = args[7]
            self.t1d3['dvz_dz'] = args[8]



    def get_t1d3_data_points(self):

        '''

        fill self.t1d3_points with the input data

        self.t1d3_points - is a list of dictionaries:

            * the length of the list = number of grid points

            * the entries of the dictionary (storing data in each grid point) assigned here:

                - grid points: x, y, z
                - vector elements: vx, vy, vz
                - gradient of vector elements (if available on input):
                    dvx_dx, dvx_dy, dvx_dz, dvy_dx, dvy_dy, dvy_dz, dvz_dx, dvz_dy, dvz_dz

        '''

        if self.t1d3_points:
            print('WARNING: t1d3_points are already assigned; will be overwritten')
            self.t1d3_points   = []

        print('checkin: type(input_data) ', type(self.input_data))
        print('checkin: cols of input_data ', self.input_data.columns)


        if (self.input_options['form_grad_tensor_1order_3d'] is not None) and (self.input_options['use_grad_from_file']):
            select = {self.t1d3['x']:'x',
                      self.t1d3['y']:'y',
                      self.t1d3['z']:'z',
                      self.t1d3['vx']:'vx',
                      self.t1d3['vy']:'vy',
                      self.t1d3['vz']:'vz',
                      self.t1d3['dvx_dx']:'dvx_dx',
                      self.t1d3['dvx_dy']:'dvx_dy',
                      self.t1d3['dvx_dz']:'dvx_dz', 
                      self.t1d3['dvy_dx']:'dvy_dx', 
                      self.t1d3['dvy_dy']:'dvy_dy', 
                      self.t1d3['dvy_dz']:'dvy_dz', 
                      self.t1d3['dvz_dx']:'dvz_dx', 
                      self.t1d3['dvz_dy']:'dvz_dy', 
                      self.t1d3['dvz_dz']:'dvz_dz'} 
        else:
            select = {self.t1d3['x']:'x',
                      self.t1d3['y']:'y',
                      self.t1d3['z']:'z',
                      self.t1d3['vx']:'vx',
                      self.t1d3['vy']:'vy',
                      self.t1d3['vz']:'vz'}

        self.work_data = self.input_data.rename(columns=select)[select.values()]
        print('checkin again: cols of input_data ', self.work_data.columns)


        #for i, r in self.input_data.iterrows():

        #    d = {}
        #    d['x']  = r[self.grid['x']]
        #    d['y']  = r[self.grid['y']]
        #    d['z']  = r[self.grid['z']]
        #    d['vx']      = r[self.t1d3['vx']]
        #    d['vy']      = r[self.t1d3['vy']]
        #    d['vz']      = r[self.t1d3['vz']]

        #    if (self.input_options['form_grad_tensor_1order_3d'] is not None) and (self.input_options['use_grad_from_file']):
        #        d['dvx_dx']  = r[self.t1d3['dvx_dx']]
        #        d['dvx_dy']  = r[self.t1d3['dvx_dy']]
        #        d['dvx_dz']  = r[self.t1d3['dvx_dz']]

        #        d['dvy_dx']  = r[self.t1d3['dvy_dx']]
        #        d['dvy_dy']  = r[self.t1d3['dvy_dy']]
        #        d['dvy_dz']  = r[self.t1d3['dvy_dz']]

        #        d['dvz_dx']  = r[self.t1d3['dvz_dx']]
        #        d['dvz_dy']  = r[self.t1d3['dvz_dy']]
        #        d['dvz_dz']  = r[self.t1d3['dvz_dz']]

        #    self.t1d3_points.append(d)

        # it might be useful to get more grid info
        # TODO: add a flag for it
        self.dim_x = len(np.unique([p['x'] for p in self.t1d3_points]))
        self.dim_y = len(np.unique([p['y'] for p in self.t1d3_points]))
        self.dim_z = len(np.unique([p['z'] for p in self.t1d3_points]))
        self.dim_cube = self.dim_x * self.dim_y * self.dim_z

        ## decide what to write to output:
        #if self.calc_options is not None:
        #    # always output the grid
        #    self.t1d3_cols.append('x')
        #    self.t1d3_cols.append('y')
        #    self.t1d3_cols.append('z')
        #    
        #    # in most detailed cases, also output the (original) vector field...
        #    if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
        #        self.t1d3_cols.append('vx')
        #        self.t1d3_cols.append('vy')
        #        self.t1d3_cols.append('vz')
        #    
        #    # ...and its gradient
        #    if self.input_options['fout_select'] == 'all':
        #        self.t1d3_cols.append('dvx_dx')
        #        self.t1d3_cols.append('dvx_dy')
        #        self.t1d3_cols.append('dvx_dz')
        #        self.t1d3_cols.append('dvy_dx')
        #        self.t1d3_cols.append('dvy_dy')
        #        self.t1d3_cols.append('dvy_dz')
        #        self.t1d3_cols.append('dvz_dx')
        #        self.t1d3_cols.append('dvz_dy')
        #        self.t1d3_cols.append('dvz_dz')

        ##else: # if self.input_options['data_out'] is None:
        ##    for col in self.data_to_export:
        ##        self.t1d3_cols.append(col)



    def find_spacing_uniform_grid(self):

        '''
        find spacing between grid points in x, y, z directions
        we assume a regular grid
        '''

        x0 = self.t1d3_points[0]['x']
        y0 = self.t1d3_points[0]['y']
        z0 = self.t1d3_points[0]['z']

        dx = abs(x0)
        dy = abs(y0)
        dz = abs(z0)

        for p in self.t1d3_points[1:]:
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

        with open(self.flog, 'a') as f:
            f.write('Grid spacing: dx, dy, dz = {}, {}, {}\n'.format(dx, dy, dz))


    def find_data_in_point_plusminus(self, d, f):

        result = {}

        x0 = d['x']
        y0 = d['y']
        z0 = d['z']

        reltol=1e-09
        abstol=1e-07

        for p in self.t1d3_points:
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


        for p in self.t1d3_points:
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

        if (x == min(p['x'] for p in self.t1d3_points)) or (x == max(p['x'] for p in self.t1d3_points)) or \
           (y == min(p['y'] for p in self.t1d3_points)) or (y == max(p['y'] for p in self.t1d3_points)) or \
           (z == min(p['z'] for p in self.t1d3_points)) or (z == max(p['z'] for p in self.t1d3_points)):
            #print('point on the border')

            which_border = [x == min(p['x'] for p in self.t1d3_points),
                            x == max(p['x'] for p in self.t1d3_points),
                            y == min(p['y'] for p in self.t1d3_points),
                            y == max(p['y'] for p in self.t1d3_points),
                            z == min(p['z'] for p in self.t1d3_points),
                            z == max(p['z'] for p in self.t1d3_points)]

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

        for i, p in enumerate(self.t1d3_points):
            grad_f = self.gradient_from_finite_elements_value_in_point(p, f)
            self.t1d3_points[i]['grad_x'] = grad_f[0]
            self.t1d3_points[i]['grad_y'] = grad_f[1]
            self.t1d3_points[i]['grad_z'] = grad_f[2]

        return grad_f


    def test_grad_numpy(self, f):

        '''
        be careful, this works OK if f has at most quadratix dependence on r
        otherwise the approximation is too hars (see jupyter notebook in test_gradient)
        TODO: requires more testing
        '''

        f_values = np.array([p[f] for p in self.t1d3_points], dtype=np.float64)
        f_array  = f_values.reshape((self.dim_x, self.dim_y, self.dim_z))

        grad_f   = np.gradient(f_array, self.dx, self.dy, self.dz, edge_order=2)
        grad_f_x = grad_f[0].reshape((self.dim_cube))
        grad_f_y = grad_f[1].reshape((self.dim_cube))
        grad_f_z = grad_f[2].reshape((self.dim_cube))


        return [grad_f_x, grad_f_y, grad_f_z]


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



    def gradient(self, f):
        '''
        calculate the gradient of f
        f is a selected element of a vector
        '''

        if self.input_options['calc_from_tensor_1order_3d_calc_grad'] == 'numpy':
            grad = self.test_grad_numpy(f)
        elif self.input_options['calc_from_tensor_1order_3d_calc_grad'] == 'finite_elements':
            grad = self.gradient_from_finite_elements(f)
        else:
            print('warning: wrong choice of calc_from_tensor_1order_3d_calc_grad')
            grad=None

        return grad


    def test_eigendecomposition(self, eig_val, eig_vec, mat):

        epsilon=1.0e-8

        for i in range(3):

            e_vec = eig_vec[:, i]
            e_val = eig_val[i]

            if isinstance(e_vec, complex) and e_vec.imag == 0:
                e_vec=e_vec.real
            if isinstance(e_val, complex) and e_val.imag == 0:
                e_val = e_val.real

            l = np.dot(mat, e_vec)
            r = e_val*e_vec
            #if not np.allclose(l, r, atol=epsilon):
            #    raise Exception('Error in eigendecomposition: A*v != lambda*v')
            diff = abs(l - r)
            for j, d in enumerate(diff):
                if d > epsilon:
                    raise Exception('Error in eigendecomposition: A*v = {} while lambda*v = {}'.format(l[j], r[j]))



    def tensor_eigendecomposition(self, point_index, fullgradtensor):

        eigenvalues, eigenvectors = la.eig(fullgradtensor)

        # eigenvectors are in columns of "eigenvectors"
        # corresponding eigenvalues are in "eigenvalues" (in the same order)
        e_vec1 = eigenvectors[:, 0]
        e_vec2 = eigenvectors[:, 1]
        e_vec3 = eigenvectors[:, 2]

        # double check: test the eigendecomposition:
        # todo: refactor using decorators
        test_eig=True
        if test_eig:
            self.test_eigendecomposition(eigenvalues, eigenvectors, fullgradtensor)

        e_val = []
        number_complex_eigenvalues = 0

        for e in eigenvalues:
            if isinstance(e, complex):
                if (e.imag==0.0):
                    # TODO: might be better to use the threshold to cut out very small values
                    e_val.append(e.real)
                    #print('WARNING: small imaginary part of eigenvalue: {}'.format(e))
                else:
                    e_val.append(e)
                    number_complex_eigenvalues += 1
            else:
                e_val.append(e)

        self.t1d3_points[point_index]['dv_eig_val1']  = e_val[0]
        self.t1d3_points[point_index]['dv_eig_val2']  = e_val[1]
        self.t1d3_points[point_index]['dv_eig_val3']  = e_val[2]

        self.t1d3_points[point_index]['dv_eig_vec1_x']  = e_vec1[0]
        self.t1d3_points[point_index]['dv_eig_vec1_y']  = e_vec1[1]
        self.t1d3_points[point_index]['dv_eig_vec1_z']  = e_vec1[2]

        self.t1d3_points[point_index]['dv_eig_vec2_x']  = e_vec2[0]
        self.t1d3_points[point_index]['dv_eig_vec2_y']  = e_vec2[1]
        self.t1d3_points[point_index]['dv_eig_vec2_z']  = e_vec2[2]

        self.t1d3_points[point_index]['dv_eig_vec3_x']  = e_vec3[0]
        self.t1d3_points[point_index]['dv_eig_vec3_y']  = e_vec3[1]
        self.t1d3_points[point_index]['dv_eig_vec3_z']  = e_vec3[2]

        self.t1d3_points[point_index]['number_complex_eigenvalues']  = number_complex_eigenvalues



    def rortex_in_point(self, point_index, point_data):


        if (point_data['number_complex_eigenvalues'] == 2):

            eigval_complex=[]
            eigvec_complex=[]

            for iv, v in enumerate([point_data['dv_eig_val1'], point_data['dv_eig_val2'], point_data['dv_eig_val3']]):

                # TODO: check how robust this is

                if isinstance(v, complex):
                    eigval_complex.append(v)
                    if (iv == 0):
                        eigvec_complex.append([point_data['dv_eig_vec1_x'], point_data['dv_eig_vec1_y'], point_data['dv_eig_vec1_z']])
                    if (iv == 1):
                        eigvec_complex.append([point_data['dv_eig_vec2_x'], point_data['dv_eig_vec2_y'], point_data['dv_eig_vec2_z']])
                    if (iv == 2):
                        eigvec_complex.append([point_data['dv_eig_vec3_x'], point_data['dv_eig_vec3_y'], point_data['dv_eig_vec3_z']])

                else:
                    eigval_real = v
                    if (iv == 0):
                        eigvec_real   = [point_data['dv_eig_vec1_x'], point_data['dv_eig_vec1_y'], point_data['dv_eig_vec1_z']]
                    if (iv == 1):
                        eigvec_real   = [point_data['dv_eig_vec2_x'], point_data['dv_eig_vec2_y'], point_data['dv_eig_vec2_z']]
                    if (iv == 2):
                        eigvec_real   = [point_data['dv_eig_vec3_x'], point_data['dv_eig_vec3_y'], point_data['dv_eig_vec3_z']]

                    # make sure that all components of this eigenvector are real (gradient tensor has real entries)
                    for e in eigvec_real:
                        if isinstance(e, complex) and e.imag != 0.0:
                            raise Exception('the real eigenvector has complex elements, check the eigendecomposition!')
                    eigvec_real = [e.real for e in eigvec_real]

            # calculate the normalized real eigenvector corresponding to the real eigenvalue:
            eigvec_real_magn       = np.sqrt(eigvec_real[0]**2 + eigvec_real[1]**2 + eigvec_real[2]**2)
            eigvec_real_normalized = [e/eigvec_real_magn for e in eigvec_real]

            # rename variables as in Xu et al. Phys Fluids 31, 095102 (2019)
            lambda_ci = abs(eigval_complex[0].imag)
            lambda_cr =     eigval_complex[0].real
            lambda_r  =     eigval_real

            # finally calculate rortex vector
            # these are eqs. 33 and 34 in Xu et al. Phys Fluids 31, 095102 (2019)

            # step 1: eq. 30
            omega_cdot_r = self.t1d3_points[point_index]['curlv_x']*eigvec_real_normalized[0] \
                         + self.t1d3_points[point_index]['curlv_y']*eigvec_real_normalized[1] \
                         + self.t1d3_points[point_index]['curlv_z']*eigvec_real_normalized[2]
            sign_changed = False
            if (omega_cdot_r < 0.0):
                # eq. 30 in Xu et al. Phys Fluids 31, 095102 (2019)
                omega_cdot_r = - omega_cdot_r
                sign_changed = True

            # step 2: eq. 33
            val = omega_cdot_r**2 - 4*(lambda_ci**2)
            if (val < 0.0):
                raise Exception('WARNING: omega_cdot_r**2 - 4*(lambda_ci**2) < 0 and equals {}'.format(val))

            rortex_magnitude = omega_cdot_r - np.sqrt(val)

            self.t1d3_points[point_index]['rortex_magnitude'] = rortex_magnitude

            # step 3: eq. 34
            if sign_changed:
                factor = -1.0
            else:
                factor = 1.0
            self.t1d3_points[point_index]['rortex_vector_x'] = factor * rortex_magnitude * eigvec_real_normalized[0]
            self.t1d3_points[point_index]['rortex_vector_y'] = factor * rortex_magnitude * eigvec_real_normalized[1]
            self.t1d3_points[point_index]['rortex_vector_z'] = factor * rortex_magnitude * eigvec_real_normalized[2]

            if self.input_options['projection_axis'] is not None:
                # project rortex vector on a selected axis
                # it is useful for plots (coloring)
                rortex_cdot_axis = self.t1d3_points[point_index]['rortex_vector_x']*self.input_options['projection_axis'][0] \
                                 + self.t1d3_points[point_index]['rortex_vector_y']*self.input_options['projection_axis'][1] \
                                 + self.t1d3_points[point_index]['rortex_vector_z']*self.input_options['projection_axis'][2]
                self.t1d3_points[point_index]['rortex_cdot_axis'] = rortex_cdot_axis


            ## rortex in tensor form (eq. 3 in Xu et al. Phys Fluids 31, 095102 (2019)):
            phi = 0.5*rortex_magnitude
            self.t1d3_points[point_index]['rortex_tensor_xx'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_xy'] = -phi
            self.t1d3_points[point_index]['rortex_tensor_xz'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_yx'] =  phi
            self.t1d3_points[point_index]['rortex_tensor_yy'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_yz'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_zx'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_zy'] =  0.0
            self.t1d3_points[point_index]['rortex_tensor_zz'] =  0.0

            if ('omega_rortex' in self.input_options['calc_from_tensor_1order_3d']):
                # we use Eq. 36 from Xu et al. Phys Fluids 31, 095102 (2019)

                omega_rortex = omega_cdot_r**2 / (2*(omega_cdot_r**2 - 2*(lambda_ci**2) + 2*(lambda_cr**2) + lambda_r**2))

                grad_vec_magn = (self.t1d3_points[point_index]['dvx_dx']**2
                               + self.t1d3_points[point_index]['dvx_dy']**2
                               + self.t1d3_points[point_index]['dvx_dz']**2
                               + self.t1d3_points[point_index]['dvy_dx']**2
                               + self.t1d3_points[point_index]['dvy_dy']**2
                               + self.t1d3_points[point_index]['dvy_dz']**2
                               + self.t1d3_points[point_index]['dvz_dx']**2
                               + self.t1d3_points[point_index]['dvz_dy']**2
                               + self.t1d3_points[point_index]['dvz_dz']**2)

                self.t1d3_points[point_index]['omega_rortex'] = omega_rortex
                self.t1d3_cols.append('omega_rortex')

        else:
            # TODO: what's best to do here?
            # first let's assign these values to None
            # in final analysis they will be set to 0
            self.t1d3_points[point_index]['rortex_vector_x'] = None
            self.t1d3_points[point_index]['rortex_vector_y'] = None
            self.t1d3_points[point_index]['rortex_vector_z'] = None
            self.t1d3_points[point_index]['rortex_magnitude'] = None

            self.t1d3_points[point_index]['rortex_tensor_xx'] = None
            self.t1d3_points[point_index]['rortex_tensor_xy'] = None
            self.t1d3_points[point_index]['rortex_tensor_xz'] = None
            self.t1d3_points[point_index]['rortex_tensor_yx'] = None
            self.t1d3_points[point_index]['rortex_tensor_yy'] = None
            self.t1d3_points[point_index]['rortex_tensor_yz'] = None
            self.t1d3_points[point_index]['rortex_tensor_zx'] = None
            self.t1d3_points[point_index]['rortex_tensor_zy'] = None
            self.t1d3_points[point_index]['rortex_tensor_zz'] = None

            self.t1d3_points[point_index]['omega_rortex'] = None
            self.t1d3_points[point_index]['omega_rortex2'] = None

            if self.input_options['projection_axis'] is not None:
                self.t1d3_points[point_index]['rortex_cdot_axis'] = None


        # decide what to save on the output file:
        self.t1d3_cols.append('rortex_vector_x')
        self.t1d3_cols.append('rortex_vector_y')
        self.t1d3_cols.append('rortex_vector_z')
        self.t1d3_cols.append('rortex_magnitude')
        if self.input_options['projection_axis'] is not None:
            self.t1d3_cols.append('rortex_cdot_axis')


        if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):

            self.t1d3_cols.append('rortex_tensor_xx')
            self.t1d3_cols.append('rortex_tensor_xy')
            self.t1d3_cols.append('rortex_tensor_xz')
            self.t1d3_cols.append('rortex_tensor_yx')
            self.t1d3_cols.append('rortex_tensor_yy')
            self.t1d3_cols.append('rortex_tensor_yz')
            self.t1d3_cols.append('rortex_tensor_zx')
            self.t1d3_cols.append('rortex_tensor_zy')
            self.t1d3_cols.append('rortex_tensor_zz')

        if self.input_options['fout_select'] == 'all':

            self.t1d3_cols.append('number_complex_eigenvalues')

            self.t1d3_cols.append('dv_eig_val1')
            self.t1d3_cols.append('dv_eig_val2')
            self.t1d3_cols.append('dv_eig_val3')

            self.t1d3_cols.append('dv_eig_vec1_x')
            self.t1d3_cols.append('dv_eig_vec1_y')
            self.t1d3_cols.append('dv_eig_vec1_z')
            self.t1d3_cols.append('dv_eig_vec2_x')
            self.t1d3_cols.append('dv_eig_vec2_y')
            self.t1d3_cols.append('dv_eig_vec2_z')
            self.t1d3_cols.append('dv_eig_vec3_x')
            self.t1d3_cols.append('dv_eig_vec3_y')
            self.t1d3_cols.append('dv_eig_vec3_z')


    def rortex_and_shear(self):

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
        
        '''

        # calculate vorticity
        self.vorticity()


        for i, d in enumerate(self.t1d3_points):

            # 1. construct the gradient tensor
            full_grad_tensor = np.array([ [d['dvx_dx'], d['dvx_dy'], d['dvx_dz']],
                                          [d['dvy_dx'], d['dvy_dy'], d['dvy_dz']],
                                          [d['dvz_dx'], d['dvz_dy'], d['dvz_dz']] ],
                                         dtype=np.float64)

            # 2. do the eigendecomposition of the gradient tensor:
            self.tensor_eigendecomposition(i, full_grad_tensor)

            # 3. calculate rortex
            self.rortex_in_point(i, d)

            # 4. calculate shear:
            # TODO: fix this!
            #self.shear_in_point(i, d, full_grad_tensor)



    def shear_in_point(self, point_index, point_data, full_grad_tensor):

        '''
        shear is calculated from Eq. (2) in Xu et al. Phys Fluids 31, 095102 (2019);
        shear is a tensor calculated as:
        'velocity gradient tensor - rortex tensor'

        however, the velocity gradient tensor needs to be expressed in Shur form;
        here we will construct the Shur form of the velocity gradient tensor
        from eq. 24 in that paper
        '''

        if (point_data['number_complex_eigenvalues'] == 2):
            #full_grad_tensor_shur = construct_shur_form_gradient_tensor(full_grad_tensor)
            phi = 0.5*self.t1d3_points[point_index]['rortex_magnitude']
            self.t1d3_points[point_index]['shear_tensor_xx'] = full_grad_tensor[0,0]
            self.t1d3_points[point_index]['shear_tensor_xy'] = 0.0
            self.t1d3_points[point_index]['shear_tensor_xz'] = 0.0
            self.t1d3_points[point_index]['shear_tensor_yx'] = full_grad_tensor[1,0] - phi
            self.t1d3_points[point_index]['shear_tensor_yy'] = full_grad_tensor[1,1]
            self.t1d3_points[point_index]['shear_tensor_yz'] = 0.0
            self.t1d3_points[point_index]['shear_tensor_zx'] = full_grad_tensor[2,0]
            self.t1d3_points[point_index]['shear_tensor_zy'] = full_grad_tensor[2,1]
            self.t1d3_points[point_index]['shear_tensor_zz'] = full_grad_tensor[2,2]
        else:
            self.t1d3_points[point_index]['shear_tensor_xx'] = None
            self.t1d3_points[point_index]['shear_tensor_xy'] = None
            self.t1d3_points[point_index]['shear_tensor_xz'] = None
            self.t1d3_points[point_index]['shear_tensor_yx'] = None
            self.t1d3_points[point_index]['shear_tensor_yy'] = None
            self.t1d3_points[point_index]['shear_tensor_yz'] = None
            self.t1d3_points[point_index]['shear_tensor_zx'] = None
            self.t1d3_points[point_index]['shear_tensor_zy'] = None
            self.t1d3_points[point_index]['shear_tensor_zz'] = None

        if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
            self.t1d3_cols.append('shear_tensor_xx')
            self.t1d3_cols.append('shear_tensor_xy')
            self.t1d3_cols.append('shear_tensor_xz')
            self.t1d3_cols.append('shear_tensor_yx')
            self.t1d3_cols.append('shear_tensor_yy')
            self.t1d3_cols.append('shear_tensor_yz')
            self.t1d3_cols.append('shear_tensor_zx')
            self.t1d3_cols.append('shear_tensor_zy')
            self.t1d3_cols.append('shear_tensor_zz')



    def construct_shur_form_gradient_tensor(t):
        alpha=0

    def v1_cdot_v2(self, v1, v2):
        '''
        scalar product of two vectors
        '''

        v1_cdot_v2  = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

        return v1_cdot_v2


    def project_v_on_projection_axis(self):

        args = [arg.strip().strip('[').strip(']') for arg in self.input_options['projection_axis'].split(',')]
        self.projection_axis['x'] = int(args[0])
        self.projection_axis['y'] = int(args[1])
        self.projection_axis['z'] = int(args[2])


        for i, d in enumerate(self.t1d3_points):
            v_cdot_axis = self.t1d3_points[i]['vx']*self.projection_axis['x'] \
                        + self.t1d3_points[i]['vy']*self.projection_axis['y'] \
                        + self.t1d3_points[i]['vz']*self.projection_axis['z']

            self.t1d3_points[i]['v_cdot_axis'] = v_cdot_axis

        #self.t1d3_cols.append('v_cdot_axis')



    def norm(self):
        '''
        calculate the norm of a 3d vector

        For vector v, whose elements are v_i:

         norm = sqrt(sum_{i} (v_{i})**2)

        '''

        # calculate

        for i, d in enumerate(self.t1d3_points):

            norm = (self.t1d3_points[i]['vx'])**2 \
                 + (self.t1d3_points[i]['vy'])**2 \
                 + (self.t1d3_points[i]['vz'])**2

            self.t1d3_points[i]['v_norm'] = np.sqrt(norm)

        # save on output

        self.t1d3_cols.append('v_norm')

        if ((self.input_options['fout_select'] == 'all') or (self.input_options['fout_select'] == 'selected')):
            self.t1d3_cols.append('vx')
            self.t1d3_cols.append('vy')
            self.t1d3_cols.append('vz')



    def mean(self):
        '''
        calculate the mean of vector elements

        For vector v, whose elements are v_i:

         mean = (sum_{i} v_{i})/3

        '''

        for i, d in enumerate(self.t1d3_points):

            mean = (self.t1d3_points[i]['vx']
                 +  self.t1d3_points[i]['vy']
                 +  self.t1d3_points[i]['vz'])/3.0

            self.t1d3_points[i]['mean'] = mean

        self.t1d3_cols.append('mean')



    def vorticity(self):

        '''
        calculate the curl of the vector:
        
        For vector v = (vx, vy, vz):

        w = \nabla \times v

        wx = d(vz)/dy - d(vy)/dz
        wy = d(vx)/dz - d(vz)/dx
        wz = d(vy)/dx - d(vx)/dy

        '''

        #todo: call need_gradient + refactor

        if not self.input_options['use_grad_from_file']:
            print('error! todo: gradient data not available on input')
        else:

            for i, d in enumerate(self.t1d3_points):

                curlv_x = d['dvz_dy'] - d['dvy_dz']
                curlv_y = d['dvx_dz'] - d['dvz_dx']
                curlv_z = d['dvy_dx'] - d['dvx_dy']
                curlv_magnitude = np.sqrt(curlv_x**2 + curlv_y**2 + curlv_z**2)

                self.t1d3_points[i]['curlv_x'] = curlv_x
                self.t1d3_points[i]['curlv_y'] = curlv_y
                self.t1d3_points[i]['curlv_z'] = curlv_z
                self.t1d3_points[i]['curlv_magnitude'] = curlv_magnitude

                if self.input_options['projection_axis'] is not None:
                    args = [arg.strip().strip('[').strip(']') for arg in self.input_options['projection_axis'].split(',')]
                    self.projection_axis['x'] = int(args[0])
                    self.projection_axis['y'] = int(args[1])
                    self.projection_axis['z'] = int(args[2])

                    curlv_cdot_axis = self.t1d3_points[i]['curlv_x']*self.projection_axis['x'] \
                                    + self.t1d3_points[i]['curlv_y']*self.projection_axis['y'] \
                                    + self.t1d3_points[i]['curlv_z']*self.projection_axis['z']

                    self.t1d3_points[i]['curlv_cdot_axis'] = curlv_cdot_axis


            self.t1d3_cols.append('curlv_x')
            self.t1d3_cols.append('curlv_y')
            self.t1d3_cols.append('curlv_z')
            self.t1d3_cols.append('curlv_magnitude')
            if self.input_options['projection_axis'] is not None:
                self.t1d3_cols.append('curlv_cdot_axis')

            if (self.input_options['fout_select'] == 'all'):
            #if (self.input_options['fout_select'] == 'selected'):
                self.t1d3_cols.append('dvx_dx')
                self.t1d3_cols.append('dvx_dy')
                self.t1d3_cols.append('dvx_dz')
                self.t1d3_cols.append('dvy_dx')
                self.t1d3_cols.append('dvy_dy')
                self.t1d3_cols.append('dvy_dz')
                self.t1d3_cols.append('dvz_dx')
                self.t1d3_cols.append('dvz_dy')
                self.t1d3_cols.append('dvz_dz')



    def omega(self):

        '''
        calculate Omega as in "Omega vortex identification method"

        e.g. Eq. 12 in Liu et. al, Journal of Hydrodynamics, 31, 205, 2019

        1. we need: gradient of a vector field: \nabla v
        2. the symmetric (S) and antisymmetric (A) parts of \nabla v
        3. the Frobenius norms of these parts, squared, |S|^2 and |A|^2, respectively
        4. Omega = |A|^2 / (|A|^2 + |S|^2)

        '''

        #todo: call need_gradient + refactor

        if not self.input_options['use_grad_from_file']:
            print('error! todo: gradient data not available on input')
        else:

            for i, d in enumerate(self.t1d3_points):

                full_grad_tensor = np.array([[d['dvx_dx'], d['dvx_dy'], d['dvx_dz']],
                                             [d['dvy_dx'], d['dvy_dy'], d['dvy_dz']],
                                             [d['dvz_dx'], d['dvz_dy'], d['dvz_dz']]],
                                             dtype=np.float64)

                sym_part      = self.get_sym_part_of_t2d3(full_grad_tensor)
                antisym_part  = self.get_antisym_part_of_t2d3(full_grad_tensor)

                sym_norm     = self.frobenius_norm_squared(sym_part)
                antisym_norm = self.frobenius_norm_squared(antisym_part)

                omega = antisym_norm/(antisym_norm + sym_norm)
                self.t1d3_points[i]['omega'] = omega


            self.t1d3_cols.append('omega')




    def get_sym_part_of_t2d3(self, m):
        '''
        m is np.array(3,3)
        '''
        s = (m + m.T)/2.0
        return s


    def get_antisym_part_of_t2d3(self, m):
        '''
        m is np.array(3,3)
        '''
        a = (m - m.T)/2.0
        return a


    def frobenius_norm_squared(self, m):

        n = 0
        for i in range(3):
            for j in range(3):
                n += m[i,j]**2

        return n


