
class global_data:

    """
    Functionalities covered:
    * A list of available functions for a selected type of input data;
      these are also the accepted labels for output data
    * A list of functions with special requirements
    * A list of types of grids
    """

    # lists of available functions
    # ============================

    # input is t0d3 (zero-rank tensor field (= scalar field) in 3D)
    all_fun_t0d3 = ['gradient']

    # input is t1d3 (first-rank tensor field (= vector field) in 3D)
    all_fun_t1d3 = ['rortex',
                    'omega_rortex',
                    'norm',
                    'mean',
                    'vorticity',
                    'omega',
                    'curlv_cdot_axis',
                    'rortex_cdot_axis']

    # input is t2d3 (second-rank tensor field in 3D)
    all_fun_t2d3 = ['trace',
                    'isotropic',
                    'deviator',
                    'antisymmetric',
                    'deviator_anisotropy',
                    'rortex_tensor_combined',
                    'omega_rortex_tensor_combined',
                    'invariant1',
                    'invariant2',
                    'invariant3']

    # functions that require the gradient of a vector
    # ===============================================
    fun_t1d3_req_grad = ['vorticity',
                         'rortex',
                         'omega_rortex',
                         'rortex_cdot_axis',
                         'omega']

    fun_t2d3_req_grad = []

    # grid types
    # ==========
    grid_types = ['uniform_rectilinear']

    # input/output data - other naming conventions
    # ============================================
    # x, y, z              - grid
    # t0                   - scalar field ("t0d3")
    # t1, t2, t3           - components of a vector field ("t1d3")
    # t11, t12, t13, ...   - components of a tensor field of rank 2 ("t2d3")

    grid_cols_to_use = {'rectilinear_3d':['x','y','z']}

    cols_to_use = {'t0d3':['t0'],
                   't1d3':['t1','t2','t3'],
                   't2d3':['t11','t12','t13','t21','t22','t23','t31','t32','t33'],
                   }

    grad_cols_to_use = {'t0d3':['t0_dx','t0_dy','t0_dz'],
                        't1d3':['t1_dx','t1_dy','t1_dz',\
                                't2_dx','t2_dy','t2_dz',\
                                't3_dx','t3_dy','t3_dz'],
                        't2d3':['t11_dx','t11_dy','t11_dz',\
                                't12_dx','t12_dy','t12_dz',\
                                't13_dx','t13_dy','t13_dz',\
                                't21_dx','t21_dy','t21_dz',\
                                't22_dx','t22_dy','t22_dz',\
                                't23_dx','t23_dy','t23_dz',\
                                't31_dx','t31_dy','t31_dz',\
                                't32_dx','t32_dy','t32_dz',\
                                't33_dx','t33_dy','t33_dz']
                        }


