
class global_data:

    """
    list of available functions for a selected type of input data
    """

    # input is t2d3 (second-rank tensor in 3D)
    all_fun_t2d3 = ['trace',
                    'isotropic',
                    'deviator',
                    'antisymmetric',
                    'deviator_anisotropy',
                    'rortex_tensor_combined',
                    'omega_rortex_tensor_combined',
                    'tensor_inv1',
                    'tensor_inv2',
                    'tensor_inv3']

    # input is t1d3 = (first-rank tensor (= vector) in 3D)
    all_fun_t1d3 = ['rortex',
                    'omega_rortex',
                    'norm',
                    'mean',
                    'vorticity',
                    'omega',
                    'curlv_cdot_axis',
                    'rortex_cdot_axis']

    # functions that require the gradient of a vector
    fun_t1d3_req_grad = ['rortex',
                         'omega_rortex',
                         'rortex_cdot_axis',
                         'omega']


    # grid types
    grid_types = ['uniform_rectilinear']


