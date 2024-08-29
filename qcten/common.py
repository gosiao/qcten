import numpy as np
import math
import pandas as pd
from pprint import pprint


#
# input: t1d3
#
    # TODO - ensure m is t1d3


def get_mean_of_t1d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to vector elements
    t1, t2, t3, ...
    """

    # TODO - ensure m is t1d3
    df = pd.DataFrame()
    df['mean'] =  (m['t1']+m['t2']+m['t3'])/3.0

    return df


def norm_of_t1d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to vector elements
    t1, t2, t3, ...
    """

    # TODO - ensure m is t1d3
    df = pd.DataFrame()
    df['norm'] =  np.sqrt(m['t1']**2 + m['t2']**2 + m['t3']**2)

    return df


#
# input: t2d3
#
    # TODO - ensure m is t2d3

def get_sym_part_of_t2d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to 3x3 tensor elements
    t11, t12, t13, t21, ...
    """

    df = pd.DataFrame()
    df['s11'] =  m['t11']
    df['s12'] = (m['t12']+m['t21'])/2.0
    df['s13'] = (m['t13']+m['t31'])/2.0
    df['s21'] = (m['t21']+m['t12'])/2.0
    df['s22'] =  m['t22']
    df['s23'] = (m['t23']+m['t32'])/2.0
    df['s31'] = (m['t31']+m['t13'])/2.0
    df['s32'] = (m['t32']+m['t23'])/2.0
    df['s33'] =  m['t33']

    return df


def get_antisym_part_of_t2d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to 3x3 tensor elements
    t11, t12, t13, t21, ...
    """

    df = pd.DataFrame()
    df['a11'] =  0.0
    df['a12'] = (m['t12']-m['t21'])/2.0
    df['a13'] = (m['t13']-m['t31'])/2.0
    df['a21'] = (m['t21']-m['t12'])/2.0
    df['a22'] =  0.0
    df['a23'] = (m['t23']-m['t32'])/2.0
    df['a31'] = (m['t31']-m['t13'])/2.0
    df['a32'] = (m['t32']-m['t23'])/2.0
    df['a33'] =  0.0

    return df


def frobenius_norm_squared_t2d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to 3x3 tensor elements;
    m needs to have exactly 9 columns
    """

    # TODO - ensure m is t2d3
    squares = m.apply(lambda x : np.square(x))
    result = squares.sum(axis=1)

    return result


def trace_of_t2d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to tensor elements
    t11, t22, t33, ...
    """

    # TODO - ensure m is t2d3
    df = pd.DataFrame()
    df['trace'] =  m['t11'] + m['t22'] + m['t33']

    return df


#
# generic input
#

def gradient(method, m, t):

    '''
    entering m is pandas dataframe and t is the data label
    corresponding to data for which we calculate the gradient (t1, t2, t3, t11, t12, ...);
    method is the argument of `--calc_grad_method`
    '''

    if method == 'numpy':
        grad = gradient_from_numpy(m, t)
    elif method == 'finite_elements':
        pass
        #grad = gradient_from_finite_elements(m,t)
    else:
        print('warning: wrong argument of `--calc_grad_method`')
        grad=None

    return grad


def gradient_from_numpy(m, t):

    '''
    entering m is pandas dataframe and t is the data label
    corresponding to data for which we calculate the gradient (t1, t2, t3, t11, t12, ...)

    be careful, this works OK if "t" has at most quadratix dependence on r
    otherwise the approximation is too harsh (see jupyter notebook in test_gradient)
    '''

    #TODO: requires more testing
    dx, dy, dz = find_spacing_of_uniform_grid(m)

    grad_t_x = np.gradient(m[t], dx, edge_order=2)
    grad_t_y = np.gradient(m[t], dy, edge_order=2)
    grad_t_z = np.gradient(m[t], dz, edge_order=2)

    return [grad_t_x, grad_t_y, grad_t_z]



#
# grid-related functions
#

def find_spacing_of_uniform_grid(m):

    '''
    entering m is pandas dataframe;
    grid points are collected in m.x, m.y, m.z
    assumes a regular grid
    '''

    #TODO: ensure a regular grid

    diffs = m.diff().dropna()
    dx = min(filter(lambda x: x > 0, diffs["x"]))
    dy = min(filter(lambda x: x > 0, diffs["y"]))
    dz = min(filter(lambda x: x > 0, diffs["z"]))

    return dx, dy, dz


def get_grid_info(m):

    '''
    entering m is pandas dataframe;
    grid points are collected in m.x, m.y, m.z
    assumes a regular grid
    '''

    #TODO: ensure a regular grid

    dim_x = len(np.unique(m['x']))
    dim_y = len(np.unique(m['y']))
    dim_z = len(np.unique(m['z']))
    dim_cube = dim_x * dim_y * dim_z

    return dim_x, dim_y, dim_z, dim_cube



