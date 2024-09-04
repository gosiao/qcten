import numpy as np
import scipy.linalg as la
import math
import pandas as pd
from pprint import pprint


#
# input: t1d3
#
    # TODO - ensure m is t1d3

thr_zero_abs = 1.0e-12
thr_zero_rel = 1.0e-6


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


def tensor_eigendecomposition(m):

    """
    entering m is 3x3 np.array



            * we use scipy.linalg package, TODO: 
                * compare with other python packages, esp. in terms of timing
                * better test checking whether the imaginary part of an
                  eigenvalue is 0 or close to 0
                * is the test_eigen necessary/sufficient?
                * check all values used as tolerance to compare numbers
    """

    eigenvalues, eigenvectors = la.eig(m)

    # eigenvectors are in columns of "eigenvectors" (eigenvectors[:, ind])
    # corresponding eigenvalues are in "eigenvalues" (in the same order)

    # double check: test the eigendecomposition:
    # todo: refactor using decorators
    test_eig=True
    if test_eig:
        test_eigendecomposition(eigenvalues, eigenvectors, m)

    number_complex_eigenvalues = 0
    res = {}
    real_eigval_ind = []
    for ind, e in enumerate(eigenvalues):
        if isinstance(e, complex):
            if (abs(e.imag) < thr_zero_rel * abs(e.real)):
                eigval = np.real(e)
                real_eigval_ind.append(ind)
            else:
                eigval = e
                number_complex_eigenvalues += 1
        else:
            eigval = e
            real_eigval_ind.append(ind)
        res['eig_pair_'+str(ind)] = [eigval, eigenvectors[:,ind]]
    res['number_complex_eigenvalues'] = number_complex_eigenvalues
    res['real_eigval_ind'] = real_eigval_ind

    return res


def test_eigendecomposition(eig_val, eig_vec, mat):

    epsilon=1.0e-8

    for i in range(len(eig_val)):

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

    #grad_t_x = np.gradient(m[t], dx, edge_order=2)

    return [grad_t_x, grad_t_y, grad_t_z]


def debug_print_df(df, msg=None):
    if msg is not None:
        print('DEBUG PRINT: ', msg)
    print('DEBUG PRINT: df.columns =  ', df.columns)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pprint(df)
    pd.reset_option('all') 


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

    #debug_print_df(m, msg = "debugging diffs; entering m")
    diffs = m.diff().dropna()
    #debug_print_df(diffs, msg = "debugging diffs")
    dx = min(filter(lambda x: x > 0, diffs["x"].to_numpy()))
    dy = min(filter(lambda x: x > 0, diffs["y"].to_numpy()))
    dz = min(filter(lambda x: x > 0, diffs["z"].to_numpy()))

    print("BUBA x: ", dx)
    pprint(diffs["x"])
    print("BUBA y: ", dy)
    pprint(diffs["y"])
    print("BUBA z: ", dz)
    pprint(diffs["z"])

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


#
# other
#


def norm_of_vec(m, label):

    tmp = 0
    for i in range(3):
        tmp = tmp + m[label][i]**2
    return np.sqrt(tmp)


def find_real(x, label, ind_range):
    i_real=[]
    for i in range(ind_range):
        if isinstance(x[label+str(i)], float):
            i_real.append(i)
    return i_real


def complex_to_real(e):
    if isinstance(e, complex):
        if (abs(e.imag) < thr_zero_rel * abs(e.real)):
            r = np.real(e)
        else:
            r = e.astype(complex)
            print('imaginary part of {} is non-negligible. Return original value'.format(e))
    else:
        r = e.astype(complex)
    return r
 
