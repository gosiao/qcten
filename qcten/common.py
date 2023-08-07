import numpy as np
import math
import pandas as pd


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

    squares = m.apply(lambda x : np.square(x))
    result = squares.sum(axis=1)

    return result


def get_mean_of_t1d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to vector elements
    t1, t2, t3, ...
    """

    df = pd.DataFrame()
    df['mean'] =  (m['t1']+m['t2']+m['t3'])/3.0

    return df


def norm_of_t1d3(m):

    """
    entering m is pandas dataframe
    with columns corresponding to vector elements
    t1, t2, t3, ...
    """

    df = pd.DataFrame()
    df['norm'] =  np.sqrt(m['t1']**2 + m['t2']**2 + m['t3']**2)

    return df


