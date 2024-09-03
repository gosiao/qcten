import sys
import numpy as np
import scipy.linalg as la
import math
import pandas as pd
from .global_data import *
from .common import *
from pprint import pprint


class t0d3():

    """
    This class holds settings and operations
    on tensors of rank 0 (scalars) in 3D space.

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

        for arg in self.input_options['calc_from_tensor_0order_3d']:

            if (arg == 'gradient'):
                self.get_t0d3_gradient(verbose)


    def get_t0d3_gradient(self, verbose):
        tmp = [x for x in global_data.grad_cols_to_use['t0d3'] if x not in self.data.columns]
        if tmp:
            self.data['t0_dx'], self.data['t0_dy'], self.data['t0_dz'] = gradient(self.input_options['calc_grad_method'], self.data, 't0')

