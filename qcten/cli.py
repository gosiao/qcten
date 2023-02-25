import os
import sys
import argparse
from pathlib import Path
from .global_data import *

class input_data:

    def __init__(self, args_list):

        self.args_list = args_list[1:]
        if args_list[0]:
            self.runinp_dir = Path(args_list[0]).resolve().parent 
        else:
            self.runinp_dir = os.getcwd()

        self.options         = {}

        # list of available functions for a selected type of input data:
        # --------------------------------------------------------------
        # input data is a second-rank tensor in 3D (t2d3)
        self.all_fun_t2d3 = global_data.all_fun_t2d3

        # input is a first-rank tensor (= vector) in 3D (t1d3)
        self.all_fun_t1d3 = global_data.all_fun_t1d3

        # input is a first-rank tensor (= vector) in 3D (t1d3)
        self.all_fun_t1d3 = global_data.all_fun_t1d3

        # import other options from global_data:
        # --------------------------------------
        self.grid_types = global_data.grid_types


    def parse_options(self):

        parser = argparse.ArgumentParser()

        required_args = parser.add_argument_group('required arguments')


        required_args.add_argument('--finp',
                                   dest='finp',
                                   action='append',
                                   metavar='file type (one of: txt, csv, hdf5, vti); file name; optional: column names ([col1, col2, ...]); optional: number of header lines to skip',
                                   required=True,
                                   help='''
                                        information on the input file with real-space data
                                        ''')

        required_args.add_argument('--fout',
                                   dest='fout',
                                   action='append',
                                   metavar='file type (one of: txt, csv, hdf5, vti); file name; optional: column names ([col1, col2:renamed_col2, ...]); optional: number of header lines to skip',
                                   required=True,
                                   help='''
                                        information on the output file with data
                                        ''')

        required_args.add_argument('--fout_select',
                                   dest='fout_select',
                                   action='store',
                                   choices=['all', 'result_only', 'selected'],
                                   required=True,
                                   help='''
                                        decide which data to write to output:
                                        choices:
                                            * all      - save all data generated on the way (massive output!)
                                            * result   - only the grid points and the value of a function that is calculated will be saved
                                            * selected - most basic data for a given function
                                        ''')

        required_args.add_argument('--flog',
                                   dest='flog',
                                   action='store',
                                   metavar='FILE (txt)',
                                   required=True,
                                   help='''
                                        log file to write to
                                        ''')

        required_args.add_argument('--grid',
                                   dest='grid',
                                   action='store',
                                   metavar='[column with x_coor, column with y_coor, column with z_coor]',
                                   required=True,
                                   help='''
                                        which columns contain grid x, y, z point coordinates
                                        ''')

        #required_args.add_argument('--grid_function',
        #                           dest='grid_function',
        #                           action='store',
        #                           metavar='[column with data 1, column with data 2, ...]',
        #                           required=True,
        #                           help='''
        #                                which columns contain data
        #                                ''')

#

        optional_args = parser.add_argument_group('optional arguments')

        # rethink this:
        optional_args.add_argument('--grid_function',
                                   dest='grid_function',
                                   action='store',
                                   metavar='[column with data 1, column with data 2, ...]',
                                   required=False,
                                   help='''
                                        which columns contain data
                                        ''')

        optional_args.add_argument('--grid_type',
                                   dest='grid_type',
                                   action='store',
                                   choices=self.grid_types,
                                   default='uniform_rectilinear',
                                   required=False,
                                   help='''
                                        mesh type (default: uniform_rectilinear)
                                        ''')


        optional_args.add_argument('--inptest',
                                   dest='inptest',
                                   action='store',
                                   required=False,
                                   help='''
                                        test input and quit
                                        ''')

        optional_args.add_argument('--form_tensor_0order_3d',
                                   dest='form_tensor_0order_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        which columns should be used to form the tensor field of order 0 (= scalar field)
                                        ''')

        optional_args.add_argument('--form_tensor_1order_3d',
                                   dest='form_tensor_1order_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        which columns should be used to form the tensor field of order 1 (= vector field)
                                        ''')

        optional_args.add_argument('--form_tensor_2order_3d',
                                   dest='form_tensor_2order_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        which columns should be used to form the tensor field of order 2
                                        ''')

        optional_args.add_argument('--calc_from_tensor_2order_3d',
                                   dest='calc_from_tensor_2order_3d',
                                   action='append',
                                   choices=self.all_fun_t2d3,
                                   required=False,
                                   help='''
                                        what to calculate from tensor_2order_3d
                                        ''')

        # verify this:
        optional_args.add_argument('--calc_from_tensor_2order_3d_fragments',
                                   dest='calc_from_tensor_2order_3d_fragments',
                                   action='store',
                                   required=False,
                                   help='''
                                        what to calculate from tensor_2order_3d; functions that apply to the fragment of the tensor
                                        ''')

        optional_args.add_argument('--form_grad_tensor_1order_3d',
                                   dest='form_grad_tensor_1order_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        which columns should be used to form the gradient of the tensorfield of order 1
                                        ''')

        optional_args.add_argument('--use_grad_from_file',
                                   dest='use_grad_from_file',
                                   action='store',
                                   required=False,
                                   help='''
                                        are the elements of the gradient of the tensor field available on file?
                                        ''')

        optional_args.add_argument('--calc_from_tensor_1order_3d',
                                   dest='calc_from_tensor_1order_3d',
                                   action='append',
                                   choices=self.all_fun_t1d3,
                                   required=False,
                                   help='''
                                        what to calculate from vector_3d
                                        ''')

        # verify this:
        optional_args.add_argument('--calc_from_tensor_1order_3d_calc_grad',
                                   dest='calc_from_tensor_1order_3d_calc_grad',
                                   action='store',
                                   choices=['numpy', 'finite_elements'],
                                   required=False,
                                   help='''
                                        calc_from_tensor_1order_3d_calc_grad
                                        ''')

        optional_args.add_argument('--projection_axis',
                                   dest='projection_axis',
                                   action='store',
                                   required=False,
                                   help='''
                                        selected external axis (on which we project vectors)
                                        ''')


        optional_args.add_argument('--fill_empty',
                                   dest='fill_empty',
                                   action='store',
                                   required=False,
                                   default='NULL',
                                   help='''
                                        how to fill empty data fields;
                                        ''')

        # options specific to TTK:
        optional_args.add_argument('--ttk_task',
                                   dest='ttk_task',
                                   action='store',
                                   required=False,
                                   choices=['start', 'calculate', 'ms', 'scatterplot', 'bottleneck', 'jacobi'],
                                   help='''
                                        which TTK task will be performed
                                        ''')


        optional_args.add_argument('--resampled_dim',
                                   dest='resampled_dim',
                                   action='store',
                                   required=False,
                                   help='''
                                        number of points in x,y,z directions for the "ResampleToImage" filter in TTK;
                                        ''')

        optional_args.add_argument('--calc_fun',
                                   dest='calc_fun',
                                   action='append',
                                   required=False,
                                   help='function to apply in the Calculator filter')

        optional_args.add_argument('--calc_gradient',
                                   dest='calc_gradient',
                                   action='append',
                                   required=False,
                                   help='apply the gradientOfUnstructuredDataSet filter')


        args = parser.parse_args(self.args_list)
        self.options = vars(args)


    def print_options(self):

        if self.options == {}:
            self.parse_options()

        print("all input options:")
        for option_key, option_value in self.options.items():
            print("{}: {}".format(option_key, option_value))


def read_input(finp=None, verbose=False):

    """
    read input options, either from a command line or from a file
    """

    args = []

    if finp is None:
        try:
            args = sys.argv[1:]
        except:
            sys.exit(1)
        if verbose:
            msg1 = 'Arguments for qcten are read from command line'
            msg2 = '\n'.join(x for x in args[1:])
            print(msg1+'\n'+msg2)
    else:
        finp_path = Path(finp)
        with open(finp_path, 'r') as f:
            args = [line.strip() for line in f if line[0] != '#' and line != '\n']
        if verbose:
            msg1 = 'Arguments for qcten are read from file {}'.format(finp_path)
            msg2 = '\n'.join(x for x in args[1:])
            print(msg1+'\n'+msg2)

    args = [finp_path] + args
            
    return args




if __name__ == '__main__':
    args = read_input()
    data = input_data(args)
    data.parse_options()
    data.print_options()




