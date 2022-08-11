import os
import sys
import argparse
from pathlib import Path

class input_data:

    def __init__(self, args_list):

        self.args_list       = args_list
        self.options         = {}

        # list of available functions for a selected type of input data:
        self.all_fun_t2d3 = ['trace',
                             'isotropic',
                             'deviator',
                             'antisymmetric',
                             'deviator_anisotropy',
                             'rortex_tensor_combined',
                             'omega_rortex_tensor_combined',
                             'tensor_inv1',
                             'tensor_inv2',
                             'tensor_inv3']

        self.all_fun_t1d3 = ['rortex',
                             'omega_rortex',
                             'norm',
                             'mean',
                             'vorticity',
                             'omega']


    def parse_options(self):

        parser = argparse.ArgumentParser()

        required_args = parser.add_argument_group('required arguments')


        required_args.add_argument('--finp',
                                   dest='finp',
                                   action='append',
                                   metavar='file name (csv); column names; column separator; number of header lines to skip',
                                   required=True,
                                   help='''
                                        input with data from quantum chem calculations;
                                        one file per one "--finp"
                                        ''')

        required_args.add_argument('--fout',
                                   dest='fout',
                                   action='store',
                                   metavar='file name (csv)',
                                   required=True,
                                   help='''
                                        output file to write to
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



        optional_args = parser.add_argument_group('optional arguments')


        optional_args.add_argument('--inptest',
                                   dest='inptest',
                                   action='store',
                                   required=False,
                                   help='''
                                        test input and quit
                                        ''')

        optional_args.add_argument('--form_tensor_2order_3d',
                                   dest='form_tensor_2order_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        form_tensor_2order_3d
                                        ''')

        optional_args.add_argument('--calc_from_tensor_2order_3d',
                                   dest='calc_from_tensor_2order_3d',
                                   action='append',
                                   choices=self.all_fun_t2d3,
                                   required=False,
                                   help='''
                                        what to calculate from tensor_2order_3d
                                        ''')

        optional_args.add_argument('--calc_from_tensor_2order_3d_fragments',
                                   dest='calc_from_tensor_2order_3d_fragments',
                                   action='store',
                                   required=False,
                                   help='''
                                        what to calculate from tensor_2order_3d; functions that apply to the fragment of the tensor
                                        ''')

        optional_args.add_argument('--form_vector_3d',
                                   dest='form_vector_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        form_vector_3d
                                        ''')

        optional_args.add_argument('--form_grad_vector_3d',
                                   dest='form_grad_vector_3d',
                                   action='store',
                                   required=False,
                                   help='''
                                        form_grad_vector_3d
                                        ''')

        optional_args.add_argument('--use_grad_from_file',
                                   dest='use_grad_from_file',
                                   action='store',
                                   required=False,
                                   help='''
                                        use_grad_from_file
                                        ''')

        optional_args.add_argument('--calc_from_vector_3d',
                                   dest='calc_from_vector_3d',
                                   action='append',
                                   choices=self.all_fun_t1d3,
                                   required=False,
                                   help='''
                                        what to calculate from vector_3d
                                        ''')

        optional_args.add_argument('--calc_from_vector_3d_calc_grad',
                                   dest='calc_from_vector_3d_calc_grad',
                                   action='store',
                                   choices=['numpy', 'finite_elements'],
                                   required=False,
                                   help='''
                                        calc_from_vector_3d_calc_grad
                                        ''')

        optional_args.add_argument('--selected_axis',
                                   dest='selected_axis',
                                   action='store',
                                   required=False,
                                   help='''
                                        selected external axis (on which we project vectors)
                                        ''')


        optional_args.add_argument('--rortex_fill_empty',
                                   dest='rortex_fill_empty',
                                   action='store',
                                   required=False,
                                   default='NULL',
                                   help='''
                                        how to fill empty fields in rortex;
                                        ''')



        args = parser.parse_args(self.args_list)

        # required arguments
        # ==================
        self.options["finp"] = args.finp
        self.options["fout"] = args.fout
        self.options["fout_select"] = args.fout_select
        self.options["flog"] = args.flog
        self.options["grid"] = args.grid

        # optional arguments
        # ==================
        if args.inptest is not None:
            self.options["inptest"] = args.inptest

        # tensors: 2nd order
        if args.form_tensor_2order_3d is not None:
            self.options["form_tensor_2order_3d"] = args.form_tensor_2order_3d

        if args.calc_from_tensor_2order_3d is not None:
            self.options["calc_from_tensor_2order_3d"] = args.calc_from_tensor_2order_3d

        if args.calc_from_tensor_2order_3d_fragments is not None:
            self.options["calc_from_tensor_2order_3d_fragments"] = args.calc_from_tensor_2order_3d_fragments

        # vectors
        if args.form_vector_3d is not None:
            self.options["form_vector_3d"] = args.form_vector_3d

        if args.form_grad_vector_3d is not None:
            self.options["form_grad_vector_3d"] = args.form_grad_vector_3d

        if args.use_grad_from_file is not None:
            self.options["use_grad_from_file"] = args.use_grad_from_file

        if args.calc_from_vector_3d is not None:
            self.options["calc_from_vector_3d"] = args.calc_from_vector_3d

        if args.calc_from_vector_3d_calc_grad is not None:
            self.options["calc_from_vector_3d_calc_grad"] = args.calc_from_vector_3d_calc_grad

        if args.selected_axis is not None:
            self.options["selected_axis"] = [float(v.strip().strip('[').strip(']')) for v in args.selected_axis.split(',')]

        if args.rortex_fill_empty is not None:
            self.options["rortex_fill_empty"] = args.rortex_fill_empty



    def print_options(self):

        if self.options == {}:
            self.parse_options()

        print("all input options:")
        for option_key, option_value in self.options.items():
            print("{}: {}".format(option_key, option_value))


def read_input(finp=None, verbose=False):

    args = []

    if finp is None:
        try:
            args = sys.argv[1:]
        except:
            sys.exit(1)
        if verbose:
            msg1 = 'Arguments for qcten are read from command line'
            msg2 = '\n'.join(x for x in args)
            print(msg1+'\n'+msg2)
    else:
        with open(finp, 'r') as f:
            args = [line.strip() for line in f if line[0] != '#' and line != '\n']
        if verbose:
            msg1 = 'Arguments for qcten are read from file {}'.format(Path(finp))
            msg2 = '\n'.join(x for x in args)
            print(msg1+'\n'+msg2)
            
    return args



if __name__ == '__main__':
    args = read_input()
    data = input_data(args)
    data.parse_options()
    data.print_options()




