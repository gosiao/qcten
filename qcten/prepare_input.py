import os
import sys
import argparse

class input_data:

    def __init__(self, args_list=None):

        self.args_list       = args_list if args_list is not None else sys.argv[1:]
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

        if args.finp is not None:
            self.options["finp"] = args.finp
        else:
            parser.error('--finp is required')

        if args.fout is not None:
            self.options["fout"] = args.fout
        else:
            parser.error('--fout is required')

        if args.fout_select is not None:
            self.options["fout_select"] = args.fout_select
        else:
            parser.error('--fout_select is required')

        if args.flog is not None:
            self.options["flog"] = args.flog
        else:
            parser.error('--flog is required')

        if args.grid is not None:
            self.options["grid"] = args.grid
        else:
            parser.error('--grid is required')

        if args.inptest is not None:
            self.options["inptest"] = args.inptest
        else:
            self.options["inptest"] = None

        # tensors: 2nd order
        if args.form_tensor_2order_3d is not None:
            self.options["form_tensor_2order_3d"] = args.form_tensor_2order_3d
        else:
            self.options["form_tensor_2order_3d"] = None

        if args.calc_from_tensor_2order_3d is not None:
            self.options["calc_from_tensor_2order_3d"] = args.calc_from_tensor_2order_3d
        else:
            self.options["calc_from_tensor_2order_3d"] = None

        if args.calc_from_tensor_2order_3d_fragments is not None:
            self.options["calc_from_tensor_2order_3d_fragments"] = args.calc_from_tensor_2order_3d_fragments
        else:
            self.options["calc_from_tensor_2order_3d_fragments"] = None

        # vectors
        if args.form_vector_3d is not None:
            self.options["form_vector_3d"] = args.form_vector_3d
        else:
            self.options["form_vector_3d"] = None

        if args.form_grad_vector_3d is not None:
            self.options["form_grad_vector_3d"] = args.form_grad_vector_3d
        else:
            self.options["form_grad_vector_3d"] = None

        if args.use_grad_from_file is not None:
            self.options["use_grad_from_file"] = args.use_grad_from_file
        else:
            self.options["use_grad_from_file"] = None

        if args.calc_from_vector_3d is not None:
            self.options["calc_from_vector_3d"] = args.calc_from_vector_3d
        else:
            self.options["calc_from_vector_3d"] = None

        if args.calc_from_vector_3d_calc_grad is not None:
            self.options["calc_from_vector_3d_calc_grad"] = args.calc_from_vector_3d_calc_grad
        else:
            self.options["calc_from_vector_3d_calc_grad"] = None

        if args.selected_axis is not None:
            self.options["selected_axis"] = [float(v.strip().strip('[').strip(']')) for v in args.selected_axis.split(',')]
        else:
            self.options["selected_axis"] = None


        if args.rortex_fill_empty is not None:
            self.options["rortex_fill_empty"] = args.rortex_fill_empty
        else:
            self.options["selected_axis"] = None



    def print_options(self):

        self.parse_options()

        print("all input options:")
        for option_key, option_value in self.options.items():
            print("{}: {}".format(option_key, option_value))



if __name__ == '__main__':
    data = input_data()
    data.print_options()




