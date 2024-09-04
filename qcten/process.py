import pandas as pd
import numpy as np
import os
import sys
import re
import collections
from pprint import pprint
from pathlib import Path
import subprocess
from .t2d3 import *
from .t1d3 import *
from .t0d3 import *
from .common import *
from .common_ttk import *

class work():

    def __init__(self, rundir, options):

        # options read from an input script or a command line
        self.options  = options

        # working directory
        self.rundir = rundir

        # IO
        self.allfinps  = ()
        self.allfouts  = ()
        self.flog      = self.options['flog']

        # grid and grid functions
        self.grid_info = {}
        self.fulldata = pd.DataFrame()


    def run(self, verbose=False):

        # 1. parse --finp; write info to self.allfinp
        self.prepare_input(verbose)

        # 2. parse --fout; write info to self.allfout
        self.prepare_output(verbose)

        # 3. prepare data, verify, check consistency
        self.prepare_data(verbose)

        # 4. calculate
        self.calculate(verbose)

        # 5. write to files
        self.write_and_close(verbose)


    def write_and_close(self, verbose):

        if not self.fulldata.empty:
            self.fulldata = self.fulldata.astype(np.float64)

            for fout in self.allfouts:

                requested_cols = []
                data_cols = []
                for col in fout.file_column_names:
                    if ':' in col:
                        old_col = col.strip().split(':')[0].strip()
                        new_col = col.strip().split(':')[1].strip()
                    else:
                        old_col = col.strip()
                        new_col = old_col

                    if old_col in self.fulldata.columns:
                        data_cols.append(old_col)
                        requested_cols.append(new_col)
                    else:
                        msg = 'ERROR: column {} not available for output'.format(col)

                if verbose:
                    print('IN write_and_close: ', fout, data_cols, requested_cols)
                df = self.fulldata[data_cols].rename(columns={k:v for k, v in zip(data_cols,requested_cols)})

                f = fout.file_path
                f.parent.mkdir(parents=True, exist_ok=True)
                if (fout.file_type == 'txt' or fout.file_type == 'csv'):
                    df.to_csv(f, index=False)
                elif fout.file_type == 'hdf5':
                    pass
                elif fout.file_type == 'vti':
                    # check if csv file exsists; 
                    ttk_support = ttk_basics(self.options, df, f, grid_info=self.grid_info)
                    ttk_support.write_data_to_vti()
                else:
                    msg = 'ERROR: unsupported file format for output; check --fout'
                    sys.exit(msg)
                    self.write_df_to_hdf5(df, f)

                if verbose:
                    print('dataframe for file ', f)
                    pprint(df)


    def prepare_input(self, verbose=False):

        """
        decode the input of '--finp'
        
        there are at least 2 and at most 4 arguments to --finp:

        obligatory:
        1. file type
        2. file name

        optional
        3. column names (only if the input is a TXT file)
        4. number of header lines (only if the input is a TXT file)
        """

        if self.allfinps is not None:
            print('WARNING: --finp arguments are already assigned; will be overwritten')

        temp = []
        for f_arg in self.options["finp"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 5:
                msg = 'ERROR: wrong number of arguments to --finp'
                sys.exit(msg)

            f_info = self.prepare_io(args)

            if f_info.file_type is None:
                msg = 'ERROR: specify format of the input file; check --finp)'
                sys.exit(msg)
            if not Path(f_info.file_path).exists():
                msg = 'ERROR: input file does not exist; check --finp)'
                sys.exit(msg)

            temp.append(f_info)

        self.allfinps = tuple(temp)

        if verbose:
            print('files with input data:')
            pprint(self.allfinps)


    def prepare_output(self, verbose=False):

        """
        decode the input of '--fout'
        
        there are at least 2 and at most 4 arguments to --fout:

        obligatory:
        1. file type
        2. file name

        optional
        3. column names
        4. number of header lines (only if the input is a TXT file)
        """

        if self.allfouts is not None:
            print('WARNING: --fout arguments are already assigned; will be overwritten')

        temp = [] 
        for f_arg in self.options["fout"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 5:
                msg = 'ERROR: wrong number of arguments to --fout'
                sys.exit(msg)

            f_info = self.prepare_io(args)

            if f_info.file_type is None:
                msg = 'ERROR: specify format of the output file; check --fout)'
                sys.exit(msg)
            if f_info.file_path is None:
                msg = 'ERROR: specify output file (name or full path); check --fout)'
                sys.exit(msg)

            temp.append(f_info)

        self.allfouts = tuple(temp)

        if verbose:
            print('files for output data:')
            pprint(self.allfouts)


    def prepare_io(self, args):

        """
        parse lines: 
        --finp/--fout format; path; [columns]; [sep]; [skip] 
        """

        # format and full path of a file
        f_type   = args[0].strip()
        f_path   = args[1].strip()

        # names of data fields:
        # they are also labels on input/output files;
        # (unless it is an hdf5 file)
        f_cols   = None
        if len(args) > 2:
            if (f_type != 'hdf5'):
                arg = args[2].strip()
                if arg != 'None':
                    if arg[0:5] == 'cols=':
                        f_cols   = [a.strip().strip('[').strip(']') for a in arg[5:].split(',')]

        # column separator; defaults to a coma
        f_sep = ','
        if len(args) > 3:
            arg = args[3].lstrip()
            if arg != 'None':
                if arg[0:4] == 'sep=':
                    f_sep = arg[4:]

        # index of a row line to skip
        f_skiprow = None
        if len(args) > 4:
            arg = args[4].strip()
            if arg != 'None':
                if arg[0:5] == 'skip=':
                    f_skiprow = int(arg[5:].strip())

        d = {}
        d['file_type'] = f_type
        d['file_path'] = Path(self.rundir, f_path).resolve()
        d['file_column_names'] = f_cols
        d['file_column_separator'] = f_sep
        d['file_skiprow'] = f_skiprow

        io_info = collections.namedtuple('io_info', ['file_type',
                                                     'file_path',
                                                     'file_column_names',
                                                     'file_column_separator',
                                                     'file_skiprow'])

        f = io_info(**d)
        return f


    def prepare_data(self, verbose):

        """
        read the input data into pandas dataframes

        TODO: 
        * deal with empty or non-float fields
        * check whether the data has been collected on the same grids
        """

        # 1. read all input data into a list of dataframes
        dfs = []
        for v in self.allfinps:

            if v.file_type.lower() == 'txt':
                if v.file_skiprow is None:
                    df = pd.read_csv(v.file_path,
                                     header = None,
                                     index_col = False,
                                     usecols = v.file_column_names,
                                     delim_whitespace = True,
                                     dtype = np.float64)[v.file_column_names]
                else:
                    df = pd.read_fwf(v.file_path, 
                                     colspecs='infer', 
                                     skiprows = v.file_skiprow, 
                                     index_col = False,
                                     usecols = v.file_column_names)[v.file_column_names]

            elif v.file_type.lower() == 'csv':
                if v.file_column_separator is None or v.file_column_separator.isspace():
                    df = pd.read_csv(v.file_path,
                                     header = 0,
                                     index_col = False,
                                     usecols = v.file_column_names,
                                     delim_whitespace = True,
                                     skiprows = v.file_skiprow,
                                     dtype = np.float64)[v.file_column_names]
                else:
                    df = pd.read_csv(v.file_path,
                                     header = 0,
                                     index_col = False,
                                     usecols = v.file_column_names,
                                     sep = v.file_column_separator,
                                     skiprows = v.file_skiprow,
                                     dtype = np.float64)[v.file_column_names]

            elif v.file_type.lower() == 'hdf5':
                data_dict=self.read_hdf5(v.file_path)
                dd = {}
                for key, val in data_dict.items():
                    if 'nr_points_dim_' in key:
                        self.grid_info[key.split('/')[-1]] = val['value'][0]
                    if isinstance(val['value'], np.ndarray) and len(val['value'])>1:
                        if 'coor_' in key:
                            dd[key.split('/')[-1]] = val['value']
                        else:
                            dd[key.split('/')[-1]] = val['value']
                            #dd[key] = val['value']
                for key, val in dd.items():
                    print('NEW ', key, dd[key])
                
                df = pd.DataFrame.from_dict(dd)
                #debug_print_df(df, msg='df from {}'.format(v.file_path))  
                pprint(dd)

            df.apply(pd.to_numeric, errors='coerce')

            dfs.append(df)
        for key, val in self.grid_info.items():
            print('GRID INFO ', key, val)

        # 2. combine a list of dataframes into one dataframe;
        #    first, remove the excess 'grid' columns (now -assuming the same grids):
        #for df in dfs[1:]:
        #    df.drop(columns=global_data.grid_cols_to_use['rectilinear_3d'], inplace=True)

        fulldata = pd.concat([df for df in dfs], axis=1, sort=False)
        if self.fulldata.empty:
            self.fulldata = fulldata.loc[:,~fulldata.columns.duplicated()].copy()

        verbose = True
        if verbose:
            print('Original data (from prepare_data): ')
            pprint(fulldata.columns)
            pprint(fulldata)


#    code copied/adapted from dirac:
    def read_hdf5(self, file_name):
        """
        Open hdf5-type file and return dictionary of its contents
        """
        import h5py
        data_dict = {}
        with h5py.File(file_name, 'r') as h5file:
            self.recursively_load_dict_contents_from_group(h5file, data_dict,'/')
        return data_dict
    
    def recursively_load_dict_contents_from_group(self, h5file, data_dict, path):
        """
        Modified from code found at Stack Exchange to get flat dictionary
        """
        import h5py
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                data_dict[path+key] = {}
                data_dict[path+key]['value'] = item[()]
            elif isinstance(item, h5py._hl.group.Group):
               self.recursively_load_dict_contents_from_group(h5file, data_dict, path + key + '/')
        return
#    end ofcode copied from dirac


    def assign_data(self, label, verbose=False):
        """
        assign data labels to the ones used internally in qcten
        """

        args=[]
        grid_args=[]
        grad_args=[]

        # always start with assigning grid labels
        grid_label = "rectilinear_3d"
        grid_args = [arg.strip().strip('[').strip(']') for arg in self.options['grid'].split(',')]

        # assign data labels
        if label == "t0d3":
            args = [arg.strip().strip('[').strip(']') for arg in self.options['form_tensor_0order_3d'].split(',')]
            if self.options["form_grad_tensor_0order_3d"] is not None:
                grad_args = [arg.strip().strip('[').strip(']') for arg in self.options['form_grad_tensor_0order_3d'].split(',')]

        elif label == "t1d3":
            args = [arg.strip().strip('[').strip(']') for arg in self.options['form_tensor_1order_3d'].split(',')]
            if self.options["form_grad_tensor_1order_3d"] is not None:
                grad_args = [arg.strip().strip('[').strip(']') for arg in self.options['form_grad_tensor_1order_3d'].split(',')]

        elif label == "t2d3":
            args = [arg.strip().strip('[').strip(']') for arg in self.options['form_tensor_2order_3d'].split(',')]
            if self.options["form_grad_tensor_2order_3d"] is not None:
                grad_args = [arg.strip().strip('[').strip(']') for arg in self.options['form_grad_tensor_2order_3d'].split(',')]
        else:
            print("Unsuported label in assign_data")
            sys.exit(1)

        _cols_to_use=[]

        for col in self.fulldata.columns:
            if grid_args is not None:
                for iarg, arg in enumerate(grid_args):
                    if arg == col and arg not in _cols_to_use:
                        self.fulldata.rename(columns={col:global_data.grid_cols_to_use[grid_label][iarg]}, inplace=True)
                        _cols_to_use.append(global_data.grid_cols_to_use[grid_label][iarg])
            if args is not None:
                for iarg, arg in enumerate(args):
                    if arg == col and arg not in _cols_to_use:
                        self.fulldata.rename(columns={col:global_data.cols_to_use[label][iarg]}, inplace=True)
                        _cols_to_use.append(global_data.cols_to_use[label][iarg])
            if grad_args is not None:
                for iarg, arg in enumerate(grad_args):
                    if arg == col and arg not in _cols_to_use:
                        self.fulldata.rename(columns={col:global_data.grad_cols_to_use[label][iarg]}, inplace=True)
                        _cols_to_use.append(global_data.grad_cols_to_use[label][iarg])

        cols_to_remove=[x for x in self.fulldata.columns if x not in _cols_to_use]

        return cols_to_remove


    def calculate(self, verbose=False):

        result_df = pd.DataFrame()

        if 'form_tensor_0order_3d' in self.options and self.options['form_tensor_0order_3d'] is not None:
        
            cols_to_remove=self.assign_data("t0d3")
            self.fulldata.drop(columns=cols_to_remove, inplace=True)
            print('BEFORE')
            pprint(self.fulldata)
            self.verify_data_for_calcs("t0d3", verbose)
            work = t0d3(self.options, self.allfouts, self.fulldata)
            work.run(verbose=verbose)
            result_df = work.data

        if 'form_tensor_1order_3d' in self.options and self.options['form_tensor_1order_3d'] is not None:

            cols_to_remove=self.assign_data("t1d3")
            self.fulldata.drop(columns=cols_to_remove, inplace=True)
            self.verify_data_for_calcs("t1d3", verbose)
            pprint(self.fulldata)
            work = t1d3(self.options, self.allfouts, self.fulldata)
            work.run(verbose=verbose)
            result_df = work.data

        if 'form_tensor_2order_3d' in self.options and self.options['form_tensor_2order_3d'] is not None:

            cols_to_remove=self.assign_data("t2d3")
            self.fulldata.drop(columns=cols_to_remove, inplace=True)
            self.verify_data_for_calcs("t2d3", verbose)
            work = t2d3(self.options, self.allfouts, self.fulldata)
            work.run(verbose=verbose)
            result_df = work.data


        self.fulldata = pd.concat((self.fulldata, result_df), axis=1)
        self.fulldata = self.fulldata.loc[:,~self.fulldata.columns.duplicated()]


    def update_df(self, new_df, new_df_cols=None):

        '''
        clean the final dataframe before writing it to output file
        '''

        if new_df_cols is not None:

            # get unique column names
            cols = set(new_df_cols)

            if self.allfout is not None:
                cols = list(cols)

                # we might want to export different data on each output file
                for k, v in self.allfout.items():
                    fulldata = new_df[cols]
                    new_cols = []
                    for old_col in cols:
                        if old_col in v['file_column_old_names']:
                            col_ind = v['file_column_old_names'].index(old_col)
                            new_col = v['file_column_new_names'][col_ind]
                        else:
                            new_col = old_col
                        new_cols.append(new_col)
                    fulldata.columns = new_cols

                    # todo: remove duplicates

                    fulldata = fulldata.reindex(columns = (['x', 'y', 'z'] + [ c for c in fulldata.columns if c not in ['x', 'y', 'z']]))
                    self.fulldata = fulldata

                with open(self.flog, 'a') as f:
                    f.write('----------------- columns written to output file -------------------\n')
                    f.write('{}'.format(cols))

        else:
            fulldata = new_df

        return fulldata



    def print_options_to_log(self):

        if self.options['flog'] is not None:

            with open(self.options['flog'], 'w') as f:

                f.write("--------------------------- main job options ---------------------------\n")
                #f.write("qcten SHA: {}\n".format(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(__file__)).strip().decode()))
                f.write("qcten SHA: {}\n".format(subprocess.check_output(["git", "describe", "--always"])))
                f.write("\n")
                for k, v in self.options.items():
                    f.write("{:<45}: {}\n".format(k, v))
                f.write("\n")

                for kf, vf in self.allfinp.items():
                    f.write("------------------ set of options for input files ------------------\n")
                    for k, v in vf.items():
                        f.write("{:<15}: {}\n".format(k, v))
                f.write("\n")



    def verify_data_for_calcs(self, label, verbose):
        """
        verify whether all data needed for the type of calculations exists;
        take care of missing data, exceptions, etc.
        """

        if self.options['grid'] is None:
            msg = 'ERROR: check `--grid` in your input'
            sys.exit(msg)
        else:
            for col in global_data.grid_cols_to_use['rectilinear_3d']:
                if col not in self.fulldata.columns:
                    msg = 'ERROR: missing assignment to `--grid`'
                    sys.exit(msg)

        if label == "t0d3":
            if self.options['calc_from_tensor_0order_3d'] is None:

                msg = 'WARNING: Nothing to calculate from the vector field. ' \
                    + 'Check --calc_from_tensor_0order_3d in your input'
                sys.exit(msg)
    
                for arg in self.options['calc_from_tensor_0order_3d']:
                    if arg not in self.all_fun_t0d3:
                        msg = 'ERROR: requested function not in the list of available functions ' \
                            + 'Check --calc_from_tensor_0order_3d in your input. ' \
                            + 'Available functions: ', self.all_fun_t0d3
                        sys.exit(msg)
    
            else:
                for arg in self.options['calc_from_tensor_0order_3d']:
                    if (arg in global_data.fun_t1d3_req_grad):
                        if self.options['use_grad_from_file']:
                            print('Gradient of t0d3 will be read from file')
                        else:
                            if self.options['calc_grad_method'] == 'numpy':
                                print('Gradient of t0d3 will be calculated using numpy')
                            else:
                                print('Other methods for calculating the gradient are not available')
                                sys.exit(1)

        if label == "t1d3":
            if self.options['calc_from_tensor_1order_3d'] is None:

                msg = 'WARNING: Nothing to calculate from the vector field. ' \
                    + 'Check --calc_from_tensor_1order_3d in your input'
                sys.exit(msg)
    
                for arg in self.options['calc_from_tensor_1order_3d']:
                    if arg not in self.all_fun_t1d3:
                        msg = 'ERROR: requested function not in the list of available functions ' \
                            + 'Check --calc_from_tensor_1order_3d in your input. ' \
                            + 'Available functions: ', self.all_fun_t1d3
                        sys.exit(msg)
    
            else:
                for arg in self.options['calc_from_tensor_1order_3d']:
                    if (arg in global_data.fun_t1d3_req_grad):
                        if self.options['use_grad_from_file']:
                            print('Gradient of t1d3 will be read from file')
                        else:
                            if self.options['calc_grad_method'] == 'numpy':
                                print('Gradient of t1d3 will be calculated using numpy')
                            else:
                                print('Other methods for calculating the gradient are not available')
                                sys.exit(1)

        elif label == "t2d3":
            if self.options['calc_from_tensor_2order_3d'] is None:

                msg = 'WARNING: Nothing to calculate from the tensor field of rank 2. ' \
                    + 'Check --calc_from_tensor_2order_3d in your input'
                sys.exit(msg)
    
                for arg in self.options['calc_from_tensor_2order_3d']:
                    if arg not in self.all_fun_t2d3:
                        msg = 'ERROR: requested function not in the list of available functions ' \
                            + 'Check --calc_from_tensor_2order_3d in your input. ' \
                            + 'Available functions: ', self.all_fun_t2d3
                        sys.exit(msg)
    
            else:
                for arg in self.options['calc_from_tensor_2order_3d']:
                    if (arg in global_data.fun_t2d3_req_grad):
                        if self.options['use_grad_from_file']:
                            print('Gradient of t2d3 will be read from file')
                        else:
                            if self.options['calc_grad_method'] == 'numpy':
                                print('Gradient of t2d3 will be calculated using numpy')
                            else:
                                print('Other methods for calculating the gradient are not available')
                                sys.exit(1)



        if self.options['projection_axis'] is not None:
            args = [arg.strip().strip('[').strip(']') for arg in self.options['projection_axis'].split(',')]
            self.fulldata["projection_axis_x"] = int(args[0])
            self.fulldata["projection_axis_y"] = int(args[1])
            self.fulldata["projection_axis_z"] = int(args[2])


