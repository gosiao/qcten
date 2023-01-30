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
from .common_ttk import *

class work():

    def __init__(self, rundir, options):

        # all options read from an input script
        self.options  = options
        self.rundir = rundir

        # io data
        self.allfinps  = ()
        self.allfouts  = ()
        self.flog      = self.options['flog']


        # print control (debug)
        #self.verbose = verbose

        # OLD
        # grid and data
        self.grid = {}
        self.grid_function = {}
        self.data = {}
        self.fulldata = pd.DataFrame()


    def run(self, verbose=True):
        # 1. parse --finp; write info to self.allfinp
        self.prepare_input(verbose=verbose)
        # 2. parse --fout; write info to self.allfout
        self.prepare_output(verbose=verbose)
        # 3. calculate
        self.prepare_data(verbose=verbose)
        self.calculate(verbose=verbose)
        # 4. write to files
        self.write_and_close(verbose=verbose)


    def write_and_close(self, verbose=False):

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
                        new_col = col.strip()

                    if old_col in self.fulldata.columns:
                        data_cols.append(old_col)
                        requested_cols.append(new_col)
                    else:
                        msg = 'ERROR: column {} not available for output'.format(col)

                df = self.fulldata[data_cols].rename(columns={k:v for k, v in zip(data_cols,requested_cols)})

                f = fout.file_path
                f.parent.mkdir(parents=True, exist_ok=True)
                if (fout.file_type == 'txt' or fout.file_type == 'csv'):
                    df.to_csv(f, index=False)
                elif fout.file_type == 'hdf5':
                    pass
                elif fout.file_type == 'vti':
                    ttk_support = ttk_basics(self.options, df, f)
                    ttk_support.write_data_to_vti()
                else:
                    msg = 'ERROR: unsupported file format for output; check --fout'
                    sys.exit(msg)

                if verbose:
                    print('dataframe for file ', f)
                    pprint(df)


    def prepare_input(self, verbose=False):

        """
        here we decode the input to '--finp'
        
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

            #self.print_options_to_log()

        self.allfinps = tuple(temp)

        if verbose:
            print('files with input data:')
            pprint(self.allfinps)

        return self.allfinps


    def prepare_output(self, verbose=False):

        """
        here we decode the input to '--fout'
        
        there are at least 2 and at most 4 arguments to --fout:

        obligatory:
        1. file type
        2. file name

        optional
        3. column names
        4. number of header lines

        """

        if self.allfouts is not None:
            print('WARNING: --fout arguments are already assigned; will be overwritten')

        temp = [] 
        for f_arg in self.options["fout"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 4:
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

        return self.allfouts


    def prepare_io(self, args):

        """
        parse lines: 
        --finp/--fout format; path; [columns]; [sep]; [skip] 
        """

        # format, path and name of a file
        f_type   = args[0].strip()
        f_path   = args[1].strip()

        # names of data fields (columns on input/output files if txt/csv)
        f_cols   = None
        if len(args) > 2:
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


    def prepare_data(self, verbose=False):

        """
        read the input data (in TXT) into pandas dataframes

        TODO: 
        * deal with empty or non-float fields
        * check whether the data has been collected on the same grids
        """

        # 1. read all input data into a list of dataframes
        dfs = []
        for v in self.allfinps:

            if v.file_type.lower() == 'txt':
                #df = pd.read_fwf(v['file_path'], colspecs='infer', header=v['file_skiprow'], names=v['file_column_names'])
                df = pd.read_fwf(v.file_path, 
                                 colspecs='infer', 
                                 skiprows = v.file_skiprow, 
                                 names=v.file_column_names)

            elif v.file_type.lower() == 'csv':
                if v.file_column_separator is None or v.file_column_separator.isspace():
                    df = pd.read_csv(v.file_path,
                                     header = 0,
                                     names  = v.file_column_names,
                                     delim_whitespace = True,
                                     skiprows = v.file_skiprow,
                                     dtype = np.float64)
                else:
                    df = pd.read_csv(v.file_path,
                                     header = 0,
                                     names  = v.file_column_names,
                                     sep = v.file_column_separator,
                                     skiprows = v.file_skiprow,
                                     dtype = np.float64)

            elif v.file_type.lower() == 'hdf5':
                pass

            df.apply(pd.to_numeric, errors='coerce')
            dfs.append(df)

        # 2. combine a list of dataframes into one dataframe;
        #    first, remove the excess 'grid' columns (now -assuming the same grids):
        for df in dfs[1:]:
            df.drop(columns=[self.grid['x'], self.grid['y'], self.grid['z']], inplace=True)

        fulldata = pd.concat([df for df in dfs], axis=1, sort=False)
        if self.fulldata.empty:
            self.fulldata = fulldata

        if verbose:
            print('Data (from prepare_data): ')
            pprint(fulldata.columns)
            pprint(fulldata)

        return fulldata


    def prepare_grid(self, verbose=False):
        """
        find which column names correspond to grid data (in csv)
        TODO: make sure the same grid is on all finp files
        """

        args = [arg.strip().strip('[').strip(']') for arg in self.options['grid'].split(',')]
        grid = {'x':args[0], 'y': args[1], 'z':args[2]}
        if self.grid == {}: 
            self.grid['x'] = args[0]
            self.grid['y'] = args[1]
            self.grid['z'] = args[2]
        if verbose:
            print('grid columns are assigned: x={}, y={}, z={}'.format(self.grid['x'],self.grid['y'],self.grid['z']))
            
        return grid


    def calculate(self, verbose=False):

        result_df = pd.DataFrame()

        if 'form_tensor_0order_3d' in self.options and self.options['form_tensor_0order_3d'] is not None:

            work = t0d3(self.options, self.grid, self.fulldata)
            work.run()

            result_df = pd.DataFrame(work.t0d3_points)
            result_df = self.update_df(result_df, new_df_cols=work.t0d3_cols)

        if 'form_tensor_2order_3d' in self.options and self.options['form_tensor_2order_3d'] is not None:

            work = t2d3(self.options, self.grid, self.fulldata)
            work.run()

            result_df = pd.DataFrame(work.t2d3_points)
            result_df = self.update_df(result_df, new_df_cols=work.t2d3_cols)


        if 'form_tensor_1order_3d' in self.options and self.options['form_tensor_1order_3d'] is not None:

            print('CHECKUP options: ', type(self.options))
            for k, v in self.options.items():
                print('     k, v: ', k, v)
            print('CHECKUP allfout: ', type(self.allfouts))
            for v in self.allfouts:
                print('     v: ', v)
            print('CHECKUP fulldata: ', type(self.fulldata))

            work = t1d3(self.options, self.allfouts, self.fulldata)
            work.run(verbose=True)
            result_df = work.work_data

        self.fulldata = pd.concat((self.fulldata, result_df), axis=1)
        self.fulldata = self.fulldata.loc[:,~self.fulldata.columns.duplicated()]

#           FIXME
            #result_df = pd.DataFrame(work.t1d3_points)
            #result_df = self.update_df(result_df, new_df_cols=work.t1d3_cols)
            #???result_df = work.update_df(result_df, new_df_cols=work.t1d3_cols)

        return result_df


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








