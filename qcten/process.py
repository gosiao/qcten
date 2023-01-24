import pandas as pd
import numpy as np
import os
import sys
import re
from pathlib import Path
import subprocess
from .t2d3 import *
from .t1d3 import *

class work():

    def __init__(self, rundir, options):

        # all options read from an input script
        self.options  = options
        self.rundir = rundir

        # options for data input and output files
        self.allfinp  = {}
        self.allfout  = {}
        self.flog     = self.options['flog']

        # grid and data
        self.grid = {}
        self.grid_function = {}
        self.data = {}
        self.fulldata = pd.DataFrame()


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

        there can be many --finp blocks (self.options["finp"] is a list)
        """

        if self.allfinp is not None:
            print('WARNING: --finp arguments are already assigned; will be overwritten')
            for k, v in self.allfinp.items():
                print('k, v : ', k, v)

        allfinp = {}
        for f_arg in self.options["finp"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 5:
                msg = 'ERROR: wrong number of arguments to --finp'
                sys.exit(msg)

            f_name, f_info = self.prepare_io(args)
            if not Path(f_info['file_path']).exists():
                sys.exit()

            allfinp[f_name] = f_info
            self.allfinp[f_name] = f_info

            self.print_options_to_log()

        if verbose:
            for k, v in self.allfinp.items():
                print('assigned self.allfinp.items = k, v : ', k, v)

        return allfinp


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

        there can be many --fout blocks (self.options["fout"] is a list)
        """

        if self.allfout is not None:
            print('WARNING: --fout arguments are already assigned; will be overwritten')

        allfout = {}
        for f_arg in self.options["fout"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 4:
                msg = 'ERROR: wrong number of arguments to --fout'
                sys.exit(msg)

            f_name, f_info = self.prepare_io(args)

            allfout[f_name] = f_info
            self.allfout[f_name] = f_info

        if verbose:
            for k, v in self.allfout.items():
                print('assigned self.allfout.items = k, v : ', k, v)

        return allfout


    def prepare_io(self, args):

        """
        --finp/--fout format; path; [columns]; [sep]; [skip] 
        """

        f_type   = args[0].strip()
        f_path   = args[1].strip()
        f_name   = os.path.basename(f_path)

        # names of data fields (columns on input/output files if txt/csv)
        f_cols   = None
        f_old_cols = []
        f_new_cols = []
        if len(args) > 2:
            arg = args[2].strip()
            if arg != 'None':
                if arg[0:5] == 'cols=':
                    f_cols   = [a.strip().strip('[').strip(']') for a in arg[5:].split(',')]
                    f_old_cols = [None for x in range(len(f_cols))]
                    f_new_cols = [None for x in range(len(f_cols))]
                    for i, f_col in enumerate(f_cols):
                        if ':' in f_col:
                            f_col_initial=f_col.split(':')[0].strip()
                            f_col_renamed=f_col.split(':')[1].strip()
                            f_old_cols[i] = f_col_initial
                            f_new_cols[i] = f_col_renamed

        # column separator; defaults to a coma
        f_sep = ','
        if len(args) > 3:
            arg = args[3].lstrip()
            if arg != 'None':
                if arg[0:4] == 'sep=':
                    f_sep = arg[4:]

        # index of a row line to skip (e.g. with file description)
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
        d['file_column_old_names'] = f_old_cols
        d['file_column_new_names'] = f_new_cols
        d['file_column_separator'] = f_sep
        d['file_skiprow'] = f_skiprow

        return f_name, d


    def prepare_data(self):

        """
        read the input data (in TXT) into pandas dataframes

        TODO: 
        * deal with empty or non-float fields
        * check whether the data has been collected on the same grids
        """

        # 1. read all input data into a list of dataframes
        dfs = []
        for k, v in self.allfinp.items():

            if v['file_type'].lower() == 'txt':
                #df = pd.read_fwf(v['file_path'], colspecs='infer', header=v['file_skiprow'], names=v['file_column_names'])
                df = pd.read_fwf(v['file_path'], 
                                 colspecs='infer', 
                                 skiprows = v['file_skiprow'], 
                                 names=v['file_column_names'])

            elif v['file_type'].lower() == 'csv':
                if v['file_column_separator'] is None or v['file_column_separator'].isspace():
                    df = pd.read_csv(v['file_path'],
                                     header = 0,
                                     names  = v['file_column_names'],
                                     delim_whitespace = True,
                                     skiprows = v['file_skiprow'],
                                     dtype = np.float64)
                else:
                    df = pd.read_csv(v['file_path'],
                                     header = 0,
                                     names  = v['file_column_names'],
                                     sep = v['file_column_separator'],
                                     skiprows = v['file_skiprow'],
                                     dtype = np.float64)

            elif v['file_type'].lower() == 'hdf5':
                pass


            df.apply(pd.to_numeric, errors='coerce')

            self.data[k] = df
            dfs.append(df)

        # 2. combine a list of dataframes into one dataframe;
        #    first, remove the excess 'grid' columns (now -assuming the same grids):
        for df in dfs[1:]:
            df.drop(columns=[self.grid['grid_x'], self.grid['grid_y'], self.grid['grid_z']], inplace=True)

        fulldata = pd.concat([df for df in dfs], axis=1, sort=False)
        if self.fulldata.empty:
            self.fulldata = fulldata


        return fulldata


    def prepare_grid(self, verbose=False):
        """
        find which column names correspond to grid data (in csv)
        TODO: make sure the same grid is on all finp files
        """

        args = [arg.strip().strip('[').strip(']') for arg in self.options['grid'].split(',')]
        grid = {'grid_x':args[0], 'grid_y': args[1], 'grid_z':args[2]}
        if self.grid == {}: 
            self.grid['grid_x'] = args[0]
            self.grid['grid_y'] = args[1]
            self.grid['grid_z'] = args[2]
        if verbose:
            print('grid columns are assigned: grid_x={}, grid_y={}, grid_z={}'.format(self.grid['grid_x'],self.grid['grid_y'],self.grid['grid_z']))
            
        return grid


    def calculate(self):

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

            work = t1d3(self.options, self.allfout, self.grid, self.fulldata)
            work.run()

            result_df = pd.DataFrame(work.t1d3_points)
            result_df = self.update_df(result_df, new_df_cols=work.t1d3_cols)

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

                    fulldata = fulldata.reindex(columns = (['grid_x', 'grid_y', 'grid_z'] + [ c for c in fulldata.columns if c not in ['grid_x', 'grid_y', 'grid_z']]))
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








