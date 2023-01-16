import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import subprocess
from .t2d3 import *
from .t1d3 import *

class work():

    def __init__(self, options):

        # all options read from an input script
        self.options  = options

        # options for data input and output files
        self.allfinp  = {}
        self.allfout  = {}
        self.flog     = self.options['flog']

        # grid and data
        self.grid = {}
        self.grid_function = {}
        self.data = {}
        self.fulldata = pd.DataFrame()


    def prepare_input(self):

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

        allfinp = {}
        for f_arg in self.options["finp"]:

            args = f_arg.split(';')
            if len(args) < 2 or len(args) > 4:
                msg = 'ERROR: wrong number of arguments to --finp'
                sys.exit(msg)

            f_name, f_info = self.prepare_io(args)

            allfinp[f_name] = f_info
            self.allfinp[f_name] = f_info

            self.print_options_to_log()

        return allfinp


    def prepare_output(self):

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

        return allfout


    def prepare_io(self, args):

        f_type   = args[0].strip()
        f_path   = args[1].strip()
        f_name   = os.path.basename(f_path)

        f_cols   = None
        if len(args) > 2:
            if args[2].strip() != 'None':
                f_cols   = [arg.strip().strip('[').strip(']') for arg in args[2].split(',')]

        f_header = None
        if len(args) > 3:
            if args[3].strip() != 'None':
                header = int(args[3].strip())

        d = {}
        d['file_type'] = f_type
        d['file_path'] = f_path
        d['file_column_names'] = f_cols
        d['file_header'] = f_header

        return f_name, d


    def prepare_data(self):

        """
        read the input data (in TXT) into pandas dataframes

        TODO: 
        * find what is better to assure floats (dtype = np.float or pd.to_numeric)
        * deal with empty or non-float fields
        * check whether the data has been collected on the same grids
        """

        # 1. read all input data into a list of dataframes
        dfs = []
        for k, v in self.allfinp.items():

            if v['file_type'].lower() == 'txt':
                df = pd.read_fwf(v['file_path'], colspecs='infer', header=v['file_header'])

            #df = pd.read_csv(v['file_name'],
            #                 sep    = v['sep'],
            #                 header = v['header'],
            #                 names  = v['column_names'],
            #                 dtype = np.float64)

            df.apply(pd.to_numeric, errors='coerce')

            self.data[k] = df
            dfs.append(df)

        # 2. combine a list of dataframes into one dataframe

        # this removes all duplicates:
        # self.fulldata = pd.concat([df for df in self.data.values()], axis=1, sort=False).T.drop_duplicates().T
        # but since it might be problematic if we have two columns with the same exact values (e.g. from calculations
        # with symmetry), it's better just to prune the excess 'grid' columns, so we do instead:
        for df in dfs[1:]:
            df.drop(columns=[self.grid['grid_x'], self.grid['grid_y'], self.grid['grid_z']], inplace=True)

        fulldata = pd.concat([df for df in dfs], axis=1, sort=False)
        if self.fulldata.empty:
            self.fulldata = fulldata

        return fulldata


    def prepare_grid(self):
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
        return grid


    def calculate(self):

        # todo: tu przenies assign vector/assign tensor

        if 'form_tensor_2order_3d' in self.options and self.options['form_tensor_2order_3d'] is not None:

            work = t2d3(self.options, self.grid, self.fulldata)
            work.run()

            result_df = pd.DataFrame(work.t2d3_points)
            result_df = self.update_df(result_df, new_df_cols=work.t2d3_cols)


        if 'form_vector_3d' in self.options and self.options['form_vector_3d'] is not None:

            work = t1d3(self.options, self.grid, self.fulldata)
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

            if self.options['data_out'] is not None:
                cols = list(cols)
                #cols = list(cols.replace('x', 'grid_x').replace('y', 'grid_y').replace('z', 'grid_z'))
                #cols = ['grid_x', 'grid_y', 'grid_z'] + [ c for c in cols if c not in ['grid_x', 'grid_y', 'grid_z']]
                for i, col in enumerate(cols):
                    if col == 'x':
                        cols[i] = col.replace('x', 'grid_x')
                    if col == 'y':
                        cols[i] = col.replace('y', 'grid_y')
                    if col == 'z':
                        cols[i] = col.replace('z', 'grid_z')
                cols = ['grid_x', 'grid_y', 'grid_z'] + [ c for c in cols if c not in ['grid_x', 'grid_y', 'grid_z']]
            else:
                # reorder, so that the grid points are always in the beginning:
                cols = ['grid_x', 'grid_y', 'grid_z'] + [ c for c in cols if c not in ['grid_x', 'grid_y', 'grid_z']]

            with open(self.flog, 'a') as f:
                f.write('----------------- columns written to output file -------------------\n')
                f.write('{}'.format(cols))

            fulldata = new_df[list(cols)]

        else:
            fulldata = new_df

        for col in fulldata.columns:
            if col in self.options["data_out_newnames"]:
                newcol = self.options["data_out_newnames"][col]
                fulldata.rename(columns={col:newcol}, inplace=True)

        self.fulldata = fulldata
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








