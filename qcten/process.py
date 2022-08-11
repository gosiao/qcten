import pandas as pd
import numpy as np
import os
from pathlib import Path
import subprocess

class work():

    def __init__(self, options):

        # all options read from an input run file
        self.options  = options

        # grid
        self.grid     = {}

        self.data     = {}
        self.fulldata = pd.DataFrame()
        self.allfinp  = {}
        self.flog     = self.options['flog']


    def parse_finp(self):

        """
        here we decode the input to '--finp'
        
        there are 4 arguments to --finp:
        1. file name
        2. column names
        3. column separator
        4. number of header lines to skip

        there can be many --finp blocks (self.options["finp"] is a list)
        """

        if self.allfinp is not None:
            print('WARNING: --finp arguments are already assigned; will be overwritten')

        allfinp = {}
        for f_arg in self.options["finp"]:

            args = f_arg.split(';')
            if len(args) < 4:
                msg = 'ERROR: not enough arguments to finp'
                sys.exit(msg)

            finp   = args[0].strip()
            cols   = [arg.strip().strip('[').strip(']') for arg in args[1].split(',')]
            sep    = args[2].strip()
            header = int(args[3].strip())

            d = {}
            d['file_name']     = finp
            d['column_names']  = cols
            d['sep']           = sep
            d['header']        = header

            allfinp[finp] = d
            self.allfinp[finp] = d

            #if self.allfinp[finp] is None:
            #    self.allfinp[finp] = d

            self.print_options_to_log()

        return allfinp


    def prepare_data(self):

        """
        read the input data into pandas dataframes

        TODO: 
        * find what is better to assure floats (dtype = np.float or pd.to_numeric)
        * deal with empty or non-float fields
        * check whether the data has been collected on the same grids
        """

        # 1. read all input data into a list of dataframes
        dfs = []
        for k, v in self.allfinp.items():

            df = pd.read_csv(v['file_name'],
                             sep    = v['sep'],
                             header = v['header'],
                             names  = v['column_names'],
                             dtype = np.float64)

            df.apply(pd.to_numeric, errors='coerce')

            self.data[v['file_name']] = df
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

        #self.print_inputdata_to_csv()

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



    def print_options_to_log(self):

        if self.options['flog'] is not None:

            with open(self.options['flog'], 'w') as f:

                f.write("--------------------------- main job options ---------------------------\n")
                f.write("qcten SHA: {}\n".format(subprocess.check_output(["git", "describe", "--always"], cwd=os.path.dirname(__file__)).strip().decode()))
                f.write("\n")
                for k, v in self.options.items():
                    f.write("{:<45}: {}\n".format(k, v))
                f.write("\n")

                for kf, vf in self.allfinp.items():
                    f.write("------------------ set of options for input files ------------------\n")
                    for k, v in vf.items():
                        f.write("{:<15}: {}\n".format(k, v))
                f.write("\n")


    def print_inputdata_to_csv(self):

        if self.options['flog'] is not None:

            finp = 'testinpdata.csv'
            finp = Path('/home/gosia/out.csv')
            finp.parent.mkdir(parents=True, exist_ok=True)
            if not self.fulldata.empty:
                self.fulldata = self.fulldata.astype(np.float64)
                self.fulldata.to_csv(finp, index=False)






