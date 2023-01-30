import pandas as pd
import numpy as np
import os
import sys
from pprint import pprint
from pathlib import Path
from paraview.simple import *


class ttk_basics():

    def __init__(self, options, data, fout_vti):
        """
        """
        self.options = options
        self.data = data 
        self.fout_vti = fout_vti # vti file to write to
        #self.inpgrid_dim = [int(x) for x in self.options['resampled_dim'].split(',')]
        self.resampled_dim = [int(x) for x in self.options['resampled_dim'].split(',') if self.options['resampled_dim']]


    def write_data_to_vti(self):

        fcsv = 'temp.csv'
        self.data.to_csv(fcsv, index=False)

        denscsv = CSVReader(FileName=fcsv)
        print('In write_data_to_vti: ', type(denscsv))
        pprint(denscsv)
        if self.options['grid_type'] == 'uniform':
            npoints = len(self.data.index)  
            n1dim = np.cbrt(npoints)
            if n1dim.is_integer():
                nx = int(n1dim)-1
                ny = int(n1dim)-1
                nz = int(n1dim)-1
            else:
                msg='ERROR: non-integer number of grid points (write_data_to_vti)', n1dim
                sys.exit(msg)

            tableToStructuredGrid = TableToStructuredGrid(Input=denscsv)
            tableToStructuredGrid.WholeExtent = [0, nx, 0, ny, 0, nz]
            tableToStructuredGrid.XColumn = 'x'
            tableToStructuredGrid.YColumn = 'y'
            tableToStructuredGrid.ZColumn = 'z'

            final_data = tableToStructuredGrid
            #final_data = AppendAttributes(Input=data)
    
            finalResampleToImage = ResampleToImage(Input=final_data)

            if self.resampled_dim is not None:
                # else - resampled dimensions are the same as original ones
                nx = self.resampled_dim[0]
                ny = self.resampled_dim[1]
                nz = self.resampled_dim[2]

            finalResampleToImage.SamplingDimensions = [nx, ny, nz]
    
            SaveData(self.fout_vti.as_posix(), proxy=finalResampleToImage)

        else:
            msg='ERROR: grids other than uniform are not supported'
            sys.exit(msg)





def resample(data, resampled_dim=[256,256,256]):

    print('resampleToImage filter applied; sampling dimensions: ', resampled_dim)

    resampleToImage = ResampleToImage(Input=data)

    resampleToImage.SamplingDimensions = [resampled_dim[0],
                                          resampled_dim[1],
                                          resampled_dim[2]]



