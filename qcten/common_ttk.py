import pandas as pd
import numpy as np
import os
import sys
from pprint import pprint
from pathlib import Path
from paraview.simple import *


class ttk_basics():

    def __init__(self, options, finp_csv, fout_vti):
        """
        """
        self.options = options
        #self.data = data 
        self.finp_csv = finp_csv # csv file with data
        self.fout_vti = fout_vti # vti file to write to
        #self.inpgrid_dim = [int(x) for x in self.options['resampled_dim'].split(',')]
        self.resampled_dim = [int(x) for x in self.options['resampled_dim'].split(',')]


    def write_data_to_vti(self):

        denscsv = CSVReader(FileName=self.finp_csv)
        print('In write_data_to_vti: ', type(denscsv))
        pprint(denscsv)
        #npoints = self.common_options['npoints'].split(',')
        #end_x = int(npoints[0]) - 1
        #end_y = int(npoints[1]) - 1
        #end_z = int(npoints[2]) - 1

        end_x = 9
        end_y = 9
        end_z = 9

        tableToStructuredGrid = TableToStructuredGrid(Input=denscsv)
        tableToStructuredGrid.WholeExtent = [0, end_x, 0, end_y, 0, end_z]
        tableToStructuredGrid.XColumn = 'x'
        tableToStructuredGrid.YColumn = 'y'
        tableToStructuredGrid.ZColumn = 'z'

        final_data = tableToStructuredGrid

        ##final_data = AppendAttributes(Input=data)

        finalResampleToImage = ResampleToImage(Input=final_data)
        finalResampleToImage.SamplingDimensions = [self.resampled_dim[0],
                                                   self.resampled_dim[1],
                                                   self.resampled_dim[2]]

        SaveData(self.fout_vti.as_posix(), proxy=finalResampleToImage)





def resample(data, resampled_dim=[256,256,256]):

    print('resampleToImage filter applied; sampling dimensions: ', resampled_dim)

    resampleToImage = ResampleToImage(Input=data)

    resampleToImage.SamplingDimensions = [resampled_dim[0],
                                          resampled_dim[1],
                                          resampled_dim[2]]



