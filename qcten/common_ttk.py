import pandas as pd
import numpy as np
import os
import sys
from pprint import pprint
from pathlib import Path
from paraview.simple import *


class ttk_basics():

    def __init__(self, options, data, fout_vti, finp_csv=None):
        """
        """
        self.options = options
        self.data = data 
        self.finp_csv = finp_csv
        self.fout_vti = fout_vti
        #self.inpgrid_dim = [int(x) for x in self.options['resampled_dim'].split(',')]
        self.resampled_dim = [int(x) for x in self.options['resampled_dim'].split(',') if self.options['resampled_dim']]


    def write_data_to_vti(self):

        if self.finp_csv is None:
            # temporary solution...
            fcsv = 'temp.csv'
            self.data.to_csv(fcsv, index=False)
        else:
            fcsv = self.finp_csv

        denscsv = CSVReader(FileName=fcsv)
        print('In write_data_to_vti: ', type(denscsv))
        pprint(denscsv)
        if self.options['grid_type'] == 'uniform_rectilinear':
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



    def apply_gradientOfUnstructuredDataSet(self, source, source_type, source_name, result_name):
        """
        source      = input TTK object
        source_type = 'POINTS' or 'CELLS'
        source_name = string, depends on the "source"
        result_name = name of the calculated gradient
        """
    
        gradient = GradientOfUnstructuredDataSet(Input=source)
        gradient.ScalarArray = [source_type, source_name]
        gradient.ResultArrayName = result_name
    
        return gradient
    
    
    def calculator(self, name, function, source_file=None, source_other=None):
    
        if source_file is not None:
            inpdata = XMLImageDataReader(FileName=source_file)
        elif source_other is not None:
            inpdata = source_other
    
        calculator = Calculator(Input=inpdata)
    
        calculator.ResultArrayName = name
        calculator.Function = function
    
        debug = False
        if debug:
            print('output from ttk_helper.calculator: source_file=  ', source_file)
            print('output from ttk_helper.calculator: source_other= ', source_other)
            print('output from ttk_helper.calculator: name=         ', name)
            print('output from ttk_helper.calculator: function=     ', function)
    
        return calculator





