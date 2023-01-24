import os
import numpy as np
import pandas as pd
from pathlib import Path
from deepdiff import DeepDiff
from .. import process

class helper:

    def __init__(self):
        self.test_space = None               # the root of 'tests' directory
        self.testdata_dir = None             # the directory with data used for tests
        self.testinp_dir = None              # the directory with one testcase
        self.scratch_dirname = "scratch"     # name of 'scratch' directory
        self.scratch_dir = None              # path to 'scratch' directory
        self.test_space_is_set = False
        self.scratch_space_is_set = False


    def set_test_space(self, verbose=True):
    
        if not self.test_space_is_set:
            self.test_space = Path(__file__).resolve().parent
            self.testdata_dir = Path(self.test_space, "testdata")
            self.test_space_is_set = True

        if verbose:
            print('test space: ')
            print('  - the root of tests directory:                    ', self.test_space)
            print('  - the directory with data used for tests:         ', self.testdata_dir)
    


    def set_scratch_space(self, test_dir_name, verbose=True):
    
        if not self.scratch_space_is_set:
            self.scratch_dir = Path(self.test_space, self.scratch_dirname, test_dir_name)

            os.makedirs(self.scratch_dir, exist_ok=True)
            self.scratch_space_is_set = True

        if verbose:
            print('scratch space: ')
            print('  - the name of scratch directory: ', self.scratch_dirname)
            print('  - the path to scratch directory: ', self.scratch_dir)
    
 

    def get_ref_aslist(self, finp):
        with open(finp, 'r') as f:
            result = f.read().splitlines()
        return result
    
    def get_ref_asdict(self, finp):
        result = {}
        result_list = self.get_ref_aslist(finp)
        for o in result_list:
            if o:
                temp=self.str_to_dict(o, occ=1)
                for k, v in temp.items():
                    result[k] = v
        return result

    def str_to_dict(self, s, occ=None):
        if occ is None:
            occ = -1
        result = {}
        k = s.split(':', maxsplit=occ)[0].strip()
        v = s.split(':', maxsplit=occ)[1].strip()
        if v != 'None':
            result[k]=v
            #if isinstance(v, str) and ':' in v:
            #    temp = self.str_to_dict(v)
            #    result[k]=temp
            #else:
            #    result[k]=v
        return result

    def get_ref_aspddataframe(self, finp):
        #result = pd.read_fwf(finp, colspecs='infer', header=None)
        #result = pd.read_csv(finp, header=0, dtype = np.float64)

        result = pd.read_csv(finp, header=0, dtype = np.float64)
        result.apply(pd.to_numeric, errors='coerce')
        return result

    def diff_dicts(self, d1, d2):
        return DeepDiff(d1, d2, ignore_string_case=True)


    def same_dataframes(self, df1, df2, atol_list=None):
        from pandas.testing import assert_frame_equal
        if assert_frame_equal(df1, df2, check_like=True) is None:
            return True
        else:
            return False

    # --- debug help ---
    def debug_dump_dataframe_to_file(self, df, fout=None):
        if fout is None:
            if self.scratch_dir is None:
                self.set_test_space()
            fout = Path(os.path.join(self.scratch_dir, 'temp.csv'))
            fout.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df = df.astype(np.float64)
            df.to_csv(fout, index=False)
            #df.dropna(axis=0,how='any',inplace=True)
            #df.to_csv(fout, index=False)





