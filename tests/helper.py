import os
import numpy as np
import pandas as pd
from pathlib import Path
from deepdiff import DeepDiff
import filecmp
import qcten

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
    
 

    def put_reflist(self, fout, ref_list):
        with open(fout, 'w') as f:
            for line in ref_list:
                f.write(line+'\n')
    
    def put_refdict(self, fout, ref_dict):
        with open(fout, 'w') as f:
            for k, v in ref_dict.items():
                f.write('{} : {}\n'.format(k, str(v)))

    def put_refdataframe(self, fout, df):
        if not df.empty:
            df = df.astype(np.float64)
            #df.dropna(axis=0,how='any',inplace=True)
            df.to_csv(fout, index=False)


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

    def same_files(self, f1, f2):
        p1 = Path(f1).resolve()
        p2 = Path(f1).resolve()
        if p1.exists() and p1.stat().st_size > 0 and p2.exists() and p2.stat().st_size > 0: 
            check = filecmp.cmp(f1,f2)
            if check:
                print("{} and {} are the same".format(f1,f2))
            else:
                print("{} and {} are different".format(f1,f2))
            return check





