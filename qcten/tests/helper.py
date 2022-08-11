import os
import numpy as np
import pandas as pd
from pathlib import Path
from deepdiff import DeepDiff

class helper:

    def __init__(self):
        self.test_space = None
        self.testdata_dirname = None
        self.testinp_dir = None
        self.scratch_dirname = None
        self.scratch_dir = None
        self.testspace_is_set = False

    def set_testspace(self):
    
        test_space = Path(__file__).resolve().parent
        testdata_dirname = "testdata"
        testinp_dir = test_space
        scratch_dirname = "scratch"
        scratch_dir = os.path.join(test_space, scratch_dirname)
        os.makedirs(scratch_dir, exist_ok=True)

        if not self.testspace_is_set:
            self.test_space = test_space
            self.testdata_dirname = testdata_dirname
            self.testinp_dir = testinp_dir
            self.scratch_dirname = scratch_dirname
            self.scratch_dir = scratch_dir
            self.testspace_is_set = True
    
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
        if assert_frame_equal(df1, df2) is None:
            return True
        else:
            return False

    # --- debug help ---
    def debug_dump_dataframe_to_file(self, df, fout=None):
        if fout is None:
            if self.scratch_dir is None:
                self.set_testspace()
            fout = Path(os.path.join(self.scratch_dir, 'temp.csv'))
            fout.parent.mkdir(parents=True, exist_ok=True)
        if not df.empty:
            df = df.astype(np.float64)
            df.to_csv(fout, index=False)





