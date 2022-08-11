import os
import sys
import pytest
from pathlib import Path

import qcten
from .helper import *

"""
Note:
tests are run in scratch directories created in each
test subdirectory; this can be redirected to any other folder
specified in "scratch_space"
"""

def FIXMEtest_prepare_data():
    """RESTART HERE"""

    testdirs = [
        "inptest_calc_t1",
        #"inptest_calc_t2"
        ]
    
    th = helper()
    if not th.testspace_is_set:
        th.set_testspace()
    os.chdir(th.scratch_dir)

    for testdir in testdirs:

        this_test = os.path.join(th.testinp_dir, testdir)

        for tf in os.listdir(this_test):
            if tf.endswith('.inp'):
                test_file = os.path.join(this_test, tf)
        args = qcten.prepare_input.read_input(finp=test_file, verbose=True)
        setup = qcten.prepare_input.input_data(args)
        setup.parse_options()
        calc = qcten.process.work(setup.options)
        grid = calc.prepare_grid()
        finp = calc.parse_finp()
        result = calc.prepare_data()
        print('RESULT')
        with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
            print(result)

        # reference arguments
        result_ref = th.get_ref_aspddataframe(Path(os.path.join(this_test, 'fulldata.ref')))
        print('RESULT_REF')
        with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
            print(result_ref)


        same = th.same_dataframes(result, result_ref)
        assert (same == True)


