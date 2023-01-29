import os
import sys
import pytest
from pathlib import Path

import qcten
from .helper import *

"""
Note:
now the tests are run in the `scratch` directory created in each
test subdirectory; this can be redirected to any other folder
specified in "scratch_space"
"""

def run_test_generic(testdirs, debug=False):

    th = helper()

    # set paths:
    # to the root of test directory = th.test_space
    # to the directory of test data = th.testdata_dir
    if not th.test_space_is_set:
        th.set_test_space()

    print('gosia bu1: ', th.test_space)
    print('gosia bu2: ', th.testdata_dir)

    for testdir in testdirs:

        testdir_path = Path(testdir).absolute()
        print('gosia bu3: ', testdir_path)

        # set scratch space for test
        if not th.scratch_space_is_set:
            th.set_scratch_space(testdir)
        os.chdir(th.scratch_dir)

        for tf in os.listdir(testdir_path):
            if tf.endswith('.inp'):
                test_file = Path(testdir_path, tf)
        args = qcten.cli.read_input(finp=test_file, verbose=True)
        setup = qcten.cli.input_data(args)
        parsed_args=setup.parse_options()
        setup.print_options()
        calc = qcten.process.work(setup.runinp_dir, setup.options)
        grid = calc.prepare_grid()
        finp = calc.prepare_input(verbose=True)
        fout = calc.prepare_output(verbose=True)
        sys.exit()
        data = calc.prepare_data()
        result = calc.calculate()
        result_ref = th.get_ref_aspddataframe(Path(testdir_path, 'result.ref'))
        #sys.exit()
        #

        if debug:
            print('RESULT')
            th.debug_dump_dataframe_to_file(result)
            with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
                print(result)
            
            print('RESULT_REF')
            with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
                print(result_ref)

        same = th.same_dataframes(result, result_ref)
        assert (same == True)


def TEMPtest_t0d3_rdg_from_rho():

    testdirs = [
        "t0d3_rdg_from_rho",
        ]
    
    run_test_generic(testdirs, debug=True)


def TEMPtest_t2d3_invariants():

    testdirs = [
        "t2d3_invariants",
        ]
    
    run_test_generic(testdirs, debug=True)


def TEMPtest_t1d3_rortex_shear():

    testdirs = [
        "t1d3_rortex_shear",
        ]
    
    run_test_generic(testdirs, debug=True)

def test_t1d3_omega():

    testdirs = [
        "t1d3_omega",
        ]
    
    run_test_generic(testdirs, debug=True)


