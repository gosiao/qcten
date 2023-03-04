import os
import sys
import pytest
from pathlib import Path
from deepdiff import DeepDiff


import qcten
from .helper import *


def run_test_generic(testdir):

    verbose=False

    th = helper()

    #
    # 1. set paths:
    #    th.test_space = path to the root of test directory
    #    th.testdata_dir = path to the directory of test data
    #
    if not th.test_space_is_set:
        th.set_test_space(verbose=verbose)


    testdir_path = Path(testdir).absolute()

    #
    # 2. set paths to the scratch space for this test
    #
    if not th.scratch_space_is_set:
        th.set_scratch_space(testdir, verbose=verbose)

    os.chdir(th.scratch_dir)

    #
    # 3. find input file
    #
    for tf in os.listdir(testdir_path):
        if tf.endswith('.inp'):
            test_file = Path(testdir_path, tf)

    os.chdir(th.test_space)

    #
    # 4. run test - test in steps
    #
    args = qcten.cli.read_input(finp=test_file, verbose=verbose)
    th.put_reflist(Path(th.scratch_dir, 'args.tmp'), args[1:])

    setup = qcten.cli.input_data(args)
    setup.parse_options()
    options = setup.options
    th.put_refdict(Path(th.scratch_dir, 'options.tmp'), options)

    calc = qcten.process.work(testdir_path, setup.options)
    grid = calc.prepare_grid()
    th.put_refdict(Path(th.scratch_dir, 'grid_names.tmp'), grid)

    #
    # 5. compare output files with reference files
    #
    same = []
    files_to_compare = ['args', 'options', 'grid_names']
    for c in files_to_compare:
        f_test = Path(th.scratch_dir, c+'.tmp')
        f_ref  = Path(testdir_path, 'reference', c+'.ref')
        if f_ref.exists() and f_test.exists():
            same.append(th.same_files(f_test, f_ref))
        else:
            sys.exit('missing file!')

    assert (all(x==True for x in same))



def list_viable_tests(test_paths=None):
    testdirs = [
        "inptest_calc_t0",
        "inptest_calc_t1",
        #"inptest_calc_t2"
        ]
    return testdirs


def test_viable_tests():
    for t in list_viable_tests():
        run_test_generic(t)




