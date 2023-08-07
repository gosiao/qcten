import os
import sys
import shutil
import pytest
from pathlib import Path

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
    print('FILES!!!! ', testdir_path)
    for tf in os.listdir(testdir_path):
        if tf.endswith('.inp'):
            test_file = Path(testdir_path, tf)
    print('FILES!!!! ', test_file)

    os.chdir(th.test_space)

    #
    # 4. run test
    #
    args = qcten.cli.read_input(finp=test_file, verbose=verbose)

    setup = qcten.cli.input_data(args)
    setup.parse_options()
    setup.print_options()

    calc = qcten.process.work(testdir_path, setup.options)
    calc.run(verbose=verbose)

    #temp
    #result=qcten.main.run(args_file=test_file, verbose=verbose)
    #th.put_refdataframe(Path(th.scratch_dir, 'fulldata.tmp'), result)


    #
    # 5. compare output files with reference files
    #
    same = []
    supported_extensions = ['.csv', '.vti']
    for e in supported_extensions:
        for f in os.listdir(testdir_path):
            if f.endswith(e):
                f_test = Path(testdir_path, f)
                f_ref  = Path(testdir_path, 'reference', f)
                if f_ref.exists():
                    same.append(th.same_files(f_test, f_ref))

    assert (all(x==True for x in same))


def cleanup(testdir):

    verbose=True

    th = helper()

    if not th.test_space_is_set:
        th.set_test_space()

    if os.path.isdir(testdir):
        testdir_path = Path(testdir).absolute()
        print('CLEANUP: testdir_path = ', testdir_path)

        if not th.scratch_space_is_set:
            th.set_scratch_space(testdir)

        print('CLEANUP: th.scratch_dir = ', th.scratch_dir)
        #shutil.rmtree(th.scratch_dir)

        supported_extensions = ['.csv', '.vti']
        for e in supported_extensions:
            for f in os.listdir(testdir_path):
                if f.endswith(e):
                    f_test = Path(testdir_path, f)
                    #os.remove(f_test)


def list_viable_tests(test_paths=None):

    testdirs = [
                "t0d3_vti_from_txt",
                "t1d3_omega",
                "t1d3_norm_mean"
                ]
    #testdirs = ["t0d3_vti_from_txt"]
    #testdirs = ["t0d3_rdg_from_rho",
    #            "t1d3_rortex_shear",
    #            "t1d3_omega",
    #            "t2d3_invariants"]

    return testdirs


def test_viable_tests():
    for t in list_viable_tests():
        print('RUNNING NOW!!! ', t)
        run_test_generic(t)
        #cleanup(t)





