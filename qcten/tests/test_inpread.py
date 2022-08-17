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

def test_read_input():

    testdirs = [
        "inptest_calc_t1",
        "inptest_calc_t2"
        ]
    
    th = helper()
    if not th.testspace_is_set:
        th.set_testspace()
    os.chdir(th.scratch_dir)

    for testdir in testdirs:

        this_test = os.path.join(th.testinp_dir, testdir)

        # test reading options from *.inp file
        for tf in os.listdir(this_test):
            if tf.endswith('.inp'):
                test_file = os.path.join(this_test, tf)
        args=qcten.main.read_input(test_file, verbose=True)

        # test reading options from command line
        # TODO

        # reference arguments
        args_ref = th.get_ref_aslist(Path(os.path.join(this_test, 'args.ref')))

        error = set(args) ^ set(args_ref)
        assert not error


def test_parse_options():

    testdirs = [
        "inptest_calc_t1",
        "inptest_calc_t2"
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
        args=qcten.main.read_input(test_file, verbose=True)

        setup = qcten.prepare_input.input_data(args)
        setup.parse_options()
        options = setup.options
        for option_key, option_value in options.items():
            if option_value is not None:
                # easier to compare strings:
                options[option_key] = str(option_value)
        print('RESULT TEST for testdir ', testdir)
        for k, v in options.items():
            print(k,":",v)

        # reference options:
        options_ref = th.get_ref_asdict(Path(os.path.join(this_test, 'options.ref')))
        print('RESULT REF for testdir ', testdir)
        for k, v in options_ref.items():
            print(k,":",v)

        diff = th.diff_dicts(options, options_ref)
        assert (diff == {})


def test_prepare_grid():

    testdirs = [
        "inptest_calc_t1",
        "inptest_calc_t2"
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
        result = calc.prepare_grid()

        # reference arguments
        result_ref = th.get_ref_asdict(Path(os.path.join(this_test, 'grid_names.ref')))

        diff = th.diff_dicts(result, result_ref)
        assert (diff == {})

def FIXMEtest_parse_finp():
    """write_me"""

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

        # test reading options from *.inp file
        for tf in os.listdir(this_test):
            if tf.endswith('.inp'):
                test_file = os.path.join(this_test, tf)
        args = qcten.prepare_input.read_input(finp=test_file, verbose=True)
        setup = qcten.prepare_input.input_data(args)
        setup.parse_options()
        calc = qcten.process.work(setup.options)
        grid = calc.prepare_grid()
        result = calc.parse_finp()
        # --- DEBUG ---
        #for k, v in result.items():
        #    if isinstance(v, dict):
        #        for k1, v1 in v.items():
        #            print('RESULT TEST ', type(k1), ' :::: ', type(v1))
        #            print('RESULT TEST ', k1, ' :::: ', v1)

        # reference arguments
        result_ref = th.get_ref_asdict(Path(os.path.join(this_test, 'finp_args.ref')))
        # --- DEBUG ---
        #result_ref = {'../testdata/lih_jbtensor_cubegrid10/data.csv':{'file_name': '../testdata/lih_jbtensor_cubegrid10/data.csv', 'column_names': ['x', 'y', 'z', 'bx_jx', 'bx_jy', 'bx_jz', 'by_jx', 'by_jy', 'by_jz', 'bz_jx', 'bz_jy', 'bz_jz'], 'sep': '\\s+', 'header': 1}}
        #for k, v in result_ref.items():
        #    #print('RESULT REF  ', type(k), ' :::: ', type(v))
        #    #print('RESULT REF  ', k, ' :::: ', v)
        #    if isinstance(v, dict):
        #        for k1, v1 in v.items():
        #            print('RESULT REF ', type(k1), ' :::: ', type(v1))
        #            print('RESULT REF ', k1, ' :::: ', v1)



        diff = th.diff_dicts(result, result_ref)
        assert (diff == {})


def test_prepare_data():

    testdirs = [
        "inptest_calc_t1",
        "inptest_calc_t2"
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
        # --- DEBUG ---
        #th.debug_dump_dataframe_to_file(result)
        #print('RESULT')
        #with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
        #    print(result)

        # reference arguments
        result_ref = th.get_ref_aspddataframe(Path(os.path.join(this_test, 'fulldata.ref')))
        # --- DEBUG ---
        #print('RESULT_REF')
        #with pd.option_context('display.max_rows', 10, 'display.max_columns', 12):
        #    print(result_ref)


        same = th.same_dataframes(result, result_ref)
        assert (same)


