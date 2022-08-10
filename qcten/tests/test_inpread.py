import os
import sys
import pytest
import qcten
from pathlib import Path

"""
here we test functions that read and process input files
tests cover functions in 
main.py, process.py and prepare_input.py

tests are run in scratch directories created in each
test subdirectory
"""

def get_ref_args(finp):
    with open(finp, 'r') as f:
        args = f.read().splitlines()
    return args

def get_ref_options(finp):
    options = {}
    with open(finp, 'r') as f:
        options_list = f.read().splitlines()
    for o in options_list:
        k = o.split(':')[0].strip()
        v = o.split(':')[1].strip()
        options[k]=v
    return options

def test_inpread():
    """Testing whether input files are read correctly."""

    testdirs = [
        "inptest_t2"
        ]
    
    for testdir in testdirs:

        here = Path(__file__).resolve().parent
        scratch = os.path.join(here, testdir, 'scratch')
        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
    
        # test reading options from *.inp file
        for tf in os.listdir(os.path.join(here, testdir)):
            if tf.endswith('.inp'):
                test_file = os.path.join(here, testdir, tf)
        args=qcten.main.read_input(test_file, verbose=True)
        #args=qcten.main.run(args_file=test_file, verbose=True)

        # test reading options from command line

        # reference arguments
        args_ref = get_ref_args(Path(os.path.join(here, testdir, 'args_t2')))

        os.chdir(here)

        error = set(args) ^ set(args_ref)
        assert not error


def test_parse_options():
    """Testing whether arguments are correctly parsed."""

    testdirs = [
        "inptest_t2"
        ]
    
    for testdir in testdirs:

        here = Path(__file__).resolve().parent
        scratch = os.path.join(here, testdir, 'scratch')
        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
    
        # test reading options from *.inp file
        for tf in os.listdir(os.path.join(here, testdir)):
            if tf.endswith('.inp'):
                test_file = os.path.join(here, testdir, tf)
        args=qcten.main.read_input(test_file, verbose=True)
        setup = qcten.prepare_input.input_data(args)
        setup.parse_options()
        options = setup.options
        for option_key, option_value in options.items():
            print("TEST - {}: {}".format(option_key, option_value))

        # reference options:
        options_ref = get_ref_options(Path(os.path.join(here, testdir, 'options_t2')))
        for option_key, option_value in options_ref.items():
            print("REF  - {}: {}".format(option_key, option_value))


        os.chdir(here)

        error = set(options) ^ set(options_ref)
        assert not error


def FIXMEtest_inptest():
    """Testing whether input files are read correctly."""

    testdirs = [
        "inptest_t2"
        ]
    
    for testdir in testdirs:

        # run test in scratch directory:
        here = Path(__file__).resolve().parent
        scratch = os.path.join(here, testdir, 'scratch')
        print('here: ', here)
        print('scratch: ', scratch)

        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
    
        #st = time.time()
    
        for tf in os.listdir(os.path.join(here, testdir)):
            if tf.endswith('.inp'):
                test_file = os.path.join(here, testdir, tf)
            if tf.endswith('.log'):
                log_ref = Path(os.path.join(here, testdir, tf))

        args=qcten.main.read_input(test_file)
        setup = qcten.prepare_input.input_data(args)
        setup.parse_options()
        work = qcten.process.work(setup.options)
        work.run()
        log_test = Path(os.path.join(scratch, work.flog))

        print('log_ref: ', log_ref)
        print('log_test: ', log_test)
        lt = log_test.read_text()[3:]
        lr = log_ref.read_text()[3:]
    
        #elapsed_time_fl = round(et - st, 4)
        #print("test {}  - runtime: {} seconds".format(d, elapsed_time_fl))

        os.chdir(here)

        assert lt == lr
