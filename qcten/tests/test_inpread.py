import os
import sys
import pytest
import qcten
from pathlib import Path


def test_inpread():
    """Testing whether input files are read correctly."""

    testdirs = [
        "inptest_t2"
        ]
    
    for testdir in testdirs:

        # run test in scratch directory:
        here = Path(__file__).resolve().parent
        scratch = os.path.join(here, testdir, 'scratch')

        os.makedirs(scratch, exist_ok=True)
        os.chdir(scratch)
    
        # test reading options from *.inp file
        for tf in os.listdir(os.path.join(here, testdir)):
            if tf.endswith('.inp'):
                test_file = os.path.join(here, testdir, tf)

        args=qcten.main.read_input(test_file)
        with open(Path(os.path.join(here, testdir, 'args_t2')), 'r') as f:
            args_ref = f.read().splitlines()

        os.chdir(here)
        error = set(args) ^ set(args_ref)
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
