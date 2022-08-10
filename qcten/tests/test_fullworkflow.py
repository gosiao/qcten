import os
import sys
import pytest
import qcten
from pathlib import Path

"""
write me
"""

def get_ref_aslist(finp):
    with open(finp, 'r') as f:
        result = f.read().splitlines()
    return result

def get_ref_asdict(finp):
    result = {}
    result_list = get_ref_aslist(finp)
    for o in result_list:
        k = o.split(':')[0].strip()
        v = o.split(':')[1].strip()
        result[k]=v
    return result


def test_run():

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
        result=qcten.main.run(args_file=test_file, verbose=True)

        # test reading options from command line

        # reference arguments
        result_ref = get_ref_asdict(Path(os.path.join(here, testdir, 'fulldata_t2')))

        os.chdir(here)

        error = set(result) ^ set(result_ref)
        assert not error

