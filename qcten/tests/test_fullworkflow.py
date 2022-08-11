import os
import sys
import pytest
from pathlib import Path

import qcten
from .helper import *


def FIXMEtest_run():

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
        for tf in os.listdir(os.path.join(th.testinp_dir, testdir)):
            if tf.endswith('.inp'):
                test_file = os.path.join(this_test, tf)
        result=qcten.main.run(args_file=test_file, verbose=True)

        # test reading options from command line

        # reference arguments
        result_ref = th.get_ref_asdict(Path(os.path.join(here, testdir, 'fulldata.ref')))

        diff = th.diff_dicts(result, result_ref)
        assert (diff == {})


