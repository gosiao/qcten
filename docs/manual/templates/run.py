import os
import sys
import pytest
from pathlib import Path
import qcten

def run_generic(testinps, debug=False):

    verbose=False
    testinp_dir = Path(__file__).resolve().parent

    for testinp in testinps:

        test_file = os.path.join(testinp_dir, testinp)

        args = qcten.cli.read_input(finp=test_file, verbose=verbose)

        setup = qcten.cli.input_data(args)
        setup.parse_options()
        setup.print_options()
        
        calc = qcten.process.work(testinp_dir, setup.options)
        calc.run(verbose=verbose)

def run_selected():

    testinps = [
        # here list input files; like that they should be present in the same directory as this run script
        "test1.inp",
        "test2.inp"
        ]
    
    run_generic(testinps, debug=True)

if __name__ == "__main__":
    run_selected()
