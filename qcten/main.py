from .cli import *
from .process import *


def run(args_file=None, verbose=False):

    args = read_input(finp=args_file, verbose=verbose)

    setup = input_data(args)
    setup.parse_options()

    calc = work(setup.runinp_dir, setup.options)
    calc.run(verbose=verbose)

    #if calc.options["inptest"] is not None and calc.options["inptest"]:
    #    msg = 'Input testing. Stopping now.'
    #    print(msg)
    #    return calc.options
    #calc.prepare_output()
    #return calc.fulldata


if __name__ == "__main__":
    run()
