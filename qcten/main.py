from .prepare_input import *
from .process import *


def run(args_file=None, verbose=False):

    """
    main engine function of the qcten
    FIXME
    """

    args = read_input(finp=args_file, verbose=verbose)
    setup = input_data(args)
    setup.parse_options()
    calc = work(setup.options)
    calc.prepare_grid()
    calc.parse_finp()
    if calc.options["inptest"] is not None and calc.options["inptest"]:
        msg = 'Input testing. Stopping now.'
        print(msg)
        return calc.options
    calc.parse_data()
    return calc.fulldata


if __name__ == "__main__":
    run()
