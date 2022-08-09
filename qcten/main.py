def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


def read_input(finp=None):
    args = []
    if finp is None:
        try:
            args = sys.argv[1:]
        except:
            sys.exit(1)
    else:
        with open(finp, 'r') as f:
            for line in f:
                if line[0] != '#' and line != '\n':
                    args.append(line.strip())
    return args


if __name__ == "__main__":
    read_input()
