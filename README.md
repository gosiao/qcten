# qcten

## About

`qcten` is a library that calculates and manipulates tensor fields in real space.
Currently, it covers tensor fields of ranks up to 2. 
Primarily developed for quantum chemistry data computed in the Quantum Chemistry Topology (QCT) applications.


## Installation

### Installation in a `conda` environment

This library is currently used together with [TTK](https://topology-tool-kit.github.io/) shipped as `Anaconda` package.
Thus, we use the following procedure to get the working version of `qcten`:

* download or clone `qcten` to your local machine: `cd my_directory && git clone git@github.com:gosiao/qcten.git .`
* go to the root directory of the `qcten` package: `cd qcten`
* create a `conda` environment with `conda env create -n qcten_env -f devtools/conda-envs/qcten_env.yaml`
* activate this environment with `conda activate qcten_env`
* build and install `qcten` with `poetry build && poetry install`
* you can verify the packages in this environment with `conda list`
* test `qcten`: `cd tests && pytest -rP -vv && cd ..` 

### Using `qcten` on your data

The easiest way to use `qcten` on your data, is to do the following:

* go to the working directory, in which you want to do the analysis, e.g., `cd  my_working_space`
* prepare scripts for `qcten` - an input file (`test.inp`) and a python script (`run.py`)
  * a template for `run.py` file is available in ...
  * use test inputs in `my_directory/qcten/tests` as a guide
* activate the environment with `conda activate qcten_env`
* run your analysis with `python run.py`

### Copyright

Copyright (c) 2023, Gosia Olejniczak

