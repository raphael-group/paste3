## Installation

Before you can use the `paste3` package, ensure that your system meets the following prerequisites:

- Python 3.12 or later
- Pip package manager

Clone the repository and enter it:
   ```
   git clone https://github.com/raphael-group/paste3.git
   cd paste3
   ```

### Creating a paste3 environment

All of `paste3`'s dependencies are on [PyPI](https://pypi.org/). We have developed and tested `paste3` on Python 3.12, but it should work on later Python versions as well.
You can create a new virtual environment using `venv`, and install dependencies using `pip`.

1. Verify that Python 3.12 or newer is installed on your system.
   ```
   python --version
   ```

2. Create a new environment and activate it.
   ```
   python -m venv .venv
   source .venv/bin/activate
   ```

3. In the activated environment, install the paste3 package in editable mode, along with its `dev` and `docs` extras:
    ```
    pip install -e .[dev,docs]
    ```

### Using a different Python version than the system default

If your Python version is not 3.12 or later, or if you're getting errors when using a non-tested Python version, we recommend using the [uv](https://github.com/astral-sh/uv) tool to create a virtual environment with the correct Python version.
`uv` is quick to [install](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and easy to use, both locally as well as on research clusters.

Once `uv` is installed:

1. Create a new environment with Python 3.12 and activate it.
   ```
   uv venv --python 3.12
   source .venv/bin/activate
   ```

2. In the activated environment, install the package in editable mode, along with its `dev` and `docs` extras:
    ```
    uv pip install -e .[dev,docs]
    ```

### paste3 environment using conda

If you prefer using `conda`, you can use the provided `environment.yml` file to create a new conda environment with all the necessary dependencies pinned to versions that have worked for us in the past.
> **Note**
>
> `paste3` can get all its dependencies from `pypi` using `pip` and does not need [conda](https://docs.anaconda.com/miniconda/) for environment management.
Nevertheless, this might be the easiest option for most users who already have access to the `conda` executable locally or through a research cluster. The provided `environment.yml` file
has the defaults channel disabled, and can be used to create a new conda environment with all the necessary dependencies.
**It can therefore be used without getting a business or enterprise license from Anaconda. (See [Anaconda FAQs](https://www.anaconda.com/pricing/terms-of-service-faqs))**

1. Create a new conda environment named `paste3` with Python version 3.12.
   ```
   conda create --name paste3 python=3.12 pip
   ```

2. Activate the environment.
   ```
   conda activate paste3
   ```
   The command prompt will change to indicate the new conda environment by prepending `(paste3)`.

3. In the activated environment, install the dependencies provided in `environment.yml`:
    ```
    conda env update --file environment.yml
    ```

4. In the activated environment, install the package in editable mode *without dependencies*.
    ```
    pip install -e . --no-deps
    ```
