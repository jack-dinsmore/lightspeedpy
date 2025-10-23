# Description

This package reduces Lightspeed data. It includes simple tools to make light curves and images.

It is currently configured as a python package with rust subroutines to speed up common processes. However, the rust subroutines have not been written. If you're installing this project, you can simply remove
```
from . import _rust
```
from `python/lightspeedpy/__init__.py`.

# Installation

If using rust, run in the shell
```
python -m pip install maturin
maturin develop
```
in the top `lightspeedpy` directory. After the project is successfully built, run in the shell
```
python -m pip install -e .
```
to install `lightspeedpy`. The `-e` tag installs the package in edit mode, so that any changes you make will be automatically implied.

Test your installation by running in the shell
```
python -c "import lightspeedpy"
```
If nothing is output, the package was properly imported and the installation was successful. If you got a `ModuleNotFoundError`, the pip installation did not work. If you got `ImportError: cannot import name '_rust'`, it cannot find the rust component of the library. Either your `maturin develop` command failed, or you forgot to delete `from . import _rust` in `__init__.py` (see above).