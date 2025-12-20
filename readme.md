# Description

This package reduces Lightspeed data. It includes simple tools to make light curves and images.

It is currently configured as a python package with rust subroutines to speed up common processes.

If you're installing this project and do not want to use Rust, you can simply remove
```
from . import _rust
```
from `python/lightspeedpy/__init__.py`.

# Installation

1 (Only for rust users). If using rust, run in the shell
```
python -m pip install maturin
maturin develop
```
in the top `lightspeedpy` directory.

2. Run in the shell
```
python -m pip install -e .
```
to install `lightspeedpy`. The `-e` tag installs the package in edit mode, so that any changes you make will be automatically implied.

3. Test your installation by running in the shell
```
python -c "import lightspeedpy"
```
If nothing is output, the package was properly imported and the installation was successful. If you got a `ModuleNotFoundError`, the pip installation did not work. If you got `ImportError: cannot import name '_rust'`, it cannot find the rust component of the library. Either your `maturin develop` command failed, or you forgot to comment out `from . import _rust` in `__init__.py` (see above).