lbmpy
=====

[![Docs](https://img.shields.io/badge/read-the_docs-brightgreen.svg)](http://pycodegen.pages.walberla.net/lbmpy)
[![pipeline status](https://i10git.cs.fau.de/pycodegen/lbmpy/badges/master/pipeline.svg)](https://i10git.cs.fau.de/pycodegen/lbmpy/commits/master)
[![coverage report](https://i10git.cs.fau.de/pycodegen/lbmpy/badges/master/coverage.svg)](http://pycodegen.pages.walberla.net/lbmpy/coverage_report)


Run fast fluid simulations based on the lattice Boltzmann method in Python.

![alt text](doc/img/logo.png)


Installation
------------

```bash
export PIP_EXTRA_INDEX_URL=https://www.walberla.net/pip
pip install lbmpy[interactive]
```


Without `[interactive]` you get a minimal version with very little dependencies.

All options:
-  `gpu`: use this if nVidia GPU is available and CUDA is installed
- `alltrafos`: pulls in additional dependencies for loop simplification e.g. libisl
- `interactive`: installs dependencies to work in Jupyter including image I/O, plotting etc.

Options can be combined e.g.
```bash
pip install lbmpy[interactive,gpu,doc]
```


Documentation
-------------

Read the docs [here](http://pycodegen.pages.walberla.net/lbmpy) and
check out the Jupyter notebooks in `doc/notebooks`.
