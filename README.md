lbmpy
=====

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
pip install pystencils[interactive,gpu,doc]
```


Documentation
-------------

Read the docs [here](http://software.pages.walberla.net/pystencils/lbmpy)
