lbmpy
=====

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mabau/lbmpy/master?filepath=doc%2Fnotebooks)
[![Docs](https://img.shields.io/badge/read-the_docs-brightgreen.svg)](http://pycodegen.pages.i10git.cs.fau.de/lbmpy)
[![pipeline status](https://i10git.cs.fau.de/pycodegen/lbmpy/badges/master/pipeline.svg)](https://i10git.cs.fau.de/pycodegen/lbmpy/commits/master)
[![coverage report](https://i10git.cs.fau.de/pycodegen/lbmpy/badges/master/coverage.svg)](http://pycodegen.pages.i10git.cs.fau.de/lbmpy/coverage_report)


Run fast fluid simulations based on the lattice Boltzmann method in Python on CPUs and GPUs.
lbmpy creates highly optimized LB compute kernels in C or CUDA, for a wide variety of different collision operators, including MRT,
entropic, and cumulant schemes.

All collision operators can be easily adapted, for example, to integrate turbulence models, custom force terms, or multi-phase models. 
It even comes with an integrated Chapman Enskog analysis based on sympy!

Common test scenarios can be set up quickly:
```python
from lbmpy.scenarios import create_channel

ch = create_channel(domain_size=(300,100, 100), force=1e-7, method="trt",
                    equilibrium_order=2, compressible=True,
                    relaxation_rates=[1.97, 1.6], optimization={'target': 'gpu'})
```

To find out more, check out the interactive [tutorial notebooks online with binder](https://mybinder.org/v2/gh/mabau/lbmpy/master?filepath=doc%2Fnotebooks).


Installation
------------

For local installation use pip:

```bash
pip install lbmpy[interactive]
```


Without `[interactive]` you get a minimal version with very little dependencies.

All options:
- `gpu`: use this if a NVIDIA GPU is available and CUDA is installed
- `opencl`: use this to enable the target `opencl` (execution using OpenCL)
- `alltrafos`: pulls in additional dependencies for loop simplification e.g. libisl
- `interactive`: installs dependencies to work in Jupyter including image I/O, plotting etc.

Options can be combined e.g.
```bash
pip install lbmpy[interactive,gpu,doc]
```


Documentation
-------------

Read the docs [here](http://pycodegen.pages.i10git.cs.fau.de/lbmpy) and
check out the Jupyter notebooks in `doc/notebooks`. 

Authors
-------

Many thanks go to the [contributors](AUTHORS.txt) of lbmpy.

### Please cite us

If you use pystencils in a publication, please cite the following articles:

Overview:
  - M. Bauer et al, lbmpy: Automatic code generation for efficient parallel lattice Boltzmann methods. Journal of Computational Science, 2021. https://doi.org/10.1016/j.jocs.2020.101269

Multiphase:
   - M. Holzer et al, Highly efficient lattice Boltzmann multiphase simulations of immiscible fluids at high-density ratios on CPUs and GPUs through code generation. The International Journal of High Performance Computing Applications, 2021. https://doi.org/10.1177/10943420211016525
