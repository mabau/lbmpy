import os
import sys
from setuptools import setup, find_packages
sys.path.insert(0, os.path.abspath('..'))
from custom_pypi_index.pypi_index import get_current_dev_version_from_git


setup(name='lbmpy',
      version=get_current_dev_version_from_git(),
      description='Code Generation for Lattice Boltzmann Methods',
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/software/pystencils/',
      packages=['lbmpy'] + ['lbmpy.' + s for s in find_packages('lbmpy')],
      install_requires=['pystencils'],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Framework :: Jupyter',
            'Topic :: Software Development :: Code Generators',
            'Topic :: Scientific/Engineering :: Physics',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
      ],
      python_requires=">=3.6",
      extras_require={
            'gpu': ['pystencils[gpu]'],
            'alltrafos': ['pystencils[alltrafos]'],
            'interactive': ['pystencils[interactive]', 'scipy', 'scikit-image', 'cython'],
            'doc': ['pystencils[doc]'],
      },
      )
