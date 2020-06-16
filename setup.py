import os
import sys
import io
from setuptools import setup, find_packages
import distutils
from contextlib import redirect_stdout
from importlib import import_module

sys.path.insert(0, os.path.abspath('doc'))

quick_tests = [
    # 'test_serial_scenarios.test_ldc_mrt',
    'test_serial_scenarios.test_channel_srt',
]


class SimpleTestRunner(distutils.cmd.Command):
    """A custom command to run selected tests"""

    description = 'run some quick tests'
    user_options = []

    @staticmethod
    def _run_tests_in_module(test):
        """Short test runner function - to work also if py.test is not installed."""
        test = 'lbmpy_tests.' + test
        mod, function_name = test.rsplit('.', 1)
        if isinstance(mod, str):
            mod = import_module(mod)

        func = getattr(mod, function_name)
        with redirect_stdout(io.StringIO()):
            func()

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """Run command."""
        for test in quick_tests:
            self._run_tests_in_module(test)

try:
    sys.path.insert(0, os.path.abspath('doc'))
    from version_from_git import version_number_from_git

    version = version_number_from_git()
    with open("RELEASE-VERSION", "w") as f:
        f.write(version)
except ImportError:
    version = open('RELEASE-VERSION', 'r').read()

def readme():
    with open('README.md') as f:
        return f.read()


setup(name='lbmpy',
      version=version,
      description='Code Generation for Lattice Boltzmann Methods',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='Martin Bauer',
      license='AGPLv3',
      author_email='martin.bauer@fau.de',
      url='https://i10git.cs.fau.de/pycodegen/lbmpy/',
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
          'gpu': ['pycuda'],
          'opencl': ['pyopencl'],
          'alltrafos': ['islpy', 'py-cpuinfo'],
          'interactive': ['scipy', 'scikit-image', 'cython', 'matplotlib',
                          'ipy_table', 'imageio', 'jupyter', 'pyevtk'],
          'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                  'sphinxcontrib-bibtex', 'sphinx_autodoc_typehints', 'pandoc'],
      },
      cmdclass={
          'quicktest': SimpleTestRunner
      }
      )
