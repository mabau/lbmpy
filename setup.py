import os
import io
from setuptools import setup, find_packages
import distutils
from contextlib import redirect_stdout
from importlib import import_module

import versioneer

try:
    import cython  # noqa

    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

quick_tests = [
    'test_quicktests.test_poiseuille_channel_quicktest',
    'test_quicktests.test_entropic_methods',
    'test_quicktests.test_cumulant_ldc',
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


def readme():
    with open('README.md') as f:
        return f.read()


def cython_extensions(*extensions):
    from distutils.extension import Extension
    if USE_CYTHON:
        ext = '.pyx'
        result = [Extension(e, [os.path.join(*e.split(".")) + ext]) for e in extensions]
        from Cython.Build import cythonize
        result = cythonize(result, language_level=3)
        return result
    elif all([os.path.exists(os.path.join(*e.split(".")) + '.c') for e in extensions]):
        ext = '.c'
        result = [Extension(e, [os.path.join(*e.split(".")) + ext]) for e in extensions]
        return result
    else:
        return None


def get_cmdclass():
    cmdclass = {"quicktest": SimpleTestRunner}
    cmdclass.update(versioneer.get_cmdclass())
    return cmdclass


major_version = versioneer.get_version().split("+")[0]
setup(name='lbmpy',
      version=versioneer.get_version(),
      description='Code Generation for Lattice Boltzmann Methods',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='Martin Bauer, Markus Holzer, Frederik Hennig',
      license='AGPLv3',
      author_email='cs10-codegen@fau.de',
      url='https://i10git.cs.fau.de/pycodegen/lbmpy/',
      packages=['lbmpy'] + ['lbmpy.' + s for s in find_packages('lbmpy')],
      install_requires=[f'pystencils>=0.4.0,<={major_version}', 'sympy>=1.5.1,<=1.11.1', 'numpy>=1.11.0'],
      package_data={'lbmpy': ['phasefield/simplex_projection.pyx', 'phasefield/simplex_projection.c']},
      ext_modules=cython_extensions("lbmpy.phasefield.simplex_projection"),
      classifiers=[
          'Development Status :: 4 - Beta',
          'Framework :: Jupyter',
          'Topic :: Software Development :: Code Generators',
          'Topic :: Scientific/Engineering :: Physics',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
      ],
      python_requires=">=3.8",
      extras_require={
          'gpu': ['pycuda'],
          'opencl': ['pyopencl'],
          'alltrafos': ['islpy', 'py-cpuinfo'],
          'interactive': ['scipy', 'scikit-image', 'cython', 'matplotlib',
                          'ipy_table', 'imageio', 'jupyter', 'pyevtk'],
          'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx',
                  'sphinxcontrib-bibtex', 'sphinx_autodoc_typehints', 'pandoc'],
          'phasefield': ['Cython']
      },
      cmdclass=get_cmdclass()
      )
