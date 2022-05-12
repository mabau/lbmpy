import os
import pytest
import tempfile
import runpy
import sys
import warnings
import platform

import nbformat
from nbconvert import PythonExporter
import sympy
# Trigger config file reading / creation once - to avoid race conditions when multiple instances are creating it
# at the same time
from pystencils.cpu import cpujit

# trigger cython imports - there seems to be a problem when multiple processes try to compile the same cython file
# at the same time
try:
    import pyximport

    pyximport.install(language_level=3)
except ImportError:
    pass
from lbmpy.phasefield.simplex_projection import simplex_projection_2d  # NOQA

SCRIPT_FOLDER = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.abspath('lbmpy'))

# the Ubuntu pipeline uses an older version of pytest which uses deprecated functionality.
# This leads to many warinings in the test and coverage pipeline.
pytest_numeric_version = [int(x, 10) for x in pytest.__version__.split('.')]
pytest_numeric_version.reverse()
pytest_version = sum(x * (100 ** i) for i, x in enumerate(pytest_numeric_version))


def add_path_to_ignore(path):
    if not os.path.exists(path):
        return
    global collect_ignore
    collect_ignore += [os.path.join(SCRIPT_FOLDER, path, f) for f in os.listdir(os.path.join(SCRIPT_FOLDER, path))]


collect_ignore = [os.path.join(SCRIPT_FOLDER, "doc", "conf.py"),
                  os.path.join(SCRIPT_FOLDER, "doc", "img", "mb_discretization", "maxwell_boltzmann_stencil_plot.py")]
add_path_to_ignore('pystencils_tests/benchmark')
add_path_to_ignore('_local_tmp')

try:
    import pycuda
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "lbmpy_tests/test_cpu_gpu_equivalence.py")]

try:
    import waLBerla
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "lbmpy_tests/test_datahandling_parallel.py")]

try:
    import blitzdb
except ImportError:
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "lbmpy_tests/benchmark"),
                       os.path.join(SCRIPT_FOLDER,
                                    "lbmpy_tests/full_scenarios/kida_vortex_flow/scenario_kida_vortex_flow.py")]

if platform.system().lower() == 'windows':
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "lbmpy_tests/test_quicktests.py")]

sver = sympy.__version__.split(".")
if int(sver[0]) == 1 and int(sver[1]) < 2:
    add_path_to_ignore('lbmpy_tests/phasefield')
    collect_ignore += [os.path.join(SCRIPT_FOLDER, "lbmpy_tests/test_n_phase_boyer_noncoupled.ipynb")]

collect_ignore += [os.path.join(SCRIPT_FOLDER, 'setup.py')]

for root, sub_dirs, files in os.walk('.'):
    for f in files:
        if f.endswith(".ipynb") and not any(f.startswith(k) for k in ['demo', 'tutorial', 'test', 'doc']):
            collect_ignore.append(f)


class IPythonMockup:
    def run_line_magic(self, *args, **kwargs):
        pass

    def run_cell_magic(self, *args, **kwargs):
        pass

    def magic(self, *args, **kwargs):
        pass

    def __bool__(self):
        return False


class IPyNbTest(pytest.Item):
    def __init__(self, name, parent, code):
        super(IPyNbTest, self).__init__(name, parent)
        self.code = code
        self.add_marker('notebook')

    def runtest(self):
        global_dict = {'get_ipython': lambda: IPythonMockup(),
                       'is_test_run': True}

        # disable matplotlib output
        exec("import matplotlib.pyplot as p; "
             "p.switch_backend('Template')", global_dict)

        # in notebooks there is an implicit plt.show() - if this is not called a warning is shown when the next
        # plot is created. This warning is suppressed here
        exec("import warnings;"
             "warnings.filterwarnings('ignore', 'Adding an axes using the same arguments as a previous.*');",
             global_dict)
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.code.encode())
            f.flush()
            runpy.run_path(f.name, init_globals=global_dict, run_name=self.name)


class IPyNbFile(pytest.File):
    def collect(self):
        exporter = PythonExporter()
        exporter.exclude_markdown = True
        exporter.exclude_input_prompt = True

        notebook_contents = self.fspath.open(encoding='utf-8')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "IPython.core.inputsplitter is deprecated")
            notebook = nbformat.read(notebook_contents, 4)
            code, _ = exporter.from_notebook_node(notebook)
            if pytest_version >= 50403:
                yield IPyNbTest.from_parent(name=self.name, parent=self, code=code)
            else:
                yield IPyNbTest(self.name, self, code)

    def teardown(self):
        pass


def pytest_collect_file(path, parent):
    glob_exprs = ["*demo*.ipynb", "*tutorial*.ipynb", "test_*.ipynb"]
    if any(path.fnmatch(g) for g in glob_exprs):
        if pytest_version >= 50403:
            return IPyNbFile.from_parent(fspath=path, parent=parent)
        else:
            return IPyNbFile(path, parent)
