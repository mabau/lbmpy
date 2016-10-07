import pyximport; pyximport.install()
from .omp_bindings import get_max_threads, set_num_threads
