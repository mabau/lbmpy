from cython.parallel cimport parallel
cimport openmp

def get_max_threads():
    return openmp.omp_get_max_threads()

def set_num_threads(int nr_of_threads):
    openmp.omp_set_num_threads(nr_of_threads)
    return nr_of_threads
