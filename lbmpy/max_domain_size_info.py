"""Information table what the maximum domain size that fits in caches, main memory and optionally GPU memory


Examples:

    Pass different memory sizes and get information about the maximum domain sizes that fit into this memory
    >>> import numpy as np
    >>> memory_sizes = {'L1': '32 KB', 'L3': '1MB'}
    >>> MaxDomainSizeInfo(memory_sizes, array_number=2, data_type=np.float64)
           Mem|      Size|      D2Q9|     D3Q19|     D3Q27
    ------------------------------------------------------
            L1|     32 KB|      15.1|       4.8|       4.2
            L3|       1MB|      85.3|      15.1|      13.4

    This means that a 2D domain of size 15^2 will fit in L1 cache, similarly a 3D domain of size 4^3.

    Instead of passing the memory sizes explicitly, Python can automatically query for the sizes of caches, main memory,
    and GPU memory. To enable these automatic query, omit the memory_size dict.

    >>> info = MaxDomainSizeInfo(array_number=2, data_type=np.float64)

    (The output is not given here, since it depends on your machine.)


"""
import warnings

import numpy as np

# Optional packages cpuinfo, pycuda and psutil for hardware queries
try:
    from cpuinfo import get_cpu_info
except ImportError:
    get_cpu_info = None

try:
    from pycuda.autoinit import device
except ImportError:
    device = None

try:
    from psutil import virtual_memory
except ImportError:
    virtual_memory = None


def square_size(memory_size, pdfs=9, array_number=2, data_type=np.float64):
    """Maximum 2D domain size fitting in memory of given size.

    Args:
        memory_size: memory size in bytes
        pdfs: how many pdfs are stored per cell
        array_number: how many pdf arrays, (2 for source-destination swap, 1 for AA pattern)
        data_type: pdf type

    Returns:
        square side length of maximum domain size
    """
    num_doubles = memory_size / data_type().itemsize
    cells = num_doubles / (pdfs * array_number)
    return cells ** (1 / 2)


def cube_size(memory_size, pdfs=19, array_number=2, data_type=np.float64):
    """Similar to square_size, but returns edge length of cube"""
    num_doubles = memory_size / data_type().itemsize
    cells = num_doubles / (pdfs * array_number)
    return cells ** (1 / 3)


def convert_memory_size(m) -> int:
    """Converts strings like '5 MB' or '1GB' into bytes."""
    if isinstance(m, str):
        m = m.lower().strip()
        if m.endswith('kb'):
            factor = 1024
        elif m.endswith('mb'):
            factor = 1024 ** 2
        elif m.endswith('gb'):
            factor = 1024 ** 3
        else:
            raise ValueError("Unknown unit")
        quantity = float(m[:-2])
        return quantity * factor
    else:
        return m


def cache_domain_size_overview(memory_name_to_size, array_number=2, data_type=np.float64):
    result = []

    for mem_name, size in memory_name_to_size.items():
        d = {'Mem': mem_name,
             'Size': size, }
        for dim, pdfs in [(2, 9), (3, 19), (3, 27)]:
            func = square_size if dim == 2 else cube_size
            stencil_name = 'D{}Q{}'.format(dim, pdfs)
            d[stencil_name] = func(convert_memory_size(size), pdfs=pdfs,
                                   array_number=array_number, data_type=data_type)

        result.append(d)
    return result


def memory_sizes_of_current_machine():
    result = {}

    if get_cpu_info:
        cpu_info = get_cpu_info()
        if 'l1_data_cache_size' in cpu_info:
            result['L1'] = cpu_info['l1_data_cache_size']
        result['L2'] = cpu_info['l2_cache_size']
        if 'l3_cache_size' in cpu_info:
            result['L3'] = cpu_info['l3_cache_size']

    if device:
        size = device.total_memory() / (1024 * 1024)
        result['GPU'] = "{0:.0f} MB".format(size)

    if virtual_memory:
        mem = virtual_memory()
        result['Free  RAM'] = "{0:.0f} MB".format((mem.total - mem.used) / (1024 * 1024))
        result['Total RAM'] = "{0:.0f} MB".format(mem.total / (1024 * 1024))

    if not result:
        warnings.warn("Couldn't query for any local memory size."
                      "Install py-cpuinfo to get cache sizes, psutil for RAM size and pycuda for GPU memory size.")

    return result


class MaxDomainSizeInfo:
    def __init__(self, memory_sizes=None, array_number=2, data_type=np.float64):
        if memory_sizes is None:
            memory_sizes = memory_sizes_of_current_machine()
        self.cache_sizes = memory_sizes
        self.data_type = data_type
        self.array_number = array_number

    def _build_table(self):
        header = ['Mem', 'Size', 'D2Q9', 'D3Q19', 'D3Q27']

        def to_str(e):
            if isinstance(e, float):
                return format(e, ">10.1f")
            elif isinstance(e, list):
                return [to_str(a) for a in e]
            else:
                return format(e, ">10")

        rows = [to_str(header)]

        data = cache_domain_size_overview(self.cache_sizes, array_number=self.array_number, data_type=self.data_type)
        for info in data:
            rows.append([to_str(info[e]) for e in header])
        return rows

    def _repr_html_(self):
        import ipy_table
        # noinspection PyProtectedMember
        return ipy_table.make_table(self._build_table())._repr_html_()

    def __str__(self):
        lines = []
        header, *content = self._build_table()

        lines.append('|'.join(header))
        lines.append('-' * len(lines[0]))

        for row in content:
            lines.append('|'.join(row))
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()


if __name__ == '__main__':
    print(MaxDomainSizeInfo())
