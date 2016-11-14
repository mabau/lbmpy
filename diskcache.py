try:
    from joblib import Memory
    memory = Memory(cachedir="/tmp/pylbm", verbose=False)
except ImportError:
    memory = None


def diskcache(function):
    if memory is not None:
        return memory.cache(function)
    else:
        # no caching of joblib is not installed
        return function
