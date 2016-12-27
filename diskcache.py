try:
    from joblib import Memory
    diskcache = Memory(cachedir="/tmp/lbmpy", verbose=False).cache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    import functools
    diskcache = functools.lru_cache(maxsize=64)


# joblibs Memory decorator does not play nicely with sphinx autodoc (decorated functions do not occur in autodoc)
# -> if this script is imported by sphinx we use functools instead
import sys
calledBySphinx = 'sphinx' in sys.modules
if calledBySphinx:
    import functools
    diskcache = functools.lru_cache(maxsize=64)