try:
    from joblib import Memory
    diskcache = Memory(cachedir="/tmp/lbmpy", verbose=False).cache
except ImportError:
    # fallback to in-memory caching if joblib is not available
    import functools
    diskcache = functools.lru_cache(maxsize=64)

