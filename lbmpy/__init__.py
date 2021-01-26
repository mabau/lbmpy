def _get_release_file():
    import os.path
    file_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(file_path, '..', 'RELEASE-VERSION')


try:
    __version__ = open(_get_release_file(), 'r').read()
except IOError:
    __version__ = 'development'
