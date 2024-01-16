#!/usr/bin/env python3

from contextlib import redirect_stdout
import io
from lbmpy_tests.test_quicktests import (
    test_poiseuille_channel_quicktest,
    test_entropic_methods,
    test_cumulant_ldc
)

quick_tests = [
    test_poiseuille_channel_quicktest,
    test_entropic_methods,
    test_cumulant_ldc,
]

if __name__ == "__main__":
    print("Running lbmpy quicktests")
    for qt in quick_tests:
        print(f"   -> {qt.__name__}")
        with redirect_stdout(io.StringIO()):
            qt()
