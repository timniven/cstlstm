import unittest
from tests import ext_tests


# > python -m unittest discover


test_cases = [
    ext_tests.HistoryTests,
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for test_case in test_cases:
        tests = loader.loadTestsFromTestCase(test_case)
        suite.addTests(tests)
    return suite