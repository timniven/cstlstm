import unittest
from ext import histories


class HistoryTests(unittest.TestCase):
    def test_last_change(self):
        empty_series = []
        single_value = [1.3]
        no_change = [1.3, 1.3]
        pos_change = [1.3, 1.4]
        neg_change = [1.3, 1.2]
        exception = False
        try:
            histories.History.last_change(empty_series)
        except ValueError:
            exception = True
        self.assertTrue(exception)
        self.assertEqual(round(histories.History.last_change(single_value), 1),
                         1.3)
        self.assertEqual(round(histories.History.last_change(no_change), 1),
                         0.0)
        self.assertEqual(round(histories.History.last_change(pos_change), 1),
                         0.1)
        self.assertEqual(round(histories.History.last_change(neg_change), 1),
                         -0.1)
