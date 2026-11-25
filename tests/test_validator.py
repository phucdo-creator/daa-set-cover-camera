import unittest
from src.data.reader import ORLibraryReader
from src.validation.validator import SolutionValidator


class TestSolutionValidator(unittest.TestCase):
    def setUp(self):
        self.problem = ORLibraryReader.read("data/vidu/vidu.txt")

    def test_valid_optimal_solution(self):
        res = SolutionValidator.validate(self.problem, [0, 1, 3])
        self.assertTrue(res.is_valid)
        self.assertEqual(res.total_cost, 16)
        self.assertEqual(res.covered_count, 7)
        self.assertEqual(res.uncovered_rows, [])

        sol = res.to_solution()
        self.assertTrue(sol.feasible)
        self.assertEqual(sol.selected_1_based(), [1, 2, 4])

    def test_invalid_missing_node(self):
        res = SolutionValidator.validate(self.problem, [0, 3])
        self.assertFalse(res.is_valid)
        self.assertEqual(res.uncovered_rows, [5])

    def test_invalid_out_of_range_index(self):
        res = SolutionValidator.validate(self.problem, [0, 1, 99])
        self.assertFalse(res.is_valid)


if __name__ == "__main__":
    unittest.main()
