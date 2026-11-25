import unittest
from src.models.problem import Problem
from src.models.solution import Solution


class TestProblemAndSolution(unittest.TestCase):
    def test_problem_post_init_builds_row_covers(self):
        problem = Problem(
            m=3,
            n=2,
            costs=[5, 8],
            sets=[{0, 1}, {1, 2}],
            name="tiny",
        )
        self.assertEqual(problem.row_covers, [{0}, {0, 1}, {1}])
        self.assertIn("HỢP LỆ", problem.summary())

    def test_problem_detects_uncovered_node(self):
        problem = Problem(
            m=3,
            n=2,
            costs=[5, 8],
            sets=[{0}, {0}],
            name="invalid_tiny",
        )
        self.assertIn("LỖI", problem.summary())

    def test_solution_1_based_conversion(self):
        sol = Solution(selected=[3, 0, 1], total_cost=16, feasible=True)
        self.assertEqual(sol.selected_1_based(), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
