import unittest
from src.algorithms.greedy import (
    ChvatalGreedySolver,
    build_cover_count,
    prune_redundant_columns,
)
from src.algorithms.local_search import (
    IteratedLocalSearchSolver,
    LocalSearchSolver,
    iter_remove_repair_moves,
    iter_swap_moves,
)
from src.algorithms.tabu_search import TabuSearchSolver
from src.data.reader import ORLibraryReader
from src.models.problem import Problem
from src.validation.validator import SolutionValidator


class TestMember2Algorithms(unittest.TestCase):
    def setUp(self):
        self.problem = ORLibraryReader.read("data/vidu/vidu.txt")

    def test_chvatal_greedy_and_prune(self):
        res = ChvatalGreedySolver(self.problem, apply_prune=True).solve()
        val = SolutionValidator.validate(self.problem, res.selected)
        self.assertTrue(val.is_valid)
        self.assertEqual(val.total_cost, res.best_cost)
        self.assertLessEqual(res.best_cost, res.raw_cost)

    def test_prune_removes_expensive_redundant_column(self):
        p = Problem(m=2, n=3, costs=[10, 3, 4], sets=[{0, 1}, {0}, {1}], name="prune_test")
        pruned, removed, cc = prune_redundant_columns(p, [0, 1, 2])
        self.assertEqual(pruned, {1, 2})
        self.assertEqual(removed, [0])
        self.assertEqual(cc, [1, 1])

    def test_delta_evaluation_preserves_cover_count(self):
        sol = {0, 1, 2, 3}
        cc = build_cover_count(self.problem, sol)
        cc_before = list(cc)

        swap_moves = list(iter_swap_moves(self.problem, sol, cc))
        self.assertEqual(cc, cc_before, "iter_swap_moves không được làm lệch cover_count")

        rr_moves = list(iter_remove_repair_moves(self.problem, sol, cc))
        self.assertEqual(cc, cc_before, "iter_remove_repair_moves không được làm lệch cover_count")

        base_cost = sum(self.problem.costs[c] for c in sol)
        for removed_cols, added_cols, delta in swap_moves + rr_moves:
            new_sol = (sol - set(removed_cols)) | set(added_cols)
            val = SolutionValidator.validate(self.problem, new_sol)
            self.assertTrue(val.is_valid)
            self.assertEqual(val.total_cost - base_cost, delta)

    def test_local_search_strategies_and_neighborhoods(self):
        for strategy in ("first", "best"):
            for nb in ("swap", "remove_repair", "both"):
                res = LocalSearchSolver(
                    self.problem, strategy=strategy, neighborhood=nb
                ).solve()
                val = SolutionValidator.validate(self.problem, res.selected)
                self.assertTrue(val.is_valid)
                self.assertEqual(val.total_cost, res.best_cost)
                self.assertLessEqual(res.best_cost, res.initial_cost)

    def test_ils_reaches_optimum_on_vidu(self):
        for strategy in ("first", "best"):
            res = IteratedLocalSearchSolver(
                self.problem,
                strategy=strategy,
                neighborhood="both",
                max_ils_iter=50,
                max_no_improve=20,
                perturbation_strength=2,
                seed=42,
            ).solve()
            val = SolutionValidator.validate(self.problem, res.selected)
            self.assertTrue(val.is_valid)
            self.assertEqual(res.best_cost, 16)

    def test_tabu_search_reaches_optimum_on_vidu(self):
        res = TabuSearchSolver(
            self.problem,
            tabu_tenure=5,
            max_iter=100,
            max_no_improve=50,
            neighborhood="both",
            seed=42,
        ).solve()
        val = SolutionValidator.validate(self.problem, res.selected)
        self.assertTrue(val.is_valid)
        self.assertEqual(res.best_cost, 16)


if __name__ == "__main__":
    unittest.main()
