import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Set
from src.models.problem import Problem
from src.models.solution import Solution


@dataclass
class SAResult:
    """Lưu kết quả chạy của thuật toán Simulated Annealing (SA)."""
    selected: List[int]
    best_cost: int
    runtime_s: float
    iterations: int
    history: List[int] = field(default_factory=list)

    def to_solution(self, feasible: bool = True) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=feasible,
        )


class SimulatedAnnealingSolver:
    """Thuật toán Mô phỏng luyện kim (Simulated Annealing) cho bài toán Set Cover."""

    def __init__(
        self,
        problem: Problem,
        t0: float = 100.0,
        t_min: float = 0.01,
        alpha: float = 0.95,
        iters_per_temp: int = 50,
        seed: int = 42,
    ):
        self.problem = problem
        self.t0 = t0
        self.t_min = t_min
        self.alpha = alpha
        self.iters_per_temp = iters_per_temp
        self.rng = random.Random(seed)

    def _cost(self, cols: Set[int]) -> int:
        return sum(self.problem.costs[c] for c in cols)

    def _prune_redundant(self, cols: Set[int]) -> Set[int]:
        """Xóa các camera thừa (bỏ đi mà mọi nút giao thông vẫn được phủ)."""
        sol = set(cols)
        cover_count = [0] * self.problem.m
        for c in sol:
            for r in self.problem.sets[c]:
                cover_count[r] += 1

        for c in sorted(sol, key=lambda x: self.problem.costs[x], reverse=True):
            if all(cover_count[r] > 1 for r in self.problem.sets[c]):
                sol.remove(c)
                for r in self.problem.sets[c]:
                    cover_count[r] -= 1
        return sol

    def _build_initial_solution(self) -> Set[int]:
        uncovered = set(range(self.problem.m))
        sol: Set[int] = set()
        while uncovered:
            r = self.rng.choice(tuple(uncovered))
            candidates = list(self.problem.row_covers[r])
            best_c = min(
                candidates,
                key=lambda c: self.problem.costs[c] / max(1, len(self.problem.sets[c] & uncovered)),
            )
            sol.add(best_c)
            uncovered -= self.problem.sets[best_c]
        return self._prune_redundant(sol)

    def _generate_neighbor(self, current_sol: Set[int]) -> Set[int]:
        neighbor = set(current_sol)
        if not neighbor:
            return self._build_initial_solution()

        k_remove = 1 if len(neighbor) <= 3 else self.rng.choice([1, 2])
        removed = set(self.rng.sample(list(neighbor), k_remove))
        neighbor -= removed

        covered: Set[int] = set()
        for c in neighbor:
            covered.update(self.problem.sets[c])
        uncovered = set(range(self.problem.m)) - covered

        while uncovered:
            r = self.rng.choice(tuple(uncovered))
            candidates = list(self.problem.row_covers[r])
            alt_candidates = [c for c in candidates if c not in removed]
            pool = alt_candidates if alt_candidates else candidates

            pool.sort(key=lambda c: self.problem.costs[c] / max(1, len(self.problem.sets[c] & uncovered)))
            chosen = self.rng.choice(pool[: min(3, len(pool))])
            neighbor.add(chosen)
            uncovered -= self.problem.sets[chosen]

        return self._prune_redundant(neighbor)

    def solve(self) -> SAResult:
        start_time = time.perf_counter()

        current_sol = self._build_initial_solution()
        current_cost = self._cost(current_sol)

        best_sol = set(current_sol)
        best_cost = current_cost

        temp = self.t0
        total_iters = 0
        history: List[int] = [best_cost]

        while temp > self.t_min:
            for _ in range(self.iters_per_temp):
                total_iters += 1
                neighbor_sol = self._generate_neighbor(current_sol)
                neighbor_cost = self._cost(neighbor_sol)
                delta = neighbor_cost - current_cost

                if delta <= 0 or self.rng.random() < math.exp(-delta / temp):
                    current_sol = neighbor_sol
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_sol = set(current_sol)
                        best_cost = current_cost

                history.append(best_cost)

            temp *= self.alpha

        runtime_s = time.perf_counter() - start_time
        return SAResult(
            selected=sorted(best_sol),
            best_cost=best_cost,
            runtime_s=runtime_s,
            iterations=total_iters,
            history=history,
        )


if __name__ == "__main__":
    from src.data.reader import ORLibraryReader
    from src.validation.validator import SolutionValidator

    problem = ORLibraryReader.read("data/vidu/vidu.txt")
    solver = SimulatedAnnealingSolver(problem, t0=50.0, t_min=0.1, alpha=0.9, iters_per_temp=20, seed=42)
    res = solver.solve()

    print("=== KẾT QUẢ SIMULATED ANNEALING (SA) ===")
    print(f"Tổng số vòng lặp: {res.iterations} | Thời gian: {res.runtime_s * 1000:.3f} ms\n")

    val = SolutionValidator.validate(problem, res.selected)
    val.print_report(problem.m)
