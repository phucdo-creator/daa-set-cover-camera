import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from src.algorithms.greedy import (
    ChvatalGreedySolver,
    build_cover_count,
    prune_redundant_columns,
)
from src.models.problem import Problem
from src.models.solution import Solution

# Move = (danh sách camera loại bỏ khỏi S, danh sách camera thêm vào S, delta chi phí)
Move = Tuple[List[int], List[int], int]


@dataclass
class LocalSearchResult:
    """Lưu kết quả chạy của thuật toán Tìm kiếm cục bộ (Local Search)."""
    selected: List[int]
    best_cost: int
    initial_cost: int
    runtime_s: float
    iterations: int
    improvements: int
    strategy: str
    neighborhood: str
    history: List[int] = field(default_factory=list)

    def to_solution(self, feasible: bool = True) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=feasible,
        )


@dataclass
class ILSResult:
    """Lưu kết quả chạy của thuật toán Tìm kiếm cục bộ lặp (Iterated Local Search - ILS)."""
    selected: List[int]
    best_cost: int
    initial_cost: int
    runtime_s: float
    ils_iterations: int
    best_iter: int
    ls_calls: int
    strategy: str
    neighborhood: str
    history: List[int] = field(default_factory=list)

    def to_solution(self, feasible: bool = True) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=feasible,
        )


def apply_move(
    problem: Problem,
    sol: Set[int],
    cover_count: List[int],
    removed_cols: List[int],
    added_cols: List[int],
) -> None:
    """Cập nhật tập nghiệm và mảng cover_count theo đúng nước đi đã chọn."""
    for c in removed_cols:
        sol.remove(c)
        for r in problem.sets[c]:
            cover_count[r] -= 1
    for c in added_cols:
        sol.add(c)
        for r in problem.sets[c]:
            cover_count[r] += 1


def iter_swap_moves(
    problem: Problem,
    sol: Set[int],
    cover_count: List[int],
) -> Iterator[Move]:
    """
    Sinh các nước đi lân cận Swap sử dụng Delta Evaluation trên cover_count.
    Khi thay c_out bằng c_in, tự động đánh giá thêm các cột cũ trở thành dư thừa.
    """
    for c_out in sorted(sol, key=lambda c: (-problem.costs[c], c)):
        unique_rows = [r for r in problem.sets[c_out] if cover_count[r] == 1]

        if not unique_rows:
            yield ([c_out], [], -problem.costs[c_out])
            continue

        unique_rows.sort(key=lambda r: len(problem.row_covers[r]))
        candidates = set(problem.row_covers[unique_rows[0]]) - sol
        for r in unique_rows[1:]:
            candidates &= problem.row_covers[r]
            if not candidates:
                break

        if not candidates:
            continue

        for r in problem.sets[c_out]:
            cover_count[r] -= 1

        try:
            for c_in in sorted(candidates, key=lambda c: (problem.costs[c], c)):
                for r in problem.sets[c_in]:
                    cover_count[r] += 1

                affected: Set[int] = set()
                for r in problem.sets[c_in]:
                    for c in problem.row_covers[r]:
                        if c in sol and c != c_out:
                            affected.add(c)

                extra_removed: List[int] = []
                saved_cost = 0
                for c in sorted(affected, key=lambda x: (-problem.costs[x], x)):
                    if all(cover_count[r] > 1 for r in problem.sets[c]):
                        extra_removed.append(c)
                        saved_cost += problem.costs[c]
                        for r in problem.sets[c]:
                            cover_count[r] -= 1

                delta = problem.costs[c_in] - problem.costs[c_out] - saved_cost
                removed_cols = [c_out] + extra_removed

                for c in extra_removed:
                    for r in problem.sets[c]:
                        cover_count[r] += 1
                for r in problem.sets[c_in]:
                    cover_count[r] -= 1

                yield (removed_cols, [c_in], delta)
        finally:
            for r in problem.sets[c_out]:
                cover_count[r] += 1


def iter_remove_repair_moves(
    problem: Problem,
    sol: Set[int],
    cover_count: List[int],
    tabu_add_until: Optional[Dict[int, int]] = None,
    current_iter: int = 0,
) -> Iterator[Move]:
    """
    Sinh các nước đi lân cận Remove and Repair sử dụng Delta Evaluation trên cover_count.
    Hỗ trợ tham số tabu_add_until (tùy chọn) để tái sử dụng trực tiếp trong Tabu Search.
    """
    for c_out in sorted(sol, key=lambda c: (-problem.costs[c], c)):
        uncovered: Set[int] = set()
        for r in problem.sets[c_out]:
            cover_count[r] -= 1
            if cover_count[r] == 0:
                uncovered.add(r)

        if not uncovered:
            for r in problem.sets[c_out]:
                cover_count[r] += 1
            yield ([c_out], [], -problem.costs[c_out])
            continue

        added: List[int] = []
        added_set: Set[int] = set()
        extra_removed: List[int] = []
        feasible_repair = True

        while uncovered:
            cand_gain: Dict[int, int] = {}
            for r in uncovered:
                for c in problem.row_covers[r]:
                    if c not in sol and c not in added_set:
                        cand_gain[c] = cand_gain.get(c, 0) + 1

            if not cand_gain:
                feasible_repair = False
                break

            if tabu_add_until is not None:
                non_tabu = [c for c in cand_gain if tabu_add_until.get(c, 0) <= current_iter]
                pool = non_tabu if non_tabu else list(cand_gain.keys())
            else:
                pool = list(cand_gain.keys())

            best_c = min(
                pool,
                key=lambda c: (problem.costs[c] / cand_gain[c], problem.costs[c], c),
            )
            added.append(best_c)
            added_set.add(best_c)
            for r in problem.sets[best_c]:
                cover_count[r] += 1
            uncovered -= problem.sets[best_c]

        move: Optional[Move] = None
        if feasible_repair:
            affected: Set[int] = set()
            for c_add in added:
                for r in problem.sets[c_add]:
                    for c in problem.row_covers[r]:
                        if (c in sol and c != c_out) or c in added_set:
                            affected.add(c)

            for c in sorted(affected, key=lambda x: (-problem.costs[x], x)):
                if all(cover_count[r] > 1 for r in problem.sets[c]):
                    extra_removed.append(c)
                    for r in problem.sets[c]:
                        cover_count[r] -= 1

            extra_removed_set = set(extra_removed)
            removed_cols = [c_out] + [c for c in extra_removed if c not in added_set]
            added_cols = [c for c in added if c not in extra_removed_set]
            delta = (
                sum(problem.costs[c] for c in added_cols)
                - sum(problem.costs[c] for c in removed_cols)
            )
            move = (removed_cols, added_cols, delta)

        for c in extra_removed:
            for r in problem.sets[c]:
                cover_count[r] += 1
        for c in added:
            for r in problem.sets[c]:
                cover_count[r] -= 1
        for r in problem.sets[c_out]:
            cover_count[r] += 1

        if move is not None:
            yield move


class LocalSearchSolver:
    """
    Thuật toán Tìm kiếm cục bộ (Local Search) cho Set Cover với Delta Evaluation bằng cover_count.
    - strategy: 'first' (First Improvement) hoặc 'best' (Best Improvement)
    - neighborhood: 'swap', 'remove_repair', hoặc 'both'
    """

    def __init__(
        self,
        problem: Problem,
        strategy: str = "best",
        neighborhood: str = "both",
        max_iter: int = 500,
        initial_sol: Optional[Iterable[int]] = None,
    ):
        if strategy not in ("first", "best"):
            raise ValueError("strategy phải là 'first' hoặc 'best'")
        if neighborhood not in ("swap", "remove_repair", "both"):
            raise ValueError("neighborhood phải là 'swap', 'remove_repair', hoặc 'both'")

        self.problem = problem
        self.strategy = strategy
        self.neighborhood = neighborhood
        self.max_iter = max_iter
        self.initial_sol = list(initial_sol) if initial_sol is not None else None

    def _iter_neighbors(self, sol: Set[int], cover_count: List[int]) -> Iterator[Move]:
        if self.neighborhood in ("swap", "both"):
            yield from iter_swap_moves(self.problem, sol, cover_count)
        if self.neighborhood in ("remove_repair", "both"):
            yield from iter_remove_repair_moves(self.problem, sol, cover_count)

    def run_on_state(
        self, sol: Set[int], cover_count: List[int]
    ) -> Tuple[Set[int], List[int], int, int, int, List[int]]:
        """Chạy Local Search trực tiếp trên trạng thái (sol, cover_count) (tái sử dụng cho ILS)."""
        current_cost = sum(self.problem.costs[c] for c in sol)
        history: List[int] = [current_cost]
        iterations = 0
        improvements = 0

        for _ in range(self.max_iter):
            iterations += 1
            chosen_move: Optional[Move] = None

            if self.strategy == "first":
                for move in self._iter_neighbors(sol, cover_count):
                    if move[2] < 0:
                        chosen_move = move
                        break
            else:
                best_delta = 0
                for move in self._iter_neighbors(sol, cover_count):
                    if move[2] < best_delta:
                        best_delta = move[2]
                        chosen_move = move

            if chosen_move is None:
                break

            removed_cols, added_cols, delta = chosen_move
            apply_move(self.problem, sol, cover_count, removed_cols, added_cols)
            current_cost += delta
            improvements += 1
            history.append(current_cost)

        return sol, cover_count, current_cost, iterations, improvements, history

    def solve(self) -> LocalSearchResult:
        start_time = time.perf_counter()

        if self.initial_sol is None:
            init_res = ChvatalGreedySolver(self.problem, apply_prune=True).solve()
            sol: Set[int] = set(init_res.selected)
            cover_count = build_cover_count(self.problem, sol)
        else:
            sol, _, cover_count = prune_redundant_columns(self.problem, self.initial_sol)

        initial_cost = sum(self.problem.costs[c] for c in sol)
        sol, _, best_cost, iterations, improvements, history = self.run_on_state(
            sol, cover_count
        )

        runtime_s = time.perf_counter() - start_time
        return LocalSearchResult(
            selected=sorted(sol),
            best_cost=best_cost,
            initial_cost=initial_cost,
            runtime_s=runtime_s,
            iterations=iterations,
            improvements=improvements,
            strategy=self.strategy,
            neighborhood=self.neighborhood,
            history=history,
        )


class IteratedLocalSearchSolver:
    """
    Tìm kiếm cục bộ lặp (Iterated Local Search - ILS) theo Lourenço, Martin, Stützle (2003).
    """

    def __init__(
        self,
        problem: Problem,
        strategy: str = "best",
        neighborhood: str = "both",
        max_ils_iter: int = 100,
        max_no_improve: int = 30,
        perturbation_strength: int = 2,
        rcl_size: int = 3,
        acceptance: str = "better",
        ls_max_iter: int = 200,
        seed: int = 42,
        initial_sol: Optional[Iterable[int]] = None,
    ):
        if strategy not in ("first", "best"):
            raise ValueError("strategy phải là 'first' hoặc 'best'")
        if neighborhood not in ("swap", "remove_repair", "both"):
            raise ValueError("neighborhood phải là 'swap', 'remove_repair', hoặc 'both'")
        if acceptance not in ("better", "strict", "rw"):
            raise ValueError("acceptance phải là 'better', 'strict', hoặc 'rw'")

        self.problem = problem
        self.strategy = strategy
        self.neighborhood = neighborhood
        self.max_ils_iter = max_ils_iter
        self.max_no_improve = max_no_improve
        self.perturbation_strength = perturbation_strength
        self.rcl_size = rcl_size
        self.acceptance = acceptance
        self.ls_max_iter = ls_max_iter
        self.rng = random.Random(seed)
        self.initial_sol = list(initial_sol) if initial_sol is not None else None

    def _perturb(
        self, sol: Set[int], cover_count: List[int]
    ) -> Tuple[Set[int], List[int]]:
        """Nhiễu loạn: loại ngẫu nhiên k camera, sửa phủ bằng RCL Chvátal và xóa cột thừa."""
        perturbed = set(sol)
        cc = list(cover_count)

        if not perturbed:
            return perturbed, cc

        k = min(self.perturbation_strength, max(1, len(perturbed) - 1))
        removed = set(self.rng.sample(sorted(perturbed), k))

        uncovered: Set[int] = set()
        for c in removed:
            perturbed.remove(c)
            for r in self.problem.sets[c]:
                cc[r] -= 1
                if cc[r] == 0:
                    uncovered.add(r)

        while uncovered:
            cand_gain: Dict[int, int] = {}
            for r in uncovered:
                for c in self.problem.row_covers[r]:
                    if c not in perturbed:
                        cand_gain[c] = cand_gain.get(c, 0) + 1

            alt_pool = [c for c in cand_gain if c not in removed]
            pool = alt_pool if alt_pool else list(cand_gain.keys())

            pool.sort(key=lambda c: (self.problem.costs[c] / cand_gain[c], self.problem.costs[c], c))
            chosen = self.rng.choice(pool[: min(self.rcl_size, len(pool))])

            perturbed.add(chosen)
            for r in self.problem.sets[chosen]:
                cc[r] += 1
            uncovered -= self.problem.sets[chosen]

        perturbed, _, cc = prune_redundant_columns(self.problem, perturbed, cc, in_place=True)
        return perturbed, cc

    def _accept(
        self,
        current_sol: Set[int],
        current_cost: int,
        cand_sol: Set[int],
        cand_cost: int,
    ) -> bool:
        """Tiêu chuẩn chấp nhận nghiệm sau mỗi chu kỳ ILS."""
        if self.acceptance == "rw":
            return True
        if self.acceptance == "strict":
            return cand_cost < current_cost
        return (cand_cost < current_cost) or (
            cand_cost == current_cost and cand_sol != current_sol
        )

    def solve(self) -> ILSResult:
        start_time = time.perf_counter()

        if self.initial_sol is None:
            init_res = ChvatalGreedySolver(self.problem, apply_prune=True).solve()
            s0: Set[int] = set(init_res.selected)
            cc0 = build_cover_count(self.problem, s0)
        else:
            s0, _, cc0 = prune_redundant_columns(self.problem, self.initial_sol)

        initial_cost = sum(self.problem.costs[c] for c in s0)

        ls_solver = LocalSearchSolver(
            self.problem,
            strategy=self.strategy,
            neighborhood=self.neighborhood,
            max_iter=self.ls_max_iter,
        )
        current_sol, current_cc, current_cost, _, _, _ = ls_solver.run_on_state(s0, cc0)
        ls_calls = 1

        best_sol: Set[int] = set(current_sol)
        best_cost = current_cost
        best_iter = 0
        history: List[int] = [best_cost]

        no_improve = 0
        executed_iters = 0

        for it in range(1, self.max_ils_iter + 1):
            executed_iters = it

            perturbed_sol, perturbed_cc = self._perturb(current_sol, current_cc)
            cand_sol, cand_cc, cand_cost, _, _, _ = ls_solver.run_on_state(
                perturbed_sol, perturbed_cc
            )
            ls_calls += 1

            if cand_cost < best_cost:
                best_sol = set(cand_sol)
                best_cost = cand_cost
                best_iter = it
                no_improve = 0
            else:
                no_improve += 1

            if self._accept(current_sol, current_cost, cand_sol, cand_cost):
                current_sol = cand_sol
                current_cc = cand_cc
                current_cost = cand_cost

            history.append(best_cost)

            if self.max_no_improve > 0 and no_improve >= self.max_no_improve:
                break

        runtime_s = time.perf_counter() - start_time
        return ILSResult(
            selected=sorted(best_sol),
            best_cost=best_cost,
            initial_cost=initial_cost,
            runtime_s=runtime_s,
            ils_iterations=executed_iters,
            best_iter=best_iter,
            ls_calls=ls_calls,
            strategy=self.strategy,
            neighborhood=self.neighborhood,
            history=history,
        )


if __name__ == "__main__":
    from src.data.reader import ORLibraryReader
    from src.validation.validator import SolutionValidator

    problem = ORLibraryReader.read("data/vidu/vidu.txt")

    print("=== 1. LOCAL SEARCH (SWAP & REMOVE-REPAIR) ===")
    for strategy in ("first", "best"):
        for nb in ("swap", "remove_repair", "both"):
            res = LocalSearchSolver(problem, strategy=strategy, neighborhood=nb).solve()
            val = SolutionValidator.validate(problem, res.selected)
            print(
                f"[LS  | strategy={strategy:5s} | nb={nb:13s}] "
                f"init={res.initial_cost} -> best={res.best_cost} | "
                f"iters={res.iterations} | valid={val.is_valid} | "
                f"time={res.runtime_s * 1000:.3f} ms"
            )

    print("\n=== 2. ITERATED LOCAL SEARCH (ILS) ===")
    for strategy in ("first", "best"):
        res = IteratedLocalSearchSolver(
            problem,
            strategy=strategy,
            neighborhood="both",
            max_ils_iter=50,
            max_no_improve=20,
            perturbation_strength=2,
            seed=42,
        ).solve()
        val = SolutionValidator.validate(problem, res.selected)
        print(
            f"[ILS | strategy={strategy:5s} | nb=both         ] "
            f"init={res.initial_cost} -> best={res.best_cost} "
            f"(vòng {res.best_iter}/{res.ils_iterations}) | valid={val.is_valid} | "
            f"time={res.runtime_s * 1000:.3f} ms"
        )
