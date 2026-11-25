import time
from dataclasses import dataclass
from typing import List, Set, Tuple
from src.models.problem import Problem
from src.models.solution import Solution


@dataclass
class ExactResult:
    """Lưu kết quả chạy của thuật toán chính xác (Branch and Bound)."""
    selected: List[int]
    best_cost: int
    runtime_s: float
    nodes_visited: int 
    is_optimal: bool
    fixed_by_reduction: int
    removed_cols: int

    def to_solution(self, feasible: bool = True) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=feasible,
        )


class BranchAndBoundSolver:
    """Thuật toán chính xác Nhánh cận (Branch & Bound) kết hợp Giảm cỡ dữ liệu cho Set Cover."""

    def __init__(self, problem: Problem, time_limit: float = 600.0):
        self.problem = problem
        self.time_limit = time_limit

    def _reduce(
        self, uncovered: Set[int], active_cols: Set[int]
    ) -> Tuple[List[int], int, Set[int], Set[int], int]:
        fixed_cols: List[int] = []
        fixed_cost = 0
        removed_cols_count = 0

        uncovered = set(uncovered)
        active_cols = set(active_cols)

        changed = True
        while changed and uncovered:
            changed = False

            empty_cols = [c for c in active_cols if not (self.problem.sets[c] & uncovered)]
            if empty_cols:
                for c in empty_cols:
                    active_cols.remove(c)
                removed_cols_count += len(empty_cols)
                changed = True

            unit_col = None
            for r in uncovered:
                covering = self.problem.row_covers[r] & active_cols
                if len(covering) == 1:
                    unit_col = next(iter(covering))
                    break
            if unit_col is not None:
                fixed_cols.append(unit_col)
                fixed_cost += self.problem.costs[unit_col]
                uncovered -= self.problem.sets[unit_col]
                active_cols.remove(unit_col)
                changed = True
                continue

            col_list = list(active_cols)
            col_eff = {c: self.problem.sets[c] & uncovered for c in col_list}
            dominated = set()
            for i in range(len(col_list)):
                c1 = col_list[i]
                if c1 in dominated:
                    continue
                for j in range(len(col_list)):
                    if i == j:
                        continue
                    c2 = col_list[j]
                    if c2 in dominated:
                        continue
                    if col_eff[c1].issubset(col_eff[c2]) and self.problem.costs[c1] >= self.problem.costs[c2]:
                        if col_eff[c1] == col_eff[c2] and self.problem.costs[c1] == self.problem.costs[c2] and c1 < c2:
                            continue
                        dominated.add(c1)
                        break
            if dominated:
                active_cols -= dominated
                removed_cols_count += len(dominated)
                changed = True

        return fixed_cols, fixed_cost, uncovered, active_cols, removed_cols_count

    def _quick_greedy_upper_bound(
        self, uncovered: Set[int], active_cols: Set[int]
    ) -> Tuple[List[int], int]:
        rem = set(uncovered)
        chosen: List[int] = []
        cost = 0
        while rem:
            best_c = None
            best_ratio = float("inf")
            for c in active_cols:
                gain = len(self.problem.sets[c] & rem)
                if gain > 0:
                    ratio = self.problem.costs[c] / gain
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_c = c
            if best_c is None:
                return [], 10**18
            chosen.append(best_c)
            cost += self.problem.costs[best_c]
            rem -= self.problem.sets[best_c]
        return chosen, cost

    def _lower_bound(self, uncovered: Set[int], active_cols: Set[int]) -> int:
        lb = 0
        for r in uncovered:
            min_c = min(
                (self.problem.costs[c] for c in (self.problem.row_covers[r] & active_cols)),
                default=10**18,
            )
            if min_c > lb:
                lb = min_c
        return lb

    def solve(self) -> ExactResult:
        start_time = time.perf_counter()
        all_rows = set(range(self.problem.m))
        all_cols = set(range(self.problem.n))

        fixed_cols, fixed_cost, rem_rows, rem_cols, removed_cols = self._reduce(all_rows, all_cols)

        greedy_cols, greedy_cost = self._quick_greedy_upper_bound(rem_rows, rem_cols)
        best_cost = fixed_cost + greedy_cost
        best_sol = fixed_cols + greedy_cols

        nodes_visited = 0
        timed_out = False

        def dfs(uncovered: Set[int], cur_cost: int, cur_sol: List[int]):
            nonlocal best_cost, best_sol, nodes_visited, timed_out

            if timed_out:
                return
            if time.perf_counter() - start_time >= self.time_limit:
                timed_out = True
                return

            nodes_visited += 1

            if not uncovered:
                if cur_cost < best_cost:
                    best_cost = cur_cost
                    best_sol = list(cur_sol)
                return

            if cur_cost + self._lower_bound(uncovered, rem_cols) >= best_cost:
                return

            target_row = min(uncovered, key=lambda r: len(self.problem.row_covers[r] & rem_cols))
            candidates = list(self.problem.row_covers[target_row] & rem_cols)
            candidates.sort(key=lambda c: self.problem.costs[c] / len(self.problem.sets[c] & uncovered))

            for c in candidates:
                new_cost = cur_cost + self.problem.costs[c]
                if new_cost >= best_cost:
                    continue
                new_uncovered = uncovered - self.problem.sets[c]
                cur_sol.append(c)
                dfs(new_uncovered, new_cost, cur_sol)
                cur_sol.pop()

        if rem_rows:
            dfs(rem_rows, fixed_cost, list(fixed_cols))

        runtime_s = time.perf_counter() - start_time
        return ExactResult(
            selected=sorted(best_sol),
            best_cost=best_cost,
            runtime_s=runtime_s,
            nodes_visited=nodes_visited,
            is_optimal=not timed_out,
            fixed_by_reduction=len(fixed_cols),
            removed_cols=removed_cols,
        )


if __name__ == "__main__":
    from src.data.reader import ORLibraryReader
    from src.validation.validator import SolutionValidator

    problem = ORLibraryReader.read("data/vidu/vidu.txt")
    solver = BranchAndBoundSolver(problem, time_limit=60.0)
    res = solver.solve()

    print("=== KẾT QUẢ THUẬT TOÁN CHÍNH XÁC (BRANCH & BOUND) ===")
    print(f"Trạng thái: {'TỐI ƯU TOÀN CỤC' if res.is_optimal else 'HẾT THỜI GIAN'}")
    print(f"Số nút đã duyệt: {res.nodes_visited} | Thời gian: {res.runtime_s * 1000:.3f} ms\n")

    val = SolutionValidator.validate(problem, res.selected)
    val.print_report(problem.m)
