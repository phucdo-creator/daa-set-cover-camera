import time
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple
from src.models.problem import Problem
from src.models.solution import Solution


@dataclass
class GreedyResult:
    """Lưu kết quả chạy của thuật toán tham lam Chvátal Greedy."""
    selected: List[int]
    best_cost: int
    raw_cost: int
    removed_redundant: List[int] = field(default_factory=list)
    runtime_s: float = 0.0
    feasible: bool = True

    def to_solution(self) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=self.feasible,
        )


def build_cover_count(problem: Problem, selected: Iterable[int]) -> List[int]:
    """Khởi tạo mảng cover_count[r] = số lượng camera đang phủ nút giao thông r."""
    cover_count = [0] * problem.m
    for c in selected:
        for r in problem.sets[c]:
            cover_count[r] += 1
    return cover_count


def prune_redundant_columns(
    problem: Problem,
    selected: Iterable[int],
    cover_count: Optional[List[int]] = None,
    in_place: bool = False,
) -> Tuple[Set[int], List[int], List[int]]:
    """
    Thủ tục xóa cột thừa (Redundancy Elimination) sử dụng cover_count.
    Ưu tiên xét xóa các camera có chi phí cao nhất trước; nếu bỏ camera c mà mọi nút
    thuộc S_c vẫn có cover_count[r] > 1 thì loại bỏ c và cập nhật cover_count.
    """
    sol = set(selected)
    if cover_count is None:
        cc = build_cover_count(problem, sol)
    else:
        cc = cover_count if in_place else list(cover_count)

    removed: List[int] = []
    ordered_cols = sorted(
        sol,
        key=lambda c: (-problem.costs[c], len(problem.sets[c]), c),
    )

    for c in ordered_cols:
        if all(cc[r] > 1 for r in problem.sets[c]):
            sol.remove(c)
            removed.append(c)
            for r in problem.sets[c]:
                cc[r] -= 1

    return sol, removed, cc


class ChvatalGreedySolver:
    """Thuật toán tham lam Chvátal kết hợp thủ tục xóa cột thừa cho bài toán Set Cover."""

    def __init__(self, problem: Problem, apply_prune: bool = True):
        self.problem = problem
        self.apply_prune = apply_prune

    def solve(self) -> GreedyResult:
        start_time = time.perf_counter()

        uncovered: Set[int] = set(range(self.problem.m))
        gain: List[int] = [len(s) for s in self.problem.sets]
        active_cols: Set[int] = {c for c in range(self.problem.n) if gain[c] > 0}

        selected: Set[int] = set()
        feasible = True

        while uncovered:
            best_c: Optional[int] = None
            best_key = (float("inf"), float("inf"), 0, 0)

            for c in active_cols:
                g = gain[c]
                if g <= 0:
                    continue
                key = (self.problem.costs[c] / g, self.problem.costs[c], -g, c)
                if key < best_key:
                    best_key = key
                    best_c = c

            if best_c is None:
                feasible = False
                break

            selected.add(best_c)
            newly_covered = self.problem.sets[best_c] & uncovered
            uncovered -= newly_covered

            for r in newly_covered:
                for c in self.problem.row_covers[r]:
                    if c in active_cols:
                        gain[c] -= 1
                        if gain[c] == 0:
                            active_cols.discard(c)

        raw_cost = sum(self.problem.costs[c] for c in selected)
        removed_redundant: List[int] = []

        if feasible and self.apply_prune:
            selected, removed_redundant, _ = prune_redundant_columns(self.problem, selected)

        best_cost = sum(self.problem.costs[c] for c in selected)
        runtime_s = time.perf_counter() - start_time

        return GreedyResult(
            selected=sorted(selected),
            best_cost=best_cost,
            raw_cost=raw_cost,
            removed_redundant=sorted(removed_redundant),
            runtime_s=runtime_s,
            feasible=feasible,
        )


if __name__ == "__main__":
    from src.data.reader import ORLibraryReader
    from src.validation.validator import SolutionValidator

    problem = ORLibraryReader.read("data/vidu/vidu.txt")
    solver = ChvatalGreedySolver(problem, apply_prune=True)
    res = solver.solve()

    print("=== KẾT QUẢ CHVÁTAL GREEDY + XÓA CỘT THỪA ===")
    print(
        f"Chi phí trước khi xóa cột thừa: {res.raw_cost} | "
        f"Đã xóa {len(res.removed_redundant)} cột thừa: {[c + 1 for c in res.removed_redundant]}"
    )
    print(f"Thời gian: {res.runtime_s * 1000:.3f} ms\n")

    val = SolutionValidator.validate(problem, res.selected)
    val.print_report(problem.m)
