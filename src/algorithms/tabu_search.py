import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set
from src.algorithms.greedy import (
    ChvatalGreedySolver,
    build_cover_count,
    prune_redundant_columns,
)
from src.algorithms.local_search import (
    Move,
    apply_move,
    iter_remove_repair_moves,
    iter_swap_moves,
)
from src.models.problem import Problem
from src.models.solution import Solution


@dataclass
class TSResult:
    """Lưu kết quả chạy của thuật toán Tìm kiếm Tabu (Tabu Search)."""
    selected: List[int]
    best_cost: int
    initial_cost: int
    runtime_s: float
    iterations: int
    best_iter: int
    history: List[int] = field(default_factory=list)

    def to_solution(self, feasible: bool = True) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.best_cost,
            feasible=feasible,
        )


class TabuSearchSolver:
    """Thuật toán Tìm kiếm Tabu (Tabu Search) với Delta Evaluation bằng cover_count."""

    def __init__(
        self,
        problem: Problem,
        tabu_tenure: int = 7,
        max_iter: int = 300,
        max_no_improve: int = 100,
        neighborhood: str = "both",
        tenure_jitter: int = 0,
        seed: int = 42,
        initial_sol: Optional[Iterable[int]] = None,
    ):
        if neighborhood not in ("swap", "remove_repair", "both"):
            raise ValueError("neighborhood phải là 'swap', 'remove_repair', hoặc 'both'")

        self.problem = problem
        self.tabu_tenure = tabu_tenure
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.neighborhood = neighborhood
        self.tenure_jitter = tenure_jitter
        self.rng = random.Random(seed)
        self.initial_sol = list(initial_sol) if initial_sol is not None else None

    def _is_tabu(
        self,
        removed_cols: List[int],
        added_cols: List[int],
        tabu_remove_until: Dict[int, int],
        tabu_add_until: Dict[int, int],
        iteration: int,
    ) -> bool:
        for c in removed_cols:
            if tabu_remove_until.get(c, 0) > iteration:
                return True
        for c in added_cols:
            if tabu_add_until.get(c, 0) > iteration:
                return True
        return False

    def _generate_candidate_moves(
        self,
        sol: Set[int],
        cover_count: List[int],
        tabu_add_until: Dict[int, int],
        iteration: int,
    ) -> List[Move]:
        moves: List[Move] = []
        if self.neighborhood in ("swap", "both"):
            moves.extend(iter_swap_moves(self.problem, sol, cover_count))
        if self.neighborhood in ("remove_repair", "both"):
            moves.extend(
                iter_remove_repair_moves(
                    self.problem,
                    sol,
                    cover_count,
                    tabu_add_until=tabu_add_until,
                    current_iter=iteration,
                )
            )
        return moves

    def solve(self) -> TSResult:
        start_time = time.perf_counter()

        if self.initial_sol is None:
            init_res = ChvatalGreedySolver(self.problem, apply_prune=True).solve()
            current_sol: Set[int] = set(init_res.selected)
            cover_count = build_cover_count(self.problem, current_sol)
        else:
            current_sol, _, cover_count = prune_redundant_columns(self.problem, self.initial_sol)

        initial_cost = sum(self.problem.costs[c] for c in current_sol)
        current_cost = initial_cost

        best_sol: Set[int] = set(current_sol)
        best_cost = current_cost
        best_iter = 0

        tabu_add_until: Dict[int, int] = {}
        tabu_remove_until: Dict[int, int] = {}

        history: List[int] = [best_cost]
        no_improve = 0
        total_iters = 0

        for it in range(1, self.max_iter + 1):
            total_iters = it
            candidates = self._generate_candidate_moves(
                current_sol, cover_count, tabu_add_until, it
            )
            if not candidates:
                break

            best_admissible: Optional[Move] = None
            best_admissible_key = (float("inf"), 0.0)

            best_fallback: Optional[Move] = None
            best_fallback_key = (float("inf"), 0.0)

            for move in candidates:
                removed_cols, added_cols, delta = move
                if not removed_cols and not added_cols:
                    continue

                tie_break = self.rng.random()
                key = (delta, tie_break)

                if key < best_fallback_key:
                    best_fallback_key = key
                    best_fallback = move

                is_tabu = self._is_tabu(
                    removed_cols, added_cols, tabu_remove_until, tabu_add_until, it
                )
                aspiration = (current_cost + delta) < best_cost

                if (not is_tabu) or aspiration:
                    if key < best_admissible_key:
                        best_admissible_key = key
                        best_admissible = move

            chosen_move = best_admissible if best_admissible is not None else best_fallback
            if chosen_move is None:
                break

            removed_cols, added_cols, delta = chosen_move
            apply_move(self.problem, current_sol, cover_count, removed_cols, added_cols)
            current_cost += delta

            tenure = self.tabu_tenure
            if self.tenure_jitter > 0:
                tenure += self.rng.randint(0, self.tenure_jitter)

            for c in removed_cols:
                tabu_add_until[c] = it + tenure
            for c in added_cols:
                tabu_remove_until[c] = it + tenure

            if current_cost < best_cost:
                best_cost = current_cost
                best_sol = set(current_sol)
                best_iter = it
                no_improve = 0
            else:
                no_improve += 1

            history.append(best_cost)

            if self.max_no_improve > 0 and no_improve >= self.max_no_improve:
                break

        runtime_s = time.perf_counter() - start_time
        return TSResult(
            selected=sorted(best_sol),
            best_cost=best_cost,
            initial_cost=initial_cost,
            runtime_s=runtime_s,
            iterations=total_iters,
            best_iter=best_iter,
            history=history,
        )


def run_ts_sensitivity_analysis(
    problem: Problem,
    tenures: Sequence[int] = (3, 5, 7, 10, 15, 20),
    max_iters: Sequence[int] = (100, 300, 500),
    neighborhoods: Sequence[str] = ("swap", "remove_repair", "both"),
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    optimal_cost: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Chạy thực nghiệm độ nhạy tham số Tabu Search (OFAT) và trả về bảng thống kê."""
    rows: List[Dict[str, object]] = []

    def _eval_config(group: str, label: str, **kwargs) -> None:
        costs: List[int] = []
        times_ms: List[float] = []
        best_iters: List[int] = []
        for s in seeds:
            res = TabuSearchSolver(problem, seed=s, **kwargs).solve()
            costs.append(res.best_cost)
            times_ms.append(res.runtime_s * 1000.0)
            best_iters.append(res.best_iter)

        best_c = min(costs)
        mean_c = statistics.mean(costs)
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
        avg_t = statistics.mean(times_ms)
        avg_bi = statistics.mean(best_iters)
        gap_pct = (
            ((mean_c - optimal_cost) / optimal_cost * 100.0)
            if optimal_cost and optimal_cost > 0
            else None
        )
        rows.append(
            {
                "group": group,
                "config": label,
                "best_cost": best_c,
                "mean_cost": mean_c,
                "std_cost": std_c,
                "gap_pct": gap_pct,
                "avg_time_ms": avg_t,
                "avg_best_iter": avg_bi,
            }
        )

    for t in tenures:
        _eval_config(
            "Tabu Tenure",
            f"tenure={t}",
            tabu_tenure=t,
            max_iter=300,
            max_no_improve=100,
            neighborhood="both",
        )

    for mi in max_iters:
        _eval_config(
            "Max Iterations",
            f"max_iter={mi}",
            tabu_tenure=7,
            max_iter=mi,
            max_no_improve=mi,
            neighborhood="both",
        )

    for nb in neighborhoods:
        _eval_config(
            "Neighborhood",
            f"nb={nb}",
            tabu_tenure=7,
            max_iter=300,
            max_no_improve=100,
            neighborhood=nb,
        )

    return rows


def format_sensitivity_markdown(rows: List[Dict[str, object]]) -> str:
    """Định dạng kết quả phân tích độ nhạy tham số TS thành bảng Markdown cho báo cáo."""
    header = (
        "| Nhóm tham số | Cấu hình | Best Cost | Mean Cost | Std | Gap (%) | "
        "Avg Time (ms) | Avg Best Iter |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for r in rows:
        gap_str = f"{r['gap_pct']:.2f}%" if r["gap_pct"] is not None else "N/A"
        lines.append(
            f"| {r['group']} | `{r['config']}` | {r['best_cost']} | "
            f"{r['mean_cost']:.2f} | {r['std_cost']:.2f} | {gap_str} | "
            f"{r['avg_time_ms']:.2f} | {r['avg_best_iter']:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from src.data.reader import ORLibraryReader
    from src.validation.validator import SolutionValidator

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/vidu/vidu.txt"
    opt_cost = int(sys.argv[2]) if len(sys.argv) > 2 else (16 if "vidu.txt" in dataset_path else None)

    problem = ORLibraryReader.read(dataset_path)
    solver = TabuSearchSolver(
        problem,
        tabu_tenure=7,
        max_iter=300,
        max_no_improve=100,
        neighborhood="both",
        seed=42,
    )
    res = solver.solve()

    print(f"=== KẾT QUẢ TABU SEARCH ({problem.name}) ===")
    print(
        f"Chi phí khởi tạo: {res.initial_cost} -> Tốt nhất: {res.best_cost} "
        f"(tại vòng {res.best_iter}/{res.iterations}) | Thời gian: {res.runtime_s * 1000:.3f} ms\n"
    )

    val = SolutionValidator.validate(problem, res.selected)
    val.print_report(problem.m)

    print("\n=== BẢNG ĐỘ NHẠY THAM SỐ TABU SEARCH ===")
    table_rows = run_ts_sensitivity_analysis(problem, optimal_cost=opt_cost)
    print(format_sensitivity_markdown(table_rows))
