from dataclasses import dataclass, field
from typing import Iterable, List, Set
from src.models.problem import Problem
from src.models.solution import Solution


@dataclass
class ValidationResult:
    """Kết quả chi tiết sau khi kiểm tra tính hợp lệ của một lời giải."""
    is_valid: bool
    selected: List[int]
    total_cost: int
    covered_count: int 
    uncovered_rows: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_solution(self) -> Solution:
        """Chuyển sang đối tượng Solution chuẩn của nhóm."""
        return Solution(
            selected=self.selected,
            total_cost=self.total_cost,
            feasible=self.is_valid,
        )

    def print_report(self, total_rows: int) -> None:
        cols_1_based = [c + 1 for c in sorted(self.selected)]
        if self.is_valid:
            print("SOLUTION VALID")
            print(f"  - Danh sách vị trí camera (1-indexed): {cols_1_based}")
            print(f"  - Tổng chi phí: {self.total_cost}")
            print(f"  - Số nút giao thông được phủ: {self.covered_count}/{total_rows}")
        else:
            print("SOLUTION INVALID")
            print(f"  - Danh sách vị trí camera (1-indexed): {cols_1_based}")
            print(f"  - Các nút chưa được phủ (1-indexed): {self.uncovered_rows}")
            print(f"  - Nguyên nhân: {'; '.join(self.reasons)}")


class SolutionValidator:
    """Bộ kiểm tra độc lập tính hợp lệ của lời giải Set Cover."""

    @staticmethod
    def validate(problem: Problem, selected_cols: Iterable[int]) -> ValidationResult:
        reasons: List[str] = []
        raw_cols = list(selected_cols)

        invalid_cols = [c for c in raw_cols if c < 0 or c >= problem.n]
        if invalid_cols:
            reasons.append(f"Tồn tại chỉ số camera không hợp lệ (0-indexed): {invalid_cols}")

        unique_cols = sorted(set(c for c in raw_cols if 0 <= c < problem.n))
        if len(unique_cols) != len(raw_cols) and not invalid_cols:
            reasons.append("Danh sách vị trí camera bị trùng lặp")

        total_cost = sum(problem.costs[c] for c in unique_cols)

        covered_rows: Set[int] = set()
        for c in unique_cols:
            covered_rows.update(problem.sets[c])

        uncovered_rows = [r + 1 for r in range(problem.m) if r not in covered_rows]
        if uncovered_rows:
            reasons.append(f"Còn {len(uncovered_rows)} nút giao thông chưa được phủ")

        return ValidationResult(
            is_valid=len(reasons) == 0,
            selected=unique_cols,
            total_cost=total_cost,
            covered_count=len(covered_rows),
            uncovered_rows=uncovered_rows,
            reasons=reasons,
        )

    @staticmethod
    def build_solution(problem: Problem, selected_cols: Iterable[int]) -> Solution:
        """Hàm tiện ích: kiểm tra và trả về trực tiếp đối tượng Solution."""
        return SolutionValidator.validate(problem, selected_cols).to_solution()


if __name__ == "__main__":
    from src.data.reader import ORLibraryReader

    problem = ORLibraryReader.read("data/vidu/vidu.txt")

    print("=== TEST 1: Nghiệm tối ưu {1, 2, 4} (0-indexed: [0, 1, 3]) ===")
    res1 = SolutionValidator.validate(problem, [0, 1, 3])
    res1.print_report(problem.m)

    print("\n=== TEST 2: Nghiệm thiếu camera {1, 4} (0-indexed: [0, 3]) ===")
    res2 = SolutionValidator.validate(problem, [0, 3])
    res2.print_report(problem.m)
