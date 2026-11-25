from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Problem:
    """Mô hình dữ liệu bài toán Phủ tập hợp (Set Cover Problem) - chuẩn 0-based."""
    m: int
    n: int
    costs: List[int]
    sets: List[Set[int]]
    row_covers: List[Set[int]] = field(default_factory=list)
    name: str = "unnamed"

    def __post_init__(self):
        if not self.row_covers:
            self.row_covers = [set() for _ in range(self.m)]
            for col, nodes in enumerate(self.sets):
                for row in nodes:
                    self.row_covers[row].add(col)

    def summary(self) -> str:
        uncovered = [i + 1 for i, cols in enumerate(self.row_covers) if len(cols) == 0]
        status = "HỢP LỆ" if not uncovered else f"LỖI (nút không có camera phủ: {uncovered})"
        return f"Dataset: {self.name} | m={self.m} nút | n={self.n} camera | Trạng thái: {status}"
