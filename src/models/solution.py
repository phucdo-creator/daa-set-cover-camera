from dataclasses import dataclass
from typing import List


@dataclass
class Solution:
    """Mô hình lời giải bài toán Set Cover - chuẩn 0-based bên trong Python."""
    selected: List[int]
    total_cost: int
    feasible: bool

    def selected_1_based(self) -> List[int]:
        """Trả về danh sách camera theo chuẩn 1-based của đề bài."""
        return [c + 1 for c in sorted(self.selected)]
