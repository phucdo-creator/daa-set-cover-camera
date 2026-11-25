from pathlib import Path
from typing import List, Set
from src.models.problem import Problem


class ORLibraryReader:
    """Đọc dữ liệu bài toán Set Cover theo chuẩn OR-Library (đầu vào 1-based -> nội bộ 0-based)."""

    @staticmethod
    def read(file_path: str) -> Problem:
        path = Path(file_path)
        tokens = list(map(int, path.read_text(encoding="utf-8").split()))
        idx = 0

        m = tokens[idx]
        n = tokens[idx + 1]
        idx += 2

        costs: List[int] = tokens[idx : idx + n]
        idx += n

        sets: List[Set[int]] = [set() for _ in range(n)]
        row_covers: List[Set[int]] = [set() for _ in range(m)]

        for row in range(m):
            k = tokens[idx]
            idx += 1
            covering_cols = tokens[idx : idx + k]
            idx += k
            for col_1_based in covering_cols:
                col = col_1_based - 1
                sets[col].add(row)
                row_covers[row].add(col)

        return Problem(
            m=m,
            n=n,
            costs=costs,
            sets=sets,
            row_covers=row_covers,
            name=path.stem,
        )


if __name__ == "__main__":
    problem = ORLibraryReader.read("data/vidu/vidu.txt")
    print(problem.summary())
    print(f"Chi phí các camera (c1..c{problem.n}): {problem.costs}")
    for j, nodes in enumerate(problem.sets):
        nodes_1_based = sorted(r + 1 for r in nodes)
        print(f"  Camera S{j + 1} (chi phí {problem.costs[j]}): phủ các nút {nodes_1_based}")
