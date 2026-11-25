import unittest
from src.data.reader import ORLibraryReader


class TestORLibraryReader(unittest.TestCase):
    def test_read_vidu_dataset(self):
        problem = ORLibraryReader.read("data/vidu/vidu.txt")
        self.assertEqual(problem.m, 7)
        self.assertEqual(problem.n, 6)
        self.assertEqual(problem.costs, [7, 5, 9, 4, 3, 7])
        self.assertEqual(problem.sets[0], {0, 3, 6})
        self.assertEqual(problem.sets[1], {4, 5})
        self.assertEqual(problem.sets[3], {1, 2, 5, 6})
        self.assertIn("HỢP LỆ", problem.summary())


if __name__ == "__main__":
    unittest.main()
