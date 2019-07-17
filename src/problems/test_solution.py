from unittest import TestCase
from Solution import Solution

class TestSolution(TestCase):
    s = Solution()

    def test_isPalindrome(self):
        self.assertTrue(self.s.isPalindrome(121))
        self.s.recoverFromPreorder("1-2--3--4-5--6--7")

