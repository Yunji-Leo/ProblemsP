from unittest import TestCase
from Solution import Solution

class TestSolution(TestCase):
    s = Solution()

    def test_isPalindrome(self):
        self.assertTrue(self.s.isPalindrome(121))

