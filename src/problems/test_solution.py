from unittest import TestCase
from Solution import Solution
from ListNode import ListNode
#from AppKit import NSWorkspace


class TestSolution(TestCase):
    s = Solution()

    def test_isPalindrome(self):
        activeAppName = NSWorkspace.sharedWorkspace().activeApplication()['NSApplicationName']
        print(activeAppName)

        nA1 = ListNode(0)
        nA2 = ListNode(2)
        nA3 = ListNode(4)
        nB1 = ListNode(1)
        nB2 = ListNode(4)
        nC1 = ListNode(3)
        nC2 = ListNode(5)
        nA1.next = nA2
        nA2.next = nA3
        nB1.next = nB2
        nC1.next = nC2
        res = self.s.mergeKLists([nA1,nB1,nC1])

        while res is not None:
            print(res.val)
            res = res.next

        self.assertTrue(self.s.isPalindrome(121))
        self.s.recoverFromPreorder("1-2--3--4-5--6--7")

    def test_findSubString(self):
        self.s.findSubstring("barfoothefoobarman", ["foo", "bar"])

