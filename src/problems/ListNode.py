# Definition for singly-linked list.

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
    def __gt__(self, other):
        return self.val > other.val
    def __eq__(self, other):
        return self.val == other.val

