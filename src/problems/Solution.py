#!/usr/bin/env python3
from typing import List
from ListNode import ListNode
import math


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        check_list = {}

        for item in range(len(nums)):
            if (target - nums[item]) in check_list:
                return check_list[target - nums[item]], item
            else:
                check_list[nums[item]] = item

    def twoSum2(self, nums: List[int], target: int) -> List[int]:
        check_list = {}
        ans: List[int] = []

        for item in range(len(nums)):
            if (target - nums[item]) in check_list:
                ans.append(check_list[target - nums[item]])
                ans.append(item)
                return ans
            else:
                check_list[nums[item]] = item

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummyNode = ListNode
        previousNode = dummyNode
        carry = 0

        while l1 is not None or l2 is not None:
            val1 = 0 if l1 is None else l1.val
            val2 = 0 if l2 is None else l2.val
            l1 = None if l1 is None else l1.next
            l2 = None if l2 is None else l2.next
            temp_sum = val1 + val2 + carry

            carry, val = divmod(temp_sum, 10)

            newNode = ListNode(val)
            previousNode.next = newNode
            previousNode = newNode

        if carry != 0:
            newNode = ListNode(carry)
            previousNode.next = newNode

        return dummyNode.next

    def lengthOfLongestSubstring(self, s: str) -> int:
        result = 0
        left = 0
        right = 0
        charSet = set()

        while right < len(s):
            if s[right] in charSet:
                charSet.remove(s[left])
                left = left + 1
            else:
                charSet.add(s[right])
                right = right + 1
                if right - left > result:
                    result = right - left
        return result

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        if (m + n) % 2 == 0:
            return (self.getKth(nums1, nums2, (m + n) // 2 + 1) + self.getKth(nums1, nums2, (m + n) // 2)) * 0.5
        else:
            return (self.getKth(nums1, nums2, (m + n) // 2 + 1)) * 1.0

    def getKth(self, A: List[int], B: List[int], k: int) -> int:
        m = len(A)
        n = len(B)

        if m > n:
            return self.getKth(B, A, k)
        if m == 0:
            return B[k - 1]
        if k == 1:
            return min(A[0], B[0])

        pa = int(min(k // 2, m))
        pb = int(k - pa)
        if A[pa - 1] <= B[pb - 1]:
            return self.getKth(A[pa:], B, pb)
        else:
            return self.getKth(A, B[pb:], pa)

    def longestPalindrome(self, s: str) -> str:
        if len(s) == 0:
            return ""
        start = 0
        end = 0
        for i in range(len(s)):
            len1 = self.expandAroundCenter(s, i, i)
            len2 = self.expandAroundCenter(s, i, i + 1)
            tmplen = max(len1, len2)
            if tmplen > end - start:
                start = i - (tmplen - 1) // 2
                end = i + tmplen // 2

        return s[start:end + 1]

    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s):
            if s[left] != s[right]:
                break
            left = left - 1
            right = right + 1
        return right - left - 1

    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        self.backtrackSubsets(result, [], nums, 0)
        return result

    def backtrackSubsets(self, result: List[List[int]], tmplist: List[int], nums: List[int], start: int):
        result.append(list(tmplist))

        for i in range(start, len(nums)):
            tmplist.append(nums[i])
            self.backtrackSubsets(result, tmplist, nums, i + 1)
            # tmplist = tmplist[:-1]
            tmplist.pop()

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        result = []
        nums.sort()
        self.backtrackSubsetsWithDup(result, [], nums, 0)
        return result

    def backtrackSubsetsWithDup(self, result: List[List[int]], tmplist: List[int], nums: List[int], start: int):
        result.append(tmplist[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue
            tmplist.append(nums[i])
            self.backtrackSubsetsWithDup(result, tmplist, nums, i + 1)
            tmplist.pop()

    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:
            return s

        rows: List[str] = []
        curRow = 0
        goingDown = False

        for i in range(len(s)):
            rows[curRow] += s[i]
            if curRow == 0 or curRow == numRows - 1:
                goingDown = not goingDown
            if goingDown:
                curRow += 1
            else:
                curRow -= 1
        return "".join([x for x in rows])

    def reverse(self, x: int) -> int:
        sign = [1, -1][x < 0]
        rev, x = 0, abs(x)
        while x:
            x, mod = divmod(x, 10)
            rev = rev * 10 + mod
        return sign * rev if - pow(2, 31) <= sign * rev <= pow(2, 31) - 1 else 0

    def myAtoi(self, str: str) -> int:
        str = str.strip()
        result = 0
        if str == "":
            return result
        sign = 1
        if str[0] == '-':
            sign = -1
            str = str[1:]
        elif str[0] == '+':
            str = str[1:]
        for i in range(len(str)):
            if str[i] >= '0' and str[i] <= '9':
                result = result * 10 + (ord(str[i]) - ord('0'))
            else:
                break

            if result > pow(2, 31) - 1 and sign == 1:
                return pow(2, 31) - 1
            elif result > pow(2, 31) and sign == -1:
                return - pow(2, 31)

        result = sign * result
        return result

    def myAtoi2(self, str: str) -> int:
        str = str.strip().split()
        if len(str) == 0:
            return 0
        idx = 0
        str = str[idx]
        if str[idx] in ['+', '-']:
            idx += 1

        for c in str[idx:]:
            if c.isdigit():
                idx += 1
            else:
                break

        str = str[:idx]
        str = int(str) if (str not in ['+', '-'] and str) else 0
        if str > 2 ** 31 - 1:
            return 2 ** 31 - 1
        elif str < -2 ** 31:
            return - 2 ** 31
        return str

    def isPalindrome(self, x: int) -> bool:
        origin = x
        if x < 0:
            return False
        result = 0
        while x > 0:
            result = result * 10 + x % 10
            x //= 10
        return result == origin

    def isMatch(self, s: str, p: str) -> bool:
        if not p:
            return not s

        first_match = bool(s) and p[0] in {s[0], '.'}

        if len(p) >= 2 and p[1] == '*':
            return (self.isMatch(s, p[2:]) or
                    first_match and self.isMatch(s[1:], p))
        else:
            return first_match and self.isMatch(s[1:], p[1:])

    def maxProfit(self, k: int, prices: List[int]):
        if k == 0 or len(prices) == 0:
            return 0
        if 2 * k > len(prices):
            result = 0
            for i in range(len(prices) - 1):
                if prices[i + 1] > prices[i]:
                    result += prices[i + 1] - prices[i]
            return result

        dp = [[[0 for k in range(2)] for j in range(k + 1)] for i in range(len(prices) + 1)]

        for i in range(1, len(prices) + 1):
            dp[i][0][1] = max(dp[i - 1][0][1], -prices[i - 1])
        for i in range(1, k + 1):
            dp[0][i][1] = -prices[0]

        for i in range(1, len(prices) + 1):
            for j in range(1, k + 1):
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j - 1][1] + prices[i - 1])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i - 1])

        return dp[len(prices)][k][0]