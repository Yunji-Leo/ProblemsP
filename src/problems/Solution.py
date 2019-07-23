#!/usr/bin/env python3
from typing import List
from ListNode import ListNode
from TreeNode import TreeNode
import math
import queue


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

        dp[0][0][1] = -2 ** 31

        for i in range(1, len(prices) + 1):
            dp[i][0][1] = max(dp[i - 1][0][1], -prices[i - 1])
        for i in range(1, k + 1):
            dp[0][i][1] = -prices[0]

        for i in range(1, len(prices) + 1):
            for j in range(1, k + 1):
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j - 1][1] + prices[i - 1])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j][0] - prices[i - 1])

        return dp[len(prices)][k][0]

    def recoverFromPreorder(self, S: str) -> TreeNode:
        nodeQueue = queue.Queue()
        depthQueue = queue.Queue()
        self.generateNodeQueue(S, nodeQueue, depthQueue)
        if nodeQueue.qsize() == 0:
            return

        nodeStack = []
        depthStack = []

        root = nodeQueue.get()
        depth = depthQueue.get()
        nodeStack.append(root)
        depthStack.append(depth)

        while nodeQueue.qsize() > 0:
            node = nodeQueue.get()
            depth = depthQueue.get()

            parent = None
            parentDepth = -1
            while True:
                parent = nodeStack.pop()
                parentDepth = depthStack.pop()
                if parentDepth == depth - 1:
                    break
            if parent.left == None:
                parent.left = node
                nodeStack.append(parent)
                depthStack.append(parentDepth)
            else:
                parent.right = node
            nodeStack.append(node)
            depthStack.append(depth)

        return root

    def generateNodeQueue(self, S: str, nodeQueue: queue.Queue, depthQueue: queue.Queue):
        if len(S) == 0:
            return
        isDigit = False
        depth = 0
        value = 0
        for i in range(len(S)):
            if '-' != S[i]:
                if isDigit:
                    value = value * 10 + (ord(S[i]) - ord('0'))
                else:
                    isDigit = True
                    value = ord(S[i]) - ord('0')
            else:
                if not isDigit:
                    depth = depth + 1
                else:
                    nodeQueue.put(TreeNode(value))
                    depthQueue.put(depth)
                    isDigit = False
                    depth = 1
        nodeQueue.put(TreeNode(value))
        depthQueue.put(depth)

    def maxArea(self, height: List[int]) -> int:
        result = 0
        left = 0
        right = len(height) - 1
        while left < right:
            result = max(result, min(height[left], height[right]) * (right - left))
            if height[left] < height[right]:
                left = left + 1
            else:
                right = right - 1
        return result

    def intToRoman(self, num: int) -> str:
        M = ["", "M", "MM", "MMM"]
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return M[num // 1000] + C[(num % 1000) // 100] + X[(num % 100) // 10] + I[num % 10]

    def romanToInt(self, s: str) -> int:
        roman = {
            'M': 1000,
            'D': 500,
            'C': 100,
            'L': 50,
            'X': 10,
            'V': 5,
            'I': 1,
        }
        result = 0
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                result = result - roman[s[i]]
            else:
                result = result + roman[s[i]]
        return result + roman[s[-1]]

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 0:
            return ""
        return self.longestCommonPrefixDC(strs, 0, len(strs) - 1)

    def longestCommonPrefixDC(self, strs: List[str], l: int, r: int) -> str:
        if l == r:
            return strs[l]
        mid = (l + r) // 2
        left = self.longestCommonPrefixDC(strs, l, mid)
        right = self.longestCommonPrefixDC(strs, mid + 1, r)
        return self.commonPrefix(left, right)

    def commonPrefix(self, left: str, right: str) -> str:
        minLength = min(len(left), len(right))
        for i in range(minLength):
            if left[i] != right[i]:
                return left[:i]
        return left[:minLength]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums) - 2):
            if i == 0 or nums[i] != nums[i - 1]:
                left = i + 1
                right = len(nums) - 1
                while left < right:
                    sum = nums[i] + nums[left] + nums[right]
                    if sum > 0:
                        right = right - 1
                    elif sum < 0:
                        left = left + 1
                    else:
                        result.append((nums[i], nums[left], nums[right]))
                        while left < right and nums[left] == nums[left + 1]:
                            left = left + 1
                        while left < right and nums[right] == nums[right - 1]:
                            right = right - 1
                        left = left + 1
                        right = right - 1
        return result

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        result = 2 ** 31 - 1
        distance = 2 ** 31 - 1
        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if abs(sum - target) < distance:
                    result = sum
                    distance = abs(sum - target)
                if sum == target:
                    return sum
                if sum > target:
                    right = right - 1
                else:
                    left = left + 1
        return result

    def letterCombinations(self, digits: str) -> List[str]:
        phone = {
            '2': ["a", "b", "c"],
            '3': ["d", "e", "f"],
            '4': ["g", "h", "i"],
            '5': ["j", "k", "l"],
            '6': ["m", "n", "o"],
            '7': ["p", "q", "r", "s"],
            '8': ["t", "u", "v"],
            '9': ["w", "x", "y", "z"],
        }

        result = []
        if len(digits) == 0:
            return result
        self.letterCombinationsRecursive(result, phone, digits, 0, "")
        return result

    def letterCombinationsRecursive(self, result: List[str], phone: {}, digits: str, pos: int, temp: str):
        if pos == len(digits):
            result.append(temp)
            return
        for i in range(len(phone[digits[pos]])):
            temp = temp + phone[digits[pos]][i]
            self.letterCombinationsRecursive(result, phone, digits, pos + 1, temp)
            temp = temp[:-1]

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        for i in range(len(nums) - 3):
            if i == 0 or nums[i] != nums[i - 1]:
                for j in range(i + 1, len(nums) - 2):
                    if j == i + 1 or nums[j] != nums[j - 1]:
                        left = j + 1
                        right = len(nums) - 1
                        while (left < right):
                            sum = nums[i] + nums[j] + nums[left] + nums[right]
                            if sum > target:
                                right = right - 1
                            elif sum < target:
                                left = left + 1
                            else:
                                result.append([nums[i], nums[j], nums[left], nums[right]])
                                while left < right and nums[left] == nums[left + 1]:
                                    left = left + 1
                                while left < right and nums[right] == nums[right - 1]:
                                    right = right - 1
                                left = left + 1
                                right = right - 1
        return result

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        first = dummy
        for i in range(n):
            first = first.next
        second = dummy
        while first.next != None:
            first = first.next
            second = second.next
        second.next = second.next.next
        return dummy.next

    def isValid(self, s: str) -> bool:
        stack = []
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(')')
            elif s[i] == '[':
                stack.append(']')
            elif s[i] == '{':
                stack.append('}')
            elif len(stack) == 0 or stack[-1] != s[i]:
                return False
            else:
                stack.pop()
        return len(stack) == 0

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0)
        prev = dummy
        while l1 != None and l2 != None:
            if l1.val > l2.val:
                prev.next = l2
                l2 = l2.next
            else:
                prev.next = l1
                l1 = l1.next
            prev = prev.next
        if l1 != None:
            prev.next = l1
        else:
            prev.next = l2
        return dummy.next

    def generateParenthesis(self, n: int) -> List[str]:
        result = []

        def backtrack(S='', left=0, right=0):
            if len(S) == 2 * n:
                result.append(S)
                return
            if left < n:
                backtrack(S + '(', left + 1, right)
            if right < left:
                backtrack(S + ')', left, right + 1)

        backtrack()
        return result

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        head = point = ListNode(0)
        q = queue.PriorityQueue()
        for l in lists:
            if l:
                q.put(l)
        while not q.empty():
            node = q.get()
            point.next = node#ListNode(node.val)
            point = point.next
            node = node.next
            if node:
                q.put(node)
        return head.next
