#!/bin/python3

import math
import os
import random
import re
import sys

from collections import Counter
from collections import deque

# Many solution below are those I have 
# concluded from taking problems in the
# Hackerrank.com practice tests.

# Minium number of swaps in an array
# in order to sort
def minimumSwaps(arr):
	arrPos = [*enumerate(arr)]
	arrPos.sort(key = lambda it : it[1])
	vis = {k : False for k in range(len(arr))}
	ans = 0

	for i in range(len(Arr)):
		if vis[i] or arrPos[i][0] == i:
			continue
		cycleSize = 0
		j = i

		while not vis[j]:
			vis[j] = True
			j = arrPos[j][0]
			cycleSize += 1

		if cycleSize > 0:
			ans += (cycleSize - 1)

	return ans



# MinimumBribes in an array 
# if too many bribes -> we say it is Too Chaotic
def minimumBribes(q):
	numeBribes = 0
	i = len(q) - 1

	while i >= 0:
		if(q[i] - (i + 1)) > 2:
			print("Too chaotic")
			return
		j = max(0, q[i] - 2)

		while j < i:
			if q[j] > q[i]:
				numBribes += 1
			j += 1
		i -= 1

	print(numBribes)

# Check Magazine - If we can create a message from the
# magazine words provided
def checkMagazine(magazine, note):
	magLength = len(magazine)
	noteLength = len(note)

	if noteLength > magLength:
		print("No")
		return

	magazineDict = {}

	for i in magazine:
		if i in magazineDict:
			magazineDict[i] = magazineDict[i] + 1
		else:
			magazineDict[i] = 1

	for n in note:
		if n in magazineDict:
			magazineDict[n] = magazineDict[n] - 1
			if magazineDict[n] == 0:
				del magazineDict[n]
		else:
			print("No")
			return

	print("Yes")
	return


# Starting with a 1-indexed array of zeros 
# and a list of operations, for each operation 
# add a value to each of the array element 
# between two given indices, inclusive. Once all 
# operations have been performed, return the maximum 
# value in the array.

def arrayManipulation(n, queries):
	arr = [0] * (n + 1)
	for q in queries:
		a = q[0]
		b = q[1]
		k = q[2]

		arr[a - 1] += k
		arr[b] -= k

	ret = 0
	high = 0

	for i in arr:
		ret += i
		if ret >= high:
			high = ret
	return high

# Sherlock And Anagrams
# Two strings are anagrams of each other if the 
# letters of one string can be rearranged to form 
# the other string. Given a string, find the number 
# of pairs of substrings of the string that are 
# anagrams of each other.
def sherlockAndAnagrams(s):
	buckets = {}
	for i in range(len(s)):
		for j in range(1, len(s) - i + 1):
			# The frozen set improves time to extract keys in O(n)
			key = frozenset(Counter(s[i:i+j]).items())
			buckets[key] = buckets.get(key, 0) + 1
	count = 0
	print(buckets)
	for key in buckets:
		count += buckets[key] * (buckets[key] - 1) // 2
	return count



# make anagrams from 2 strings
def makeAnagrams(a, b):
	CHARS = 26

	ctr1 = [0] * CHARS
	ctr2 = [0] * CHARS

	i = 0

	while i < len(a):
		ctr1[ord(a[i]) - ord('a')] += 1
		i += 1

	i = 0

	while i < len(b):
		ctr2[ord(b[i]) - ord('a')] += 1
		i += 1		

	result = 0

	for i in range(26):
		result += abs(ctr1[i] - ctr2[i])

	return result


# Frequenncy Query - given a squence of
# operations in the form of 2d arrays
# we will have an operation that we must
# determine the frequency of a given
# input from a previous set of operations
# 3 is the frequency inquiry operation
def freqQuery(queries):
	dict = {}
	ret = []

	freqs = defauldict(set)

	for oper, elem in queries:
		freq = dict.get(elem, 0)
		if oper ==1:
			dict[elem] = dict.get(elem, 0) + 1
			freqs[freq].discard(elem)
			freqs[freq + 1].add(elem)
		elif oper == 2:
			dict[elem] = max(0, freq-1)
			freqs[freq].discard(elem)
			freqs[freq - 1].add(elem)
		elif oper == 3:
			ret.append(1 if freqs[elem] else 0)
	return ret


# the count triplets function taks a value
# multiplys it by a value r and then looks
# for that value at pos n+1. If found we repeat
# once more. If found then we have a triplet.
def countTriplets(arr, r):
	count = 0
	dict = {}
	dictPairs = {}

	for i in reversed(arr):
		if i * r in dictPairs:
			count += dictPairs[i * r]
		if i * r in dict:
			dictpairs[i] = dictPairs.get(i, 0) + dict[i * r]
		dict[i] = dict.get(i, 0)
		print(dict[i])

	return count


# working with classes and comparators
# below is a sample class used to compare
# a name and score

from functools import cmp_to_key

class Player:
	def __init__(self, name, score):
		self.name = name
		self.score = score

	def __eq__(self, other):
		return (self.score == other.score)

	def __repr__(self):
		return "%s %s" % (self.name, self.score)

	def comparator(Self, other):
		if self.__eq__(other):
			return (self.name > other.name) - (self.name < other.name)
		else:
			return (self.score < other.score) - (self.score > other.score)


# Simple function to determine the maximum
# toys to be purchased based on prices and 
# some amount k
def maximumToys(prices, k):
	prices = sorted(prices)
	total = 0
	ctr = 0
	for p in prices:
		total += p
		if total <= k:
			ctr += 1
		else:
			return ctr
	return ctr

# count the number of swaps to sort an array

# helper function to do the swap
def swap(arr, i, j):
	tmp = arr[i]
	arr[i] = arr[j]
	arr[j] = tmp
	return arr

def countSwaps(a):
	swapCtr = 0
	for i in range(len(a)):
		for j in range(len(a) - 1):
			if a[j] > a[j + 1]:
				a = swap(a, j, (j + 1))
				swapCtr += 1

	print("Array is sorted in " + str(swapCtr) + " swaps.")
	print("First Element: " + str(a[0]))
	print("Last Element: " + str(a[len(a) - 1]))


# Counting Sort function using bisect
# this is quicker than the built in 
# mergesort that python uses

def countSort(arr):
	n = len(arr)
	k = max(arr) + 1

	position = [0] * k

	for v in arr:
		position[v] += 1

	s = 0
	for i in range(0, k):
		temp = position[i]
		position[i] = s
		s += temp

	result = [None] * n
	for v in arr:
		result[position[v]] = v
		position[v] + 1

	return result

# using the above countSort we sort an array
# and determine if the 2 * the mid point/avg
# is greater than the exenditure. If so we send a
# notification which we count

def activityNotifications(expenditure, d):
	ret = 0
	length = len(expenditure)
	arr = expenditure[length - d - 1:length - 1]
	arr = countSort(arr)
	mid = d//2
	e = d % 2
	i = length - 1
	while i >= d:
		if e == 0:
			m = float((arr[mid] + arr[mid - 1]) /2)
		else:
			m = arr[mid]

		if expenditure[i] >= 2 * m:
			ret += 1

		del arr[bisect.bisect_left(arr, expenditure[i - 1])]
		bisect.insort(arr, expenditure[i - d - 1])
		i = i - 1
	return ret


# Mergesort Inversions - count the number of swaps to 
# complete a mergesort algorithm
# two elements are considered to be inversions if they
# are in opposite locations of where they should be.
# 3 > 2 and 3 is before 2 then they are inverted.
def mergeSortInversions(arr):
	if len(arr) == 1:
		return arr, 0 # 0 represents no swaps
	else:
		mid = len(arr) // 2
		a = arr[:mid]
		b = arr[mid:]

		# we can use recursion to recall the same function
		# to break down to smaller pairs returning the swap count 
		# each time.
		a, ai = mergeSortInversions(a)
		b, bi = mergeSortInversions(b)

		c = []
		i = 0
		j = 0

		inversion = 0 + ai + bi

		while i < len(a) and j < len(b):
			if a[i] < b[j]:
				c.append(a[i])
				i += 1
			else:
				c.append(b[j])
				j += 1
				inversion += (len(a) - i)
		# putting together halves from each
		c += a[i:]
		c += b[j:]

		return c, inversions

def countInversions(arr):
	count = 0
	arr, count = mergeSortInversions(arr)
	return count


#determine how many characters you need to 
# remove such that no two characters beside 
# one another are the same
def alternatingCharacters(s):
	if "A" not in s:
		return (len(s) - 1)
	elif "B" not in s:
		return (len(s) - 1)
	else:
		ctr = 0
		i = 0
		while i < len(s) - 1:
			if s[i + 1] == s[i]:
				ctr += 1
			i += 1

		return ctr

# below we will take a string and determine 
# if all characters in the string occur the
# same number of times. We are allowed 1 delete
# in order to make this rule valid.
# given some string we will state if it is
# valid (YES) or not valid (NO)
def isValid(s):
	ret = ""
	string = Counter(Counter(s).values())
	if len(string.keys()) == 0:
		ret = "YES"
	elif len(string.values()) == 2:
		key1, key2 = string.keys()
		if string[key1] == 1 and (key1 - 1 == key2 or key1 - 1 == 0):
			ret = "YES"
		elif string[key2] == 1 and (key2 - 1 == key1 or key2 - 1 == 0):
			ret = "YES"
		else:
			ret = "NO"

	else:
		ret = "NO"

	return ret


# below is the substring Count where it must satisfy the 
# following: a string must be either all the same characters
# or the middle character must have the same characters on 
# the left as it does its right. A single character also 
# satisfies this requirement. 
# Example aadaa - > a, a, d, a, a, ada, aadaa
def substrCount(n, s):
	total = 0
	ctr = 0
	prev = ''

	for i, v in enumerate(s):
		ctr += 1
		if i and (prev != v):
			j = 1
			while((i - j) >= 0) and ((i + j) < len(s)) and j <<= ctr:
				if s[i - j] == prev == s[i + j]:
					total += 1
					j += 1
				else:
					break
			ctr = 1
		total += ctr
		prev = v
	return total


# the below tells us what elements exist in 
# both strings (equal length) such that
# the longest consecutive values that
# are common to both is returned as the 
# length value
def commonChild(s1, s2):
	prev = [0] * (len(S2) + 1)
	curr = [0] * (len(s2) + 1)

	for r in s1:
		for i, c in enumerate(s2, 1):
			curr[i] = prev[i - 1] + 1 if r == c else max(prev[i], curr[i - 1])
		prev, curr = curr, prev

	return prev[-1]


# a perutations problem and getting the smallest lexicographical string in array
from collections import defaultdict
from itertools import permutations

def reverse(s):
    return s[::-1]

def frequency(s):
    res = defaultdict(int)
    for c in s:
        res[c] += 1
    return res
    
# Complete the reverseShuffleMerge function below.
def reverseShuffleMerge(s):
    freq = frequency(s)
    used = defaultdict(int)
    remaining = dict(freq)
    
    # is usable if we have remaining chars in the array
    # divide floor by 2 since array is split
    def usable(c):
        return (freq[c] // 2 - used[c]) > 0
    
    # popable only if we have the number of times
    # a char is used plus the ramaing is greater 
    # than needed, then we can still pop it. If we need
    # more than what is used and remining, nothing to
    # to pop
    def popable(c):
        needed = freq[c] // 2
        return used[c] + remaining[c] - 1 >= needed
    
    res = []
    
    for c in reverse(s):
        if usable(c):
            while res and res[-1] > c and popable(res[-1]):
                removed = res.pop()
                used[removed] -= 1
                
            used[c] += 1
            res.append(c)
            
        remaining[c] -= 1
        
    return "".join(res)

# permutations
from itertools import permutations
def getPermuations(s):
	p = [''.join(i) for i in permutations(s)]
	return p

# reverse a string
def reverse(s):
	return s[::-1]

# lexicographical least string
def lexiLeast(arr):
	return (min(w for w in p if isinstance(w, str)))

# Problem - minimum candies given scores in array
# Alice is a kindergarten teacher. She wants to 
# give some candies to the children in her class.  
# All the children sit in a line and each of them 
# has a rating score according to his or her performance 
# in the class.  Alice wants to give at least 1 candy 
# to each child. If two children sit next to each other, 
# then the one with the higher rating must get more 
# candies. Alice wants to minimize the total number 
# of candies she must buy.
# dynamic programming problem
# O(2N)
def candies(n, arr):
    i = 0
    prev = 0
    ret = [0] * n
    # walk through left to right and
    # give 1 plus the previous value if
    # the score is greater than the previous
    while i < n:
        if arr[i] > arr[prev]:
            ret[i] = ret[prev] + 1
        else:
            ret[i] = 1
        prev = i
        
        i += 1
            
    i = n - 1
    # do the same but in reverse correcting for any next value 
    # that is greater than the current
    while i > 0:
        if arr[i - 1] > arr[i]:
            if ret[i - 1] <= ret[i]:
                ret[i - 1] = ret[i] + 1
        i -= 1
    print(ret)
    return sum(ret)

# Simple Tree Traversal
# helper
def helper(node, st):
    if node.left == None and node.right == None:
        # is leaf Node
        return st, node.data
    else:
        if st[0] == "1":
            # we go right
            return helper(node.right, st[1:])
        else:
            # we go left
            return helper(node.left, st[1:])

# Enter your code here. Read input from STDIN. Print output to STDOUT
def decodeHuff(root, s):
    #Enter Your Code Here
    i = 0
    ret = ""
    head = root
    while i <= len(s):
        i = 0
        s, c = helper(root, s)
        ret += c
        i += 1
    print(ret)

# BFS

visitied = []
queue = []

def bfs (visitied, graph, node):
	visited.append(node)
	queue.append(node)

	while queue:
		s = queue.pop(0)
		print (s, end = '')

		for neighbor in graph[s]:
			if neighbor not in visitied:
				visited.append(neighbor)
				queue.append(neighbor)
# driver would be bfs(visitied, graph, 'A')

class Graph:
  def __init__(self):
    self.nodes = set()
    self.edges = defaultdict(list)
    self.distances = {}

  def add_node(self, value):
    self.nodes.add(value)

  def add_edge(self, from_node, to_node, distance):
    self.edges[from_node].append(to_node)
    self.edges[to_node].append(from_node)
    self.distances[(from_node, to_node)] = distance


def dijsktra(graph, initial):
  visited = {initial: 0}
  path = {}

  nodes = set(graph.nodes)

  while nodes: 
    min_node = None
    for node in nodes:
      if node in visited:
        if min_node is None:
          min_node = node
        elif visited[node] < visited[min_node]:
          min_node = node

    if min_node is None:
      break

    nodes.remove(min_node)
    current_weight = visited[min_node]

    for edge in graph.edges[min_node]:
      weight = current_weight + graph.distance[(min_node, edge)]
      if edge not in visited or weight < visited[edge]:
        visited[edge] = weight
        path[edge] = min_node

  return visited, path