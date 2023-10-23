from lib import *
from typing import Callable

def parity(secret: int) -> Callable[[int], int]:
	"""
	This is the function that generates the functions used in Bernstein-Vazirani.
	"""
	def f(x: int) -> int:
		result = 0
		s = secret
		while s > 0:
			if s & 1 == 1:
				result ^= x & 1
			s >>= 1
			x >>= 1
		return result
	return f

def product(secret: int) -> Callable[[int], int]:
	"""
	This is our original attempt at a variation of Bernstein-Vazirani.

	Here, we replace XOR with AND.
	"""
	def f(x: int) -> int:
		result = 1
		s = secret
		while s > 0:
			if s & 1 == 1:
				result &= x & 1
			s >>= 1
			x >>= 1
		return result
	return f

def sum(secret: int) -> Callable[[int], int]:
	"""
	Unexplored possibility suggested by @Ashnah.

	Here, we will replace XOR with OR.
	"""
	def f(x: int) -> int:
		result = 0
		s = secret
		while s > 0:
			if s & 1 == 1:
				result |= x & 1
			s >>= 1
			x >>= 1
		return result
	return f

def svetlichny():
	"""
	The function structure that was discussed in the last meeting
	"""




# Number of input bits.
m = 2
# Number of possible inputs.
M = (1 << m)
# Number of secret bits.
n = 2
# Number of possible secrets.
N = (1 << n)
# The function we are evaluating.
function = sum
# The matrix that holds the overall truth tables.
matrix = Matrix.new(M, N)

for s in range(N):
	f = function(secret = s)
	for x in range(M):
		matrix[x, s] = 0.5 if (f(x) == 0) else -0.5

print(matrix)
print(f"Is the matrix orthogonal? {matrix.is_orthogonal}.")
