def bits(x: int) -> list[int]:
	"""
	Returns the bits of the given integer, in reverse order, left-padded with zeros.
	"""
	bitString = bin(x)[2:].zfill(4)
	result = map(int, bitString)
	return list(result)

def subsets(set: list[int]) -> list[list[int]]:
	"""
	Returns the subsets of the given set.
	"""
	result: list[list[int]] = [[]]
	for x in set:
		result += [subset + [x] for subset in result]
	return result

def expression(positions: list[int]) -> str:
	"""
	Returns the equation for the given positions.
	"""
	sub = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈"]
	def term(variables: list[int]) -> str:
		if len(variables) == 0:
			return "0"
		result = ""
		for i in variables:
			result += f"𝑥{sub[i]}"
		return result
	terms = subsets(list(range(4)))
	result: list[str] = []
	for (i, position) in enumerate(positions):
		result.append(f"s{sub[i]}{term(terms[position])}")
	return " ⊕ ".join(result)


class Vector:
	"""
	A vector is a mathematical object that has a magnitude and a direction.

	A vector can be represented as a list of numbers, where each number represents
	the magnitude of the vector in a particular dimension.
	"""

	def __init__(self, components: list[float] = []):
		"""
		Initialises a vector with the given components.
		"""
		from numpy import array
		self.components = array(components)


	@property
	def magnitude(self) -> float:
		"""
		Returns the magnitude of the vector.
		"""
		from numpy.linalg import norm
		return float(norm(self.components))

	@property
	def direction(self) -> "Vector":
		"""
		Returns the direction of the vector.
		"""
		result = Vector()
		result.components = self.components / self.magnitude
		return result

	@property
	def dimensions(self) -> int:
		"""
		Returns the number of dimensions of the vector.
		"""
		return self.components.size

	@property
	def as_list(self) -> list[float]:
		"""
		Returns the vector as a list of numbers.
		"""
		return self.components.tolist()

	@property
	def is_zero(self) -> bool:
		"""
		Returns whether the vector is the zero vector.
		"""
		from math import isclose
		return isclose(self.magnitude, 0.0)

	@property
	def is_unit(self) -> bool:
		"""
		Returns whether the vector is a unit vector.
		"""
		from math import isclose
		return isclose(self.magnitude, 1.0)


	def __str__(self) -> str:
		"""
		Returns a string representation of the vector.
		"""
		return str(self.components)

	def __getitem__(self, index: int) -> float:
		"""
		Returns the component of the vector at the given index.
		"""
		return self.components[index]

	def __setitem__(self, index: int, value: float):
		"""
		Sets the component of the vector at the given index to the given value.
		"""
		self.components[index] = value

	def __mul__(self, other: float) -> "Vector":
		"""
		Returns the product of this vector and a scalar.
		"""
		result = Vector()
		result.components = self.components * other
		return result

	def __rmul__(self, other: float) -> "Vector":
		"""
		Returns the product of this vector and a scalar.
		"""
		return self * other

	def __truediv__(self, other: float) -> "Vector":
		"""
		Returns the quotient of this vector and a scalar.
		"""
		result = Vector()
		result.components = self.components / other
		return result

	def __add__(self, other: "Vector") -> "Vector":
		"""
		Returns the sum of this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be added."
		result = Vector()
		result.components = self.components + other.components
		return result

	def __sub__(self, other: "Vector") -> "Vector":
		"""
		Returns the difference of this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be subtracted."
		result = Vector()
		result.components = self.components - other.components
		return result

	def __eq__(self, other: object) -> bool:
		"""
		Checks if this Vector is equal to another Vector.
		"""
		assert (isinstance(other, Vector)), "Vectors can only be compared to other Vectors."
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be compared."
		from numpy import isclose
		return bool(isclose(self.components, other.components).all())

	def __ne__(self, other: object) -> bool:
		"""
		Checks if this Vector is not equal to another Vector.
		"""
		return not (self == other)

	def __matmul__(self, other: "Vector") -> "Matrix":
		"""
		Returns the outer product of this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions for an outer product."
		from numpy import outer
		result = Matrix()
		result.elements = outer(self.components, other.components)
		return result


	def dot(self, other: "Vector") -> float:
		"""
		Returns the dot product of this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be dotted."
		from numpy import dot
		return dot(self.components, other.components)

	def cross(self, other: "Vector") -> "Vector":
		"""
		Returns the cross product of this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions == 3), "Vectors must have 3 dimensions to be crossed."
		from numpy import cross
		result = Vector()
		result.components = cross(self.components, other.components)
		return result

	def angle(self, other: "Vector") -> float:
		"""
		Returns the angle between this Vector and another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be angled."
		from numpy import arccos
		return arccos(self.dot(other) / (self.magnitude * other.magnitude))

	def projection(self, other: "Vector") -> "Vector":
		"""
		Returns the projection of this Vector onto another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be projected."
		return (self.dot(other) / other.magnitude) * other.direction

	def is_orthogonal(self, other: "Vector") -> bool:
		"""
		Checks if this Vector is orthogonal to another Vector.
		"""
		assert (self.dimensions == other.dimensions), "Vectors must have the same dimensions to be checked for orthogonality."
		from math import isclose
		return isclose(self.dot(other),  0.0)


	@staticmethod
	def zero(n: int):
		"""
		Returns the zero vector of the given size.
		"""
		import numpy
		result = Vector()
		result.components = numpy.zeros(n)
		return result

	@staticmethod
	def new(n: int):
		"""
		Returns a new vector of the given size.
		"""
		import numpy
		result = Vector()
		result.components = numpy.empty(n)
		return result



class Matrix:
	"""
	Represents a matrix of numbers.
	"""

	def __init__(self, rows: list[list[float]] = []):
		"""
		Initialises a matrix with the given rows.
		"""
		from numpy import array
		self.elements = array(rows)


	def subset_is_orthogonal(self, indices: list[int]) -> bool:
		"""
		Returns the maximum set of orthogonal vectors in the given matrix.
		"""
		vectors = self.columns
		subset = [vector.as_list for (i, vector) in enumerate(vectors) if i in indices]
		return Matrix(subset).transpose.is_orthogonal


	@property
	def dimensions(self) -> tuple[int, int]:
		"""
		Returns the dimensions of the matrix.
		"""
		rows = len(self.elements.tolist())
		columns = len(self.elements.tolist()[0])
		return (rows, columns)

	@property
	def rows(self) -> list["Vector"]:
		"""
		Returns a list of the rows of the matrix.
		"""
		return [Vector(row.tolist()) for row in self.elements]

	@property
	def columns(self) -> list[Vector]:
		"""
		Returns a list of the columns of the matrix.
		"""
		return [Vector(column.tolist()) for column in self.elements.T]

	@property
	def determinant(self) -> float:
		"""
		Returns the determinant of the matrix.
		"""
		from numpy.linalg import det
		return det(self.elements)

	@property
	def frobenius_norm(self) -> float:
		"""
		Returns the Frobenius norm of the matrix.
		"""
		from numpy.linalg import norm
		return float(norm(self.elements,  ord = "fro"))

	@property
	def transpose(self) -> "Matrix":
		"""
		Returns the transpose of the matrix.
		"""
		result = Matrix()
		result.elements = self.elements.T
		return result

	@property
	def is_square(self):
		"""
		Returns whether the matrix is square.
		"""
		(r, c) = self.dimensions
		return r == c

	@property
	def is_symmetric(self):
		"""
		Returns whether the matrix is symmetric.
		"""
		return self == self.transpose

	@property
	def is_orthogonal(self):
		"""
		Returns whether the matrix is orthogonal.
		"""
		columns = self.columns
		c = len(columns)
		for i in range(c):
			for j in range(c):
				if i == j:
					continue
				if not columns[i].is_orthogonal(columns[j]):
					return False
		return True

	@property
	def max_orthogonal_subset(self) -> list[int]:
		"""
		Returns the maximum set of orthogonal vectors in the given matrix.
		"""
		(_, c) = self.dimensions
		powerset = subsets(list(range(c)))[1:]
		max_subset: list[int] = []
		for indices in powerset:
			if self.subset_is_orthogonal(indices):
				max_subset = indices
		return max_subset

	@property
	def inverse(self) -> "Matrix":
		"""
		Returns the inverse of the matrix.
		"""
		from numpy.linalg import inv
		assert (self.is_square), "Matrix must be square to be inverted."
		assert (self.determinant != 0), "Matrix must have a non-zero determinant to be inverted."
		result = Matrix()
		result.elements = inv(self.elements)
		return result

	@property
	def as_list(self) -> list[list[float]]:
		"""
		Returns the matrix as a list of lists of numbers.
		"""
		return self.elements.tolist()


	def __str__(self) -> str:
		"""
		Returns a string representation of the matrix.
		"""
		return str(self.elements)

	def __getitem__(self, index: tuple[int, int]) -> float:
		"""
		Returns the element of the matrix at the given index.
		"""
		return self.elements[index]

	def __setitem__(self, index: tuple[int, int], value: float):
		"""
		Sets the element of the matrix at the given index to the given value.
		"""
		self.elements[index] = value

	def __mul__(self, other: float) -> "Matrix":
		"""
		Returns the product of this matrix and a scalar.
		"""
		result = Matrix()
		result.elements = self.elements * other
		return result

	def __rmul__(self, other: float) -> "Matrix":
		"""
		Returns the product of this matrix and a scalar.
		"""
		return self * other

	def __truediv__(self, other: float) -> "Matrix":
		"""
		Returns the quotient of this matrix and a scalar.
		"""
		result = Matrix()
		result.elements = self.elements / other
		return result

	def __add__(self, other: "Matrix") -> "Matrix":
		"""
		Returns the sum of this matrix and another matrix.
		"""
		assert (self.dimensions == other.dimensions), "Matrices must have the same dimensions to be added."
		result = Matrix()
		result.elements = self.elements + other.elements
		return result

	def __sub__(self, other: "Matrix") -> "Matrix":
		"""
		Returns the difference of this matrix and another matrix.
		"""
		assert (self.dimensions == other.dimensions), "Matrices must have the same dimensions to be subtracted."
		result = Matrix()
		result.elements = self.elements - other.elements
		return result

	def __eq__(self, other: object) -> bool:
		"""
		Checks if this matrix is equal to another matrix.
		"""
		assert (isinstance(other, Matrix)), "Matrices can only be compared to other Matrices."
		(r1, c1) = self.dimensions
		(r2, c2) = other.dimensions
		assert ((r1 == r2) and (c1 == c2)), "Matrices must have the same dimensions to be compared."
		from numpy import isclose
		return bool(isclose(self.elements, other.elements).all())

	def __ne__(self, other: object) -> bool:
		"""
		Checks if this matrix is not equal to another matrix.
		"""
		return not (self == other)

	def __matmul__(self, other: "Matrix") -> "Matrix":
		"""
		Returns the product of this matrix and another matrix.
		"""
		(_, c1) = self.dimensions
		(r2, _) = other.dimensions
		assert (c1 == r2), "Matrices must have compatible dimensions to be multiplied."
		from numpy import matmul
		result = Matrix()
		result.elements = matmul(self.elements, other.elements)
		return result

	def __pow__(self, power: int) -> "Matrix":
		"""
		Returns the power of this matrix to the given power.
		"""
		assert (self.is_square), "Matrix must be square to be powered."
		from numpy.linalg import matrix_power
		result = Matrix()
		result.elements = matrix_power(self.elements, power)
		return result


	@staticmethod
	def identity(n: int) -> "Matrix":
		"""
		Returns the identity matrix of the given size.
		"""
		import numpy
		result = Matrix()
		result.elements = numpy.identity(n)
		return result

	@staticmethod
	def zero(m: int, n: int) -> "Matrix":
		"""
		Returns the zero matrix of the given size.
		"""
		import numpy
		result = Matrix()
		result.elements = numpy.zeros((m, n))
		return result

	@staticmethod
	def new(m: int, n: int) -> "Matrix":
		"""
		Returns a new matrix of the given size.
		"""
		import numpy
		result = Matrix()
		result.elements = numpy.empty((m, n))
		return result
