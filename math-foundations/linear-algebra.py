import numpy as np


# Vectors from Scratch
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dimension = len(components)

    # Dunder methods
    # Called automatically when using the + operator
    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(a**2 for a in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([a / mag for a in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    # Called when python needs a string representation of this object
    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
print("---------------------------")


# Matrix from scratch


# TODOs:
# Implement addition, subtraction, determinant, inverse 2x2, scalar multiply, element wise multiply
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __add__(self, other):
        return Matrix(
            [
                [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def __sub__(self, other):
        return Matrix(
            [
                [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        # nxn
        # Laplace expansion
        # Break a bigger determinant into smaller determinants
        # Base case 1x1 and 2x2 returns det values
        for j in range(self.cols):
            minor = Matrix(
                [
                    [self.data[i][k] for k in range(self.cols) if k != j]
                    for i in range(1, self.rows)
                ]
            )
            det += (-1 * j) * self.data[0][j] * minor.determinant()

    def predict_transformation(self):
        det = self.determinant()
        if det == 0:
            return "The matrix loses shape"
        if det < 0:
            return f"The matrix would be scaled down by {det}times"
        if det > 0:
            return f"The matrix would be scaled up by {det}times"

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.data[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(
                    sum(
                        self.data[i][k] * other.data[k][j] for k in range(self.shape[1])
                    )
                )
            rows.append(row)
        return Matrix(rows)

    def matmul(self, other):
        return Matrix(
            [
                [
                    sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    for j in range(other.cols)
                ]
                for i in range(self.rows)
            ]
        )

    def element_wise_multiply(self, other):
        return Matrix(
            [
                [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
                for i in range(self.rows)
            ]
        )

    def inverse2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix(
            [
                [self.data[1][1] / det, -self.data[0][1] / det],
                [-self.data[1][0] / det, self.data[0][0] / det],
            ]
        )

    def transpose(self):
        return Matrix(
            [
                [self.data[i][j] for i in range(self.shape[0])]
                for j in range(self.shape[1])
            ]
        )

    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
print("---------------------------")

# Linear Independence and Projection

# Find pivot elements in columns and move to nullify below and above rows


def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


# In numpy
a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a+b = {a + b}")
print(f"a . b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")
print("---------------------------")

rng = np.random.default_rng(seed=42)
print(rng.standard_normal((2, 3)))
rng_recreated = np.random.default_rng(seed=42)
print(rng_recreated.standard_normal((2, 3)))
W = rng.standard_normal((2, 3)) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")

print("--------------------------")
print("Matrix operations")
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse2x2().data)

print("--------------------------")
I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse2x2()).data)
print("Resultant of using this matrix for transformation", A.predict_transformation())


# A simple neural network dense layer
# Every dense layer in a neural network does this
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)])
bias = Matrix([[0.1], [0.1]])


# Activation function
# Rectified Linear Unit
# Negative values in the matrix become 0, while positive values remain unchanged
def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])


pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print("--------------------------")

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
