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
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(
                [
                    sum(
                        self.rows[i][j] * other.components[j]
                        for j in range(self.shape[1])
                    )
                    for i in range(self.shape[0])
                ]
            )
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(
                    sum(self.rows[i][k] * other.rows[k][j])
                    for k in range(self.shape[1])
                )
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix(
            [
                [self.rows[i][j] for i in range(self.shape[0])]
                for j in range(self.shape[1])
            ]
        )

    def __repr__(self):
        return f"Matrix({self.rows})"


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
