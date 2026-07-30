// Linear Algebra Notes

Random vs Standard Normal
Randn vs default_rng

->  Random values are generated out of distributions, and not real random randomly
    .random - uniform dist
    .normal - normal dist
    .exponential - exponential dist
    .binomial - binomial dist
    .poisson - poisson dist
    .integer 

->  Why distributions - Each dataset for a specific purpose would need the random dataset generated to follow a specific distribution according to the needs. Not all real world quantities are uniformly distributed

->  Numpy internals of generating random values according to distribution specified:

    Step 1: Generate uniform random integers irrespective of the distribution
    Step 2: Different algorithms convert samples from requested distribution
        A transformation formula is applied, which converts them into the necessary distribution

-> default_rng vs randn
    -> default_rng  -    new api to generate randoms
                        creates a random generator class    
                        you call the random generator class with the necessary dist
                        Faster than randn and supports multiple distributions
    -> randn        -   Legacy, and supported only standard normal distribution

// Matrix

->  Determinant

    ->  Is a single number that summarizes how much the matrix scales in area(2D) and in volume(3D)
    ->  It describes what the matrix does as a transformation to any vectors and shapes that pass through it
    ->  Det != 0 - Columns are linearly independent
    ->  Det == 0 - Columns are linearly dependent - Transformation reduces to a lower dimension and loses info
    ->  2x2 matrix, row-col operations
        3x3 matrix, Cofactor method
        det=a⋅(leftover)−b⋅(leftover)+c⋅(leftover)−…
        higher nxn matrix, Gaussian elimination
    -> Product of matrices - det(AB) = det(A)det(B)

->  Element Wise vs Matrix Multiplication

    ->  Element-wise: multiply matching positions. Both matrices must be in same shape
        | 1 2 |   | 5 6 |   | 5 12 |
        | 3 4 | * | 7 8 | = | 21 32|

->  Broadcasting
    
    -> When adding a bias vector to a matrix of outputs, the shapes do not match, broadcasting stretches the smaller array to fit
    | 1 2 3 | + [10, 20, 30]    -> Broadcasting ->  | 1 2 3 |   |10, 20, 30|    | 11 22 33 |
    | 4 5 6 |                                       | 4 5 6 | + |10, 20, 30| =  | 14 25 36 |

-> Laplace expansion

->  Inverse of a matrix
    A^−1 = adj(A)/det(A)

    adjoint()
    ->  Build the cofactor matrix: for each entry, take its minor's determinant times the checkerboard sign(-1)^(i+j) the same signs and minors from your determinant code.
    ->  Transpose it (that gives the adjugate).
