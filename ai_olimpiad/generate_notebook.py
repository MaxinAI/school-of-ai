import nbformat as nbf

# Function to create markdown cells
def create_markdown(text):
    return nbf.v4.new_markdown_cell(text)

# Function to create code cells
def create_code(code):
    return nbf.v4.new_code_cell(code)

# -------------------------------
# Create Tasks Notebook
# -------------------------------
tasks_nb = nbf.v4.new_notebook()

tasks_cells = []

# Title and Introduction
tasks_cells.append(create_markdown("# Matrix Operations and Transformations with NumPy\n\n## Introduction\n\nWelcome to the **Matrix Operations and Transformations with NumPy** notebook! This interactive guide will help you understand and apply various matrix concepts using Python's NumPy library. You'll engage with tasks that reinforce matrix definitions, operations, and transformations, including linear and affine transformations.\n\n---"))

# Table of Contents
tasks_cells.append(create_markdown("## Table of Contents\n\n1. [Matrix Basics](#Matrix-Basics)\n2. [Matrix Operations](#Matrix-Operations)\n3. [Diagonal Matrices](#Diagonal-Matrices)\n4. [Linear Transformations](#Linear-Transformations)\n5. [Affine Transformations](#Affine-Transformations)\n6. [Visualization of Transformations](#Visualization-of-Transformations)\n7. [Conclusion](#Conclusion)\n\n---"))

# Matrix Basics Section
tasks_cells.append(create_markdown("## Matrix Basics\n\n### Definition of a Matrix\n\nA **matrix** is a two-dimensional array of numbers arranged in rows and columns. Matrices are fundamental in various fields, including computer graphics, data science, and engineering.\n\n### Matrix Dimensions\n\nThe **dimensions** of a matrix are defined by the number of rows and columns it contains, denoted as \( n \\times m \) (rows \\(\times\\) columns).\n\n### Task 1: Create Your Own Matrices\n\n1. **Create a 3x3 matrix** named `C` with values from 1 to 9.\n2. **Create a 4x2 matrix** named `D` with any integer values of your choice.\n3. **Print** both matrices to verify their structure."))

# Task 1 Code Cell
tasks_cells.append(create_code("""# Task 1: Create Your Own Matrices

# 1. Create a 3x3 matrix C with values from 1 to 9
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("Matrix C (3x3):\\n", C)

# 2. Create a 4x2 matrix D with integer values of your choice
D = np.array([[10, 11],
              [12, 13],
              [14, 15],
              [16, 17]])
print("\\nMatrix D (4x2):\\n", D)
"""))

# Matrix Operations Section
tasks_cells.append(create_markdown("---\n\n## Matrix Operations\n\n### Matrix Addition and Subtraction\n\nMatrices can be added or subtracted element-wise if they have the same dimensions.\n\n### Matrix Multiplication\n\nMatrix multiplication (dot product) requires that the number of columns in the first matrix equals the number of rows in the second matrix.\n\n### Transpose of a Matrix\n\nThe **transpose** of a matrix flips it over its diagonal, swapping the row and column indices.\n\n### Task 2: Perform Matrix Operations\n\n1. **Add** matrices `M1` and `M2`.\n2. **Subtract** `M1` from `M2`.\n3. **Multiply** matrix `A` with matrix `B`.\n4. **Transpose** the resulting matrix `C`."))

# Task 2 Code Cell
tasks_cells.append(create_code("""# Task 2: Perform Matrix Operations

# Define matrices M1 and M2
M1 = np.array([[1, 2],
               [3, 4]])
M2 = np.array([[5, 6],
               [7, 8]])

# 1. Add matrices M1 and M2
M_add = M1 + M2
print("Matrix Addition (M1 + M2):\\n", M_add)

# 2. Subtract M1 from M2
M_sub = M2 - M1
print("\\nMatrix Subtraction (M2 - M1):\\n", M_sub)

# Define matrices A and B for multiplication
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])   # 3x2 matrix

# 3. Multiply matrices A and B
C = A @ B
print("\\nMatrix Multiplication (A @ B):\\n", C)

# 4. Transpose the resulting matrix C
C_transpose = C.T
print("\\nTranspose of Matrix C:\\n", C_transpose)
"""))

# Diagonal Matrices Section
tasks_cells.append(create_markdown("---\n\n## Diagonal Matrices\n\n### Definition of a Diagonal Matrix\n\nA **diagonal matrix** is a square matrix in which the entries outside the main diagonal are all zero.\n\n### Properties of Diagonal Matrices\n\n- **Multiplicative**: The product of two diagonal matrices is also a diagonal matrix.\n- **Inverse**: If all diagonal entries are non-zero, the inverse of a diagonal matrix is obtained by taking the reciprocal of each diagonal entry.\n- **Transpose**: The transpose of a diagonal matrix is the matrix itself.\n\n### Task 3: Work with Diagonal Matrices\n\n1. **Create a 4x4 diagonal matrix** named `E` with diagonal elements `[1, 2, 3, 4]`.\n2. **Compute the inverse** of matrix `D` if possible.\n3. **Verify** that the transpose of `D` is equal to `D` itself."))

# Task 3 Code Cell
tasks_cells.append(create_code("""# Task 3: Work with Diagonal Matrices

# 1. Create a 4x4 diagonal matrix E with diagonal elements [1, 2, 3, 4]
E = np.diag([1, 2, 3, 4])
print("Diagonal Matrix E:\\n", E)

# 2. Compute the inverse of matrix D if possible
try:
    D_inv = np.linalg.inv(D)
    print("\\nInverse of Matrix D:\\n", D_inv)
except np.linalg.LinAlgError:
    print("\\nMatrix D is singular and cannot be inverted.")

# 3. Verify the transpose of D
D_transpose = D.T
print("\\nTranspose of Matrix D:\\n", D_transpose)
print("\\nIs D equal to its transpose?", np.array_equal(D, D_transpose))
"""))

# Linear Transformations Section
tasks_cells.append(create_markdown("---\n\n## Linear Transformations\n\n### Definition of a Linear Transformation\n\nA **linear transformation** is a function \( T: \\mathbb{R}^n \\to \\mathbb{R}^m \\) that satisfies:\n\n1. **Additivity**:\n   \[\n   T(\\mathbf{u} + \\mathbf{v}) = T(\\mathbf{u}) + T(\\mathbf{v})\n   \]\n2. **Homogeneity**:\n   \[\n   T(c\\mathbf{u}) = cT(\\mathbf{u})\n   \]\n\nEvery linear transformation can be represented by a matrix.\n\n### Task 4: Implement Linear Transformations\n\n1. **Create a shearing matrix** that shears along the x-axis with a shearing factor of `k = 1.5`.\n2. **Apply** the scaling matrix `A_scale` to a vector `v = [1, 1]`.\n3. **Apply** the rotation matrix `A_rotate` to the same vector `v`.\n4. **Print** the results of these transformations."))

# Task 4 Code Cell
tasks_cells.append(create_code("""# Task 4: Implement Linear Transformations

# 1. Create a shearing matrix that shears along the x-axis with k = 1.5
k = 1.5
A_shear = np.array([[1, k],
                    [0, 1]])
print("Shearing Matrix A_shear:\\n", A_shear)

# 2. Apply the scaling matrix A_scale to vector v = [1, 1]
v = np.array([1, 1])
v_scaled = A_scale @ v
print("\\nScaled Vector (A_scale @ v):\\n", v_scaled)

# 3. Apply the rotation matrix A_rotate to the same vector v
v_rotated = A_rotate @ v
print("\\nRotated Vector (A_rotate @ v):\\n", v_rotated)

# 4. Apply the shearing matrix A_shear to vector v
v_sheared = A_shear @ v
print("\\nSheared Vector (A_shear @ v):\\n", v_sheared)
"""))

# Affine Transformations Section
tasks_cells.append(create_markdown("---\n\n## Affine Transformations\n\n### Definition of an Affine Transformation\n\nAn **affine transformation** is a function \( T: \\mathbb{R}^m \\to \\mathbb{R}^n \\) that can be expressed as:\n\n\[\nT(\\mathbf{x}) = A\\mathbf{x} + \\mathbf{b}\n\]\n\nWhere:\n- \( A \\) is an \( n \\times m \\) **transformation matrix**.\n- \( \\mathbf{b} \\) is a fixed vector in \( \\mathbb{R}^n \\) known as the **translation vector**.\n\nAffine transformations combine **linear transformations** with **translations**.\n\n### Task 5: Implement Affine Transformations\n\n1. **Define a translation vector** \( \\mathbf{b} = [3, -2] \\).\n2. **Create an affine transformation** that first rotates a vector by 90 degrees and then translates it using \( \\mathbf{b} \\).\n3. **Apply** this affine transformation to the vector \( \\mathbf{v} = [2, 1] \\).\n4. **Print** the result."))

# Task 5 Code Cell
tasks_cells.append(create_code("""# Task 5: Implement Affine Transformations

# 1. Define a translation vector b = [3, -2]
b = np.array([3, -2])
print("Translation Vector b:\\n", b)

# 2. Create a rotation matrix for 90 degrees
theta = np.pi / 2  # 90 degrees in radians
A_rotate_90 = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
print("\\nRotation Matrix A_rotate_90 (90 degrees):\\n", A_rotate_90)

# 3. Define the affine transformation function
def affine_transform(v, A, b):
    return A @ v + b

# 4. Define vector v
v = np.array([2, 1])
print("\\nOriginal Vector v:\\n", v)

# 5. Apply affine transformation
v_affine = affine_transform(v, A_rotate_90, b)
print("\\nAffine Transformed Vector (A_rotate_90 @ v + b):\\n", v_affine)
"""))

# Visualization of Transformations Section
tasks_cells.append(create_markdown("---\n\n## Visualization of Transformations\n\n### Task 6: Visualize an Affine Transformation\n\n1. **Define an affine transformation** that rotates a vector by 45 degrees and translates it by \( \\mathbf{b} = [2, 3] \\).\n2. **Apply** this transformation to the vector \( \\mathbf{v} = [1, 2] \\).\n3. **Plot** both the original and transformed vectors on a 2D plane."))

# Task 6 Code Cell
tasks_cells.append(create_code("""# Task 6: Visualize an Affine Transformation

# 1. Define a rotation matrix for 45 degrees
theta = np.pi / 4  # 45 degrees in radians
A_rotate_45 = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
print("Rotation Matrix A_rotate_45 (45 degrees):\\n", A_rotate_45)

# 2. Define translation vector b = [2, 3]
b = np.array([2, 3])
print("\\nTranslation Vector b:\\n", b)

# 3. Define the affine transformation function
def affine_transform(v, A, b):
    return A @ v + b

# 4. Define vector v
v = np.array([1, 2])
print("\\nOriginal Vector v:\\n", v)

# 5. Apply affine transformation
v_affine = affine_transform(v, A_rotate_45, b)
print("\\nAffine Transformed Vector (A_rotate_45 @ v + b):\\n", v_affine)

# 6. Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Original vector
plt.arrow(0, 0, v[0], v[1], color='gray', width=0.05, label='Original Vector')

# Transformed vector
plt.arrow(0, 0, v_affine[0], v_affine[1], color='purple', width=0.05, label='Affine Transformed Vector')

plt.xlim(-1, 5)
plt.ylim(-1, 6)
plt.grid(True)
plt.legend()
plt.title('Visualization of Affine Transformation (Rotation + Translation)')
plt.show()
"""))

# Conclusion Section
tasks_cells.append(create_markdown("---\n\n## Conclusion\n\nIn this notebook, you've explored:\n\n- **Matrix Basics**: Definitions and creation of matrices.\n- **Matrix Operations**: Addition, subtraction, multiplication, and transpose.\n- **Diagonal Matrices**: Creation and properties.\n- **Linear Transformations**: Definitions and implementation of scaling, rotation, shearing.\n- **Affine Transformations**: Combining linear transformations with translations.\n- **Visualization**: Graphical representation of transformations.\n\nThese exercises provide a solid foundation for understanding and applying matrix operations and transformations using NumPy. Continue experimenting with different matrices and transformations to deepen your comprehension!\n\n---\n\n# Appendix: Additional Resources\n\n- [NumPy Official Documentation](https://numpy.org/doc/)\n- [Matplotlib Official Documentation](https://matplotlib.org/stable/contents.html)\n- [Linear Algebra - Khan Academy](https://www.khanacademy.org/math/linear-algebra)\n\n---"))

# Assign cells to tasks notebook
tasks_nb['cells'] = tasks_cells

# Define metadata
tasks_nb['metadata'] = {
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.5",
        "mimetype": "text/x-python",
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py"
    }
}

tasks_nb['nbformat'] = 4
tasks_nb['nbformat_minor'] = 5

# Write the tasks notebook to a file
with open('Matrix_Operations_and_Transformations_Tasks.ipynb', 'w') as f:
    nbf.write(tasks_nb, f)

print("Tasks Notebook 'Matrix_Operations_and_Transformations_Tasks.ipynb' has been created successfully.")

# -------------------------------
# Create Solutions Notebook
# -------------------------------
solutions_nb = nbf.v4.new_notebook()

solutions_cells = []

# Title and Introduction
solutions_cells.append(create_markdown("# Matrix Operations and Transformations with NumPy - Solutions\n\n## Introduction\n\nThis notebook provides the solutions to the tasks outlined in the **Matrix Operations and Transformations with NumPy** exercises. It includes detailed implementations and explanations to aid your understanding.\n\n---"))

# Table of Contents
solutions_cells.append(create_markdown("## Table of Contents\n\n1. [Matrix Basics](#Matrix-Basics)\n2. [Matrix Operations](#Matrix-Operations)\n3. [Diagonal Matrices](#Diagonal-Matrices)\n4. [Linear Transformations](#Linear-Transformations)\n5. [Affine Transformations](#Affine-Transformations)\n6. [Visualization of Transformations](#Visualization-of-Transformations)\n7. [Conclusion](#Conclusion)\n\n---"))

# Matrix Basics Section
solutions_cells.append(create_markdown("## Matrix Basics\n\n### Task 1: Create Your Own Matrices - Solutions"))

# Task 1 Solutions Code Cell
solutions_cells.append(create_code("""# Task 1: Create Your Own Matrices - Solutions

# 1. Create a 3x3 matrix C with values from 1 to 9
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
print("Matrix C (3x3):\\n", C)

# 2. Create a 4x2 matrix D with integer values of your choice
D = np.array([[10, 11],
              [12, 13],
              [14, 15],
              [16, 17]])
print("\\nMatrix D (4x2):\\n", D)
"""))

# Matrix Operations Section
solutions_cells.append(create_markdown("---\n\n## Matrix Operations\n\n### Task 2: Perform Matrix Operations - Solutions"))

# Task 2 Solutions Code Cell
solutions_cells.append(create_code("""# Task 2: Perform Matrix Operations - Solutions

# Define matrices M1 and M2
M1 = np.array([[1, 2],
               [3, 4]])
M2 = np.array([[5, 6],
               [7, 8]])

# 1. Add matrices M1 and M2
M_add = M1 + M2
print("Matrix Addition (M1 + M2):\\n", M_add)

# 2. Subtract M1 from M2
M_sub = M2 - M1
print("\\nMatrix Subtraction (M2 - M1):\\n", M_sub)

# Define matrices A and B for multiplication
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3 matrix
B = np.array([[7, 8],
              [9, 10],
              [11, 12]])   # 3x2 matrix

# 3. Multiply matrices A and B
C = A @ B
print("\\nMatrix Multiplication (A @ B):\\n", C)

# 4. Transpose the resulting matrix C
C_transpose = C.T
print("\\nTranspose of Matrix C:\\n", C_transpose)
"""))

# Diagonal Matrices Section
solutions_cells.append(create_markdown("---\n\n## Diagonal Matrices\n\n### Task 3: Work with Diagonal Matrices - Solutions"))

# Task 3 Solutions Code Cell
solutions_cells.append(create_code("""# Task 3: Work with Diagonal Matrices - Solutions

# 1. Create a 4x4 diagonal matrix E with diagonal elements [1, 2, 3, 4]
E = np.diag([1, 2, 3, 4])
print("Diagonal Matrix E:\\n", E)

# 2. Compute the inverse of matrix D if possible
try:
    D_inv = np.linalg.inv(D)
    print("\\nInverse of Matrix D:\\n", D_inv)
except np.linalg.LinAlgError:
    print("\\nMatrix D is singular and cannot be inverted.")

# 3. Verify the transpose of D
D_transpose = D.T
print("\\nTranspose of Matrix D:\\n", D_transpose)
print("\\nIs D equal to its transpose?", np.array_equal(D, D_transpose))
"""))

# Linear Transformations Section
solutions_cells.append(create_markdown("---\n\n## Linear Transformations\n\n### Task 4: Implement Linear Transformations - Solutions"))

# Task 4 Solutions Code Cell
solutions_cells.append(create_code("""# Task 4: Implement Linear Transformations - Solutions

# 1. Create a shearing matrix that shears along the x-axis with k = 1.5
k = 1.5
A_shear = np.array([[1, k],
                    [0, 1]])
print("Shearing Matrix A_shear:\\n", A_shear)

# 2. Apply the scaling matrix A_scale to vector v = [1, 1]
v = np.array([1, 1])
v_scaled = A_scale @ v
print("\\nScaled Vector (A_scale @ v):\\n", v_scaled)

# 3. Apply the rotation matrix A_rotate to the same vector v
v_rotated = A_rotate @ v
print("\\nRotated Vector (A_rotate @ v):\\n", v_rotated)

# 4. Apply the shearing matrix A_shear to vector v
v_sheared = A_shear @ v
print("\\nSheared Vector (A_shear @ v):\\n", v_sheared)
"""))

# Affine Transformations Section
solutions_cells.append(create_markdown("---\n\n## Affine Transformations\n\n### Task 5: Implement Affine Transformations - Solutions"))

# Task 5 Solutions Code Cell
solutions_cells.append(create_code("""# Task 5: Implement Affine Transformations - Solutions

# 1. Define a translation vector b = [3, -2]
b = np.array([3, -2])
print("Translation Vector b:\\n", b)

# 2. Create a rotation matrix for 90 degrees
theta = np.pi / 2  # 90 degrees in radians
A_rotate_90 = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
print("\\nRotation Matrix A_rotate_90 (90 degrees):\\n", A_rotate_90)

# 3. Define the affine transformation function
def affine_transform(v, A, b):
    return A @ v + b

# 4. Define vector v
v = np.array([2, 1])
print("\\nOriginal Vector v:\\n", v)

# 5. Apply affine transformation
v_affine = affine_transform(v, A_rotate_90, b)
print("\\nAffine Transformed Vector (A_rotate_90 @ v + b):\\n", v_affine)
"""))

# Visualization of Transformations Section
solutions_cells.append(create_markdown("---\n\n## Visualization of Transformations\n\n### Task 6: Visualize an Affine Transformation - Solutions"))

# Task 6 Solutions Code Cell
solutions_cells.append(create_code("""# Task 6: Visualize an Affine Transformation - Solutions

# 1. Define a rotation matrix for 45 degrees
theta = np.pi / 4  # 45 degrees in radians
A_rotate_45 = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta),  np.cos(theta)]])
print("Rotation Matrix A_rotate_45 (45 degrees):\\n", A_rotate_45)

# 2. Define translation vector b = [2, 3]
b = np.array([2, 3])
print("\\nTranslation Vector b:\\n", b)

# 3. Define the affine transformation function
def affine_transform(v, A, b):
    return A @ v + b

# 4. Define vector v
v = np.array([1, 2])
print("\\nOriginal Vector v:\\n", v)

# 5. Apply affine transformation
v_affine = affine_transform(v, A_rotate_45, b)
print("\\nAffine Transformed Vector (A_rotate_45 @ v + b):\\n", v_affine)

# 6. Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Original vector
plt.arrow(0, 0, v[0], v[1], color='gray', width=0.05, label='Original Vector')

# Transformed vector
plt.arrow(0, 0, v_affine[0], v_affine[1], color='purple', width=0.05, label='Affine Transformed Vector')

plt.xlim(-1, 5)
plt.ylim(-1, 6)
plt.grid(True)
plt.legend()
plt.title('Visualization of Affine Transformation (Rotation + Translation)')
plt.show()
"""))

# Conclusion Section
solutions_cells.append(create_markdown("---\n\n## Conclusion\n\nIn this notebook, you've explored the solutions to the following tasks:\n\n- **Matrix Basics**: Definitions and creation of matrices.\n- **Matrix Operations**: Addition, subtraction, multiplication, and transpose.\n- **Diagonal Matrices**: Creation and properties.\n- **Linear Transformations**: Definitions and implementation of scaling, rotation, shearing.\n- **Affine Transformations**: Combining linear transformations with translations.\n- **Visualization**: Graphical representation of transformations.\n\nThese solutions provide a comprehensive understanding of matrix operations and transformations using NumPy. Reviewing and experimenting with these examples will enhance your proficiency in handling matrices in Python.\n\n---\n\n# Appendix: Additional Resources\n\n- [NumPy Official Documentation](https://numpy.org/doc/)\n- [Matplotlib Official Documentation](https://matplotlib.org/stable/contents.html)\n- [Linear Algebra - Khan Academy](https://www.khanacademy.org/math/linear-algebra)\n\n---"))

# Assign cells to solutions notebook
solutions_nb['cells'] = solutions_cells

# Define metadata
solutions_nb['metadata'] = {
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.5",
        "mimetype": "text/x-python",
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py"
    }
}

solutions_nb['nbformat'] = 4
solutions_nb['nbformat_minor'] = 5

# Write the solutions notebook to a file
with open('Matrix_Operations_and_Transformations_Solutions.ipynb', 'w') as f:
    nbf.write(solutions_nb, f)

print("Solutions Notebook 'Matrix_Operations_and_Transformations_Solutions.ipynb' has been created successfully.")