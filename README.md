# Final Year Project:

## Introduction

The Bernstein-Vazirani algorithm is a quantum algorithm that efficiently solves a specific type of problem, exploiting quantum parallelism. In this project, we aim to modify the Bernstein-Vazirani algorithm by replacing the XOR operation with another function that yields orthogonal vectors.

Orthonormal vectors can be discriminated perfectly, so such a modification may provide a potentially powerful tool for quantum information processing and quantum computing tasks.

## Modification Strategies

There are two potential strategies we thought of for modifying the Bernstein-Vazirani algorithm:

1. **Direct Generation of Orthogonal Vectors:**

   - Develop a function that directly generates orthogonal vectors as outputs.
2. **Generation of Linearly Independent Vectors and Application of Quantum Gram-Schmidt (QGS):**

   - Create a function that produces linearly independent vectors.
   - Apply an algorithm, possibly Quantum Gram-Schmidt, to convert these vectors into an orthonormal basis.

## Buildup to Quantum Gram-Schmidt (QGS)

### Two-State Discrimination

- Illustrate with an example, such as |0⟩ and |+⟩ states.
- Use calculus to derive an error bound for discrimination.

### N-State Discrimination

- Extend the discrimination to multiple states.
- Derive a generalized error bound formula for discriminating N states.

### Quantum Gram-Schmidt (QGS)

#### Research Question

- Investigate the potential of Quantum Gram-Schmidt to achieve perfect discrimination for any linearly independent set of vectors.
- Explore the theoretical aspects and practical implications of using QGS in the context of quantum algorithms.
