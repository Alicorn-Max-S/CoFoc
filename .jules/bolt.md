## 2025-02-18 - Optimized Transform Matrix Calculation
**Learning:** Manually constructing the transformation matrix (TRS) instead of chaining matrix multiplications (`T @ R @ S`) yields a ~3.3x speedup in Python/Numpy. This avoids allocating 3 intermediate 4x4 matrices and performing 2 matrix multiplications (which are expensive in Python due to overhead, even with Numpy).
**Action:** Look for similar matrix composition patterns in the rendering loop where intermediate objects can be avoided.
