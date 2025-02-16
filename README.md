
# CUDA Matrix Operations

## Introduction
This project focuses on optimizing CUDA programs for two key operations: vector addition and matrix multiplication. It demonstrates how modifying memory access patterns and optimizing parallelism can significantly enhance the performance of these operations on NVIDIA GPUs. The project explores various techniques, such as improving memory coalescence and implementing optimizations for matrix multiplication.

## Methodology
The project is structured around two main tasks:

### Vector Addition (Non-Coalesced and Coalesced):
- **Non-Coalesced Vector Addition (`vecadd00`):** This is the base implementation of vector addition, where memory access is not optimized for coalescing. It is implemented in `vecaddKernel00.cu`.
- **Coalesced Vector Addition (`vecadd01`):** This is an optimized version that improves performance by ensuring coalesced memory access. It is implemented in `vecaddKernel01.cu`.

### Matrix Multiplication:
- **Basic Matrix Multiplication (`matmult00`):** This implementation performs matrix multiplication using basic CUDA operations, without any optimizations.
- **Optimized Matrix Multiplication (`matmult01`):** This version improves the performance of `matmult00` by optimizing memory access patterns and incorporating techniques like tiling.

## Observations
### Vector Addition
- **Non-Coalesced vs. Coalesced:** The optimized, coalesced vector addition (`vecadd01`) showed a significant improvement in performance compared to the non-coalesced version (`vecadd00`). This improvement is mainly due to better alignment of memory access with the GPU's architecture, which helps in utilizing the GPU’s bandwidth more effectively.

### Matrix Multiplication
- **Performance Improvement:** The optimized `matmult01` demonstrated a substantial reduction in execution time and an increase in GFlops/s, especially when dealing with larger matrix sizes, compared to the basic `matmult00`. This optimization was achieved through advanced techniques like memory coalescing and tiling, which enhanced memory access efficiency and reduced computational overhead.

## How to Run the Code
Before running the code, ensure that you have CUDA installed along with the appropriate NVIDIA drivers. You can use the following commands to compile and run the programs:

### Compilation
1. Navigate to the folder containing the code.
   ```bash
   cd path/to/folder
   ```
2. Clean and compile the code using the `make` command.
   ```bash
   make clean
   make
   ```

### Running the Programs

- **Vector Addition:**
   - For non-coalesced vector addition:
     ```bash
     ./vecadd00 [arguments]
     ```
   - For coalesced vector addition:
     ```bash
     ./vecadd01 [arguments]
     ```

- **Matrix Multiplication:**
   - For basic matrix multiplication:
     ```bash
     ./matmul00 [arguments]
     ```
   - For optimized matrix multiplication:
     ```bash
     ./matmul01 [arguments]
     ```

Replace `[arguments]` with the required inputs, as defined by the program's usage.

## Conclusion
This project highlights the importance of memory access patterns and parallel optimization techniques in improving the performance of CUDA applications. The results demonstrate how these optimizations, such as memory coalescing and tiling, can significantly enhance the performance of vector addition and matrix multiplication operations, providing valuable insights into the scalability and efficiency of GPU programming.
