# OpenCL Image Processing Benchmark Suite

## Overview
A high-performance benchmarking tool comparing **CPU (Serial)** vs **GPU (Parallel)** image processing pipelines. 
Built with C++17 and OpenCL 1.2 to analyze the trade-offs between computational throughput and PCIe data transfer latency.

## 📊 Key Findings (Benchmark Results)

**Test Environment:**
* **Image:** 6691×4281 pixels (28 Megapixels, ~85MB raw data)
* **GPU:** NVIDIA GeForce GTX 1650
* **CPU:** Intel Core i5/i7 (Serial Implementation)

| Algorithm | CPU Time | GPU Time | Speedup | Verdict |
|-----------|----------|----------|---------|---------|
| **Grayscale** (Memory Bound) | 57.4 ms | 151.7 ms | **0.38x** (Slower) ⚠️ | CPU Wins |
| **Gaussian Blur** (Compute Bound) | 627.9 ms | 122.7 ms | **5.1x** (Faster) 🚀 | GPU Wins |

### 💡 Critical Engineering Insight
**"GPU acceleration is not a silver bullet."**

1.  **The Transfer Bottleneck:** For O(1) operations like **Grayscale**, the overhead of copying 85MB of data over the PCIe bus (~35ms+) outweighs the parallel processing speed. The CPU processes the data faster than the GPU can receive it.
2.  **The Compute Threshold:** For O(N) operations like **Gaussian Blur** (where every pixel requires 9 floating-point multiplications), the GPU's massive parallelism crushes the CPU, easily amortizing the data transfer cost.

## 🛠 Tech Stack
* **Language:** C++17
* **Compute API:** OpenCL 1.2 (Heterogeneous Computing)
* **Image I/O:** `stb_image` (Header-only library)
* **Build System:** CMake
* **Visualization:** Python (Pandas + Matplotlib)

## 📂 Project Structure
```text
/src
    ├── main.cpp          # Host code (CPU logic + OpenCL runtime wrapper)
/kernels
    ├── filters.cl        # OpenCL C kernels (Grayscale & Gaussian Blur)
/scripts
    ├── plot_results.py   # Python script to visualize benchmark CSV
/include
    ├── stb_image.h       # Image loading
    └── stb_image_write.h # Image saving
🚀 Build Instructions
Prerequisites
C++ Compiler (MSVC or GCC)

CMake 3.10+

OpenCL SDK (NVIDIA CUDA Toolkit or Intel SDK)

Build & Run
Bash

mkdir build && cd build
cmake ..
cmake --build . --config Release

# Run the benchmark
.\Release\Benchmarker.exe

# Visualize results (requires Python)
cd ..
python scripts/plot_results.py
🔮 Roadmap & Optimizations
The current implementation uses standard clEnqueueWriteBuffer (blocking copy). Future improvements will target the memory bottleneck:

[ ] Phase 2: Implement Pinned Memory (Zero-Copy) to reduce PCIe transfer overhead.

[ ] Phase 3: Implement Local Memory Tiling to optimize cache hits for convolution kernels.

[ ] Phase 4: Add Sobel Edge Detection filter.


---

### **Step 3: Push to GitHub**

Run these commands in your terminal (root folder):

```powershell
git init
git add .
git commit -m "Initial commit: Baseline CPU vs GPU benchmark with standard memory transfer"
git branch -M main
# (Create a new repo on GitHub and get the URL)
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main