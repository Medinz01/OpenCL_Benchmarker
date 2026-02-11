# OpenCL Image Processing Benchmark Suite

## 🚀 Overview
A high-performance benchmarking tool designed to analyze the trade-offs between **CPU (Serial)** and **GPU (Parallel)** image processing pipelines.

Built with **C++17** and **OpenCL 1.2**, this project quantifies the "Systems Bottleneck" in heterogeneous computing: specifically, at what point does the cost of transferring data over the PCIe bus outweigh the benefits of massive parallel computation?

## 📊 Key Findings & Benchmarks

**Test Environment:**
* **Image:** 6691×4281 pixels (28 Megapixels, ~85MB raw data)
* **GPU:** NVIDIA GeForce GTX 1650
* **CPU:** Intel Core i5/i7 (Serial Implementation)

### 1. Baseline Performance (Standard Memory Transfer)
| Algorithm | CPU Time | GPU Time | Speedup | Verdict |
|-----------|----------|----------|---------|---------|
| **Grayscale** (Memory Bound) | **57.4 ms** | 163.7 ms | **0.35x** (Slower) ⚠️ | **CPU Wins** |
| **Gaussian Blur** (Compute Bound) | 627.9 ms | **117.5 ms** | **5.3x** (Faster) 🚀 | **GPU Wins** |

### 2. Optimization: Pinned Memory (Zero-Copy)
To mitigate the PCIe bottleneck observed in the Grayscale test, I implemented **Pinned Memory** (`CL_MEM_ALLOC_HOST_PTR`). This allows the GPU to access host memory directly via DMA, bypassing the CPU staging buffer copy.

| Optimization Strategy | Execution Time | Improvement |
|-----------------------|----------------|-------------|
| Standard Transfer     | 163.7 ms       | Baseline    |
| **Pinned Memory** | **129.3 ms** | **~21% Faster** ⚡ |

---

## 💡 Engineering Insights

**1. The "Transfer Tax":**
For O(1) operations like **Grayscale** (where 1 pixel in = 1 pixel out), the GPU is effectively starved. The overhead of copying 85MB of data to VRAM (~35-40ms latency) is higher than the time it takes the CPU to just process the data in L3 cache.
* *Insight:* Do not offload simple math to the GPU unless the dataset is already on the GPU.

**2. The Parallelism Win:**
For O(N) operations like **Gaussian Blur** (where 1 pixel out requires 9 inputs and multiplications), the computational density is high enough to hide the transfer latency. The GPU's thousands of cores crush the serial CPU implementation, delivering a **530% speedup**.

---

## 🛠 Tech Stack
* **Language:** C++17
* **Compute API:** OpenCL 1.2 (Heterogeneous Computing)
* **Image I/O:** `stb_image` (Header-only library for raw pixel access)
* **Build System:** CMake
* **Visualization:** Python (Pandas + Matplotlib)
* **Optimization:** Pinned Memory / Zero-Copy Mapping

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