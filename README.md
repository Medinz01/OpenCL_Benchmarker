## Performance Results

**Test Image:** 6691×4281 pixels (28 megapixels)  
**Hardware:** NVIDIA GeForce GTX 1650  
**Averaged over 2 runs**

### Gaussian Blur Performance

| Implementation | Total Time | Speedup vs CPU | Improvement |
|----------------|------------|----------------|-------------|
| CPU Baseline | 445.5ms | 1.0x | - |
| GPU (Global Memory) | 124.1ms | **3.6x** | Baseline GPU |
| GPU (Local Memory) | 67.7ms | **6.6x** | **2.0x faster than naive GPU** |

### Optimization Impact

**Global Memory (Naive):**
- Each thread makes 25 reads from VRAM (3×3 kernel × RGB + neighbors)
- Bandwidth-bound: 124ms

**Local Memory (Optimized):**
- Cooperative tile loading: 1 read per pixel shared across threads
- Compute-bound: **67ms (46% faster)**

**PCIe Transfer Overhead:** ~40ms (constant across both GPU implementations)

### Key Insight

Halving global memory accesses doubled GPU throughput. Local memory optimization had 2x impact on kernel performance, proving that **memory locality > thread count** for bandwidth-bound operations.