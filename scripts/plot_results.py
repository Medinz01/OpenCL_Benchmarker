import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data from your latest run
data = {
    'Algorithm': ['Grayscale', 'Grayscale', 'Grayscale', 'Gaussian Blur', 'Gaussian Blur', 'Gaussian Blur'],
    'Device': ['CPU', 'GPU (Standard)', 'GPU (Pinned)', 'CPU', 'GPU (Standard)', 'GPU (Local Mem)'],
    'Time_ms': [52.06, 155.10, 129.78, 445.47, 129.46, 17.85]
}

df = pd.DataFrame(data)

# Separate into two charts for clarity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: The "Bandwidth Bottleneck" (Grayscale)
gray_df = df[df['Algorithm'] == 'Grayscale']
colors_g = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
bars1 = ax1.bar(gray_df['Device'], gray_df['Time_ms'], color=colors_g)
ax1.set_title('Memory Bound: Grayscale (PCIe Bottleneck)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Time (ms) - Lower is Better')
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.bar_label(bars1, fmt='%.1f ms', padding=3)

# Chart 2: The "Compute Victory" (Blur)
blur_df = df[df['Algorithm'] == 'Gaussian Blur']
colors_b = ['#d62728', '#1f77b4', '#9467bd'] # Red, Blue, Purple
bars2 = ax2.bar(blur_df['Device'], blur_df['Time_ms'], color=colors_b)
ax2.set_title('Compute Bound: Gaussian Blur (25x Speedup)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.3)
ax2.bar_label(bars2, fmt='%.1f ms', padding=3)

plt.suptitle('Final Benchmark: Heterogeneous Computing Performance', fontsize=16)
plt.tight_layout()
plt.savefig('final_benchmark_result.png')
print("Graph saved!")
plt.show()