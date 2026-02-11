import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load Data
try:
    df = pd.read_csv('build/benchmark_results.csv')
except FileNotFoundError:
    print("Error: Could not find benchmark_results.csv. Did you run the C++ app?")
    sys.exit(1)

# Pivot data for plotting
pivot_df = df.pivot(index='Algorithm', columns='Device', values='Time_ms')

# Plot
ax = pivot_df.plot(kind='bar', figsize=(10, 6), rot=0)

# Styling
plt.title('Performance Benchmark: CPU vs GPU (OpenCL)', fontsize=14)
plt.ylabel('Execution Time (ms) - Lower is Better', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

plt.tight_layout()
plt.savefig('benchmark_chart.png')
print("Chart saved as benchmark_chart.png")
plt.show()