import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('build/multi_res_results.csv')

# Create the plot
plt.figure(figsize=(12, 7))

# Plot CPU and GPU lines
plt.plot(df['Resolution'], df['CPU_ms'], marker='o', linewidth=2, label='CPU (Serial)', color='#d62728')
plt.plot(df['Resolution'], df['GPU_Local_ms'], marker='s', linewidth=2, label='GPU (Local Memory)', color='#2ca02c')

# Log scale makes the crossover at small resolutions visible
plt.yscale('log')

# Adding labels and titles
plt.title('Performance Scaling & Crossover Analysis', fontsize=16, fontweight='bold')
plt.xlabel('Resolution', fontsize=12)
plt.ylabel('Execution Time (ms) - [Log Scale]', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="-", alpha=0.3)

# Highlight crossover point
plt.annotate('Crossover Point', xy=(1, df.iloc[1]['GPU_Local_ms']), xytext=(0.5, 100),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

plt.tight_layout()
plt.savefig('performance_crossover.png')
plt.show()