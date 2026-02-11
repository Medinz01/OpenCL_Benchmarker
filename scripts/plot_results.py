import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Data
try:
    df = pd.read_csv('build/benchmark_results.csv', header=None, names=['Algorithm', 'Device', 'Time_ms'])
    # Skip the header row if it exists (simple check)
    if df.iloc[0]['Algorithm'] == 'Algorithm':
        df = df.iloc[1:]
        df['Time_ms'] = pd.to_numeric(df['Time_ms'])
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# 2. Setup the Plot
plt.figure(figsize=(10, 6))
colors = []
labels = []
values = []

# Logic to assign colors
for index, row in df.iterrows():
    label = f"{row['Algorithm']} ({row['Device']})"
    labels.append(label)
    values.append(row['Time_ms'])
    
    if "CPU" in row['Device']:
        colors.append('#d62728') # Red for CPU
    elif "Pinned" in row['Device']:
        colors.append('#2ca02c') # Green for Optimization
    else:
        colors.append('#1f77b4') # Blue for Standard GPU

# 3. Create Bar Chart
bars = plt.barh(labels, values, color=colors)

# 4. Styling
plt.title('Optimization Impact: Pinned Memory vs Standard', fontsize=14, fontweight='bold')
plt.xlabel('Execution Time (ms) - Lower is Better', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.5)

# 5. Add Value Labels
for bar in bars:
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f} ms', 
             va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_final.png')
print("Graph saved to benchmark_final.png")
plt.show()  