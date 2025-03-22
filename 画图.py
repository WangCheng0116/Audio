import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# Sample data (replace with your actual data)
models = ["Standard", "CoT", "Audio Priority", "Bias Aware"]

# Accuracy data
acc_neutral = [60.5, 65.2, 67.8, 63.1]  # Neutral condition
acc_faithful = [87.2, 89.4, 88.0, 86.5]  # Faithful condition 
acc_adversarial = [43.6, 52.3, 58.7, 56.2]  # Adversarial condition
acc_irrelevant = [53.7, 58.1, 62.4, 60.9]  # Irrelevant condition

# Text Influence Rate (TIF) data
tif_faithful = [86.1, 83.2, 75.4, 79.1]  # Faithful condition
tif_adversarial = [53.7, 42.5, 31.8, 36.3]  # Adversarial condition
tif_irrelevant = [15.3, 12.1, 8.7, 10.2]  # Irrelevant condition

# Create a figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True)

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Colors for different conditions
colors = {
    'neutral': '#1f77b4',  # blue
    'faithful': '#2ca02c',  # green
    'adversarial': '#ff7f0e',  # orange
    'irrelevant': '#9467bd'  # purple
}

# Plot Accuracy data (top graph)
x = np.arange(len(models))
width = 0.18

ax1.plot(x, acc_neutral, 'o-', color=colors['neutral'], label='Neutral', linewidth=2, markersize=8)
ax1.plot(x, acc_faithful, 'o-', color=colors['faithful'], label='Faithful', linewidth=2, markersize=8)
ax1.plot(x, acc_adversarial, 'o-', color=colors['adversarial'], label='Adversarial', linewidth=2, markersize=8)
ax1.plot(x, acc_irrelevant, 'o-', color=colors['irrelevant'], label='Irrelevant', linewidth=2, markersize=8)

# Customize the top graph
ax1.set_ylabel('Accuracy (%)', fontsize=14, labelpad=1)
ax1.set_ylim(30, 100)
ax1.grid(True, linestyle='--', alpha=0.7)

# Only add the legend to the top graph
legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=True, fontsize=12)

# Plot Text Influence Rate data (bottom graph)
ax2.plot(x, tif_faithful, 'o-', color=colors['faithful'], linewidth=2, markersize=8)
ax2.plot(x, tif_adversarial, 'o-', color=colors['adversarial'], linewidth=2, markersize=8)
ax2.plot(x, tif_irrelevant, 'o-', color=colors['irrelevant'], linewidth=2, markersize=8)

# Customize the bottom graph
ax2.set_ylabel('Text Influence Rate (%)', fontsize=14, labelpad=1)
ax2.set_xlabel('Prompt Strategy', fontsize=14)
ax2.set_ylim(0, 100)
ax2.grid(True, linestyle='--', alpha=0.7)

# Set x-ticks for both graphs
plt.xticks(x, models, fontsize=12)

# Add data labels to points
for i, ax in enumerate([ax1, ax2]):
    lines = ax.get_lines()
    for line in lines:
        y_data = line.get_ydata()
        x_data = line.get_xdata()
        for x, y in zip(x_data, y_data):
            ax.annotate(f'{y:.1f}', (x, y), 
                        textcoords="offset points", 
                        xytext=(0, 10), 
                        ha='center',
                        fontsize=10)

plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust top spacing for the legend

# Save the figure
plt.savefig('lalm_evaluation_graphs.png', dpi=300, bbox_inches='tight')
