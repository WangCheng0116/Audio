import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Sample data - replace with your actual uncertainty measurements
# Number of samples per condition
n_samples = 100

# Generate sample data for each uncertainty metric and condition
conditions = ['Faithful', 'Adversarial', 'Irrelevant', 'Neutral']

# Creating sample data for each metric
data = []

# p(True) - lower values indicate greater uncertainty
for condition in conditions:
    base = 0.85 if condition == 'Faithful' else 0.65 if condition == 'Neutral' else 0.45
    spread = 0.10 if condition == 'Adversarial' else 0.05
    values = np.clip(np.random.normal(base, spread, n_samples), 0, 1)
    for val in values:
        data.append({
            'Condition': condition,
            'Metric': 'p(True)',
            'Value': val
        })

# Verbalized uncertainty
for condition in conditions:
    base = 0.05 if condition == 'Faithful' else 0.10 if condition == 'Neutral' else 0.20
    spread = 0.08 if condition == 'Adversarial' else 0.04
    values = np.clip(np.random.normal(base, spread, n_samples), 0, 1)
    for val in values:
        data.append({
            'Condition': condition,
            'Metric': 'Verbalized Uncertainty',
            'Value': val
        })

# Embedding distance
for condition in conditions:
    base = 0.15 if condition == 'Faithful' else 0.25 if condition == 'Neutral' else 0.40
    spread = 0.10 if condition == 'Adversarial' else 0.07
    values = np.clip(np.random.normal(base, spread, n_samples), 0, 1)
    for val in values:
        data.append({
            'Condition': condition,
            'Metric': 'Embedding Distance',
            'Value': val
        })

# Entropy
for condition in conditions:
    base = 0.30 if condition == 'Faithful' else 0.45 if condition == 'Neutral' else 0.65
    spread = 0.15 if condition == 'Adversarial' else 0.10
    values = np.clip(np.random.normal(base, spread, n_samples), 0, 1)
    for val in values:
        data.append({
            'Condition': condition,
            'Metric': 'Entropy',
            'Value': val
        })

# Create DataFrame
df = pd.DataFrame(data)

# Set up the figure with Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Create figure with subplots in one row - more compact
fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=False)

# Custom colors for conditions
colors = {
    'Faithful': '#2ca02c',   # green
    'Adversarial': '#ff7f0e', # orange
    'Irrelevant': '#9467bd',  # purple
    'Neutral': '#1f77b4'      # blue
}

# Define the order of conditions
condition_order = ['Neutral', 'Faithful', 'Adversarial', 'Irrelevant']

metrics = ['p(True)', 'Verbalized Uncertainty', 'Embedding Distance', 'Entropy']
titles = ['p(True)', 'Verbalized Uncertainty', 'Embedding Distance', 'Entropy']

for i, (metric, title) in enumerate(zip(metrics, titles)):
    # Filter data for current metric
    metric_df = df[df['Metric'] == metric]
    
    # Create boxplot
    ax = axes[i]
    sns.boxplot(x='Condition', y='Value', data=metric_df, 
                order=condition_order,
                palette=[colors[c] for c in condition_order],
                ax=ax)
    
    # Add strip plot (individual points)
    sns.stripplot(x='Condition', y='Value', data=metric_df,
                 order=condition_order,
                 color='black', size=2, alpha=0.2,
                 jitter=True, ax=ax)
    
    # Customize the plot
    ax.set_title(title, fontsize=12)
    
    # Only add y-label for the first plot
    if i == 0:
        ax.set_ylabel('Uncertainty Measure', fontsize=11)
    else:
        ax.set_ylabel('')
    
    # Remove x-axis labels and ticks for all plots
    ax.set_xlabel('')
    ax.set_xticklabels([])
    
    # Add grid but make it lighter
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Reduce whitespace
    ax.margins(x=0.1)

# Create legend for the conditions
legend_elements = [
    Patch(facecolor=colors['Neutral'], label='Neutral'),
    Patch(facecolor=colors['Faithful'], label='Faithful'),
    Patch(facecolor=colors['Adversarial'], label='Adversarial'),
    Patch(facecolor=colors['Irrelevant'], label='Irrelevant')
]

# Add legend below the subplots
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, 
           bbox_to_anchor=(0.5, 0), frameon=True)

# Make the layout more compact
plt.tight_layout()
plt.subplots_adjust(bottom=0.18, wspace=0.15)  # Reduce space between plots

# Save the figure
plt.savefig('uncertainty_distribution.png', dpi=300, bbox_inches='tight')
