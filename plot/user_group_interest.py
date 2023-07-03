import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
interest_relevance_data = {
    'g1': [0.4, 0.48, 0.68, 0.69, 0.7],
    'g2': [0.38, 0.43, 0.6, 0.68, 0.72],
    'g3': [0.39, 0.47, 0.6, 0.71, 0.72],
    'g4': [0.5, 0.45, 0.52, 0.69, 0.71],
    'g5': [0.42, 0.455, 0.45, 0.64, 0.706],
}
interest_relevance_df = pd.DataFrame(interest_relevance_data, index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

tail_exposure_data = {
    'g1': [-0.25, -0.21, -0.18, -0.16, -0.16],
    'g2': [-0.21, -0.13, -0.11, -0.15, -0.115],
    'g3': [-0.19, -0.07, -0.1, -0.08, -0.065],
    'g4': [-0.11, -0.005, -0.05, -0.02, 0.02],
    'g5': [-0.06, 0.02, -0.02, 0.01, 0.08],
}
tail_exposure_df = pd.DataFrame(tail_exposure_data, index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

# Set font sizes
title_font_size = 14
label_font_size = 12
legend_font_size = 9
tick_font_size = 10

# Create a colormap
unique_categories = pd.concat([interest_relevance_df['index'], tail_exposure_df['index']]).unique()
cmap = sns.cubehelix_palette(5, gamma=0.6)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

sns.barplot(x='variable', y='value', hue='index', data=interest_relevance_df, palette=cmap, ax=axes[0])
axes[0].set_title('Interest Relevance@20', size=title_font_size)
axes[0].set_xlabel('User groups by number of interests', size=label_font_size)
axes[0].set_ylabel('Performance', size=label_font_size)
axes[0].legend(fontsize=legend_font_size, ncol=3)
axes[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[0].set_ylim(0, 0.8)  # Set y-axis limits for the left plot

sns.barplot(x='variable', y='value', hue='index', data=tail_exposure_df, palette=cmap, ax=axes[1])
axes[1].set_title('Tail Exposure Improvement@20', size=title_font_size)
axes[1].set_xlabel('User groups by number of interests', size=label_font_size)
axes[1].set_ylabel('Performance', size=label_font_size)
axes[1].legend(fontsize=legend_font_size, ncol=3, loc='lower right')
axes[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[1].set_ylim(-0.3, 0.1)  # Set y-axis limits for the right plot


plt.tight_layout()
plt.savefig('user_group_interet.png')
plt.show()
