import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
interest_relevance_data = {
    'g1': [0.34, 0.44, 0.65, 0.68, 0.7],
    'g2': [0.35, 0.48, 0.64, 0.69, 0.71],
    'g3': [0.39, 0.47, 0.57, 0.71, 0.72],
    'g4': [0.4, 0.471, 0.53, 0.72, 0.75],
    'g5': [0.46, 0.475, 0.46, 0.64, 0.73],
}
interest_relevance_df = pd.DataFrame(interest_relevance_data, index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

tail_exposure_data = {
    'g1': [-0.28, -0.22, -0.17, -0.16, -0.172],
    'g2': [-0.24, -0.14, -0.11, -0.13, -0.12],
    'g3': [-0.17, -0.08, -0.1, -0.09, -0.09],
    'g4': [-0.1, -0.02, -0.05, -0.02, 0.03],
    'g5': [-0.05, 0.05, -0.02, 0.04, 0.09],
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
axes[0].set_xlabel('User groups by number of interactions', size=label_font_size)
axes[0].set_ylabel('Performance', size=label_font_size)
axes[0].legend(fontsize=legend_font_size, ncol=3)
axes[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[0].set_ylim(0, 1)  # Set y-axis limits for the left plot

sns.barplot(x='variable', y='value', hue='index', data=tail_exposure_df, palette=cmap, ax=axes[1])
axes[1].set_title('Tail Exposure Improvement@20', size=title_font_size)
axes[1].set_xlabel('User groups by number of interactions', size=label_font_size)
axes[1].set_ylabel('Performance', size=label_font_size)
axes[1].legend(fontsize=legend_font_size, ncol=3, loc='lower right')
axes[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[1].set_ylim(-0.3, 0.1)  # Set y-axis limits for the right plot

plt.tight_layout()
plt.savefig('user_group_degree.png')

plt.show()
