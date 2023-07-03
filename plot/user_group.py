import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
interest_relevance_data_1 = {
    'g1': [0.34, 0.44, 0.65, 0.68, 0.7],
    'g2': [0.35, 0.48, 0.64, 0.69, 0.71],
    'g3': [0.39, 0.47, 0.57, 0.71, 0.72],
    'g4': [0.4, 0.471, 0.53, 0.72, 0.75],
    'g5': [0.46, 0.475, 0.46, 0.64, 0.73],
}
interest_relevance_df_1 = pd.DataFrame(interest_relevance_data_1,
                                       index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

tail_exposure_data_1 = {
    'g1': [-0.28, -0.22, -0.17, -0.16, -0.172],
    'g2': [-0.24, -0.14, -0.11, -0.13, -0.12],
    'g3': [-0.17, -0.08, -0.1, -0.09, -0.09],
    'g4': [-0.1, -0.02, -0.05, -0.02, 0.03],
    'g5': [-0.05, 0.05, -0.02, 0.04, 0.09],
}
tail_exposure_df_1 = pd.DataFrame(tail_exposure_data_1,
                                  index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

interest_relevance_data_2 = {
    'g1': [0.4, 0.48, 0.68, 0.69, 0.7],
    'g2': [0.38, 0.43, 0.6, 0.68, 0.72],
    'g3': [0.39, 0.47, 0.6, 0.71, 0.72],
    'g4': [0.5, 0.45, 0.52, 0.69, 0.71],
    'g5': [0.42, 0.455, 0.45, 0.64, 0.706],
}
interest_relevance_df_2 = pd.DataFrame(interest_relevance_data_2,
                                       index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

tail_exposure_data_2 = {
    'g1': [-0.25, -0.21, -0.18, -0.16, -0.16],
    'g2': [-0.21, -0.13, -0.11, -0.15, -0.115],
    'g3': [-0.19, -0.07, -0.1, -0.08, -0.065],
    'g4': [-0.11, -0.005, -0.05, -0.02, 0.02],
    'g5': [-0.06, 0.02, -0.02, 0.01, 0.08],
}
tail_exposure_df_2 = pd.DataFrame(tail_exposure_data_2,
                                  index=['Random', 'MostPop', 'SUR', 'MUR', 'DUR']).reset_index().melt('index')

# Set font sizes
title_font_size = 14
label_font_size = 14
legend_font_size = 10
tick_font_size = 14
subtitle_fontsize = 20

# Create a colormap
cmap = sns.cubehelix_palette(5, gamma=0.6)

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=False)

sns.barplot(x='variable', y='value', hue='index', data=interest_relevance_df_1, palette=cmap, ax=axes[0, 0])
axes[0, 0].set_title('Interest Relevance@20 (IR@20)', size=title_font_size)
axes[0, 0].set_xlabel('', size=label_font_size)
axes[0, 0].set_ylabel('Performance', size=label_font_size)
axes[0, 0].legend(fontsize=legend_font_size, ncol=3)
axes[0, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[0, 0].set_ylim(0, 1)  # Set y-axis limits for the left plot

sns.barplot(x='variable', y='value', hue='index', data=tail_exposure_df_1, palette=cmap, ax=axes[0, 1])
axes[0, 1].set_title('Tail Exp. Improv.@20 (TEI@20)', size=title_font_size)
axes[0, 1].set_xlabel('', size=label_font_size)
axes[0, 1].set_ylabel('Performance', size=label_font_size)
axes[0, 1].legend(fontsize=legend_font_size, ncol=3, loc='lower right')
axes[0, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[0, 1].set_ylim(-0.3, 0.1)  # Set y-axis limits for the right plot

sns.barplot(x='variable', y='value', hue='index', data=interest_relevance_df_2, palette=cmap, ax=axes[1, 0])
axes[1, 0].set_title('Interest Relevance@20 (IR@20)', size=title_font_size)
axes[1, 0].set_xlabel('', size=label_font_size)
axes[1, 0].set_ylabel('Performance', size=label_font_size)
axes[1, 0].legend(fontsize=legend_font_size, ncol=3)
axes[1, 0].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[1, 0].set_ylim(0, 1)  # Set y-axis limits for the left plot

sns.barplot(x='variable', y='value', hue='index', data=tail_exposure_df_2, palette=cmap, ax=axes[1, 1])
axes[1, 1].set_title('Tail Exp. Improv.@20 (TEI@20)', size=title_font_size)
axes[1, 1].set_xlabel('', size=label_font_size)
axes[1, 1].set_ylabel('Performance', size=label_font_size)
axes[1, 1].legend(fontsize=legend_font_size, ncol=3, loc='lower right')
axes[1, 1].tick_params(axis='both', which='major', labelsize=tick_font_size)
axes[1, 1].set_ylim(-0.3, 0.1)  # Set y-axis limits for the right plot



# Add subtitles
fig.text(0.5, 0.51, '(a) Users grouped by number of interactions', ha='center', va='center', fontsize=subtitle_fontsize, fontproperties='Times New Roman', fontweight="bold")
fig.text(0.5, 0.03, '(b) Users grouped by number of interests', ha='center', va='center', fontsize=subtitle_fontsize, fontproperties='Times New Roman', fontweight="bold")

fig.tight_layout(pad=4.0)
plt.savefig('user_group.png', bbox_inches='tight')
plt.show()
