import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for line plot
data_line = {
    'Dimension': [128, 64, 32, 16, 8, 4],
    'Random': [0.6909, 0.6930, 0.6859, 0.6902, 0.6956, 0.6951],
    'MostPop': [0.7037, 0.7037, 0.7037, 0.7037, 0.7037, 0.7037],
    'SUR': [0.7173, 0.6930, 0.6748, 0.6720, 0.6709, 0.6455],
    'MUR': [0.7121, 0.6921, 0.6821, 0.6558, 0.6548, 0.6357],
    'DUR': [0.7535, 0.7533, 0.7375, 0.7337, 0.7365, 0.7356]
}

df_line = pd.DataFrame(data_line)

# Marker types
markers = ['o', 'v', '^', '<', '>']

# Data for bar plot
data_bar = {
    'Dimension': [128, 64, 32, 16, 8, 4],
    'Random': [0.1505, 0.1872, 0.2036, 0.2511, 0.2991, 0.3965],
    'MostPop': [0.2125, 0.2493, 0.2728, 0.3241, 0.3638, 0.5172],
    'SUR': [0.3423, 0.3880, 0.4160, 0.4323, 0.4768, 0.5326],
    'MUR': [0.3379, 0.3876, 0.4156, 0.4486, 0.4880, 0.5245],
    'DUR': [0.3697, 0.4376, 0.4703, 0.4997, 0.5487, 0.5900]
}

df_bar = pd.DataFrame(data_bar)

legend_fontsize = 11
label_fontsize = 15
tick_fontsize = 13
line_width = 2

# Set figure size
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Set color palette
cmap = sns.cubehelix_palette(5, gamma=0.6)

# Plot line plot
# sns.lineplot(x='Dimension', y='value', hue='variable', data=df_line.melt('Dimension'), marker='o', palette=cmap,
#              linewidth=line_width, ax=axes[0])
for i, col in enumerate(df_line.columns[1:]):
    axes[0].plot(df_line['Dimension'], df_line[col], marker=markers[i], color=cmap[i], linewidth=line_width, label=col)

axes[0].set_title('Interest Coverage@20 (IC@20)', fontsize=14)
axes[0].set_xlabel('Dimension Size', fontsize=label_fontsize)
axes[0].set_ylabel('Performance', fontsize=label_fontsize)
axes[0].legend(fontsize=legend_fontsize, ncol=3)
axes[0].grid(True)

# Plot bar plot
# Define the order of the bars
order = ['DUR', 'MUR', 'SUR', 'MostPop', 'Random']

# Plot the bar plot with hue_order
sns.barplot(x='Dimension', y='value', hue='variable', data=df_bar.melt('Dimension'), palette=cmap[::-1], ax=axes[1],
            hue_order=order)

axes[1].set_title('Interest Relevance@20 (IR@20)', fontsize=14)
axes[1].set_xlabel('Dimension Size', fontsize=label_fontsize)
axes[1].set_ylabel('Performance', fontsize=label_fontsize)
# axes[1].legend(fontsize=legend_fontsize, ncol=3)


# Get the legend handles and labels from the plot
handles, labels = axes[1].get_legend_handles_labels()

# Manually specify legend order
axes[1].legend_.remove()  # Remove the current legend

# Create a new legend with specified order
legend_order = ['Random', 'MostPop', 'SUR', 'MUR', 'DUR']  # the order you want for legend labels
handles_ordered = [handles[labels.index(l)] for l in legend_order]  # order handles (colored lines) accordingly
labels_ordered = [labels[labels.index(l)] for l in legend_order]  # order labels accordingly

axes[1].legend(handles_ordered, labels_ordered, fontsize=legend_fontsize, ncol=3)

axes[1].set_ylim(0, 0.7)  # Set y-axis limits

axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

# Highlight x-axis tick positions in the first subfigure
axes[0].set_xticks([128, 64, 32, 16, 8, 4])

# Reverse the x-axis
axes[0].invert_xaxis()
axes[1].invert_xaxis()

plt.tight_layout()
plt.savefig('robustness.png', bbox_inches='tight')
plt.show()
