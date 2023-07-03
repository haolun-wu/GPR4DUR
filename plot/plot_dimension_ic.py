import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'Dimension': [128, 64, 32, 16, 8, 4],
    'Random': [0.6909, 0.6930, 0.6859, 0.6902, 0.7056, 0.6971],
    'MostPop': [0.7037, 0.7037, 0.7037, 0.7037, 0.7037, 0.7037],
    'SUR': [0.7173, 0.6930, 0.6748, 0.6720, 0.6709, 0.6455],
    'MUR': [0.6721, 0.6921, 0.6821, 0.6558, 0.6548, 0.6357],
    'DUR': [0.7435, 0.7533, 0.7325, 0.7407, 0.7305, 0.7366]
}

df = pd.DataFrame(data)

# Set color palette
cmap = sns.cubehelix_palette(5, gamma=0.6)

# Set font and line width sizes
legend_font_size = 12
label_font_size = 10
tick_font_size = 8
line_width = 2

# Plotting
plt.figure(figsize=(8, 6))
sns.lineplot(x='Dimension', y='value', hue='variable', data=df.melt('Dimension'), marker='o', palette=cmap, linewidth=line_width)

plt.title('Comparison of IC@20 across Different Dimensions', fontsize=14)
plt.xlabel('Dimension Size', fontsize=label_font_size)
plt.ylabel('IC@20', fontsize=label_font_size)

plt.legend(fontsize=legend_font_size)
plt.grid(True)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.show()
