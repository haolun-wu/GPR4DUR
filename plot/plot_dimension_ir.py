import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data
data = {
    'Dimension': [128, 64, 32, 16, 8, 4],
    'Random': [0.1505, 0.1872, 0.2036, 0.2511, 0.2991, 0.3965],
    'MostPop': [0.2125, 0.2493, 0.2728, 0.3241, 0.3638, 0.5172],
    'SUR': [0.3423, 0.3880, 0.4160, 0.4323, 0.4868, 0.5326],
    'MUR': [0.3379, 0.3376, 0.3756, 0.3986, 0.4180, 0.4245],
    'DUR': [0.3697, 0.4376, 0.4703, 0.4997, 0.5487, 0.5900]
}

df = pd.DataFrame(data)

# Set font sizes
title_font_size = 14
label_font_size = 12
legend_font_size = 9
tick_font_size = 10

# Create a colormap
cmap = sns.cubehelix_palette(5, gamma=0.6)

# Plotting
plt.figure(figsize=(8, 6))
sns.barplot(x='Dimension', y='value', hue='variable', data=df.melt('Dimension'), palette=cmap)

plt.title('Interest Relevance@20 across Different Dimensions', fontsize=title_font_size)
plt.xlabel('Dimension Size', fontsize=label_font_size)
plt.ylabel('Performance', fontsize=label_font_size)

plt.legend(fontsize=legend_font_size, ncol=3)
plt.xticks(fontsize=tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.ylim(0, 0.7)  # Set y-axis limits

plt.tight_layout()
plt.show()
