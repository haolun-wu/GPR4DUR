import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set a random seed for reproducibility
np.random.seed(0)

# (1) Generate random 5-dimensional embeddings for 3000 items and 1 user
item_embeddings = np.random.rand(100, 5)
user_embedding = np.random.rand(1, 5)

# Compute the prediction scores as the dot product of the embeddings
scores = np.dot(item_embeddings, user_embedding.T).flatten()

# Mark 20 items as interacted items
interacted_items = np.random.choice(range(100), size=20, replace=False)

# Assign items to 4 categories
categories = np.random.choice(range(4), size=100)

# Use t-SNE to reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=0)
item_embeddings_2d = tsne.fit_transform(item_embeddings)

# Create the scatter plot
plt.figure(figsize=(10, 8))

# Plot non-interacted items
plt.scatter(item_embeddings_2d[~np.isin(range(100), interacted_items), 0],
            item_embeddings_2d[~np.isin(range(100), interacted_items), 1],
            c=categories[~np.isin(range(100), interacted_items)],
            cmap='viridis', alpha=0.5)

# Plot interacted items
for category in range(4):
    mask = np.logical_and(np.isin(range(100), interacted_items), categories == category)
    plt.scatter(item_embeddings_2d[mask, 0], item_embeddings_2d[mask, 1],
                c=categories[mask], marker='^', cmap='viridis', edgecolor='k', s=100)

plt.colorbar(label='Category')
plt.title('t-SNE visualization of item embeddings (interacted items marked with triangles)')
plt.show()
