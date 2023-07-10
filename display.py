import numpy as np
from collections import Counter

# Load the .npy file
item_clusters = np.load('synthetic_data/synthetic_data_alpha0.8_gamma0.1/item_clusters.npy')
item_counter = Counter(item_clusters.tolist())
items = np.load('synthetic_data/synthetic_data_alpha0.8_gamma0.1/items.npy')
user_interests = np.load('synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests.npy')
user_item_sequences = np.load('synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_item_sequences.npy')

# Print the contents of the array
print("item_clusters:", item_clusters)
print("item_counter:", item_counter)
print("items:", items)
print("user_interests:", user_interests.shape, user_interests)
print("user_item_sequences:", user_item_sequences.shape, user_item_sequences)
