import numpy as np

# Read the data from the dataset
item_clusters = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/item_clusters.npy')
items = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/items.npy')
# user_interests = np.load(
#     '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests.npy')
user_interests_slice_1 = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests_slice_1.npy')
user_interests_slice_2 = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests_slice_2.npy')
user_interests_slice_3 = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests_slice_3.npy')
user_interests_slice_4 = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_interests_slice_4.npy')
user_item_sequences = np.load(
    '/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/synthetic_data_alpha0.8_gamma0.1/user_item_sequences.npy')


def dataset_loader():
    dataset = {}
    dataset['item_clusters'] = item_clusters
    dataset['items'] = items
    # dataset['user_interests'] = user_interests
    dataset['user_interests_slice_1'] = user_interests_slice_1
    dataset['user_interests_slice_2'] = user_interests_slice_2
    dataset['user_interests_slice_3'] = user_interests_slice_3
    dataset['user_interests_slice_4'] = user_interests_slice_4
    dataset['user_item_sequences'] = user_item_sequences
    return dataset
