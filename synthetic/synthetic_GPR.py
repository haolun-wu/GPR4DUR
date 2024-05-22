import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import numpy.random as npr
from synthetic_load import dataset_loader
import tensorflow_probability as tfp

tf.get_logger().setLevel('ERROR')

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

# hyperparameter
amplitude = 1.0
length_scale = 0.1
gamma = 1.0
o_noise = 10.0
p_noise = 0.1

# Load the dataset
dataset = dataset_loader()

item_clusters = dataset['item_clusters']
items = dataset['items']
user_interests = dataset['user_interests_slice_1']
user_item_sequences = dataset['user_item_sequences'][:25]
print("user_interests:", user_interests.shape)
print("user_item_sequences:", user_item_sequences.shape)

num_users = user_interests.shape[0]
num_clusters = len(np.unique(item_clusters))  # Number of unique interest clusters
num_items = len(items)
embedding_dim = 32

# 1. Obtain item embedding
cluster_embeddings = npr.normal(size=(num_clusters, embedding_dim))  # Gaussian distribution for each cluster
item_embedding_values = np.empty((num_items, embedding_dim))
for i, cluster in enumerate(item_clusters):
    item_embedding_values[i] = cluster_embeddings[int(cluster)]  # Sample item embedding from its cluster distribution
print("item_embedding_values:", item_embedding_values.shape)

# 2. Obtain user embedding
power_law_weights = npr.power(a=3, size=(num_users, num_clusters))  # Weight vector for each user
user_embedding_values = np.empty((num_users, embedding_dim))
for i, interests in enumerate(user_interests):
    user_embedding_values[i] = np.sum([power_law_weights[i, j] * cluster_embeddings[j] for j in interests], axis=0)
print("user_embedding_values:", user_embedding_values.shape)

# 3. Obtain ground truth
user_representations = tf.reduce_sum(user_embedding_values[:, None, :] * power_law_weights[:, :, None], axis=1)
print("user_representations:", user_representations.shape)
user_representations = tf.cast(user_representations, dtype=tf.double)
item_embeddings = tf.cast(item_embedding_values.T, dtype=tf.double)
ground_truth_scores = tf.linalg.matmul(user_representations, item_embeddings)
ground_truth_items = tf.math.top_k(ground_truth_scores, k=5)
print("ground_truth_scores:", ground_truth_scores.shape)

# 4. Obtain user history
history = item_embedding_values[user_item_sequences]

print("item_embedding_values:", item_embedding_values.shape)
print("history:", history.shape)

# 5. GPR fit
# Note: GPR works on a per-user basis. If there are multiple users, we need multiple GPRs.
gp_kernel = psd_kernels.Linear(bias_amplitude=length_scale, slope_amplitude=amplitude)
observations = np.ones((num_users, user_item_sequences.shape[1])).astype(np.float32)  # Initialize observations
gprms = [tfd.GaussianProcessRegressionModel(
    kernel=gp_kernel,
    index_points=item_embedding_values.astype(np.float32),
    observation_index_points=history[i].astype(np.float32),
    observations=observations[i],
    observation_noise_variance=o_noise,
    predictive_noise_variance=p_noise,
) for i in range(num_users)]

# 6. GPR predict
# The GPR predicts the expected reward of each item for each user
prediction_means = [gprm.sample() for gprm in gprms]
prediction_vars = [np.var(sample) for sample in
                   prediction_means]  # Variance of the samples as a proxy for prediction variance
prediction_means = np.array(prediction_means)
prediction_vars = np.array(prediction_vars)
print("prediction_means:", prediction_means.sum())
print("prediction_vars:", prediction_vars.sum())

# UCB and Thompson Sampling
n_rounds = 10
confidence_level = 1.96
recommendations_UCB = []
recommendations_TS = []
topk = 5  # Set the topk value

for i in range(num_users):
    for round_num in range(1, n_rounds + 1):  # Start from round 1 to avoid division by zero
        # UCB
        ucb_score = prediction_means[i] + confidence_level * prediction_vars[i]
        ucb_recommendation = np.argsort(ucb_score)[-topk:]  # Only keep the top k items
        recommendations_UCB.append(ucb_recommendation)

        # Thompson Sampling
        sample = np.random.normal(loc=prediction_means[i], scale=np.sqrt(prediction_vars[i]))
        ts_recommendation = np.argsort(sample)[-topk:]  # Only keep the top k items
        recommendations_TS.append(ts_recommendation)

recommendations_UCB = np.array(recommendations_UCB).reshape(num_users, n_rounds, -1)
recommendations_TS = np.array(recommendations_TS).reshape(num_users, n_rounds, -1)

print("Recommendations (UCB):\n", recommendations_UCB)
print("Recommendations (Thompson Sampling):\n", recommendations_TS)
