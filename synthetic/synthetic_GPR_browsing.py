import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import numpy.random as npr
from synthetic_load import dataset_loader
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow_probability as tfp

tf.get_logger().setLevel('ERROR')

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


def evaluate_interest_coverage(recommendations, user_interests, item_clusters):
    n_rounds = recommendations.shape[1]
    num_users = recommendations.shape[0]

    # Prepare interest coverage array
    interest_coverage = np.zeros(n_rounds)

    for round_num in range(n_rounds):
        total_coverage = 0.0
        for user_num in range(num_users):
            # Get the clusters of the items recommended to the user in this round
            recommended_item_clusters = set(item_clusters[recommendations[user_num, round_num]])

            # Get the clusters of the user's interests
            user_interest_clusters = set(user_interests[user_num])

            # Check the coverage of user's interests
            covered_interests = recommended_item_clusters.intersection(user_interest_clusters)

            # Compute the coverage rate and add to the total
            total_coverage += float(len(covered_interests) / len(user_interest_clusters))

        # Calculate the average coverage rate in this round
        interest_coverage[round_num] = total_coverage / num_users

    return interest_coverage


def evaluate_interest_relevance(recommendations, item_embedding_values, cluster_embeddings, item_clusters):
    item_clusters = item_clusters.astype(int)
    n_rounds = recommendations.shape[1]
    num_users = recommendations.shape[0]

    # Prepare interest relevance array
    interest_relevance = np.zeros(n_rounds)

    for round_num in range(n_rounds):
        total_relevance = 0.0
        for user_num in range(num_users):
            # Get the embeddings of the items recommended to the user in this round
            recommended_item_embeddings = item_embedding_values[recommendations[user_num, round_num]]

            # Get the cluster embeddings corresponding to the recommended items
            recommended_item_clusters = item_clusters[recommendations[user_num, round_num]]
            corresponding_cluster_embeddings = cluster_embeddings[recommended_item_clusters]

            # Compute cosine similarities between recommended items and their corresponding cluster embeddings
            similarity_scores = cosine_similarity(recommended_item_embeddings, corresponding_cluster_embeddings)

            # Compute the sum of similarity scores and add to the total
            total_relevance += np.sum(similarity_scores)

        # Calculate the average relevance in this round
        interest_relevance[round_num] = total_relevance / num_users

    return interest_relevance


def update_observation_and_history(recommended_ids, feedback, updated_user_item_ids, updated_user_observations):
    user_feedback = feedback
    new_user_items = np.array([item for item, clicked in zip(recommended_ids, user_feedback) if clicked is not None])
    new_observations = np.array([1 if clicked else -1 for clicked in user_feedback if clicked is not None])

    num_interaction = len(new_user_items)
    if len(updated_user_item_ids) < 50:
        updated_user_item_ids = np.concatenate((updated_user_item_ids, new_user_items))
        updated_user_observations = np.concatenate((updated_user_observations, new_observations))
    else:
        # remove oldest, add latest
        updated_user_item_ids = np.concatenate((updated_user_item_ids[num_interaction:], new_user_items))
        updated_user_observations = np.concatenate((updated_user_observations[num_interaction:], new_observations))

    return updated_user_item_ids, updated_user_observations


# Define function to simulate user browsing and click behavior based on the cascade model
# def simulate_user_feedback(recommended_items, ground_truth_scores, base_click_prob, stop_prob):
#     feedback = [None] * len(recommended_items)
#     for i in range(len(recommended_items)):
#         # User examines an item
#         if np.random.rand() < stop_prob:
#             feedback[i] = False  # user skips this item
#         else:
#             # User decides whether to click on the item
#             # Make click probability proportional to the ground truth score
#             click_prob = base_click_prob * ground_truth_scores[recommended_items[i]]
#             if np.random.rand() < click_prob:
#                 feedback[i] = True  # user clicks this item
#                 break  # user stops browsing
#             else:
#                 feedback[i] = False  # user skips this item
#     return feedback
def simulate_user_feedback(recommended_items, ground_truth_scores, item_clusters, base_click_prob, stop_prob,
                           continue_prob, alpha):
    feedback = [None] * len(recommended_items)
    ground_truth_scores_tf = tf.cast(ground_truth_scores, dtype=tf.float32)
    recommended_items_tf = tf.convert_to_tensor(recommended_items, dtype=tf.int32)

    # Compute penalty term for each item based on the number of items in its cluster
    item_clusters = item_clusters.astype(int)
    cluster_occurrences = np.bincount(item_clusters)
    penalty_terms = alpha ** cluster_occurrences[item_clusters[recommended_items]]

    attractiveness_scores_tf = base_click_prob * tf.gather(ground_truth_scores_tf, recommended_items_tf) / penalty_terms
    attractiveness_scores = attractiveness_scores_tf.numpy().tolist()

    for i in range(len(recommended_items)):
        # User examines an item
        if i > 0 and np.random.rand() < stop_prob:  # User does not stop for the first item
            feedback[i] = False  # user skips this item
        else:
            # User decides whether to click on the item based on attractiveness and previous clicks
            click_prob = attractiveness_scores[i]
            for j in range(0, i):
                if feedback[j] == True:  # If user clicked previous item
                    click_prob *= 1 - attractiveness_scores[
                        j]  # Decrease click prob based on attractiveness of clicked item
            if np.random.rand() < click_prob:
                feedback[i] = True  # user clicks this item
                if np.random.rand() < continue_prob:  # user decides to continue browsing
                    continue
                else:  # user decides to stop browsing
                    break
            else:
                feedback[i] = False  # user skips this item
    return feedback


if __name__ == '__main__':
    # hyperparameters
    amplitude = 1.0
    length_scale = 0.1
    gamma = 1.0
    o_noise = 10.0
    p_noise = 0.1
    recommend_mode = 'UCB'
    embedding_dim = 128

    # others
    n_rounds = 20
    topk = 10  # Set the topk value
    click_prob = 0.9  # Click probability
    stop_prob = 0.3  # Stop probability
    continue_prob = 0.7  # Continue probability
    alpha = 0.5  # penalty term

    # Load the dataset
    dataset = dataset_loader()

    item_clusters = dataset['item_clusters']
    items = dataset['items']
    user_interests = dataset['user_interests']
    user_item_sequences = dataset['user_item_sequences']
    print("user_interests:", user_interests.shape)
    print("user_item_sequences:", user_item_sequences.shape)

    num_users = user_interests.shape[0]
    num_clusters = len(np.unique(item_clusters))  # Number of unique interest clusters
    num_items = len(items)

    # 1. Obtain item embedding
    cluster_embeddings = npr.normal(
        size=(num_clusters, embedding_dim))  # Mean for Gaussian distribution for each cluster
    std_dev = 1.0  # Standard deviation for the Gaussian distributions
    item_embedding_values = np.empty((num_items, embedding_dim))
    # Sample item embedding from its cluster distribution
    for i, cluster in enumerate(item_clusters):
        item_embedding_values[i] = npr.normal(loc=cluster_embeddings[int(cluster)], scale=std_dev)
    print("item_embedding_values:", item_embedding_values.shape)

    # 2. Obtain user embedding
    user_embedding_values = np.empty((num_users, embedding_dim))
    for i, interests in enumerate(user_interests):
        # Count the occurrence of each interest in the user's interests
        unique, counts = np.unique(interests, return_counts=True)
        interest_count_dict = dict(zip(unique, counts))

        # Normalize counts to get weights
        total = sum(interest_count_dict.values())
        interest_weights = {k: v / total for k, v in interest_count_dict.items()}

        # Compute weighted average of cluster embeddings
        user_embedding_values[i] = np.sum(
            [interest_weights[j] * cluster_embeddings[j] for j in interests], axis=0)
    print("user_embedding_values:", user_embedding_values.shape)

    # 3. Obtain ground truth
    user_representations = tf.cast(user_embedding_values, dtype=tf.double)
    item_embeddings = tf.cast(item_embedding_values.T, dtype=tf.double)
    ground_truth_scores = tf.linalg.matmul(user_representations, item_embeddings)
    ground_truth_items = tf.math.top_k(ground_truth_scores, k=5)
    print("ground_truth_scores:", ground_truth_scores.shape)

    # 4. Obtain user history
    history = item_embedding_values[user_item_sequences]
    print("history:", history.shape)

    # Initialize recommendation arrays
    full_recommendations = np.empty((num_users, n_rounds, topk), dtype=int)

    for i in range(num_users):
        user_item_ids = user_item_sequences[i]
        observation_index_points = history[i].astype(np.float32)
        observations = np.ones(user_item_ids.shape).astype(np.float32),  # Initialize observations

        updated_user_item_ids = user_item_ids
        updated_user_item_embs = observation_index_points
        updated_user_observations = observations[0]
        print("original user_item_ids:", user_item_ids)
        print("original user_observations:", updated_user_observations)

        for round_num in range(n_rounds):  # Start from round 0 to include all rounds
            # 5. GPR fit
            # Note: GPR works on a per-user basis. If there are multiple users, we need multiple GPRs.
            gp_kernel = psd_kernels.Linear(bias_amplitude=length_scale, slope_amplitude=amplitude)

            gprm = tfd.GaussianProcessRegressionModel(
                kernel=gp_kernel,
                index_points=item_embedding_values.astype(np.float32),
                observation_index_points=updated_user_item_embs.astype(np.float32),
                observations=updated_user_observations.astype(np.float32),
                observation_noise_variance=o_noise,
                predictive_noise_variance=p_noise,
            )

            # 6. GPR predict
            # The GPR predicts the expected reward of each item for each user
            prediction_mean = gprm.mean()
            prediction_var = gprm.stddev()

            if recommend_mode == 'UCB':
                ucb_score = prediction_mean + 0.1 * prediction_var
                cur_recommendation = np.argsort(ucb_score)[-topk:]  # Only keep the top k items
            elif recommend_mode == 'TS':
                ts_score = gprm.sample()
                cur_recommendation = np.argsort(ts_score)[-topk:]  # Only keep the top k items
            full_recommendations[i, round_num] = cur_recommendation

            # Simulate user feedback
            feedback = simulate_user_feedback(cur_recommendation, ground_truth_scores[i], item_clusters, click_prob,
                                              stop_prob,
                                              continue_prob, alpha)

            # update
            updated_user_item_ids, updated_user_observations = update_observation_and_history(cur_recommendation,
                                                                                              feedback,
                                                                                              updated_user_item_ids,
                                                                                              updated_user_observations)
            updated_user_item_embs = item_embedding_values[updated_user_item_ids]

            print(f"User {i + 1}, Round {round_num + 1}")
            # print("ucb_recommendation:", ucb_recommendation)
            print("cur_recommendation:", cur_recommendation)
            # print("feedback:", feedback)
            # print("updated_user_item_ids:", updated_user_item_ids)
            # print("updated_user_observations:", updated_user_observations)
            print("------")

    print("full_recommendation:", full_recommendations)
    interest_coverage = evaluate_interest_coverage(full_recommendations, user_interests, item_clusters)
    interest_relevance = evaluate_interest_relevance(full_recommendations, item_embedding_values, cluster_embeddings,
                                                     item_clusters)
    print("Interest coverage per round:", interest_coverage)
    # print("Interest relevance per round:", interest_relevance)
