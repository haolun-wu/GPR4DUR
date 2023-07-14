"""Generates and saves synthetic data for analysis.
"""
import math
import numpy as np
import os
from typing import Any, Dict, List, Sequence, Text
from collections import Counter

import numpy as np
from tensorflow.io import gfile

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="GPR4DUR_synthetic_data")
    parser.add_argument('--platform', type=str, default='CPU', choices=['CPU', 'GPU', 'TPU'])
    parser.add_argument('--output_file_path', type=str,
                        default='/home/haolun/projects/def-cpsmcgil/haolun/GPR4DUR/synthetic_data/')
    parser.add_argument('--type', type=str, default='sparse_conditional',
                        choices=['global', 'conditional', 'mixture_num_interests', 'mixture_interest_volatility',
                                 'sparse_conditional'])
    parser.add_argument('--num_clusters', type=int, default=10, help='Number of clusters for items.')
    parser.add_argument('--num_data_points', type=int, default=100, help='Number of user sequences to generate.')
    parser.add_argument('--items_per_cluster', type=int, default=20, help='Items in each cluster.')
    parser.add_argument('--interests_per_user', type=int, default=5, help='Number of interests assigned to each user.')
    parser.add_argument('--num_items_in_seq', type=int, default=20, help='Number of items in the sequence.')
    parser.add_argument('--alpha', type=float, default=0.8, help='Probability of staying in a cluster.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Probability of transition to a different interesting cluster.')
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='Probability of staying in a "not-interesting" cluster.')
    parser.add_argument('--interest_power', type=float, default=1.0, help='Exponent for user interests power law.')
    parser.add_argument('--item_power', type=float, default=1.0, help='Exponent for item power law within a cluster.')

    return parser.parse_args()


def save_file(path: Text, **data):
    """Saves numpy data arrays."""

    gfile.makedirs(path)
    for arr_name, arr in data.items():
        with gfile.GFile(os.path.join(path, arr_name + '.npy'), 'wb') as f:
            np.save(f, arr)


def run_markov_chain(transition_matrix: List[List[float]],
                     initial_state_prob: List[float],
                     time_steps: int) -> List[int]:
    """Runs the markov chain for a given transition matrix.

    Args:
      transition_matrix: a 2d matrix specifying the probability of transition
        across states.
      initial_state_prob: A probability vector specifying the probability of state
        at t=0.
      time_steps: Number of steps for which the markov chain will run.

    Returns:
      user_interest_sequence: A list of integers of length time_steps, specifying
        the state at each time_step.

    Raise:
      ValueError:
        - If rows of transition matrix do not sum to 1, or
        - If initial_state_prob doesn't sum to 1.
    """

    transition_matrix = np.array(transition_matrix)
    num_states = transition_matrix.shape[0]
    if not np.allclose(np.sum(transition_matrix, axis=1), np.ones(num_states, )):
        raise ValueError('Invalid transition matrix. Rows do not sum to 1.')
    elif not math.isclose(np.sum(initial_state_prob), 1.0):
        raise ValueError(
            'Initial probability distribution (Sum: {}) does not sum to 1.'.format(
                np.sum(initial_state_prob)))

    interests = np.arange(len(transition_matrix))
    user_interest_sequence = np.ones((time_steps,), dtype=int) * -1
    user_interest_sequence[0] = np.random.choice(interests, p=initial_state_prob)

    for t in range(1, time_steps):
        prev_interest = user_interest_sequence[t - 1]
        user_interest_sequence[t] = np.random.choice(
            interests, p=transition_matrix[prev_interest])

    return user_interest_sequence


def generate_item_sequence_from_interest_sequence(
        user_interest_sequence: List[int],
        items_per_interest: int,
        item_power: float = 0.0) -> List[int]:
    """Generates a user item sequence from a list of user interests.

    Args:
      user_interest_sequence: A sequence of user interests at different timesteps.
      items_per_interest: Number of item in each interest cluster.
      item_power: The exponent for power-law distribution of items in an interest.
        If zero, the distribution will be uniform.

    Returns:
      user_item_sequence: A user item sequence corresponding to user interests.
    """

    user_item_sequence = []
    prob = np.arange(1, items_per_interest + 1) ** (-1.0 * item_power)
    prob /= np.sum(prob)
    for interest in user_interest_sequence:
        user_item_sequence.append(np.random.choice(
            np.arange(items_per_interest * interest,
                      items_per_interest * (interest + 1)),
            p=prob))

    return np.array(user_item_sequence)


def load_data(data_path: str) -> Dict[str, Any]:
    """Loads synthetic dataset given the path to generated dataset.

    Args:
      data_path: Path to the data directory, which contains the dataset. The path
        should be the output directory used when the synthetic dataset was
        generated. When the generated synthetic data has type = `global`,
        `user_interests.npy` doesn't exist. For details see
        synthetic_data/generate_synthetic_data.py.

    Returns:
      dataset: A dictionary containing the synthetic dataset.
    """

    dataset = dict()
    filenames = [
        'items', 'user_item_sequences', 'item_clusters', 'user_interests'
    ]

    for fname in filenames:
        file_path = os.path.join(data_path, fname + '.npy')
        if not gfile.exists(file_path):
            continue

        with gfile.GFile(file_path, 'rb') as f:
            dataset[fname] = np.load(f)

    user_interests = dataset.get('user_interests', None)

    if user_interests is not None:
        item_clusters = dataset['item_clusters']
        dataset['ground_truth_scores'] = extract_ground_truth_scores_for_ndcg(
            item_clusters, user_interests)

    return dataset


def extract_ground_truth_scores_for_ndcg(item_clusters, user_interests):
    """Returns the ground truth relevance scores for synthetic dataset.

    The scores are used as the ground truth for the NDCG@K metric. For synthetic
    data, since all the items are uniformly drawn from a cluster, we set the same
    score for all items in a cluster. For instance, if a user is interested in
    clusters [1, 2], then all items in [1, 2] will have a score of 1.0, whereas
    all other items will have score of 0.0.

    Args:
      item_clusters: The cluster ids for all the items in the dataset.
      user_interests: Array of user interests.

    Returns:
      ground_truth_scores: An array of size [num_users, num_items], where each
      element depicts the relevance score of items for each user.
    """

    ground_truth_scores = np.expand_dims(item_clusters, 0)
    ground_truth_scores = np.tile(ground_truth_scores,
                                  (user_interests.shape[0], 1))
    ground_truth_scores = [
        np.isin(item_clusters, user_interest)
        for (item_clusters,
             user_interest) in zip(ground_truth_scores, user_interests)
    ]
    ground_truth_scores = np.array(ground_truth_scores).astype(dtype=float)
    return ground_truth_scores


def generate_global_markovian_data(num_clusters: int,
                                   items_per_cluster: int,
                                   num_data_points: int = 10000,
                                   time_steps: int = 20,
                                   alpha: float = 0.9) -> Dict[Text, Any]:
    """Generates user-item histories using a hierarchical markovian process.

    Assumes that each cluster of items depict a user interest and each cluster has
    a fixed number of items. The user-item interactions are generated by
    considering a discrete stochastic process for cluster transitions. The
    function assumes that the transition matrix has `alpha` as the diagonal and
    `(1-alpha)/(num_interests - 1)` as the off-diagonal elements.

    Args:
      num_clusters: The total number of clusters. The clusters will correspond to
        a state in the discrete markov process.
      items_per_cluster: The number of items belonging to each cluster.
      num_data_points: The number of data points (user sequences) to genertate.
      time_steps: The number of steps for which the markov chain is run.
      alpha: The diagonal entries of the interest transition matrix.

    Returns:
      A dict containg keys 'user_item_sequences' corresponding to user histories,
      'items' corresponding to item ids, `item_labels` corresponding to the
      mapping of items to their respecive clusters.
    """

    items = np.arange(items_per_cluster * num_clusters)
    item_clusters = np.ones((items_per_cluster * num_clusters,)) * -1

    # Transition matrix for the discrete markov process.
    transition_matrix = np.ones((num_clusters, num_clusters)) * (1.0 - alpha) / (
            num_clusters - 1)
    for ix in range(num_clusters):
        transition_matrix[ix, ix] = alpha
        item_clusters[ix * items_per_cluster:(ix + 1) * items_per_cluster] = ix

    initial_state_dist = np.ones((num_clusters,)) * 1.0 / num_clusters
    user_item_sequences = np.ones((num_data_points, time_steps), dtype=int) * -1
    for u in range(num_data_points):
        user_interest_sequence = run_markov_chain(transition_matrix,
                                                  initial_state_dist,
                                                  time_steps)
        user_item_sequences[u] = generate_item_sequence_from_interest_sequence(
            user_interest_sequence, items_per_cluster)

    dataset = dict(
        user_item_sequences=user_item_sequences,
        items=items,
        item_clusters=item_clusters)

    return dataset


def generate_user_specific_markovian_data(
        num_clusters: int,
        items_per_cluster: int,
        clusters_per_user: int,
        num_data_points: int = 10000,
        time_steps: int = 20,
        alpha: float = 0.8,
        gamma: float = 0.1,
        epsilon: float = 0.2,
        interest_power: float = 0.0,
        item_power: float = 0.0) -> Dict[Text, Any]:
    """Generates user-item histories using a conditional hierarchical markovian process.

    Assumes that each cluster of items depict a user interest and each cluster has
    a fixed number of items. Each user is assumed to have `cluster_per_user`
    interests. The user-item interactions are generated by considering a discrete
    stochastic process for cluster transitions. The transition matrix is user
    specific and depends on the interests of the user. While running the markov
    chain, the transition matrix is assumed to be:

    transition[i,j] = alpha, if `i=j` corresponds to a user interest cluster.
    transition[i,j] = gamma / (clusters_per_user-1), if `i!=j` and both `i` and
      `j` correspond to user's interest clusters.
    transition[i,j] = (1-alpha-gamma)/(H-clusters_per_user), if only `i`
      corresponds to one of user's interest clusters.
    transition[i,j] = epsilon, if `i=j` corresponds to a cluster not in user's
      interests.
    transition[i,j] = 0.0, if `i` and `j` corresponds to clusters not in user's
      interests.
    transition[i,j] = (1-epsilon) / clusters_per_user, if `i` corresponds to a
      cluster not in user's interests and 'j' corresponds to a user interest
      cluster.

    Args:
      num_clusters: The total number of clusters. The clusters will correspond to
        a state in the discrete markov process.
      items_per_cluster: The number of items belonging to each cluster.
      clusters_per_user: The number of interests a user may have. Should be less
        than `num_clusters`.
      num_data_points: The number of data points (user sequences) to genertate.
      time_steps: The number of steps for which the markov chain is run.
      alpha: The transition probability to stay in a user interest cluster.
      gamma: The total probability of transition to a different user interest
        cluster. Note alpha + gamma <= 1
      epsilon: The transition probability to stay in a cluster which is not a user
        interest.
      interest_power: The exponent for power-law distirbution of interests. If
        zero, the distribution will be uniform.
      item_power: The exponent for power-law distribution of items in an interest.
        If zero, the distribution will be uniform.

    Returns:
      A dict containg keys 'user_item_sequences' corresponding to user histories,
      `user_interests` corresponding to a subset of clusters that reflect the
      interest of users, 'items' corresponding to item ids, and `item_clusters`
      corresponding to the mapping of items to their respecive clusters.

    Raises:
      ValueError: If alpha + gamma > 1.
    """

    if alpha + gamma > 1:
        raise ValueError('Probability of transitions cannot be greater than 1.')

    items = np.arange(items_per_cluster * num_clusters)
    interests = np.arange(num_clusters, dtype=int)

    # Allocate interests to users by random cluster assignment.
    def random_choice():
        return np.random.choice(interests, size=clusters_per_user, replace=False)

    if interest_power > 0.0:
        # Sample with replacement to get power-law distribution.
        prob = (interests + 1) ** (-1.0 * interest_power)
        prob /= np.sum(prob)
        assigned_interests = np.random.choice(
            interests,
            size=(num_data_points, clusters_per_user),
            replace=True,
            p=prob)
    else:
        # Sample without replacement for each user using a uniform distribution.
        assigned_interests = np.array(
            [random_choice() for ix in range(num_data_points)])

    def get_transition_probability(user_interest_set: List[int],
                                   src_interest_state: int) -> List[float]:
        """Returns the `src_state` row of the user-specific transition matrix."""

        src_state_interesting = src_interest_state in user_interest_set
        num_interests = len(user_interest_set)

        if src_state_interesting:

            exploration_prob = (1 - alpha - gamma) / (num_clusters - num_interests)
            transition_probs = np.ones((num_clusters,)) * exploration_prob
            for j in range(num_clusters):
                if j == src_interest_state:
                    if num_interests == 1:
                        transition_probs[j] = alpha + gamma
                    else:
                        transition_probs[j] = alpha
                elif j in user_interest_set:
                    transition_probs[j] = gamma / (num_interests - 1)
        else:
            transition_probs = np.zeros((num_clusters,))
            transition_probs[src_interest_state] = epsilon
            transition_probs[user_interest_set] = (1 - epsilon) / num_interests

        return transition_probs

    def get_transition_matrix(user_interest_set: List[int]) -> List[List[float]]:
        """Returns the transition matrix of a user given user_interest_set."""

        transition_matrix = np.zeros((num_clusters, num_clusters))
        for src_state in range(num_clusters):
            transition_matrix[src_state] = get_transition_probability(
                user_interest_set, src_state)
        return transition_matrix

    # Assume state distribution at time=0 is uniform.
    initial_state_dist = np.ones((num_clusters,)) * 1.0 / num_clusters
    user_item_sequences = np.ones((num_data_points, time_steps), dtype=int) * -1
    for u in range(num_data_points):
        interest_set = list(set(assigned_interests[u]))
        transition_matrix = get_transition_matrix(interest_set)
        user_interest_sequence = run_markov_chain(transition_matrix,
                                                  initial_state_dist,
                                                  time_steps)
        user_item_sequences[u] = generate_item_sequence_from_interest_sequence(
            user_interest_sequence, items_per_cluster, item_power)

    item_clusters = np.ones((items_per_cluster * num_clusters,)) * -1
    for ix in range(num_clusters):
        item_clusters[ix * items_per_cluster:(ix + 1) * items_per_cluster] = ix

    dataset = dict(
        user_item_sequences=user_item_sequences,
        user_interests=assigned_interests,
        items=items,
        item_clusters=item_clusters)

    return dataset


def generate_heterogeneuos_user_slices(num_clusters: int,
                                       items_per_cluster: int,
                                       clusters_per_user_list: List[int],
                                       num_data_points: int = 10000,
                                       time_steps: int = 20,
                                       alpha: float = 0.8,
                                       gamma: float = 0.1,
                                       epsilon: float = 0.2) -> Dict[Text, Any]:
    """Generate dataset with multiple slices each with distinct clusters_per_user.

    Each dataset slice has equal number of users. Example: If num_data_points=100
    and number of slices is 4, then each slice will have 25 users.

    Args:
      num_clusters: The total number of clusters. The clusters will correspond to
        a state in the discrete markov process.
      items_per_cluster: The number of items belonging to each cluster.
      clusters_per_user_list: A list depicting number of interests to assign for
        each user. For each element of the list, we generate one dataset slice.
        For example, if clusters_per_user_list = [2,3,5], then there will be 3
        dataset slices and the users in the dataset slices will have 2,3, and 5
        interests per user.
      num_data_points: The number of data points (user sequences) to generate.
      time_steps: The number of steps for which the markov chain is run.
      alpha: The transition probability to stay in a user interest cluster.
      gamma: The total probability of transition to a different user interest
        cluster. Note alpha + gamma <= 1
      epsilon: The transition probability to stay in a cluster which is not a user
        interest.

    Returns:
      dataset: A dictionary containing the following keys
        'user_item_sequences': User item sequence corresponding to user histories.
        `user_interests_slice_k`: User interests for the kth slice of the dataset.
        'items' corresponding to item ids.
        `item_clusters` corresponding to the mapping of items to their respective
          clusters.
    """

    dataset = dict()
    user_item_sequences = []
    for k, cluster_per_user in enumerate(clusters_per_user_list):
        dataset_slice = generate_user_specific_markovian_data(
            num_clusters=num_clusters,
            items_per_cluster=items_per_cluster,
            clusters_per_user=cluster_per_user,
            num_data_points=num_data_points // len(clusters_per_user_list),
            time_steps=time_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon)

        dataset[f'user_interests_slice_{k + 1}'] = dataset_slice['user_interests']
        user_item_sequences.append(dataset_slice['user_item_sequences'])

    dataset['user_item_sequences'] = np.concatenate(user_item_sequences, axis=0)
    # items and item_clusters remain same across dataset slices.
    dataset['items'] = dataset_slice['items']
    dataset['item_clusters'] = dataset_slice['item_clusters']

    return dataset


def generate_mixture_interest_volatility_users(
        num_clusters: int,
        items_per_cluster: int,
        clusters_per_user: int,
        alpha_list: List[float],
        num_data_points: int = 10000,
        time_steps: int = 20,
        epsilon: float = 0.2) -> Dict[Text, Any]:
    """Generate dataset with multiple slices each with distinct user volatility.

    Each dataset slice has equal number of users. Example: If num_data_points=100
    and number of slices is 4, then each slice will have 25 users.

    Args:
      num_clusters: The total number of clusters. The clusters will correspond to
        a state in the discrete markov process.
      items_per_cluster: The number of items belonging to each cluster.
      clusters_per_user: The number of interests a user has.
      alpha_list: The transition probability to stay in a user interest cluster.
        For each alpha in list, one user slice is created.
      num_data_points: The number of data points (user sequences) to generate.
      time_steps: The number of steps for which the markov chain is run.
      epsilon: The transition probability to stay in a cluster which is not a user
        interest.

    Returns:
      dataset: A dictionary containg the following keys
        'user_item_sequences': User item sequence corresponding to user histories.
        `user_interests`: User interests.
        'items' corresponding to item ids.
        `item_clusters`: corresponding to the mapping of items to their respective
          clusters.
    """

    dataset = dict()
    user_item_sequences = []
    user_interests = []
    for alpha in alpha_list:
        gamma = 0.9 - alpha
        dataset_slice = generate_user_specific_markovian_data(
            num_clusters=num_clusters,
            items_per_cluster=items_per_cluster,
            clusters_per_user=clusters_per_user,
            num_data_points=num_data_points // len(alpha_list),
            time_steps=time_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon)

        user_interests.append(dataset_slice['user_interests'])
        user_item_sequences.append(dataset_slice['user_item_sequences'])

    dataset['user_interests'] = np.concatenate(user_interests, axis=0)
    dataset['user_item_sequences'] = np.concatenate(user_item_sequences, axis=0)
    # items and item_clusters remain same across dataset slices.
    dataset['items'] = dataset_slice['items']
    dataset['item_clusters'] = dataset_slice['item_clusters']

    return dataset


if __name__ == '__main__':
    args = parse_args()

    dataset_type = args.type
    if dataset_type == 'conditional':
        dataset = generate_user_specific_markovian_data(
            num_clusters=args.num_clusters,
            clusters_per_user=args.interests_per_user,
            items_per_cluster=args.items_per_cluster,
            num_data_points=args.num_data_points,
            time_steps=args.num_items_in_seq,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon)
        alpha_str = 'alpha{}_gamma{}'.format(args.alpha, args.gamma)
    elif dataset_type == 'mixture_num_interests':
        dataset = generate_heterogeneuos_user_slices(
            num_clusters=args.num_clusters,
            clusters_per_user_list=[4, 3, 2, 1],
            items_per_cluster=args.items_per_cluster,
            num_data_points=args.num_data_points,
            time_steps=args.num_items_in_seq,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon)
        alpha_str = 'alpha{}_gamma{}'.format(args.alpha, args.gamma)
    elif dataset_type == 'mixture_interest_volatility':
        alpha_mixture = [0.9, 0.8, 0.7, 0.6]
        dataset = generate_mixture_interest_volatility_users(
            num_clusters=args.num_clusters,
            clusters_per_user=args.interests_per_user,
            items_per_cluster=args.items_per_cluster,
            num_data_points=args.num_data_points,
            time_steps=args.num_items_in_seq,
            alpha_list=alpha_mixture,
            epsilon=args.epsilon)
        alpha_str = 'mixture_alpha{}'.format('_'.join(map(str, alpha_mixture)))
    elif dataset_type == 'sparse_conditional':
        dataset = generate_user_specific_markovian_data(
            num_clusters=args.num_clusters,
            clusters_per_user=args.interests_per_user,
            items_per_cluster=args.items_per_cluster,
            num_data_points=args.num_data_points,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            time_steps=args.num_items_in_seq,
            interest_power=args.interest_power,
            item_power=args.item_power)
        alpha_str = 'interest-power{}_item-power{}_alpha{}_gamma{}'.format(
            args.interest_power, args.item_power, args.alpha, args.gamma)
    else:
        dataset = generate_global_markovian_data(
            num_clusters=args.num_clusters,
            items_per_cluster=args.items_per_cluster,
            num_data_points=args.num_data_points,
            time_steps=args.num_items_in_seq,
            alpha=args.alpha)
        alpha_str = 'alpha{}'.format(args.alpha)

    print('Dataset created. Saving the data...')
    data_path = os.path.join(args.output_file_path,
                             'synthetic_data_{}'.format(alpha_str))
    save_file(data_path, **dataset)
    print('Dataset saved at %s.', data_path)

    """Show"""
    item_clusters = dataset['item_clusters']
    items = dataset['items']
    try:
        user_interests = dataset['user_interests']
    except:
        user_interests = dataset['user_interests_slice_2']
    user_item_sequences = dataset['user_item_sequences']

    item_counter = Counter(item_clusters.tolist())

    print("item_clusters:", item_clusters)
    print("item_counter:", item_counter)
    print("items:", items)
    print("user_interests:", user_interests.shape, user_interests)
    print("user_item_sequences:", user_item_sequences.shape, user_item_sequences)
