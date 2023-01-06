import math
import numpy as np
from scipy.spatial.distance import pdist, cdist
from copy import deepcopy
from collections import Counter, defaultdict


def alpha_ndcg_k(actual, predicted, num_cate, item_cate_dict, topk):
    ndcg_values_list = []
    for user_id in range(len(actual)):
        topics_num_of_occurrences = dict(zip(np.arange(num_cate), np.zeros(num_cate)))
        topic_dict_actual = {k: item_cate_dict[k] for k in actual[user_id]}
        topic_num_of_occur_actual = Counter(list(topic_dict_actual.values()))
        # print("topic_num_of_occur_actual:", topic_num_of_occur_actual)

        k = min(topk, len(predicted[user_id]))
        dcg_values = np.zeros(k)
        k_actual = min(topk, len(actual[user_id]))

        value = 0.0
        for i in range(k):
            topic = item_cate_dict[predicted[user_id][i]]
            if predicted[user_id][i] in set(actual[user_id]):
                value += (0.5 ** topics_num_of_occurrences[topic]) / math.log(i + 2, 2)
                topics_num_of_occurrences[topic] += 1
            else:
                value += 0
            dcg_values[i] = value

        idcg_values = alpha_idcg_k(topic_num_of_occur_actual, k, k_actual)
        ndcg_values = dcg_values / idcg_values

        ndcg_values_list.append(ndcg_values[-1])

    return sum(ndcg_values_list) / float(len(actual))


def alpha_idcg_k(topic_num_of_occur_actual, k, k_actual):
    topic_times = list(topic_num_of_occur_actual.values())
    topic_times = [i for i in topic_times if i != 0]
    idcg_values = np.zeros(k)
    num_rel = int(sum(topic_times))

    # get the optimal decay list
    occur_list = []
    for i in topic_times:
        occur_list.extend(np.arange(i))
    occur_list = sorted(occur_list)

    value = 0.0
    for i in range(k):
        if i < k_actual:
            value += (0.5 ** occur_list[i]) / math.log(i + 2, 2)
        else:
            value += 0
        idcg_values[i] = value

    return idcg_values


def compute_alpha_ndcg(test_set, pred_list, num_cate, item_cate_dict):
    alpha_ndcg = []
    for k in [10, 20, 50]:
        alpha_ndcg.append(round(alpha_ndcg_k(test_set, pred_list, num_cate, item_cate_dict, k), 6))

    return alpha_ndcg
