import math
import numpy as np
from scipy.spatial.distance import pdist, cdist
from copy import deepcopy
from collections import Counter, defaultdict


def ext_alpha_ndcg_k(actual, predicted, num_cate, item_cate_dict, cate_item_map, item_emb, topk):
    ndcg_values_list = []
    for user_id in range(len(actual)):
        cate_num_occur = dict(zip(np.arange(num_cate), np.zeros(num_cate)))
        dcg_values = np.zeros(topk)

        curr_actual_emb = item_emb[actual[user_id], :]
        curr_predicted_emb = item_emb[predicted[user_id], :]
        relevance = 1 - cdist(curr_predicted_emb, curr_actual_emb, 'cosine')
        relevance = relevance.max(1)
        relevance[relevance < 0] = 0

        value = 0.0
        for i in range(topk):
            topic = item_cate_dict[predicted[user_id][i]]
            value += relevance[i] * (0.5 ** cate_num_occur[topic]) / math.log(i + 2, 2)
            cate_num_occur[topic] += 1
            dcg_values[i] = value

        idcg_values = ext_alpha_idcg_k(actual[user_id], item_emb, topk, item_cate_dict, cate_item_map)
        ndcg_values = dcg_values / idcg_values
        ndcg_values_list.append(ndcg_values[-1])

    return sum(ndcg_values_list) / float(len(actual))


def ext_alpha_idcg_k(ind_actual, item_emb, topk, item_cate_dict, cate_item_map):
    num_cate = len(cate_item_map)
    cate_num_occur = dict(zip(np.arange(num_cate), np.zeros(num_cate)))

    ideal_ranking = []
    idcg_values = np.zeros(topk)

    curr_actual_emb = item_emb[ind_actual, :]
    relevance = 1 - cdist(item_emb, curr_actual_emb, 'cosine')
    relevance = relevance.max(1)
    relevance[relevance < 0] = 0

    # sort relevance score from high to low for each category
    cate_item_map_copy = deepcopy(cate_item_map)
    for i, (k, v) in enumerate(cate_item_map.items()):
        cate_item_map_copy[k] = [x for _, x in sorted(zip(relevance[v], v))][::-1]

    for i in range(topk):
        bestValue = float("-inf")
        whoIsBest = "noOne"
        item_candidates = []
        for k in list(cate_item_map_copy.keys()):
            if len(cate_item_map_copy[k]) > 0:
                item_candidates.append(cate_item_map_copy[k][0])

        for iid in item_candidates:
            cate = item_cate_dict[iid]
            value = relevance[iid] * (0.5 ** cate_num_occur[cate]) / math.log(i + 2, 2)
            if value > bestValue:
                bestValue = value
                whoIsBest = iid

        cate_of_best = item_cate_dict[whoIsBest]
        cate_num_occur[cate_of_best] += 1
        ideal_ranking.append(iid)
        cate_item_map_copy[cate_of_best].pop(0)  # remove the selected one

        idcg_values[i] = bestValue

    idcg_values = np.cumsum(idcg_values)

    return idcg_values


def compute_ext_alpha_ndcg(test_set, pred_list, num_cate, item_cate_dict, cate_item_map, item_emb):
    alpha_ndcg = []
    for k in [10, 20, 50]:
        alpha_ndcg.append(
            round(ext_alpha_ndcg_k(test_set, pred_list, num_cate, item_cate_dict, cate_item_map, item_emb, k), 6))

    return alpha_ndcg
