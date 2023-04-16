# @title evaluation
"""Evaluation on all metrics."""
from collections import Counter
from collections import defaultdict
from copy import deepcopy
import math
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist


# from scipy.special import rel_entr
# from scipy.special import softmax
# from scipy.stats import entropy
###
### Traditional relevance metrics.
###
def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)
    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def recall_at_k_list(actual, predicted, topk):
    res = []
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            res.append(round(len(act_set & pred_set) / float(len(act_set)), 4))
    return res


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    # print("actual:", len(actual))
    # print("predicted:", np.shape(predicted))
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id]))
                / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


###
### S_Alpha ndcg metric.
###
def ext_alpha_ndcg_k(
        actual, predicted, num_cate, item_cate_dict, cate_item_map, item_emb, topk
):
    """ext_alpha_ndcg_k.
    Args:
      actual:
      predicted:
      num_cate:
      item_cate_dict:
      cate_item_map:
      item_emb:
      topk:
    Returns:
    """
    ndcg_values_list = []
    for user_id in range(len(actual)):
        cate_num_occur = dict(zip(np.arange(num_cate), np.zeros(num_cate)))
        dcg_values = np.zeros(topk)
        curr_actual_emb = item_emb[actual[user_id], :]
        curr_predicted_emb = item_emb[predicted[user_id], :]
        relevance = 1 - cdist(curr_predicted_emb, curr_actual_emb, "cosine")
        relevance = relevance.max(1)
        relevance[relevance < 0] = 0
        value = 0.0
        for i in range(topk):
            topic = item_cate_dict[predicted[user_id][i]]
            value += (
                    relevance[i] * (0.5 ** cate_num_occur[topic]) / math.log(i + 2, 2)
            )
            cate_num_occur[topic] += 1
            dcg_values[i] = value
        idcg_values = ext_alpha_idcg_k(
            actual[user_id], item_emb, topk, item_cate_dict, cate_item_map
        )
        ndcg_values = dcg_values / idcg_values
        ndcg_values_list.append(ndcg_values[-1])
    return sum(ndcg_values_list) / float(len(actual))


def ext_alpha_idcg_k(ind_actual, item_emb, topk, item_cate_dict, cate_item_map):
    """ext_alpha_idcg_k.
    Args:
      ind_actual:
      item_emb:
      topk:
      item_cate_dict:
      cate_item_map:
    Returns:
    """
    num_cate = len(cate_item_map)
    cate_num_occur = dict(zip(np.arange(num_cate), np.zeros(num_cate)))
    ideal_ranking = []
    idcg_values = np.zeros(topk)
    curr_actual_emb = item_emb[ind_actual, :]
    relevance = 1 - cdist(item_emb, curr_actual_emb, "cosine")
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
            value = (
                    relevance[iid] * (0.5 ** cate_num_occur[cate]) / math.log(i + 2, 2)
            )
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


###
### Simpson's Diversity Index.
###
def simpson_div_k(pred_list, item_cate_dict, topk):
    nominator = 0
    for i in range(len(pred_list)):
        curr_cate_list = [item_cate_dict[ind] for ind in pred_list[i][:topk]]
        curr_cate_count = np.array(list(Counter(curr_cate_list).values()))
        nominator += (curr_cate_count * (curr_cate_count - 1)).sum()
    denominator = topk * (topk - 1)
    res = float(nominator / denominator) / len(pred_list)
    return 1 - res


###
### Function for metrics computation.
###
def compute_metrics(test_set, pred_list):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [5, 10, 20, 50]:
        precision.append(round(precision_at_k(test_set, pred_list, k), 6))
        recall.append(round(recall_at_k(test_set, pred_list, k), 6))
        MAP.append(round(mapk(test_set, pred_list, k), 6))
        ndcg.append(round(ndcg_k(test_set, pred_list, k), 6))
    return precision, recall, MAP, ndcg


def compute_ilad(rec_item_emb):  # num_user, topk, 64
    res_cos, ilad_cos = [], []
    res_euc, ilad_euc = [], []
    num_user = rec_item_emb.shape[0]
    for k in [5, 10, 20, 50]:
        for i in range(num_user):
            ilad_cos.append(pdist(rec_item_emb[i].squeeze()[:k], "cosine").mean())
            ilad_euc.append(pdist(rec_item_emb[i].squeeze()[:k], "euclidean").mean())
        res_cos.append(round(np.mean(ilad_cos), 6))
        res_euc.append(round(np.mean(ilad_euc), 6))
    return res_cos, res_euc


def invert_dict(d):
    d_inv = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv


def compute_ext_alpha_ndcg(
        test_set, pred_list, num_cate, item_cate_dict, cate_item_map, item_emb
):
    alpha_ndcg = []
    for k in [5, 10, 20, 50]:
        alpha_ndcg.append(
            round(
                ext_alpha_ndcg_k(
                    test_set,
                    pred_list,
                    num_cate,
                    item_cate_dict,
                    cate_item_map,
                    item_emb,
                    k,
                ),
                6,
            )
        )
    return alpha_ndcg


def compute_simpson_div(pred_list, item_cate_dict):
    simpson_div = []
    for k in [5, 10, 20, 50]:
        simpson_div.append(round(simpson_div_k(pred_list, item_cate_dict, k), 6))
    return simpson_div


###
### Relevance.
###
def soft_recall_at_k(actual, predicted, item_emb, topk):
    sum_rel = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        curr_actual_emb = item_emb[actual[i], :]
        curr_predicted_emb = item_emb[predicted[i][:topk], :]
        relevance = 1 - cdist(curr_actual_emb, curr_predicted_emb, "cosine")
        relevance = relevance.max(1)
        sum_rel += np.sum(relevance > 0.95) / float(len(set(actual[i])))
    return sum_rel / num_users


def compute_soft_recall(test_set, pred_list, item_emb):
    rel = []
    for k in [5, 10, 20, 50]:
        rel.append(round(soft_recall_at_k(test_set, pred_list, item_emb, k), 6))
    return rel


def soft_precision_at_k(actual, predicted, item_emb, topk):
    sum_rel = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        curr_actual_emb = item_emb[actual[i], :]
        curr_predicted_emb = item_emb[predicted[i][:topk], :]
        relevance = 1 - cdist(curr_predicted_emb, curr_actual_emb, "cosine")
        relevance = relevance.max(1)
        sum_rel += np.sum(relevance > 0.95) / float(topk)
    return sum_rel / num_users


def compute_soft_precision(test_set, pred_list, item_emb):
    rel = []
    for k in [5, 10, 20, 50]:
        rel.append(round(soft_precision_at_k(test_set, pred_list, item_emb, k), 6))
    return rel


###
### AUC
###
def compute_auc(test_set, rec_list_all):
    num_users, num_items = len(test_set), len(rec_list_all[0])
    auc = 0
    for i in range(num_users):
        num_pos = len(test_set[i])
        num_neg = num_items - num_pos
        indices = np.where(np.in1d(rec_list_all[i], test_set[i]))[0]
        # ref: https://www.jianshu.com/p/03a11a083a6d
        auc += float(
            (num_items - 1 - indices).sum() - num_pos * (num_pos + 1) / 2.0
        ) / (num_pos * num_neg)
    return auc / num_users
