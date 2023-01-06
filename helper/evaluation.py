import math
import numpy as np
from scipy.spatial.distance import pdist, cdist
from copy import deepcopy
from collections import Counter, defaultdict


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    # print("actual:", len(actual))
    # print("predicted:", len(predicted))
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
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def alpha_ndcg_k(actual, predicted, num_topics, doc_topics_dict, topk):
    ndcg_values_list = []
    for user_id in range(len(actual)):
        topics_num_of_occurrences = dict(zip(np.arange(num_topics), np.zeros(num_topics)))
        topic_dict_actual = {k: doc_topics_dict[k] for k in actual[user_id]}
        topic_num_of_occur_actual = Counter(list(topic_dict_actual.values()))

        k = min(topk, len(predicted[user_id]))
        dcg_values = np.zeros(k)
        k_actual = min(topk, len(actual[user_id]))

        value = 0.0
        for i in range(k):
            topic = doc_topics_dict[predicted[user_id][i]]
            if predicted[user_id][i] in set(actual[user_id]):
                value += (0.6 ** topics_num_of_occurrences[topic]) / math.log(i + 2, 2)
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
            value += (0.6 ** occur_list[i]) / math.log(i + 2, 2)
        else:
            value += 0
        idcg_values[i] = value

    return idcg_values


def ext_alpha_ndcg_k(actual, predicted, num_cate, item_cate_dict, item_emb, topk):
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

        idcg_values = ext_alpha_idcg_k(actual[user_id], item_emb, topk, cate_item_map)
        ndcg_values = dcg_values / idcg_values
        ndcg_values_list.append(ndcg_values[-1])

    return ndcg_values_list


def ext_alpha_idcg_k(ind_actual, item_emb, topk, cate_item_map):
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


def compute_metrics(test_set, pred_list):
    precision, recall, MAP, ndcg = [], [], [], []
    for k in [10, 20, 50]:
        precision.append(round(precision_at_k(test_set, pred_list, k), 6))
        recall.append(round(recall_at_k(test_set, pred_list, k), 6))
        MAP.append(round(mapk(test_set, pred_list, k), 6))
        ndcg.append(round(ndcg_k(test_set, pred_list, k), 6))

    return precision, recall, MAP, ndcg


def compute_ilad(rec_item_emb):  # num_user, topk, 64
    res_cos, ilad_cos = [], []
    res_euc, ilad_euc = [], []
    num_user = rec_item_emb.shape[0]
    for k in [10, 20, 50]:
        for i in range(num_user):
            ilad_cos.append(pdist(rec_item_emb[i].squeeze()[:k], 'cosine').mean())
            ilad_euc.append(pdist(rec_item_emb[i].squeeze()[:k], 'euclidean').mean())
        res_cos.append(np.mean(ilad_cos))
        res_euc.append(np.mean(ilad_euc))

    return res_cos, res_euc


def invert_dict(d):
    d_inv = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv


if __name__ == '__main__':
    actual = [[1, 2, 4], [3, 4, 5]]
    predicted = [[1, 6, 2, 4, 7], [7, 5, 1, 6, 3]]

    item_cate_dict = {}
    item_cate_dict[1] = 0
    item_cate_dict[2] = 0
    item_cate_dict[3] = 0
    item_cate_dict[4] = 1
    item_cate_dict[5] = 1
    item_cate_dict[6] = 2
    item_cate_dict[7] = 3
    item_cate_dict[0] = 0
    print(ndcg_k(actual, predicted, 5))

    print("item_cate_dict:", item_cate_dict)

    cate_item_map = invert_dict(item_cate_dict)
    cate_item_map[0] = cate_item_map[0][:-1]  # exclude 0
    num_cate = len(cate_item_map)

    # tag = [[0, 2, 0, 3, 1], [2, 1, 3, 1, 0]]
    topk = 5
    ndcg = ndcg_k(actual, predicted, topk)
    print("ndcg:", ndcg)
    # alpha_ndcg = alpha_ndcg_k(actual, predicted, num_topics, doc_topics_dict, topk)
    # print("alpha_ndcg:", alpha_ndcg)
    item_emb = np.random.randn(50, 4)
    ext_alpha_ndcg = ext_alpha_ndcg_k(actual, predicted, num_cate, item_cate_dict, item_emb, topk)
    # ext_alpha_idcg = ext_alpha_idcg_k(item_emb, topk, cate_item_map)
    print("ext_alpha_ndcg:", ext_alpha_ndcg)
    # print("ext_alpha_idcg:", ext_alpha_idcg)
