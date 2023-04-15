#@title eval_new
from collections import Counter
import math
import numpy as np
from scipy.spatial.distance import cdist
def compute_rmse(rating_matrix, holdout, holdout_rate):
  num_users = rating_matrix.shape[0]
  sum_rmse = 0
  for i in range(num_users):
    holdout_id = holdout[i]
    err = rating_matrix[i][holdout_id] - holdout_rate[i]
    sum_rmse += np.sqrt(np.mean(err**2))
  return sum_rmse / num_users
#############
###### NDCG
#############
def compute_ndcg(rating_matrix, pred_list, holdout, holdout_rate, k):
  num_users = rating_matrix.shape[0]
  sum_ndcg = 0
  ind_ndcg_list = []
  for i in range(num_users):
    k_cur = min(k, len(holdout[i]))
    dcg = sum(
        [
            int(pred_list[i][j] in set(holdout[i])) / math.log(j + 2, 2)
            for j in range(k)
        ]
    )
    idcg = sum([1.0 / math.log(i + 2, 2) for i in range(k_cur)])
    ind_ndcg = dcg / idcg
    sum_ndcg += ind_ndcg
    ind_ndcg_list.append(ind_ndcg)
  return np.mean(ind_ndcg_list), np.std(ind_ndcg_list) / math.sqrt(num_users)
def compute_ndcg_list(rating_matrix, pred_list, holdout, holdout_rate):
  ndcg_list, std_list = [], []
  k_list = [10, 20, 50, 100]
  for k in k_list:
    ndcg, std = compute_ndcg(
        rating_matrix,
        pred_list,
        holdout,
        holdout_rate,
        k,
    )
    ndcg_list.append(round(ndcg, 4))
    std_list.append(round(std, 4))
  return ndcg_list, std_list
#############
###### Recall
#############
def compute_recall(rating_matrix, pred_list, holdout, holdout_rate, k):
  num_users = rating_matrix.shape[0]
  sum_recall = 0
  ind_recall_list = []
  for i in range(num_users):
    act_set = set(holdout[i])
    pred_set = set(pred_list[i][:k])
    ind_recall = len(act_set & pred_set) / float(len(act_set))
    sum_recall += ind_recall
    ind_recall_list.append(ind_recall)
  return np.mean(ind_recall_list), np.std(ind_recall_list) / math.sqrt(
      num_users
  )
def compute_recall_list(rating_matrix, pred_list, holdout, holdout_rate):
  recall_list, std_list = [], []
  k_list = [10, 20, 50, 100]
  for k in k_list:
    recall, std = compute_recall(
        rating_matrix,
        pred_list,
        holdout,
        holdout_rate,
        k,
    )
    recall_list.append(round(recall, 4))
    std_list.append(round(std, 4))
  return recall_list, std_list
#############
###### Soft - Recall
#############
def compute_soft_recall(
    rating_matrix, pred_list, holdout, holdout_rate, item_pairwise_sim, k
):
  num_users = rating_matrix.shape[0]
  sum_recall = 0
  ind_recall_list = []
  for i in range(num_users):
    act_index = np.array(holdout[i])
    pred_index = np.array(pred_list[i][:k])
    relevance = item_pairwise_sim[act_index, :][:, pred_index]
    soft_recall = relevance.max(1).sum()
    ind_recall = soft_recall / float(len(act_index))
    sum_recall += ind_recall
    ind_recall_list.append(ind_recall)
  return np.mean(ind_recall_list), np.std(ind_recall_list) / math.sqrt(
      num_users
  )
def compute_soft_recall_list(
    rating_matrix, pred_list, holdout, holdout_rate, item_pairwise_sim
):
  recall_list, std_list = [], []
  k_list = [10, 20, 50, 100]
  for k in k_list:
    recall, std = compute_soft_recall(
        rating_matrix,
        pred_list,
        holdout,
        holdout_rate,
        item_pairwise_sim,
        k,
    )
    recall_list.append(round(recall, 4))
    std_list.append(round(std, 4))
  return recall_list, std_list
#############
###### Cut - Recall
#############
def compute_cut_recall(
    rating_matrix, pred_list, holdout, holdout_rate, item_pairwise_sim, k
):
  num_users = rating_matrix.shape[0]
  sum_recall = 0
  ind_recall_list = []
  for i in range(num_users):
    act_index = np.array(holdout[i])
    pred_index = np.array(pred_list[i][:k])
    relevance = item_pairwise_sim[act_index, :][:, pred_index]
    cut_recall = (relevance.max(1) > 0.5).sum()
    ind_recall = cut_recall / float(len(act_index))
    sum_recall += ind_recall
    ind_recall_list.append(ind_recall)
  return np.mean(ind_recall_list), np.std(ind_recall_list) / math.sqrt(
      num_users
  )
def compute_cut_recall_list(
    rating_matrix, pred_list, holdout, holdout_rate, item_pairwise_sim
):
  recall_list, std_list = [], []
  k_list = [10, 20, 50, 100]
  for k in k_list:
    recall, std = compute_cut_recall(
        rating_matrix,
        pred_list,
        holdout,
        holdout_rate,
        item_pairwise_sim,
        k,
    )
    recall_list.append(round(recall, 4))
    std_list.append(round(std, 4))
  return recall_list, std_list
#############
###### Interest Coverage
#############
def int_cov_k(
    rating_matrix,
    actual_list,
    actual_rate_list,
    pred_list,
    item_cate_dict,
    topk,
    eval='full',
):
  sum_cate_recall, sum_cate_precision = 0, 0
  ic_recall_list, ic_precision_list = [], []
  num_users = len(actual_list)
  for i in range(len(actual_list)):
    pos_actual_items = np.array(actual_list[i])[
        np.array(actual_rate_list[i]) > 0
    ].astype('int64')
    actual_cate_list = [item_cate_dict[ind] for ind in pos_actual_items]
    actual_cate = [i for l in actual_cate_list for i in l]
    flag = 1
    if len(set(actual_cate)) > 0:
      if eval == 'full':
        rank_list = pred_list[i]
      elif eval == 'cond':
        if len(pos_actual_items) >= topk:
          rating = rating_matrix[i][actual_list[i]]
          _, rank_list = zip(*sorted(zip(rating, actual_list[i]), reverse=True))
        else:
          num_users -= 1
          flag = 0
      if flag:
        pred_cate_list = [item_cate_dict[ind] for ind in rank_list[:topk]]
        pred_cate = [i for l in pred_cate_list for i in l]
        ic_recall_list.append(
            len(set(actual_cate) & set(pred_cate))
            / float(len(set(actual_cate)))
        )
        ic_precision_list.append(
            len(set(actual_cate) & set(pred_cate)) / float(len(set(pred_cate)))
        )
        # if i< 10:
        #   print("actual:{}, pred:{}, inter:{}".format(len(set(actual_cate)), len(set(pred_cate)), len(set(actual_cate) & set(pred_cate))))
    else:
      num_users -= 1
  return (
      np.mean(ic_recall_list),
      np.mean(ic_precision_list),
      np.std(ic_recall_list) / math.sqrt(num_users),
      np.std(ic_precision_list) / math.sqrt(num_users),
  )
def compute_int_cov(
    rating_matrix,
    actual_list,
    actual_rate_list,
    pred_list,
    item_cate_dict,
    eval='cond',
):
  icr_list, icp_list = [], []
  icr_std_list, icp_std_list = [], []
  if eval == 'cond':
    k_list = [3, 5, 10, 20]
  elif eval == 'full':
    k_list = [10, 20, 50, 100]
  for k in k_list:
    icr, icp, icr_std, icp_std = int_cov_k(
        rating_matrix,
        actual_list,
        actual_rate_list,
        pred_list,
        item_cate_dict,
        k,
        eval,
    )
    icr_list.append(round(icr, 4))
    icp_list.append(round(icp, 4))
    icr_std_list.append(round(icr_std, 4))
    icp_std_list.append(round(icp_std, 4))
  return icr_list, icp_list, icr_std_list, icp_std_list
#############
###### Interest Relevance
#############
def int_rel_k(
    rating_matrix,
    actual_list,
    actual_rate_list,
    pred_list,
    item_cate_dict,
    item_pairwise_sim,
    item_emb,
    topk,
    eval,
    cutoff,
):
  num_users = len(actual_list)
  ir_recall_list, ir_precision_list = [], []
  for i in range(len(actual_list)):
    pos_actual_items = np.array(actual_list[i])[
        np.array(actual_rate_list[i]) > 0
    ].astype('int64')
    cate_recall, cate_precision = 0, 0
    actual_cate_list = [item_cate_dict[ind] for ind in pos_actual_items]
    actual_cate = [i for l in actual_cate_list for i in l]
    flag = 1
    if len(set(actual_cate)) > 0:
      if eval == 'full':
        rank_list = pred_list[i]
      elif eval == 'cond':
        if len(pos_actual_items) >= topk:
          rating = rating_matrix[i][actual_list[i]]
          _, rank_list = zip(*sorted(zip(rating, actual_list[i]), reverse=True))
        else:
          num_users -= 1
          flag = 0
      if flag:
        pred_cate_list = [item_cate_dict[ind] for ind in rank_list[:topk]]
        pred_cate = [i for l in pred_cate_list for i in l]
        relevance_sum = 0
        for cate in set(actual_cate) & set(pred_cate):
          actual_position, pred_position = [], []
          for i in range(len(actual_cate_list)):
            if cate in actual_cate_list[i]:
              actual_position.append(i)
          actual_index = pos_actual_items[actual_position]
          # actual_position = np.where(np.array(actual_cate) == cate)[0]
          for i in range(len(pred_cate_list)):
            if cate in pred_cate_list[i]:
              pred_position.append(i)
          pred_index = np.array(rank_list)[pred_position]
          # pred_position = np.where(np.array(pred_cate) == cate)[0]
          curr_actual_emb = item_emb[actual_index, :]
          curr_predicted_emb = item_emb[pred_index, :]
          relevance = 1 - cdist(curr_predicted_emb, curr_actual_emb, 'cosine')
          # relevance = relevance.max(1)
          # relevance = item_pairwise_sim[actual_index, :][:, pred_index]
          if cutoff == 1:
            if relevance.max() >= 0.8:
              relevance_sum += 1
          else:
            relevance_sum += relevance.max()
        ir_recall_list.append(relevance_sum / float(len(set(actual_cate))))
        ir_precision_list.append(relevance_sum / float(len(set(pred_cate))))
    else:
      num_users -= 1
  return (
      np.mean(ir_recall_list),
      np.mean(ir_precision_list),
      np.std(ir_recall_list) / math.sqrt(num_users),
      np.std(ir_precision_list) / math.sqrt(num_users),
  )
def compute_int_rel(
    rating_matrix,
    actual_list,
    actual_rate_list,
    pred_list,
    item_cate_dict,
    item_pairwise_sim,
    item_emb,
    eval,
    cutoff,
):
  irr_list, irp_list = [], []
  irr_std_list, irp_std_list = [], []
  if eval == 'cond':
    k_list = [3, 5, 10, 20]
  elif eval == 'full':
    k_list = [10, 20, 50, 100]
  for k in k_list:
    irr, irp, irr_std, irp_std = int_rel_k(
        rating_matrix,
        actual_list,
        actual_rate_list,
        pred_list,
        item_cate_dict,
        item_pairwise_sim,
        item_emb,
        k,
        eval,
        cutoff,
    )
    irr_list.append(round(irr, 4))
    irp_list.append(round(irp, 4))
    irr_std_list.append(round(irr_std, 4))
    irp_std_list.append(round(irp_std, 4))
  return irr_list, irp_list, irr_std_list, irp_std_list
#############
###### Exposure Deviation
#############
def compute_exposure(actual_list, pred_list, item_cate_dict, num_cate, k):
  all_cate_list = [
      i for l in list(item_cate_dict.values()) for i in l
  ]  # flat all cates
  count = Counter(all_cate_list)
  count = {
      k: v
      for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)
  }
  cate_sort = list(count.keys())  # pop to not-pop
  num_users = len(actual_list)
  ind_exposure_dev, tail_exposure_dev = [], []
  for i in range(len(actual_list)):
    actual_exposure_dict = dict.fromkeys(np.arange(num_cate), 0)
    pred_exposure_dict = dict.fromkeys(np.arange(num_cate), 0)
    """All categories"""
    actual_cate_list = [item_cate_dict[ind] for ind in actual_list[i]]
    actual_cate = [i for l in actual_cate_list for i in l]  # flatten
    actual_cate_count = Counter(actual_cate)
    for cate in actual_cate_count:
      actual_exposure_dict[cate] = actual_cate_count[cate]
    pred_cate_list = [item_cate_dict[ind] for ind in pred_list[i][:k]]
    pred_cate = [i for l in pred_cate_list for i in l]  # flatten
    pred_cate_count = Counter(pred_cate)
    for cate in pred_cate_count:
      pred_exposure_dict[cate] = pred_cate_count[cate]
    # """Only those rated category"""
    # actual_cate_list = [item_cate_dict[ind] for ind in actual_list[i]]
    # actual_cate = [i for l in actual_cate_list for i in l] # flatten
    # actual_cate_count = Counter(actual_cate)
    # actual_exposure_dict = actual_cate_count
    # pred_cate_list = [item_cate_dict[ind] for ind in pred_list[i][:k]]
    # pred_cate = [i for l in pred_cate_list for i in l] # flatten
    # pred_cate_count = Counter(pred_cate)
    # for cate in pred_cate_count:
    #   pred_exposure_dict[cate] = pred_cate_count[cate]
    # pred_exposure_dict = {k: pred_exposure_dict[k] for k in actual_exposure_dict}
    actual_exposure = np.array(list(actual_exposure_dict.values()))
    pred_exposure = np.array(list(pred_exposure_dict.values()))
    actual_exposure = actual_exposure / actual_exposure.sum()
    pred_exposure = pred_exposure / pred_exposure.sum()
    ind_exposure_dev.append(
        np.sqrt(np.sum(np.power(actual_exposure - pred_exposure, 2)))
    )
    if num_cate < 25:
      tail_length = int(num_cate * 0.8)
    else:
      tail_length = int(num_cate * 0.4)
    actual_exposure = actual_exposure[cate_sort][-tail_length:]
    pred_exposure = pred_exposure[cate_sort][-tail_length:]
    pred_exposure[np.where(actual_exposure == 0)[0]] = 0
    tail_exposure_dev.append(np.sum(pred_exposure - actual_exposure))
  return (
      np.mean(ind_exposure_dev),
      np.mean(tail_exposure_dev),
      np.std(ind_exposure_dev) / math.sqrt(num_users),
      np.std(tail_exposure_dev) / math.sqrt(num_users),
  )
def compute_exposure_list(actual_list, pred_list, item_cate_dict, num_cate):
  ind_exposure_list, tail_exposure_list, ind_std_list, tail_std_list = (
      [],
      [],
      [],
      [],
  )
  k_list = [10, 20, 50, 100]
  for k in k_list:
    ind_exposure, tail_exposure, ind_std, tail_std = compute_exposure(
        actual_list, pred_list, item_cate_dict, num_cate, k
    )
    ind_exposure_list.append(round(ind_exposure, 4))
    tail_exposure_list.append(round(tail_exposure, 4))
    ind_std_list.append(round(ind_std, 4))
    tail_std_list.append(round(tail_std, 4))
  return ind_exposure_list, tail_exposure_list, ind_std_list, tail_std_list
#############
###### Compute all metrics
#############
def compute_all_metrics(
    full_rec_list,
    rating_pred_full,
    item_cate_dict,
    item_pairwise_sim,
    item_emb,
    num_cate,
    args,
    mode='tune',
):
  if mode == 'tune':
    holdout = args.gpr_holdout
    holdout_rate = args.gpr_holdout_rate
  elif mode == 'test':
    holdout = args.test_holdout
    holdout_rate = args.test_holdout_rate
  icr_full, _, _, _ = compute_int_cov(
      rating_pred_full,
      holdout,
      holdout_rate,
      full_rec_list,
      item_cate_dict,
      eval='full',
  )
  (irr_full, _, _, _) = compute_int_rel(
      rating_pred_full,
      holdout,
      holdout_rate,
      full_rec_list,
      item_cate_dict,
      item_pairwise_sim,
      item_emb,
      eval='full',
      cutoff=0,
  )
  ind_exposure, tail_exposure, _, _ = compute_exposure_list(
      holdout, full_rec_list, item_cate_dict, num_cate
  )
  return icr_full, irr_full, ind_exposure, tail_exposure
