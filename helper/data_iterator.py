#@title data_iterator
"""Data Iterator."""
import random
import numpy as np
import tensorflow.google.compat.v1 as tf
from scipy import sparse
class DataIterator:
  def __init__(
      self,
      source,
      item_count,
      item_cate_dict,
      batch_size=128,
      maxlen=100,
      train_flag=0,
  ):
    self.read(source)
    self.users = list(self.users)
    self.batch_size = batch_size
    self.eval_batch_size = batch_size
    self.train_flag = train_flag
    self.maxlen = maxlen
    self.index = 0
    self.item_count = item_count
    self.item_cate_dict = item_cate_dict
    all_cate_list = list(self.item_cate_dict.values())
    row_ind = [k for k, v in self.item_cate_dict.items() for _ in range(len(v))]
    col_ind = [i for l in all_cate_list for i in l] # flat all cates
    self.item_cate_matrix = sparse.csr_matrix(([1] * len(row_ind), (row_ind, col_ind)))
    self.num_cate = len(set(col_ind))
  def __iter__(self):
    return self
  def next(self):
    return self.__next__()
  def read(self, source):
    self.graph = {}
    self.users = set()
    self.items = set()
    with tf.io.gfile.GFile(source, "r") as f:
      for line in f:
        conts = line.strip().split(",")
        user_id = int(conts[0])
        item_id = int(conts[1])
        time_stp = int(conts[2])
        exp_rating = float(conts[3])
        imp_rating = float(conts[4])
        self.users.add(user_id)
        self.items.add(item_id)
        if user_id not in self.graph:
          self.graph[user_id] = []
        self.graph[user_id].append((item_id, time_stp, exp_rating, imp_rating))
    for user_id, value in self.graph.items():
      value.sort(key=lambda x: x[1])
      self.graph[user_id] = [
          [x[0] for x in value],  # item_id
          [x[2] for x in value],  # exp_rating
          [x[3] for x in value],  # imp_rating
      ]
    self.users = list(self.users)
    self.items = list(self.items)
  def get_matrix(self):
    row, col, rate = [], [], []
    graph = self.graph
    self.min_index = np.min(list(self.graph.keys()))
    for i in range(len(graph)):
      item_id = graph[i + self.min_index][0]
      exp_rating = graph[i + self.min_index][1]
      num_inter = len(item_id)
      row.extend([i] * num_inter)
      col.extend(item_id)
      rate.extend(exp_rating)
    return row, col, rate
  def __next__(self):
    if self.train_flag == 1:
      user_id_list = random.sample(self.users, self.batch_size)
    else:
      total_user = len(self.users)
      if self.index >= total_user:
        self.index = 0
        raise StopIteration
      user_id_list = self.users[self.index : self.index + self.eval_batch_size]
      self.index += self.eval_batch_size
    item_id_list = []
    rate_score_list = []
    cate_list = []
    cate_rating = []
    hist_item_list = []
    hist_mask_list = []
    hist_rate_list = []
    for user_id in user_id_list:
      item_list = self.graph[user_id][0]
      exp_rating_list = self.graph[user_id][1]
      imp_rating_list = self.graph[user_id][2]
      if self.train_flag == 1:
        k = random.choice(range(0, len(item_list)))
        item_id_list.append(item_list[k])
        rate_score_list.append(exp_rating_list[k])
        cate = random.choice(range(0, self.num_cate))
        cate_list.append(cate)
        cate_rating.append(self.item_cate_matrix[item_list[k], cate])
      else:
        k = int(len(item_list) * 0.8)
        item_id_list.append(item_list[k:])
        rate_score_list.append(exp_rating_list[k:])
      if k >= self.maxlen:
        hist_item_list.append(item_list[k - self.maxlen : k])
        # hist_mask_list.append(imp_rating_list[k - self.maxlen : k])
        hist_mask_list.append([1] * self.maxlen)
        hist_rate_list.append(exp_rating_list[k - self.maxlen : k])
      else:
        hist_item_list.append(
            [item_list[0]] * (self.maxlen - k) + item_list[:k]
        )
        # hist_mask_list.append(
        #     [imp_rating_list[0]] * (self.maxlen - k) + imp_rating_list[:k]
        # )
        hist_mask_list.append([1] * self.maxlen)
        hist_rate_list.append(
            [exp_rating_list[0]] * (self.maxlen - k) + exp_rating_list[:k]
        )
    return (
        user_id_list,
        item_id_list,
        rate_score_list,
        cate_list,
        cate_rating,
    ), (
        hist_item_list,
        hist_mask_list,
        hist_rate_list,
    )
