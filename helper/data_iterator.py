import numpy
import json
import random
import numpy as np


class DataIterator:

    def __init__(self, source,
                 item_count,
                 batch_size=128,
                 maxlen=100,
                 train_flag=1
                 ):
        self.read(source)
        self.users = list(self.users)

        self.batch_size = batch_size
        self.eval_batch_size = batch_size
        self.train_flag = train_flag
        self.maxlen = maxlen
        self.index = 0
        self.item_count = item_count

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def read(self, source):
        self.graph = {}
        self.users = set()
        self.items = set()
        with open(source, 'r') as f:
        # with tf.io.gfile.GFile(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                time_stamp = int(conts[2])
                rating = int(conts[3])
                self.users.add(user_id)
                self.items.add(item_id)
                if user_id not in self.graph:
                    self.graph[user_id] = []
                self.graph[user_id].append((item_id, time_stamp, rating))
        for user_id, value in self.graph.items():
            value.sort(key=lambda x: x[1])
            self.graph[user_id] = [[x[0] for x in value], [x[2] for x in value]]
        self.users = list(self.users)
        self.items = list(self.items)

    def get_matrix(self):
        row, col, rate = [], [], []
        graph = self.graph
        self.min_index = np.min(list(self.graph.keys()))
        for i in range(len(graph)):
            data = graph[i + self.min_index][0]
            data_rating = graph[i + self.min_index][1]
            num_inter = len(data)
            row.extend([i] * num_inter)
            col.extend(data)
            rate.extend(data_rating)
        # rate = [1] * len(row)
        return row, col, rate

    def __next__(self):
        if self.train_flag == 1:
            user_id_list = random.sample(self.users, self.batch_size)
        else:
            total_user = len(self.users)
            if self.index >= total_user:
                self.index = 0
                raise StopIteration
            user_id_list = self.users[self.index: self.index + self.eval_batch_size]
            self.index += self.eval_batch_size

        pos_id_list = []
        neg_id_list = []
        rate_score_list = []
        hist_item_list = []
        hist_mask_list = []
        neg_item_list = []
        hist_rate_list = []
        for user_id in user_id_list:
            item_list = self.graph[user_id][0]
            rating_list = list(np.array(self.graph[user_id][1]) - 3)
            non_item_list = list(set(np.arange(self.item_count)) - set(item_list))

            if self.train_flag == 1:
                k = random.choice(range(4, len(item_list)))
                pos_id_list.append(item_list[k])
                rate_score_list.append(rating_list[k])
                pick = random.choice(range(1, len(non_item_list)))
                neg_id_list.append(non_item_list[pick])
            else:
                k = int(len(item_list) * 0.8)
                pos_id_list.append(item_list[k:])
                rate_score_list.append(rating_list[k:])

            if k >= self.maxlen:
                hist_item_list.append(item_list[k - self.maxlen: k])
                hist_mask_list.append([1.0] * self.maxlen)
                hist_rate_list.append(rating_list[k - self.maxlen: k])
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.maxlen - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.maxlen - k))
                hist_rate_list.append(rating_list[:k] + [0] * (self.maxlen - k))

            neg_item_list.append(random.sample(non_item_list, self.maxlen))

        # print("user_id_list:", len(user_id_list), user_id_list[0])
        # print("item_id_list:", len(item_id_list), item_id_list[0])
        # print("hist_item_list:", len(hist_item_list), hist_item_list[0])
        # print("hist_mask_list:", len(hist_mask_list), hist_mask_list[0])

        return (user_id_list, pos_id_list, neg_id_list, rate_score_list), (
        hist_item_list, hist_mask_list, neg_item_list, hist_rate_list)
