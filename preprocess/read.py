import os
import sys
import json
import random
from collections import defaultdict

source = '../data/book_data/book_train.txt'


def statistics_data(source):
    with open(source, 'r') as f:
        u_list, i_list = [], []
        for line in f:
            r = line.strip().split(',')
            uid = r[0]
            iid = r[1]
            u_list.append(uid)
            i_list.append(iid)
    # print("u_list:", len(u_list), len(set(u_list)))
    # print("i_list:", len(i_list))
    return u_list, i_list


u_list_train, i_list_train = statistics_data('../data/book_data/book_train.txt')
u_list_val, i_list_val = statistics_data('../data/book_data/book_valid.txt')
u_list_test, i_list_test = statistics_data('../data/book_data/book_test.txt')

num_interaction = len(u_list_train) + len(u_list_val) + len(u_list_test)
num_user = len(set(u_list_train + u_list_val + u_list_test))
num_item = len(set(i_list_train + i_list_val + i_list_test))


print("interaction:", num_interaction)
print("num_user:", num_user)
print("num_item:", num_item)
print("density:", num_interaction / num_user / num_item)
print("avg degree:", num_interaction / num_user)