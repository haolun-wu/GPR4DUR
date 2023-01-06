import warnings

warnings.filterwarnings('ignore')
import random
# import jax
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
# import tensorflow.google.compat.v1 as tf
# from tensorflow.compat.v1.nn.rnn_cell import GRUCell
# import tf_slim as slim

from scipy.sparse import csr_matrix

# from sklearn.metrics.pairwise import eucliFeb 17dean_distances
import argparse
from collections import defaultdict
from helper.data_iterator import DataIterator
from helper.preparation import *
from func_train_test.func_train import train
from func_train_test.func_test import test

# from colabtools import adhoc_import


# with adhoc_import.Google3():
#     from tensorboardX import SummaryWriter

# from google3.third_party.tensorflow.python.keras.google_utils import gfile_utils

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='test-gpr', help='train | test-sur | test-gpr')
parser.add_argument('--dataset', type=str, default='ml1m_exp', help='book | taobao | beauty')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--embedding_dim', type=int, default=16)
parser.add_argument('--hidden_size', type=int, default=16)
parser.add_argument('--num_interest', type=int, default=4)
parser.add_argument('--model_type', type=str, default='SVD', help='DNN | GRU4REC | ..')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--max_iter', type=int, default=20, help='')
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--coef', default=None)
parser.add_argument('--topN', type=int, default=50)
parser.add_argument('--perc', type=float, default=0.2)


def invert_dict(d):
    d_inv = defaultdict(list)
    for k, v in d.items():
        d_inv[v].append(k)
    return d_inv


if __name__ == '__main__':
    args = parser.parse_args(args=[])
    SEED = args.random_seed

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    train_name = 'train'
    valid_name = 'valid'
    test_name = 'test'

    if args.dataset == 'cd':
        # path = '/cns/yo-d/home/haolunwu/data/cd_data/'
        path = 'data/cd_data/'
        args.user_train_count = 30067
        args.item_count = 78357
        args.batch_size = 1024
        args.maxlen = 20
        args.test_iter = 10
    elif args.dataset == 'ml1m':
        # path = '/cns/yo-d/home/haolunwu/data/ml1m_data/'
        path = 'data/ml1m_data/'
        args.user_train_count = 4827
        args.item_count = 3126
        args.batch_size = 1024
        args.maxlen = 20
        args.test_iter = 10
    elif args.dataset == 'ml1m_exp':
        # path = '/cns/yo-d/home/haolunwu/data/ml1m_exp_data/'
        path = 'data/ml1m_exp_data/'
        args.user_train_count = 4832
        args.item_count = 3261
        args.batch_size = 1024
        args.maxlen = 20
        args.test_iter = 10

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    cate_file = path + args.dataset + '_item_cate.txt'
    dataset = args.dataset

    train_data = DataIterator(valid_file, args.item_count, args.batch_size, args.maxlen, train_flag=1)
    valid_data = DataIterator(valid_file, args.item_count, args.batch_size, args.maxlen, train_flag=0)
    test_data = DataIterator(test_file, args.item_count, args.batch_size, args.maxlen, train_flag=0)

    row, col, rate = train_data.get_matrix()
    train_rating = csr_matrix((rate, (row, col)), shape=(len(set(row)), args.item_count))

    row, col, rate = valid_data.get_matrix()
    valid_rating = csr_matrix((rate, (row, col)), shape=(len(set(row)), args.item_count))

    row, col, rate = test_data.get_matrix()
    test_rating = csr_matrix((rate, (row, col)), shape=(len(set(row)), args.item_count))

    valid_set_full = np.array([])
    for src, tgt in valid_data:
        valid_set = prepare_data(src, tgt)[1]
        if len(valid_set_full) == 0:
            valid_set_full = valid_set
        else:
            valid_set_full = np.concatenate((valid_set_full, valid_set), axis=0)

    test_set_full = np.array([])
    for src, tgt in test_data:
        test_set = prepare_data(src, tgt)[1]
        if len(test_set_full) == 0:
            test_set_full = test_set
        else:
            test_set_full = np.concatenate((test_set_full, test_set), axis=0)

    max_test_length = max(len(row) for row in test_set_full)
    test_set_full_padded = np.array([row + [row[-1]] * (max_test_length - len(row)) for row in test_set_full])
    test_set_len = np.array(list(map(lambda x: len(x), test_set_full)))

    item_cate_dict = {}
    with tf.io.gfile.GFile(cate_file, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id, cate = int(conts[0]), int(conts[1])
            item_cate_dict[item_id] = cate
    item_cate_dict[0] = 0

    cate_item_map = invert_dict(item_cate_dict)
    cate_item_map[0] = cate_item_map[0][:-1]
    num_cate = len(cate_item_map)

    best_metric = 0

    print("model:{}, dim:{}, perc:{}".format(args.model_type, args.embedding_dim, args.perc))

    tf.reset_default_graph()

    if args.p == 'train':
        train(train_file, valid_file, test_file, valid_set_full, test_set_full, args)
    elif args.p == 'test-sur':
        test(test_data, test_set_full, test_rating, num_cate, item_cate_dict,cate_item_map,  args, gpr=False)
    elif args.p == 'test-gpr':
        test(test_data, test_set_full, test_rating, num_cate, item_cate_dict,cate_item_map,  args, gpr=True, kernel_type='expon')
