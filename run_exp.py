"""The Main file."""
import ast
import pickle
import random
import time
import warnings

warnings.simplefilter('ignore')
# from absl import app
# from absl import flags
import numpy as np
from scipy import sparse
# import tensorflow as tf
import tensorflow.compat.v1 as tf

from func.train_sur import run_train_sur
from func.test_sur import run_test_sur
from func.tune_gpr import run_tune_gpr
from func.test_gpr import run_test_gpr
from helper.data_iterator import DataIterator
from helper.preparation import prepare_data

# tf.compat.v1.disable_eager_execution()
from argparse import ArgumentParser

best_metric = 0.0


def parse_args():
    parser = ArgumentParser(description="GPR4DUR")
    parser.add_argument('--platform', type=str, default='CPU', choices=['CPU', 'GPU', 'TPU'])
    parser.add_argument(
        '--p', type=str, default='train-sur', choices=['train-sur', 'tune-gpr', 'test-sur', 'test-gpr']
    )
    parser.add_argument('--dataset', type=str, default='ml1m', choices=['ml1m', 'cd', 'beauty', 'book', 'ml20m'])
    parser.add_argument('--model_type', type=str, default='DNN')
    parser.add_argument('--random_seed', type=int, default=19, help='random_seed.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding_dim.')
    parser.add_argument('--hidden_size', type=int, default=16, help='hidden_size for DNN.')
    parser.add_argument('--num_interest', type=int, default=4, help='num_interest.')
    parser.add_argument('--lr', type=float, default=0.001, help='lr.')
    parser.add_argument('--wd', type=float, default=1e-5, help='wd.')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch_size.')
    parser.add_argument('--max_iter', type=int, default=1000, help='max_iter.')
    parser.add_argument('--test_iter', type=int, default=50, help='test_iter.')
    parser.add_argument('--patience', type=int, default=10, help='patience.')
    parser.add_argument('--maxlen', type=int, default=500, help='maxlen of user history.')
    parser.add_argument('--topN', type=int, default=50, help='topN.')
    # parameters for gpr
    parser.add_argument('--density_type', type=str, default='expon', help='kernel for KDE.')
    parser.add_argument('--kernel_type', type=str, default='dot', help='kernel for GPR.')
    parser.add_argument('--gpr_method', type=str, default='ucb', help='once | mean | thompson | ucb')
    parser.add_argument('--amp', type=float, default=1.0, help='amplitude: controls the max value of kernel.')
    parser.add_argument('--scale', type=float, default=0.1, help='length_scale:  how sharp or wide the kernel.')
    parser.add_argument('--gamma', type=float, default=1.0, help='mean + gamma * std')
    parser.add_argument('--o_noise', type=float, default=10.0, help='observation noise')
    parser.add_argument('--p_noise', type=float, default=0.1, help='prediction noise')
    parser.add_argument('--time', type=str, default='Y', help='time decay for GPR.')
    parser.add_argument('--decay', type=float, default=0.01, help='time decay factor.')

    return parser.parse_args()


def get_data_and_rating(data_file, args, flag):
    data = DataIterator(
        data_file,
        args.item_count,
        args.item_cate_dict,
        args.batch_size,
        args.maxlen,
        train_flag=flag,
    )
    row, col, rate = data.get_matrix()
    rating = sparse.csr_matrix(
        (rate, (row, col)), shape=(len(set(row)), args.item_count)
    )
    # if flag == 1:
    #   rating[rating < 0] = 0
    return data, rating


def get_full_set(data):
    holdout_full, holdout_rate_full = [], []
    hist_full, hist_rate_full = None, None
    for src, tgt in data:
        data_cur = prepare_data(src, tgt)
        holdout, holdout_rate = data_cur[1], data_cur[2]
        hist, hist_rate = data_cur[5], data_cur[7]
        if hist_full is None:
            holdout_full.extend(holdout)
            holdout_rate_full.extend(holdout_rate)
            hist_full = hist
            hist_rate_full = hist_rate
        else:
            holdout_full.extend(holdout)
            holdout_rate_full.extend(holdout_rate)
            hist_full = np.concatenate((hist_full, hist), axis=0)
            hist_rate_full = np.concatenate((hist_rate_full, hist_rate), axis=0)
    return holdout_full, holdout_rate_full, hist_full, hist_rate_full


def main():
    args = parse_args()
    # args = FLAGS
    random_seed = args.random_seed
    args.hidden_size = args.embedding_dim
    tf.set_random_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if args.dataset == 'cd':
        path = 'data/cd_data/'
        args.user_train_count = 4978
        args.item_count = 32830
        args.maxlen = 60
    elif args.dataset == 'kindle':
        path = 'data/kindle_data/'
        args.user_train_count = 10088
        args.item_count = 45689
        args.maxlen = 60
    elif args.dataset == 'ml1m':
        path = 'data/ml1m_data/'
        args.user_train_count = 4488
        args.item_count = 2934
        args.maxlen = 175
    elif args.dataset == 'ml20m':
        path = 'data/ml20m_data/'
        args.user_train_count = 98401
        args.item_count = 12532
        args.maxlen = 160
    elif args.dataset == 'taobao':
        path = 'data/taobao_data/'
        args.user_train_count = 605513
        args.item_count = 570350
        args.maxlen = 100
    ### Load data: Category. ###
    cate_file = path + args.dataset + '_item_cate.txt'
    item_cate_dict = {}
    with tf.io.gfile.GFile(cate_file, 'r') as f:
        for line in f:
            conts = line.strip().split(',', maxsplit=1)
            item_id = int(conts[0])
            cate = ast.literal_eval(conts[1])
            item_cate_dict[item_id] = cate
    args.item_cate_dict = item_cate_dict
    all_cate_list = list(item_cate_dict.values())
    # row_ind = [k for k, v in self.item_cate_dict.items() for _ in range(len(v))]
    col_ind = [i for l in all_cate_list for i in l]  # flatten all cates
    num_cate = len(set(col_ind))
    args.num_cate = num_cate
    cate_item_map = None
    ### Load data: Train-Valid-Test. ###
    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    args.train_data, args.train_rating = get_data_and_rating(train_file, args, 1)
    args.valid_data, args.valid_rating = get_data_and_rating(valid_file, args, 0)
    args.test_data, args.test_rating = get_data_and_rating(test_file, args, 0)
    (
        args.valid_holdout,
        args.valid_holdout_rate,
        args.valid_hist,
        args.valid_hist_rate,
    ) = get_full_set(args.valid_data)
    (
        args.test_holdout,
        args.test_holdout_rate,
        args.test_hist,
        args.test_hist_rate,
    ) = get_full_set(args.test_data)
    ### Prepare train data for GPR. ###
    train_gpr_file = path + args.dataset + '_train_valid.txt'
    args.gpr_data, args.gpr_rating = get_data_and_rating(train_gpr_file, args, 0)
    (
        args.gpr_holdout,
        args.gpr_holdout_rate,
        args.gpr_hist,
        args.gpr_hist_rate,
    ) = get_full_set(args.gpr_data)
    print('user in train:', args.train_rating.shape[0])
    print('user in valid:', len(args.valid_holdout))
    print('user in test:', len(args.test_holdout))
    print('user in train gpr:', len(args.gpr_holdout))
    # ### Load Item-Item similarity. ###
    # if args.p == 'train-sur':
    #   pass
    # else:
    #   start = time.time()
    #   if args.dataset in ['ml1m']:
    #     cell = 'oi'
    #   if args.dataset in ['cd']:
    #     cell = 'is'
    #   elif args.dataset in ['kindle', 'ml20m']:
    #     cell = 'el'
    #   with tf.io.gfile.GFile(
    #       '/cns/{}-d/home/haolunwu/data/{}_data/'.format(cell, args.dataset)
    #       + args.dataset
    #       + '_item_sim.pkl',
    #       mode='rb',
    #   ) as f:
    #     item_pairwise_sim = pickle.load(f)
    #   args.item_pairwise_sim = item_pairwise_sim
    #   print('load sim time:', time.time() - start)
    print(
        'p:{}, model:{}, dim:{}'.format(
            args.p,
            args.model_type,
            args.embedding_dim,
        )
    )
    tf.reset_default_graph()
    if args.p == 'train-sur':
        run_train_sur(num_cate, item_cate_dict, cate_item_map, args)
    elif args.p == 'test-sur':
        run_test_sur(num_cate, item_cate_dict, cate_item_map, args)
    elif args.p == 'test-gpr':
        run_test_gpr(num_cate, item_cate_dict, cate_item_map, args)
    elif args.p == 'tune-gpr':
        run_tune_gpr(num_cate, item_cate_dict, cate_item_map, args)


if __name__ == '__main__':
    main()
