
"""The Main file."""
import ast
import pickle
import random
import time
import warnings
from absl import app
from absl import flags
import numpy as np
from scipy import sparse
import tensorflow as tf
import tensorflow.google.compat.v1 as tf
from google3.experimental.users.haolunwu.GPR_MUR.func.test_gpr import run_test_gpr
from google3.experimental.users.haolunwu.GPR_MUR.func.test_sur import run_test_sur
from google3.experimental.users.haolunwu.GPR_MUR.func.train_sur import run_train_sur
from google3.experimental.users.haolunwu.GPR_MUR.func.tune_gpr import run_tune_gpr
from google3.experimental.users.haolunwu.GPR_MUR.helper.data_iterator import DataIterator
from google3.experimental.users.haolunwu.GPR_MUR.helper.preparation import prepare_data
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')
best_metric = 0.0
FLAGS = flags.FLAGS
flags.DEFINE_string('platform', 'GPU', 'TPU, GPU, or CPU.')
flags.DEFINE_string(
    'p', 'tune-gpr', 'train-sur | tune-gpr | test-sur | test-gpr.'
)
flags.DEFINE_string('dataset', 'ml1m', 'ml1m, cd, beauty, book')
flags.DEFINE_string('model_type', 'DNN', 'MF | DNN | ComiRec | ..')
flags.DEFINE_integer('random_seed', 19, 'random_seed.')
flags.DEFINE_integer('embedding_dim', 16, 'embedding_dim.')
flags.DEFINE_integer('hidden_size', 16, 'hidden_size for DNN.')
flags.DEFINE_integer('num_interest', 4, 'num_interest.')
flags.DEFINE_float('lr', 0.001, 'lr.')
flags.DEFINE_float('wd', 1e-5, 'wd.')
flags.DEFINE_integer('batch_size', 1024, 'batch_size.')
flags.DEFINE_integer('max_iter', 1000, 'max_iter.')
flags.DEFINE_integer('test_iter', 50, 'test_iter.')
flags.DEFINE_integer('patience', 10, 'patience.')
flags.DEFINE_integer('maxlen', 500, 'maxlen of user history.')
flags.DEFINE_integer('topN', 50, 'topN.')
# parameters for gpr
flags.DEFINE_string('density_type', 'expon', 'kernel for KDE.')
flags.DEFINE_string('kernel_type', 'expon', 'kernel for GPR.')
flags.DEFINE_string('gpr_method', 'once', 'once | mean | thompson')
flags.DEFINE_float('amp', 1, 'amplitude: controls the max value of kernel.')
flags.DEFINE_float('scale', 0.1, 'length_scale:  how sharp or wide the kernel.')
flags.DEFINE_float('gamma', 0.1, 'mean + gamma * std')
flags.DEFINE_float('o_noise', 0.1, 'observation noise')
flags.DEFINE_float('p_noise', 0.1, 'prediction noise')
flags.DEFINE_string('time', 'Y', 'time decay for GPR.')
flags.DEFINE_float('decay', 0.02, 'time decay factor.')
flags.DEFINE_integer('user_train_count', None, 'user_train_count.')
flags.DEFINE_integer('item_count', None, 'item_count.')
flags.DEFINE_integer('item_cate_dict', None, 'item_cate_dict.')
flags.DEFINE_integer('num_cate', None, 'num_cate.')
flags.DEFINE_integer('item_pairwise_sim', None, 'item_pairwise_sim.')
flags.DEFINE_integer('train_data', None, 'train_data.')
flags.DEFINE_integer('train_rating', None, 'train_rating.')
flags.DEFINE_integer('valid_data', None, 'valid_data.')
flags.DEFINE_integer('valid_rating', None, 'valid_rating.')
flags.DEFINE_integer('valid_hist', None, 'valid_hist.')
flags.DEFINE_integer('valid_hist_rate', None, 'valid_hist_rate.')
flags.DEFINE_integer('valid_holdout', None, 'valid_holdout.')
flags.DEFINE_integer('valid_holdout_rate', None, 'valid_holdout_rate.')
flags.DEFINE_integer('test_data', None, 'test_data.')
flags.DEFINE_integer('test_rating', None, 'test_rating.')
flags.DEFINE_integer('test_hist', None, 'test_hist.')
flags.DEFINE_integer('test_hist_rate', None, 'test_hist_rate.')
flags.DEFINE_integer('test_holdout', None, 'test_holdout.')
flags.DEFINE_integer('test_holdout_rate', None, 'test_holdout_rate.')
flags.DEFINE_integer('gpr_data', None, 'gpr_data.')
flags.DEFINE_integer('gpr_rating', None, 'gpr_rating.')
flags.DEFINE_integer('gpr_hist', None, 'gpr_hist.')
flags.DEFINE_integer('gpr_hist_rate', None, 'gpr_hist_rate.')
flags.DEFINE_integer('gpr_holdout', None, 'gpr_holdout.')
flags.DEFINE_integer('gpr_holdout_rate', None, 'gpr_holdout_rate.')
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
def main(argv):
  # args = parse_args()
  args = FLAGS
  random_seed = args.random_seed
  args.hidden_size = args.embedding_dim
  tf.set_random_seed(random_seed)
  np.random.seed(random_seed)
  random.seed(random_seed)
  if args.dataset == 'cd':
    path = '/cns/jq-d/home/haolunwu/data/cd_data/'
    args.user_train_count = 4978
    args.item_count = 32830
    args.maxlen = 60
  elif args.dataset == 'kindle':
    path = '/cns/jq-d/home/haolunwu/data/kindle_data/'
    args.user_train_count = 10088
    args.item_count = 45689
    args.maxlen = 60
  elif args.dataset == 'ml1m':
    path = '/cns/jq-d/home/haolunwu/data/ml1m_data/'
    args.user_train_count = 4488
    args.item_count = 2934
    args.maxlen = 175
  elif args.dataset == 'ml20m':
    path = '/cns/jq-d/home/haolunwu/data/ml20m_data/'
    args.user_train_count = 98401
    args.item_count = 12532
    args.maxlen = 160
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
  app.run(main)
