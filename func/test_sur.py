#@title test_sur

import warnings
import tensorflow as tf
import tensorflow.google.compat.v1 as tf
from google3.experimental.users.haolunwu.GPR_MUR.helper.eval_new import compute_all_metrics
from google3.experimental.users.haolunwu.GPR_MUR.helper.prediction import generate_full_sur
from google3.experimental.users.haolunwu.GPR_MUR.helper.preparation import get_exp_name
from google3.experimental.users.haolunwu.GPR_MUR.model.model_comirec import Model_ComiRec_DR
from google3.experimental.users.haolunwu.GPR_MUR.model.model_comirec_sa import Model_ComiRec_SA
from google3.experimental.users.haolunwu.GPR_MUR.model.model_dnn import Model_DNN
from google3.experimental.users.haolunwu.GPR_MUR.model.model_gru4rec import Model_GRU4REC
# from google3.experimental.users.haolunwu.GPR_MUR.model.model_mf import Model_MF
from google3.experimental.users.haolunwu.GPR_MUR.model.model_mind import Model_MIND
from google3.experimental.users.haolunwu.GPR_MUR.model.model_ydnn import Model_YDNN
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')
def run_test_sur(num_cate, item_cate_dict, cate_item_map, args):
  # load from args
  item_pairwise_sim = args.item_pairwise_sim
  dataset, model_type, batch_size, lr, maxlen = (
      args.dataset,
      args.model_type,
      args.batch_size,
      args.lr,
      args.maxlen,
  )
  test_data = args.test_data
  # user_train_count, item_count = args.user_train_count, args.item_count
  # emb_dim, hid_dim = args.embedding_dim, args.hidden_size
  exp_name = get_exp_name(args)
  best_model_path = (
      '/cns/jq-d/home/haolunwu/GPR_MUR/best_model/' + exp_name + '/'
  )
  print('best_model_path:', best_model_path)
  gpu_options = tf.GPUOptions(allow_growth=True)
  # if model_type == 'MF':
  #   model = Model_MF(
  #       user_train_count, item_count, emb_dim, hid_dim, batch_size, maxlen
  #   )
  if model_type == 'DNN':
    model = Model_DNN(args)
  if model_type == 'YDNN':
    model = Model_YDNN(args)
  elif model_type == 'GRU4REC':
    model = Model_GRU4REC(args)
  elif model_type == 'MIND':
    model = Model_MIND(args)
  elif model_type == 'ComiRec_SA':
    model = Model_ComiRec_SA(args)
  elif model_type == 'ComiRec_DR':
    model = Model_ComiRec_DR(args)
  if args.model_type in ['pop', 'random']:
    model = Model_GRU4REC(args)
    cc = args.model_type
    args.model_type = 'GRU4REC'
    exp_name = get_exp_name(args)
    best_model_path = (
        '/cns/jq-d/home/haolunwu/GPR_MUR/best_model/' + exp_name + '/'
    )
    args.model_type = cc
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model.restore(sess, best_model_path)
    rec_list_full_test, rating_pred_full_test, item_embs = generate_full_sur(
        sess,
        test_data,
        model,
        args,
    )
    (
        cover_test,
        relevance_test,
        exposure_test,
        tail_exposure_test,
    ) = compute_all_metrics(
        rec_list_full_test,
        rating_pred_full_test,
        item_cate_dict,
        item_pairwise_sim,
        item_embs,
        num_cate,
        args,
        mode='test',
    )
    print('cover_test:', ', '.join(str(e) for e in cover_test))
    print('relevance_test:', ', '.join(str(e) for e in relevance_test))
    print('exposure_test:', ', '.join(str(e) for e in exposure_test))
    print('tail_exposure_test:', ', '.join(str(e) for e in tail_exposure_test))
