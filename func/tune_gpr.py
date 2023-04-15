#@title tune_gpr

import pickle
import warnings
import tensorflow as tf
import tensorflow.google.compat.v1 as tf
from google3.experimental.users.haolunwu.GPR_MUR.helper.eval_new import compute_all_metrics
from google3.experimental.users.haolunwu.GPR_MUR.helper.prediction import generate_full_gpr
from google3.experimental.users.haolunwu.GPR_MUR.helper.preparation import get_exp_name
from google3.experimental.users.haolunwu.GPR_MUR.model.model_dnn import Model_DNN
from google3.experimental.users.haolunwu.GPR_MUR.model.model_gru4rec import Model_GRU4REC
from google3.experimental.users.haolunwu.GPR_MUR.model.model_mf import Model_MF
from google3.experimental.users.haolunwu.GPR_MUR.model.model_ydnn import Model_YDNN
tf.compat.v1.disable_eager_execution()
warnings.filterwarnings('ignore')
def run_tune_gpr(num_cate, item_cate_dict, cate_item_map, args):
  # load from args
  gpr_data, test_data = args.gpr_data, args.test_data
  item_pairwise_sim = args.item_pairwise_sim
  model_type = args.model_type
  exp_name = get_exp_name(args)
  best_model_path = (
      '/cns/jq-d/home/haolunwu/GPR_MUR/best_model/' + exp_name + '/'
  )
  print('best_model_path:', best_model_path)
  gpu_options = tf.GPUOptions(allow_growth=True)
  if model_type == 'MF':
    model = Model_MF(args)
  elif model_type == 'DNN' or model_type == 'DNN+':
    model = Model_DNN(args)
  elif model_type == 'YDNN':
    model = Model_YDNN(args)
  elif model_type == 'GRU4REC':
    model = Model_GRU4REC(args)
  with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model.restore(sess, best_model_path)
    rec_list_full_tune, rating_pred_full_tune, _, _, item_embs = (
        generate_full_gpr(
            sess,
            gpr_data,
            model,
            args,
        )
    )
    (
        cover_tune,
        relevance_tune,
        exposure_tune,
        tail_exposure_tune,
    ) = compute_all_metrics(
        rec_list_full_tune,
        rating_pred_full_tune,
        item_cate_dict,
        item_pairwise_sim,
        item_embs,
        num_cate,
        args,
        mode='tune',
    )
    rec_list_full_test, rating_pred_full_test, _, _, item_embs = (
        generate_full_gpr(
            sess,
            test_data,
            model,
            args,
        )
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
    ### Save the results.###
    res_dict = {
        'cover_tune': cover_tune,
        'relevance_tune': relevance_tune,
        'exposure_tune': exposure_tune,
        'tail_exposure_tune': tail_exposure_tune,
        'cover_test': cover_test,
        'relevance_test': relevance_test,
        'exposure_test': exposure_test,
        'tail_exposure_test': tail_exposure_test,
    }
    res_save_path = '/cns/jq-d/home/haolunwu/GPR_MUR/tune_gpr/' + exp_name
    if tf.io.gfile.exists(res_save_path) is False:
      tf.io.gfile.makedirs(res_save_path)
    exp_setting = '{}-{}-{}-{}-{}-{}-{}-{}'.format(
        args.gpr_method,
        args.kernel_type,
        args.amp,
        args.scale,
        args.gamma,
        args.o_noise,
        args.p_noise,
        args.decay,
    )
    with tf.io.gfile.GFile(
        res_save_path + '/tune_{}.pickle'.format(exp_setting), mode='wb'
    ) as f:
      pickle.dump(res_dict, f)
