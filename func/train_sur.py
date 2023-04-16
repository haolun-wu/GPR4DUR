# @title train_sur

import sys
import time
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
from helper.evaluation import compute_metrics
from helper.prediction import generate_full_mur
from helper.prediction import generate_full_sur
from helper.preparation import get_exp_name
from helper.preparation import prepare_data
from model.model_comirec import Model_ComiRec_DR
from model.model_comirec_sa import Model_ComiRec_SA
from model.model_dnn import Model_DNN
from model.model_gru4rec import Model_GRU4REC
from model.model_mf import Model_MF
from model.model_mind import Model_MIND
from model.model_ydnn import Model_YDNN

# tf.compat.v1.disable_eager_execution()
best_metric = 0


def run_train_sur(num_cate, item_cate_dict, cate_item_map, args):
    # load from args
    dataset, model_type, batch_size, lr, maxlen = (
        args.dataset,
        args.model_type,
        args.batch_size,
        args.lr,
        args.maxlen,
    )
    train_data, valid_data, valid_holdout = (
        args.train_data,
        args.valid_data,
        args.valid_holdout,
    )
    item_cate_dict = args.item_cate_dict
    test_iter, max_iter = args.test_iter, args.max_iter
    patience = args.patience
    exp_name = get_exp_name(args)
    best_model_path = (
            'best_model/' + exp_name + '/'
    )
    print('best_model_path:', best_model_path)
    if tf.io.gfile.exists(best_model_path) is False:
        tf.io.gfile.makedirs(best_model_path)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
    ) as sess:
        if model_type == 'MF':
            model = Model_MF(args)
        elif model_type == 'DNN' or model_type == 'DNN+':
            model = Model_DNN(args)
        elif model_type == 'YDNN':
            model = Model_YDNN(args)
        elif model_type == 'ComiRec_SA':
            model = Model_ComiRec_SA(args)
        elif model_type == 'ComiRec_DR':
            model = Model_ComiRec_DR(args)
        elif model_type == 'GRU4REC':
            model = Model_GRU4REC(args)
        elif model_type == 'MIND':
            model = Model_MIND(args)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('training begin')
        sys.stdout.flush()
        start_time = time.time()
        iter = 0
        try:
            loss_sum = 0.0
            trials = 0
            for src, tgt in train_data:
                data_iter = prepare_data(src, tgt)
                loss = model.train(sess, list(data_iter) + [lr])
                loss_sum += loss
                iter += 1
                if iter % test_iter == 0:
                    print(
                        'iter: {}, train loss:{:.4f}'.format(iter, loss_sum / test_iter)
                    )
                    # user_embs_valid = infer_user_embs(sess, model, args, mode='valid')
                    # item_embs = model.output_item(sess)
                    # if len(user_embs_valid.shape) == 2:
                    #   rating_matrix = np.dot(user_embs_valid, item_embs.T)
                    # else:
                    #   rating_matrix = np.dot(user_embs_valid, item_embs.T).max(1)
                    if args.model_type in ['DNN+', 'DNN', 'YDNN', 'GRU4REC']:
                        full_rec_list, _, _ = generate_full_sur(
                            sess, valid_data, model, args
                        )
                    elif args.model_type in ['ComiRec_SA', 'ComiRec_DR', 'MIND']:
                        full_rec_list, _, _ = generate_full_mur(
                            sess, valid_data, model, args
                        )
                    precision_list, recall_list, _, _ = compute_metrics(
                        valid_holdout, full_rec_list
                    )
                    print('prec:', ', '.join(str(e) for e in precision_list))
                    print('rec :', ', '.join(str(e) for e in recall_list))
                    cur_metric = recall_list[-1]
                    global best_metric
                    if cur_metric > best_metric:
                        best_metric = cur_metric
                        model.save(sess, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            break
                    loss_sum = 0.0
                    test_time = time.time()
                    print(
                        'time interval: {:.4f} min \n'.format(
                            (test_time - start_time) / 60.0
                        )
                    )
                    sys.stdout.flush()
                if iter >= max_iter:
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')
        model.restore(sess, best_model_path)
