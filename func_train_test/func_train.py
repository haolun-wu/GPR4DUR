import tensorflow as tf
from helper.data_iterator import DataIterator
from helper.preparation import *
from helper.prediction import *
from helper.evaluation import compute_metrics, compute_ilad
from helper.ext_alpha_ndcg import compute_ext_alpha_ndcg
from scipy.sparse import csr_matrix
import sys
import time

best_metric = 0

def train(train_file, valid_file, test_file, valid_set_full, test_set_full, args):
    dataset, model_type, batch_size, lr, maxlen = args.dataset, args.model_type, args.batch_size, args.lr, args.maxlen
    user_train_count, item_count = args.user_train_count, args.item_count
    emb_dim, hid_dim = args.embedding_dim, args.hidden_size
    test_iter, max_iter = args.test_iter, args.max_iter
    patience = args.patience
    perc = args.perc


    exp_name = get_exp_name(dataset, model_type, batch_size, lr, emb_dim, maxlen, perc)
    best_model_path = "best_model/" + exp_name + '/'

    summary_path = 'runs/' + exp_name

    if os.path.exists(best_model_path) == False:
        os.makedirs(best_model_path)

    gpu_options = tf.GPUOptions(allow_growth=True)
    # train_data = DataIterator(train_file, item_count, batch_size, maxlen, train_flag=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
        train_data = DataIterator(train_file, item_count, batch_size, maxlen, train_flag=1)
        # row, col, rate = train_data.get_matrix()
        # train_matrix = csr_matrix((rate, (row, col)), shape=(user_train_count, item_count))
        # print("train_matrix:", train_matrix.shape)

        valid_data = DataIterator(valid_file, item_count, batch_size, maxlen, train_flag=0)
        row, col, rate = valid_data.get_matrix()
        val_rating = csr_matrix((rate, (row, col)), shape=(len(set(row)), args.item_count))

        model = get_model(dataset, model_type, user_train_count, item_count, emb_dim, hid_dim, batch_size, maxlen)

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

                if args.model_type == 'MF':
                    loss = model.train_MF(sess, list(data_iter) + [lr])
                elif args.model_type == 'DNN':
                    loss = model.train(sess, list(data_iter) + [lr])
                elif args.model_type == 'SVD':
                    loss = model.train_SVD(sess, list(data_iter) + [lr])

                loss_sum += loss
                iter += 1

                if iter % test_iter == 0:
                    full_rec_list, rating_pred = generate_full(sess, valid_data, val_rating, model, best_model_path,
                                                               batch_size,
                                                               args)
                    print('iter: {}, train loss:{:.4f}'.format(iter, loss_sum / test_iter))

                    precision_list, recall_list, MAP_list, ndcg_list = compute_metrics(valid_set_full, full_rec_list)
                    print("Recall@10, 20, 50:", ', '.join(str(e) for e in recall_list))
                    print("ndcg  @10, 20, 50:", ', '.join(str(e) for e in ndcg_list))

                    recall = recall_list[-1]
                    global best_metric

                    if recall > best_metric:
                        best_metric = recall
                        saver = tf.train.Saver(max_to_keep=1)
                        model.save(sess, best_model_path)
                        trials = 0
                    else:
                        trials += 1
                        if trials > patience:
                            break

                    loss_sum = 0.0
                    test_time = time.time()
                    print("time interval: {:.4f} min \n".format((test_time - start_time) / 60.0))
                    sys.stdout.flush()

                if iter >= max_iter:
                    break
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

        model.restore(sess, best_model_path)

        test_data = DataIterator(test_file, item_count, batch_size, maxlen, train_flag=0)
        row, col, rate = test_data.get_matrix()
        test_rating = csr_matrix((rate, (row, col)), shape=(len(set(row)), args.item_count))
        full_rec_list_test, rating_pred_test = generate_full(sess, test_data, test_rating, model, best_model_path,
                                                             batch_size, args)
        precision_list, recall_list, MAP_list, ndcg_list = compute_metrics(test_set_full, full_rec_list_test)
        item_embs = model.output_item(sess)
        rec_item_embs = item_embs[full_rec_list_test]
        ilad_cos, ilad_euc = compute_ilad(rec_item_embs)
        print("rec_item_embs:", rec_item_embs.shape)
        print("overall test:")
        print("Recall@10, 20, 50:", ', '.join(str(e) for e in recall_list))
        print("ndcg  @10, 20, 50:", ', '.join(str(e) for e in ndcg_list))
        print("d_cos @10, 20, 50: {:.6f}, {:.6f}, {:.6f}".format(ilad_cos[0], ilad_cos[1], ilad_cos[2]))
        print("d_euc @10, 20, 50: {:.6f}, {:.6f}, {:.6f}".format(ilad_euc[0], ilad_euc[1], ilad_euc[2]))
        print("-------------------")
