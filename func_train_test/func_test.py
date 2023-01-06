import tensorflow as tf
from helper.data_iterator import DataIterator
from helper.preparation import *
from helper.prediction import *
from helper.evaluation import compute_metrics, compute_ilad
from helper.ext_alpha_ndcg import compute_ext_alpha_ndcg
from scipy.sparse import csr_matrix

def test(test_data, test_set_full, test_rating, num_cate, item_cate_dict, cate_item_map, args, gpr=False, density_type='cosine', kernel_type='dot'):
    dataset, model_type, batch_size, lr, maxlen = args.dataset, args.model_type, args.batch_size, args.lr, args.maxlen
    user_train_count, item_count = args.user_train_count, args.item_count
    emb_dim, hid_dim = args.embedding_dim, args.hidden_size
    perc = args.perc

    exp_name = get_exp_name(dataset, model_type, batch_size, lr, emb_dim, maxlen, perc, save=False)
    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)
    model = get_model(dataset, model_type, user_train_count, item_count, emb_dim, hid_dim, batch_size, maxlen)
    # item_cate_map = load_item_cate(cate_file)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)

        full_rec_list_test, rating_pred = generate_full(sess, test_data, test_rating, model, best_model_path, batch_size, args, gpr, density_type, kernel_type)
        precision_list, recall_list, MAP_list, ndcg_list = compute_metrics(test_set_full, full_rec_list_test)
        item_embs = model.output_item(sess)
        rec_item_embs = item_embs[full_rec_list_test]
        ilad_cos, ilad_euc = compute_ilad(rec_item_embs)
        print("rec_item_embs:", rec_item_embs.shape)
        print("overall test:")
        print("Prec  @10, 20, 50:", ', '.join(str(e) for e in precision_list))
        print("Recall@10, 20, 50:", ', '.join(str(e) for e in recall_list))
        print("ndcg  @10, 20, 50:", ', '.join(str(e) for e in ndcg_list))

        print("-------------------")
        print("d_cos @10, 20, 50: {:.6f}, {:.6f}, {:.6f}".format(ilad_cos[0], ilad_cos[1], ilad_cos[2]))
        print("d_euc @10, 20, 50: {:.6f}, {:.6f}, {:.6f}".format(ilad_euc[0], ilad_euc[1], ilad_euc[2]))
        # alpha_ndcg_list = compute_alpha_ndcg(test_set_full, full_rec_list_test, num_cate, item_cate_dict)
        # print("a_ndcg@10, 20, 50:", ', '.join(str(e) for e in alpha_ndcg_list))
        ext_alpha_ndcg_list = compute_ext_alpha_ndcg(test_set_full, full_rec_list_test, num_cate, item_cate_dict, cate_item_map, item_embs)
        print("a*ndcg@10, 20, 50:", ', '.join(str(e) for e in ext_alpha_ndcg_list))
        print("-------------------")