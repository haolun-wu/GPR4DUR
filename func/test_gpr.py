# @title test_gpr
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow.compat.v1 as tf
from helper.eval_new import compute_all_metrics
from helper.prediction import generate_full_gpr
from helper.preparation import get_exp_name
from model.model_dnn import Model_DNN
from model.model_ydnn import Model_YDNN

# tf.compat.v1.disable_eager_execution()



def run_test_gpr(num_cate, item_cate_dict, cate_item_map, args):
    # load from args
    test_data = args.test_data
    # item_pairwise_sim = args.item_pairwise_sim
    model_type = args.model_type
    exp_name = get_exp_name(args)
    best_model_path = (
            'best_model/' + exp_name + '/'
    )
    print('best_model_path:', best_model_path)
    gpu_options = tf.GPUOptions(allow_growth=True)
    if model_type == 'DNN':
        model = Model_DNN(args)
    elif model_type == 'YDNN':
        model = Model_YDNN(args)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, best_model_path)
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
            # item_pairwise_sim,
            item_embs,
            num_cate,
            args,
            mode='test',
        )
        print('cover_test:', ', '.join(str(e) for e in cover_test))
        print('relevance_test:', ', '.join(str(e) for e in relevance_test))
        print('exposure_test:', ', '.join(str(e) for e in exposure_test))
        print('tail_exposure_test:', ', '.join(str(e) for e in tail_exposure_test))
