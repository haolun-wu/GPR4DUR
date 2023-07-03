# @title prediction

"""Generate prediction score and full_rec_list."""
import time
import jax
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from helper.preparation import prepare_data


def infer_user_embs(sess, model, args, mode='valid'):
    if mode == 'valid':
        data = args.valid_data
    elif mode == 'test':
        data = args.test_data
    item_embs = model.output_item(sess)
    # min_index = data.min_index
    user_embs_full = None
    for src, tgt in data:
        (
            nick_id,
            item_id,
            _,
            _,
            _,
            hist_item,
            hist_mask,
            hist_rate,
        ) = prepare_data(src, tgt)
        print('# user val/test:', len(nick_id))
        if args.model_type in ['MF']:
            min_index = min(nick_id)
            user_index = sorted(np.array(nick_id) - min_index)
            user_embs = None
            for u in user_index:
                hist_rating = np.array(hist_rate[u]).reshape(1, -1)
                item_emb = item_embs[hist_item[u]]
                user_emb = np.dot(
                    np.dot(hist_rating, item_emb),
                    np.linalg.inv(np.dot(item_emb.T, item_emb)),
                )
                if u == 0:
                    user_embs = user_emb
                else:
                    user_embs = np.concatenate((user_embs, user_emb), 0)
        else:
            user_embs = model.output_user(sess, [hist_item, hist_mask])
        if user_embs_full is None:
            user_embs_full = user_embs
        else:
            user_embs_full = np.concatenate((user_embs_full, user_embs), 0)
    # print('user_embs_full:', user_embs_full.shape)
    return user_embs_full


def get_topn(rating_pred, hist_item, topn):
    # print('rating_pred:', np.shape(rating_pred))
    # print('hist_item:', np.shape(hist_item))
    rating_pred_final = rating_pred.copy()
    np.put_along_axis(
        rating_pred_final, np.array(hist_item), -1e5, axis=1
    )  # set those history items with score -1e5
    ind = np.argpartition(rating_pred_final, -topn)
    ind = ind[:, -topn:]
    arr_ind = rating_pred_final[np.arange(len(rating_pred_final))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_final)), ::-1]
    rec_index = ind[np.arange(len(rating_pred_final))[:, None], arr_ind_argsort]
    rec_rating = np.take_along_axis(rating_pred_final, rec_index, axis=-1)
    return rec_rating, rec_index


def generate_full_sur(sess, test_data, model, args):
    topn = 100
    item_embs = model.output_item(sess)
    user_embs_full = None
    hist_item_full = None
    for src, tgt in test_data:
        (
            nick_id,
            item_id,
            _,
            _,
            _,
            hist_item,
            hist_mask,
            hist_rate,
        ) = prepare_data(src, tgt)
        if args.model_type == 'MF':
            min_index = min(nick_id)
            user_index = sorted(np.array(nick_id) - min_index)
            user_embs = None
            for u in user_index:
                hist_rating = np.array(hist_rate[u]).reshape(1, -1)
                item_emb = item_embs[hist_item[u]]
                user_emb = np.dot(
                    np.dot(hist_rating, item_emb),
                    np.linalg.inv(np.dot(item_emb.T, item_emb)),
                )
                if u == 0:
                    user_embs = user_emb
                else:
                    user_embs = np.concatenate((user_embs, user_emb), 0)
        else:
            user_embs = model.output_user(sess, [hist_item, hist_mask])
        if user_embs_full is None:
            user_embs_full = user_embs
            hist_item_full = hist_item
        else:
            user_embs_full = np.concatenate((user_embs_full, user_embs), 0)
            hist_item_full = np.concatenate((hist_item_full, hist_item), 0)
    rating_pred_full = np.matmul(user_embs_full, item_embs.T)
    _, rec_list_full = get_topn(rating_pred_full, hist_item_full, topn)
    # print('rating_pred_full:', rating_pred_full.shape)
    # print('rec_list_full:', np.shape(rec_list_full))
    return rec_list_full, rating_pred_full, item_embs


def generate_full_mur(sess, test_data, model, args):
    topn = 100
    item_embs = model.output_item(sess)
    user_embs_full = np.array(['None'])
    hist_item_full = None
    full_rec_list = np.array(['None'])
    for src, tgt in test_data:
        (
            nick_id,
            item_id,
            _,
            _,
            _,
            hist_item,
            hist_mask,
            hist_rate,
        ) = prepare_data(src, tgt)
        user_embs = model.output_user(sess, [hist_item, hist_mask])
        if user_embs_full == 0:
            user_embs_full = user_embs
            hist_item_full = hist_item
        else:
            user_embs_full = np.concatenate((user_embs_full, user_embs), 0)
            hist_item_full = np.concatenate((hist_item_full, hist_item), 0)
    user_embs_copy = user_embs_full.copy()
    ni = user_embs_copy.shape[1]
    rating_pred_full = np.matmul(user_embs_full, item_embs.T).max(1)
    user_embs_full = np.reshape(user_embs_full, [-1, user_embs_full.shape[-1]])
    rec_list_full = []
    rating_pred = np.matmul(user_embs_full, item_embs.T)
    hist_item_repeat = np.repeat(hist_item_full, 4, axis=0)
    d_mur, i_mur = get_topn(rating_pred, hist_item_repeat, topn)
    for i in range(user_embs_copy.shape[0]):
        item_list_set = set()
        item_cor_list = []
        item_list = list(
            zip(
                np.reshape(i_mur[i * ni: (i + 1) * ni], -1),
                np.reshape(d_mur[i * ni: (i + 1) * ni], -1),
            )
        )
        item_list.sort(key=lambda x: x[1], reverse=True)
        for j in range(len(item_list)):
            if item_list[j][0] not in item_list_set:  # we do not pad 0.
                item_list_set.add(item_list[j][0])
                item_cor_list.append(item_list[j][0])
                if len(item_list_set) >= topn:
                    break
        while len(item_cor_list) < topn:
            item_cor_list.append(item_cor_list[-1])
        rec_list_full.append(item_cor_list)
    rec_list_full = np.array(rec_list_full)
    return rec_list_full, rating_pred_full, item_embs


def get_topn_thompson(rating_pred, hist_item):
    rec_index, rating_pred_all = ['None'], ['None']
    rating_pred_tom = None
    num_users, num_items = rating_pred[0].shape[0], rating_pred[0].shape[1]
    mask = np.zeros((num_users, num_items))
    np.put_along_axis(mask, np.array(hist_item), -1e5, axis=1)
    for i in range(rating_pred.shape[0] - 1):
        rating_pred_sample = rating_pred[i].copy()
        # Add mask: filter out hist_items (and recommended items)
        rating_pred_sample += mask
        # Get the index of the max value by row and save
        max_index = np.argmax(rating_pred_sample, 1).reshape(-1, 1)
        raing_pred_cur = np.take_along_axis(rating_pred_sample, max_index, axis=1)
        if rec_index == ['None']:
            rec_index = max_index
            rating_pred_tom = raing_pred_cur
        else:
            rec_index = np.concatenate((rec_index, max_index), axis=1)
            rating_pred_tom = np.concatenate(
                (rating_pred_tom, raing_pred_cur), axis=1
            )
        # Generate mask
        np.put_along_axis(mask, max_index, -1e5, axis=1)
    not_select_full = []
    for i in range(num_users):
        not_select = list(set(np.arange(num_items)) - set(rec_index[i]))
        not_select_full.append(not_select)
    rating_pred_not_select = np.take_along_axis(
        rating_pred[-1], np.array(not_select_full).astype(np.int64), axis=1
    )
    rating_pred_tom = np.concatenate(
        (rating_pred_tom, rating_pred_not_select), axis=1
    )
    print('rating_pred_tom:', rating_pred_tom.shape)
    print(rating_pred_tom.min())
    return rating_pred_tom, rec_index


def fit_gp(
        item_embs,
        user_item_sequences,
        hist_rate,
        item_batch_index,
        args,
):
    kernel_type, length_scale, amplitude = (
        args.kernel_type,
        args.scale,
        args.amp,
    )
    # tfb = tfp.bijectors
    tfd = tfp.distributions
    index_points = item_embs[item_batch_index]
    data = item_embs[np.array(user_item_sequences)]
    observe_value = np.array(hist_rate)
    psd_kernels = tfp.math.psd_kernels
    if kernel_type == 'dot':
        gp_kernel = psd_kernels.Linear(
            bias_amplitude=length_scale, slope_amplitude=amplitude
        )
    elif kernel_type == 'expon':
        gp_kernel = psd_kernels.ExponentiatedQuadratic(
            amplitude=amplitude, length_scale=length_scale
        )
    elif kernel_type == 'matern':
        gp_kernel = psd_kernels.MaternOneHalf(
            amplitude=amplitude, length_scale=length_scale
        )
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=gp_kernel,
        index_points=index_points.astype(np.float32),
        observation_index_points=data.astype(np.float32),
        observations=observe_value.astype(np.float32),
        observation_noise_variance=args.o_noise,
        predictive_noise_variance=args.p_noise,
    )
    return gprm


def generate_full_gpr(sess, data_for_use, model, args):
    topn = 100
    batch_size = args.batch_size
    item_embs = model.output_item(sess)
    rating_pred_full = np.array([])
    rec_list_full = np.array([])
    mean_gpr_full, std_gpr_full = np.array([]), np.array([])
    init_key, _ = jax.random.split(jax.random.PRNGKey(np.random.randint(200)))
    for src, tgt in data_for_use:
        (
            nick_id,
            item_id,
            _,
            _,
            _,
            hist_item,
            hist_mask,
            hist_rate,
        ) = prepare_data(src, tgt)
        # if args.model_type in ['MF']:
        #     min_index = min(nick_id)
        #     user_index = sorted(np.array(nick_id) - min_index)
        #     user_embs = None
        #     for u in user_index:
        #         hist_rating = np.array(hist_rate[u]).reshape(1, -1)
        #         item_emb = item_embs[hist_item[u]]
        #         user_emb = np.dot(
        #             np.dot(hist_rating, item_emb),
        #             np.linalg.inv(np.dot(item_emb.T, item_emb)),
        #         )
        #         if u == 0:
        #             user_embs = user_emb
        #         else:
        #             user_embs = np.concatenate((user_embs, user_emb), 0)
        # else:
        #     user_embs = model.output_user(sess, [hist_item, hist_mask])

        """start gpr process"""
        st_time = time.time()
        rating_pred = np.array(['None'])
        mean_gpr, std_gpr = None, None
        batch_size = 256
        item_count = item_embs.shape[0]
        num_batches = item_count // batch_size + 1
        for batch_id in range(num_batches):
            start = batch_id * batch_size
            end = start + batch_size
            if batch_id == num_batches - 1:
                if start < item_count:
                    end = item_count
                else:
                    break
            item_batch_index = np.arange(item_count)[start:end]
            hist_rate = np.ones_like(hist_rate)  # input all 1 into GPR
            degrade_fact = (np.exp(-np.arange(args.maxlen) * args.decay))[::-1]
            hist_rate_augment = hist_rate * degrade_fact
            jax_gpr = fit_gp(
                # user_embs,
                item_embs,
                hist_item,
                hist_rate_augment,
                item_batch_index,
                args,
            )
            batch_mean_gpr = jax_gpr.mean()
            batch_std_gpr = jax_gpr.stddev()
            if args.gpr_method == 'thompson':
                rating_pred_batch = jax_gpr.sample(seed=init_key)
            elif args.gpr_method == 'thompson++':
                rating_pred_batch = jax_gpr.sample(sample_shape=51, seed=init_key)
            elif args.gpr_method == 'ucb':
                rating_pred_batch = batch_mean_gpr + args.gamma * batch_std_gpr
            if batch_id == 0:
                rating_pred = rating_pred_batch
                mean_gpr = batch_mean_gpr
                std_gpr = batch_std_gpr
            else:
                mean_gpr = np.concatenate((mean_gpr, batch_mean_gpr), axis=1)
                std_gpr = np.concatenate((std_gpr, batch_std_gpr), axis=1)
                if args.gpr_method == 'thompson++':
                    rating_pred = np.concatenate((rating_pred, rating_pred_batch), axis=2)
                else:
                    rating_pred = np.concatenate((rating_pred, rating_pred_batch), axis=1)
        print('rating_pred gpr:', rating_pred.shape)
        print('GPR time:', time.time() - st_time)
        assert np.isnan(rating_pred.sum()) == 0
        if args.gpr_method == 'thompson++':
            rating_pred_cum, rec_list = get_topn_thompson(rating_pred, hist_item)
        else:
            _, rec_list = get_topn(rating_pred, hist_item, topn)
            rating_pred_cum = rating_pred
        if len(mean_gpr_full) == 0:
            rating_pred_full = rating_pred_cum
            mean_gpr_full = mean_gpr
            std_gpr_full = std_gpr
            rec_list_full = rec_list
        else:
            rating_pred_full = np.concatenate(
                (rating_pred_full, rating_pred_cum), axis=0
            )
            mean_gpr_full = np.concatenate((mean_gpr_full, mean_gpr), axis=0)
            std_gpr_full = np.concatenate((std_gpr_full, std_gpr), axis=0)
            rec_list_full = np.concatenate((rec_list_full, rec_list), axis=0)
    return rec_list_full, rating_pred_full, mean_gpr_full, std_gpr_full, item_embs
