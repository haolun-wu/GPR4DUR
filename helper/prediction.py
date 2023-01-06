import tensorflow as tf
import numpy as np
from helper.preparation import prepare_data
from scipy.sparse import csr_matrix
import random

from tensorflow_probability.substrates import jax as tfp
from sklearn.neighbors import KernelDensity
from scipy import stats


def help_knn(rating_pred, hist_item, topN):
    # pair_distance = euclidean_distances(user_embs, item_embs)
    rating_pred_final = rating_pred.copy()
    np.put_along_axis(rating_pred_final, np.array(hist_item), -1e5, axis=1)  # set those history items with score -1e5
    # start = time.time()
    ind = np.argpartition(rating_pred_final, -topN)
    ind = ind[:, -topN:]
    arr_ind = rating_pred_final[np.arange(len(rating_pred_final))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred_final)), ::-1]
    I = ind[np.arange(len(rating_pred_final))[:, None], arr_ind_argsort]

    # end = time.time()
    # print("Time for sorting:{:.4f} min".format((end-start) / 60.0))
    D = np.take_along_axis(rating_pred_final, I, axis=-1)

    # D is the topN inner-product, sorted from highest to lowest
    # I is the topN index, corresponding to D
    return D, I


def fit_gp(user_embs, item_embs, user_item_sequences, hist_rate, item_batch_index, density_type='cosine',
           kernel_type='dot', user_id=-1, user_embedding_weight=1.0):
    tfb = tfp.bijectors
    tfd = tfp.distributions
    data = item_embs[np.array(user_item_sequences)]
    # print("data:", data.shape)

    # density = []
    # for i in range(data.shape[0]):
    #   if density_type == 'cosine':
    #       density_curr = 1 - cdist(data[i], data[i], 'cosine')
    #       density_curr = density_curr.mean(1)
    #       # bw_scott = 1e-2
    #       # kernel = KernelDensity(bandwidth=bw_scott, kernel='cosine').fit(data[i])
    #       # density_curr = kernel.score_samples(data[i])
    #   elif density_type == 'gaussian':
    #       # bw_scott = data.shape[0] ** (-1./(data.shape[1]+4)) * data.std(ddof=1)
    #       bw_scott = 1e-2
    #       kernel = KernelDensity(bandwidth=bw_scott, kernel='gaussian').fit(data[i])
    #       density_curr = kernel.score_samples(data[i])
    #       # kernel = stats.gaussian_kde(data[i].T)
    #       # density_curr = kernel(data[i].T)

    #   density.append(density_curr)

    density = np.array(hist_rate) * 100

    # density = np.array(density) * 1

    index_points = item_embs[item_batch_index]

    psd_kernels = tfp.math.psd_kernels
    if kernel_type == 'dot':
        gp_kernel = psd_kernels.Linear()
    elif kernel_type == 'expon':
        gp_kernel = psd_kernels.ExponentiatedQuadratic(length_scale=0.1)
    elif kernel_type == 'matern':
        gp_kernel = psd_kernels.MaternOneHalf(length_scale=0.1)

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=gp_kernel,
        index_points=index_points.astype(np.float32),
        observation_index_points=data.astype(np.float32),
        observations=density.astype(np.float32),
        observation_noise_variance=1.0)

    return gprm


def generate_full(sess, test_data, rating_matrix, model, model_path, batch_size, args, gpr=False, density_type='cosine',
                  kernel_type='dot'):
    topN = 50

    item_embs = model.output_item(sess)
    min_index = test_data.min_index

    full_rec_list = np.array([])

    for src, tgt in test_data:
        nick_id, item_id, neg_id, rating, hist_item, hist_mask, neg_items, hist_rate = prepare_data(src, tgt)

        # print("hist_rate:", hist_rate)

        # nick_id = list(np.array(nick_id)[index])
        # item_id = [item_id[i] for i in index]
        # hist_item = [hist_item[i] for i in index]
        # hist_mask = [hist_mask[i] for i in index]
        # neg_items = [neg_items[i] for i in index]

        if args.model_type == 'DNN':
            # user_embs = model.output_user(sess, [hist_item, hist_mask])[index]
            user_embs = model.output_user(sess, [hist_item, hist_mask])
        else:
            user_index = np.array(nick_id - min_index)
            batch_rating = rating_matrix.toarray()[user_index][:, :int(args.item_count * args.perc)]
            item_embs_use = item_embs[:int(args.item_count * args.perc)]

            user_embs = np.dot(np.dot(batch_rating, item_embs_use),
                               np.linalg.inv(np.dot(item_embs_use.T, item_embs_use)))

        if gpr:
            st_time = time.time()
            rating_pred = None
            init_key, sample_key = jax.random.split(jax.random.PRNGKey(np.random.randint(200)))
            batch_size = 256
            item_count = item_embs.shape[0]
            num_batches = item_count // batch_size + 1
            for batchID in range(num_batches):
                start = batchID * batch_size
                end = start + batch_size

                if batchID == num_batches - 1:
                    if start < item_count:
                        end = item_count
                    else:
                        break

                item_batch_index = np.arange(item_count)[start:end]
                jax_gpr = fit_gp(user_embs, item_embs, hist_item, hist_rate, item_batch_index, density_type,
                                 kernel_type)
                rating_pred_batch = jax_gpr.sample(seed=init_key)

                if batchID == 0:
                    rating_pred = rating_pred_batch
                else:
                    rating_pred = np.concatenate((rating_pred, rating_pred_batch), axis=1)

            print("rating_pred gpr:", rating_pred.shape)
            print("GPR time:", time.time() - st_time)
        else:
            rating_pred = np.matmul(user_embs, item_embs.T)

        D, rec_list = help_knn(rating_pred, hist_item, topN)

        rec_list = np.array(rec_list)
        if len(full_rec_list) == 0:
            full_rec_list = rec_list
        else:
            full_rec_list = np.concatenate((full_rec_list, rec_list), axis=0)

    return full_rec_list, rating_pred
