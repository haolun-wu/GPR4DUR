import os
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_v2_behavior()
import tensorflow.contrib.slim as slim
from tensorflow.nn.rnn_cell import GRUCell

class Model(object):
    def __init__(self, n_uid, n_mid, embedding_dim, hidden_size, batch_size, seq_len, flag="DNN"):
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.decay = 1e-5
        self.n_mid = n_mid
        self.n_uid = n_uid

        # placeholder definition

        # self.rating = tf.placeholder(tf.int32, shape=(None,))

        # self.n_uid = n_uid
        self.neg_num = 10
        with tf.name_scope('Inputs'):
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='mid_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.mid_batch_ph = tf.placeholder(tf.int32, [None, ], name='mid_batch_ph')
            self.neg_items = tf.placeholder(tf.int32, [None, ], name='mid_neg_batch')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])
            self.rating = tf.placeholder(tf.float32, [None, ], name='rating')

        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)

        # Embedding layer
        with tf.name_scope('Embedding_layer'):
            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, embedding_dim],
                                                      initializer=tf.random_normal_initializer(stddev=0.01),
                                                      trainable=True)
            self.mid_embeddings_bias = tf.get_variable("mid_bias_lookup_table", [n_mid],
                                                       initializer=tf.zeros_initializer(),
                                                       trainable=False)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)

            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, embedding_dim],
                                                      initializer=tf.random_normal_initializer(stddev=0.01),
                                                      trainable=True)
            self.uid_embeddings_bias = tf.get_variable("uid_bias_lookup_table", [n_uid],
                                                       initializer=tf.zeros_initializer(),
                                                       trainable=False)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(self.mask, (-1, seq_len, 1))

        self.u_g_embeddings = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.mid_embeddings_var, self.uid_batch_ph)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)

    def build_svd_loss(self):
        predicted_rating = tf.reduce_sum(tf.multiply(self.uid_batch_embedded, self.mid_batch_embedded), 1) \
                           + tf.nn.embedding_lookup(self.uid_embeddings_bias, self.uid_batch_ph) \
                           + tf.nn.embedding_lookup(self.mid_embeddings_bias, self.mid_batch_ph)
        svd_loss = tf.reduce_mean(tf.nn.l2_loss(predicted_rating - self.rating))
        regularizer = self.decay * (tf.nn.l2_loss(self.uid_batch_embedded) + tf.nn.l2_loss(self.mid_batch_embedded)
                                    + tf.nn.l2_loss(self.uid_embeddings_bias) + tf.nn.l2_loss(self.mid_embeddings_bias))
        regularizer = regularizer / self.batch_size

        return svd_loss, regularizer

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        return mf_loss, emb_loss

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(self.mid_embeddings_var, self.mid_embeddings_bias,
                                                              tf.reshape(self.mid_batch_ph, [-1, 1]), user_emb,
                                                              self.neg_num * self.batch_size, self.n_mid))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def train_DNN(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.neg_items: inps[2],
            self.mid_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.lr: inps[-1]
        }
        # loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        loss, _ = sess.run([self.bpr_loss, self.opt], feed_dict=feed_dict)
        return loss

    def train_MF(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.neg_items: inps[2],
            self.lr: inps[-1]
        }
        loss, _ = sess.run([self.bpr_loss, self.opt], feed_dict=feed_dict)
        return loss

    def train_SVD(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.rating: inps[3],
            self.lr: inps[-1]
        }
        loss, _ = sess.run([self.svd_loss, self.svd_opt], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(self.user_eb, feed_dict={
            self.mid_his_batch_ph: inps[0],
            self.mask: inps[1]
        })
        return user_embs

    def save(self, sess, path):
        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + 'model.ckpt')

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + 'model.ckpt')
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, n_uid, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_DNN, self).__init__(n_uid, n_mid, embedding_dim, hidden_size,
                                        batch_size, seq_len, flag="DNN")

        masks = tf.concat([tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1)

        self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (
                tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9)
        self.user_eb = tf.layers.dense(self.item_his_eb_mean, hidden_size, activation=None)
        # self.build_sampled_softmax_loss(self.item_eb, self.user_eb)

        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.user_eb,
                                                           self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)
        self.bpr_loss = self.mf_loss + self.emb_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.bpr_loss)


class Model_MF(Model):
    def __init__(self, n_uid, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_MF, self).__init__(n_uid, n_mid, embedding_dim, hidden_size,
                                       batch_size, seq_len, flag="MF")

        self.mf_loss, self.emb_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                           self.pos_i_g_embeddings,
                                                           self.neg_i_g_embeddings)
        self.bpr_loss = self.mf_loss + self.emb_loss
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.bpr_loss)


class Model_SVD(Model):
    def __init__(self, n_uid, n_mid, embedding_dim, hidden_size, batch_size, seq_len=256):
        super(Model_SVD, self).__init__(n_uid, n_mid, embedding_dim, hidden_size,
                                        batch_size, seq_len, flag="SVD")

        svd_loss, regularizer = self.build_svd_loss()

        self.svd_loss = svd_loss + regularizer
        self.svd_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.svd_loss)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])

    return shape
