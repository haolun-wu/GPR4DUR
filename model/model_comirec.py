# @title Model_ComiRec_DR
import os
import numpy as np
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)
import tensorflow as tf
import tensorflow.compat.v1 as tf


class Model_ComiRec_DR(object):
    def __init__(self, args, flag="ComiRec_DR"):
        n_uid = args.user_train_count
        n_mid = args.item_count
        embedding_dim = args.embedding_dim
        hidden_size = embedding_dim
        batch_size = args.batch_size
        seq_len = args.maxlen
        self.decay = args.decay
        self.model_flag = flag
        self.reg = False
        self.batch_size = batch_size
        self.n_mid = n_mid
        self.n_uid = n_uid
        hard_readout = True
        relu_layer = False
        num_interest = 4
        # placeholder definition
        # self.rating = tf.placeholder(tf.int32, shape=(None,))
        self.neg_num = 1
        with tf.name_scope("Inputs"):
            self.mid_his_batch_ph = tf.placeholder(
                tf.int32, [None, None], name="mid_his_batch_ph"
            )
            self.uid_batch_ph = tf.placeholder(
                tf.int32,
                [
                    None,
                ],
                name="uid_batch_ph",
            )
            self.mid_batch_ph = tf.placeholder(
                tf.int32,
                [
                    None,
                ],
                name="mid_batch_ph",
            )
            self.mask = tf.placeholder(tf.float32, [None, None], name="mask_batch_ph")
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name="target_ph")
            self.lr = tf.placeholder(tf.float64, [])
        self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)
        # Embedding layer
        with tf.name_scope("Embedding_layer"):
            self.mid_embeddings_var = tf.get_variable(
                "mid_embedding_var", [n_mid, embedding_dim], trainable=True
            )
            self.mid_embeddings_bias = tf.get_variable(
                "bias_lookup_table",
                [n_mid],
                initializer=tf.zeros_initializer(),
                trainable=False,
            )
            self.mid_batch_embedded = tf.nn.embedding_lookup(
                self.mid_embeddings_var, self.mid_batch_ph
            )
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(
                self.mid_embeddings_var, self.mid_his_batch_ph
            )
        self.item_eb = self.mid_batch_embedded
        self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(
            self.mask, (-1, seq_len, 1)
        )
        item_his_emb = self.item_his_eb
        capsule_network = CapsuleNetwork(
            hidden_size,
            seq_len,
            bilinear_type=2,
            num_interest=num_interest,
            hard_readout=hard_readout,
            relu_layer=relu_layer,
        )
        self.user_eb, self.readout = capsule_network(
            item_his_emb, self.item_eb, self.mask
        )
        self.build_sampled_softmax_loss(self.item_eb, self.readout)

    def build_sampled_softmax_loss(self, item_emb, user_emb):
        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=self.mid_embeddings_var,
                biases=self.mid_embeddings_bias,
                labels=tf.reshape(self.mid_batch_ph, [-1, 1]),
                inputs=user_emb,
                num_sampled=self.neg_num * self.batch_size,
                num_classes=self.n_mid,
            )
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss
        )

    # def create_bpr_loss(self, users, pos_items, neg_items):
    #   pos_scores = tf.reduce_sum(
    #       tf.multiply(users, tf.expand_dims(pos_items, 1)), axis=2
    #   )
    #   neg_scores = tf.reduce_sum(
    #       tf.multiply(users, tf.expand_dims(neg_items, 1)), axis=2
    #   )
    #   regularizer = (
    #       tf.nn.l2_loss(users)
    #       + tf.nn.l2_loss(pos_items)
    #       + tf.nn.l2_loss(neg_items)
    #   )
    #   regularizer = regularizer / self.batch_size
    #   mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
    #   emb_loss = self.decay * regularizer
    #   return mf_loss, emb_loss
    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            # self.neg_items: inps[2],
            self.mid_his_batch_ph: inps[5],
            self.mask: inps[6],
            self.lr: inps[-1],
        }
        loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        # loss, _ = sess.run([self.bpr_loss, self.opt], feed_dict=feed_dict)
        return loss

    def output_item(self, sess):
        item_embs = sess.run(self.mid_embeddings_var)
        return item_embs

    def output_user(self, sess, inps):
        user_embs = sess.run(
            self.user_eb,
            feed_dict={self.mid_his_batch_ph: inps[0], self.mask: inps[1]},
        )
        return user_embs

    def save(self, sess, path):
        if tf.io.gfile.exists(path) is False:
            tf.io.gfile.makedirs(path)
        saver = tf.train.Saver()
        saver.save(sess, path + "model.ckpt")

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path + "model.ckpt")
        print("model restored from %s" % path)


def get_shape(inputs):
    dynamic_shape = tf.shape(inputs)
    static_shape = inputs.get_shape().as_list()
    shape = []
    for i, dim in enumerate(static_shape):
        shape.append(dim if dim is not None else dynamic_shape[i])
    return shape


class CapsuleNetwork(tf.layers.Layer):
    def __init__(
        self,
        dim,
        seq_len,
        bilinear_type=2,
        num_interest=4,
        hard_readout=True,
        relu_layer=True,
    ):
        super(CapsuleNetwork, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.bilinear_type = bilinear_type
        self.num_interest = num_interest
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True

    def call(self, item_his_emb, item_eb, mask):
        with tf.variable_scope("bilinear", reuse=tf.AUTO_REUSE):
            if self.bilinear_type == 0:
                item_emb_hat = tf.layers.dense(
                    item_his_emb, self.dim, activation=None, bias_initializer=None
                )
                item_emb_hat = tf.tile(item_emb_hat, [1, 1, self.num_interest])
            elif self.bilinear_type == 1:
                item_emb_hat = tf.layers.dense(
                    item_his_emb,
                    self.dim * self.num_interest,
                    activation=None,
                    bias_initializer=None,
                )
            else:
                w = tf.get_variable(
                    "weights",
                    shape=[1, self.seq_len, self.num_interest * self.dim, self.dim],
                    initializer=tf.random_normal_initializer(),
                )
                # [N, T, 1, C]
                u = tf.expand_dims(item_his_emb, axis=2)
                # [N, T, num_caps * dim_caps]
                item_emb_hat = tf.reduce_sum(w[:, : self.seq_len, :, :] * u, axis=3)
        item_emb_hat = tf.reshape(
            item_emb_hat, [-1, self.seq_len, self.num_interest, self.dim]
        )
        item_emb_hat = tf.transpose(item_emb_hat, [0, 2, 1, 3])
        item_emb_hat = tf.reshape(
            item_emb_hat, [-1, self.num_interest, self.seq_len, self.dim]
        )
        if self.stop_grad:
            item_emb_hat_iter = tf.stop_gradient(item_emb_hat, name="item_emb_hat_iter")
        else:
            item_emb_hat_iter = item_emb_hat
        if self.bilinear_type > 0:
            capsule_weight = tf.stop_gradient(
                tf.zeros([get_shape(item_his_emb)[0], self.num_interest, self.seq_len])
            )
        else:
            capsule_weight = tf.stop_gradient(
                tf.truncated_normal(
                    [get_shape(item_his_emb)[0], self.num_interest, self.seq_len],
                    stddev=1.0,
                )
            )
        for i in range(3):
            atten_mask = tf.tile(
                tf.expand_dims(mask, axis=1), [1, self.num_interest, 1]
            )
            paddings = tf.zeros_like(atten_mask)
            capsule_softmax_weight = tf.nn.softmax(capsule_weight, axis=1)
            capsule_softmax_weight = tf.where(
                tf.equal(atten_mask, 0), paddings, capsule_softmax_weight
            )
            capsule_softmax_weight = tf.expand_dims(capsule_softmax_weight, 2)
            if i < 2:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat_iter)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule
                delta_weight = tf.matmul(
                    item_emb_hat_iter, tf.transpose(interest_capsule, [0, 1, 3, 2])
                )
                delta_weight = tf.reshape(
                    delta_weight, [-1, self.num_interest, self.seq_len]
                )
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = tf.matmul(capsule_softmax_weight, item_emb_hat)
                cap_norm = tf.reduce_sum(tf.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / tf.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule
        interest_capsule = tf.reshape(
            interest_capsule, [-1, self.num_interest, self.dim]
        )
        if self.relu_layer:
            interest_capsule = tf.layers.dense(
                interest_capsule, self.dim, activation=tf.nn.relu, name="proj"
            )
        atten = tf.matmul(interest_capsule, tf.reshape(item_eb, [-1, self.dim, 1]))
        atten = tf.nn.softmax(tf.pow(tf.reshape(atten, [-1, self.num_interest]), 1))
        if self.hard_readout:
            readout = tf.gather(
                tf.reshape(interest_capsule, [-1, self.dim]),
                tf.argmax(atten, axis=1, output_type=tf.int32)
                + tf.range(tf.shape(item_his_emb)[0]) * self.num_interest,
            )
        else:
            readout = tf.matmul(
                tf.reshape(atten, [get_shape(item_his_emb)[0], 1, self.num_interest]),
                interest_capsule,
            )
            readout = tf.reshape(readout, [get_shape(item_his_emb)[0], self.dim])
        return interest_capsule, readout
