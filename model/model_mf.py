#@title Model_MF
"""Define models."""
import os
import numpy as np
import tensorflow as tf
import tensorflow.google.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
class Model_MF(object):
  def __init__(self, args, flag='MF'):
    n_uid = args.user_train_count
    n_mid = args.item_count
    embedding_dim = args.embedding_dim
    hidden_size = embedding_dim
    batch_size = args.batch_size
    seq_len = args.maxlen
    num_cate = args.num_cate
    self.wd = args.wd
    self.model_flag = flag
    self.reg = False
    self.batch_size = batch_size
    # placeholder definition
    with tf.name_scope('Inputs'):
      self.mid_his_batch_ph = tf.placeholder(
          tf.int32, [None, None], name='mid_his_batch_ph'
      )
      self.uid_batch_ph = tf.placeholder(
          tf.int32,
          [
              None,
          ],
          name='uid_batch_ph',
      )
      self.mid_batch_ph = tf.placeholder(
          tf.int32,
          [
              None,
          ],
          name='mid_batch_ph',
      )
      self.tid_batch_ph = tf.placeholder(
          tf.int32,
          [
              None,
          ],
          name='tid_batch_ph',
      )
      self.lr = tf.placeholder(tf.float64, [])
      self.rating = tf.placeholder(
          tf.float32,
          [
              None,
          ],
          name='rating',
      )
      self.cate_rating = tf.placeholder(
          tf.float32,
          [
              None,
          ],
          name='cate_rating',
      )
    # Embedding layer
    with tf.name_scope('Embedding_layer'):
      self.mid_embeddings_var = tf.get_variable(
          'mid_embedding_var',
          [n_mid, embedding_dim],
          initializer=tf.random_normal_initializer(stddev=0.01),
          trainable=True,
      )
      self.mid_embeddings_bias = tf.get_variable(
          'mid_bias_lookup_table',
          [n_mid],
          initializer=tf.zeros_initializer(),
          trainable=False,
      )
      self.mid_batch_embedded = tf.nn.embedding_lookup(
          self.mid_embeddings_var, self.mid_batch_ph
      )
      self.uid_embeddings_var = tf.get_variable(
          'uid_embedding_var',
          [n_uid, embedding_dim],
          initializer=tf.random_normal_initializer(stddev=0.01),
          trainable=True,
      )
      self.uid_embeddings_bias = tf.get_variable(
          'uid_bias_lookup_table',
          [n_uid],
          initializer=tf.zeros_initializer(),
          trainable=False,
      )
      self.uid_batch_embedded = tf.nn.embedding_lookup(
          self.uid_embeddings_var, self.uid_batch_ph
      )
      self.tid_embeddings_var = tf.get_variable(
          'tid_embedding_var',
          [num_cate, embedding_dim],
          initializer=tf.random_normal_initializer(stddev=0.01),
          trainable=True,
      )
      self.tid_embeddings_bias = tf.get_variable(
          'tid_bias_lookup_table',
          [num_cate],
          initializer=tf.zeros_initializer(),
          trainable=False,
      )
      self.tid_batch_embedded = tf.nn.embedding_lookup(
          self.tid_embeddings_var, self.tid_batch_ph
      )
    ### main ###
    gamma = 5
    ui_loss, ui_regularizer = self.build_l2_loss()
    it_loss, it_regularizer = self.build_l2_loss_tag()
    self.loss = ui_loss + ui_regularizer + (it_loss + it_regularizer) * gamma
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
        self.loss
    )
    # pair_loss = self.pairwise_loss(
    #     self.mid_batch_embedded, self.pos_tags_embedded, self.neg_tags_embedded
    # )
    # self.loss = regr_loss + regularizer + pair_loss * 20
    # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
    #     self.loss
    # )
  def build_l2_loss(self):
    prediction = (
        tf.reduce_sum(
            tf.multiply(self.uid_batch_embedded, self.mid_batch_embedded), 1
        )
        + tf.nn.embedding_lookup(self.uid_embeddings_bias, self.uid_batch_ph)
        + tf.nn.embedding_lookup(self.mid_embeddings_bias, self.mid_batch_ph)
    )
    regr_loss = tf.reduce_mean(tf.nn.l2_loss(prediction - self.rating))
    # regr_loss = tf.losses.mean_squared_error(self.rating, predicted_rating)
    regularizer = self.wd * (
        tf.nn.l2_loss(self.uid_batch_embedded)
        + tf.nn.l2_loss(self.mid_batch_embedded)
    )
    regularizer = regularizer / self.batch_size
    return regr_loss, regularizer
  def build_l2_loss_tag(self):
    prediction = (
        tf.reduce_sum(
            tf.multiply(self.mid_batch_embedded, self.tid_batch_embedded), 1
        )
        + tf.nn.embedding_lookup(self.mid_embeddings_bias, self.mid_batch_ph)
        + tf.nn.embedding_lookup(self.tid_embeddings_bias, self.tid_batch_ph)
    )
    regr_loss = tf.reduce_mean(tf.nn.l2_loss(prediction - self.cate_rating))
    # regr_loss = tf.losses.mean_squared_error(
    #     tf.ones_like(predicted_rating), predicted_rating
    # )
    regularizer = self.wd * (
        # tf.nn.l2_loss(self.mid_batch_embedded)
        tf.nn.l2_loss(self.tid_batch_embedded)
    )
    regularizer = regularizer / self.batch_size
    return regr_loss, regularizer
  def pairwise_loss(self, inds, pos, neg):
    pos_scores = tf.reduce_sum(tf.multiply(inds, pos), axis=1)
    neg_scores = tf.reduce_sum(tf.multiply(inds, neg), axis=1)
    regularizer = tf.nn.l2_loss(inds) + tf.nn.l2_loss(pos) + tf.nn.l2_loss(neg)
    regularizer = regularizer / self.batch_size
    mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
    emb_loss = self.wd * regularizer
    pair_loss = mf_loss + emb_loss
    return pair_loss
  def train(self, sess, inps):
    feed_dict = {
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        self.rating: inps[2],
        self.tid_batch_ph: inps[3],
        self.cate_rating: inps[4],
        self.lr: inps[-1],
    }
    loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
    return loss
  def output_item(self, sess):
    item_embs = sess.run(self.mid_embeddings_var)
    return item_embs
  def save(self, sess, path):
    if tf.io.gfile.exists(path) is False:
      tf.io.gfile.makedirs(path)
    # if not os.path.exists(path):
    #   os.makedirs(path)
    saver = tf.train.Saver()
    saver.save(sess, path + 'model.ckpt')
  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, path + 'model.ckpt')
    print('model restored from %s' % path)
