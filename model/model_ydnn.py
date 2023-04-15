#@title Model_YDNN

"""Define models."""
import os
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.google.compat.v1 as tf
class Model_YDNN(object):
  def __init__(self, args, flag='YDNN'):
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
    self.neg_num = 1
    self.n_mid = n_mid
    self.n_uid = n_uid
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
      self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
      # self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
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
    self.mask_length = tf.cast(tf.reduce_sum(self.mask, -1), dtype=tf.int32)
    # Embedding layer
    with tf.name_scope('Embedding_layer'):
      self.mid_embeddings_var = tf.get_variable(
          'mid_embedding_var',
          [n_mid, embedding_dim],
          initializer=tf.random_normal_initializer(stddev=0.01),
          trainable=True,
      )
      self.mid_embeddings_bias = tf.get_variable(
          'bias_lookup_table',
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
    self.item_his_eb = self.mid_his_batch_embedded * tf.reshape(
        self.mask, (-1, seq_len, 1)
    )
    masks = tf.concat(
        [tf.expand_dims(self.mask, -1) for _ in range(embedding_dim)], axis=-1
    )
    self.item_his_eb_mean = tf.reduce_sum(self.item_his_eb, 1) / (
        tf.reduce_sum(tf.cast(masks, dtype=tf.float32), 1) + 1e-9
    )
    self.uid_batch_embedded = tf.layers.dense(
        self.item_his_eb_mean, hidden_size, activation=None
    )
    self.build_sampled_softmax_loss(
        self.mid_batch_embedded, self.uid_batch_embedded
    )
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
  def train(self, sess, inps):
    feed_dict = {
        self.uid_batch_ph: inps[0],
        self.mid_batch_ph: inps[1],
        # self.rating: inps[2],
        # self.tid_batch_ph: inps[3],
        # self.cate_rating: inps[4],
        self.mid_his_batch_ph: inps[5],
        self.mask: inps[6],
        self.lr: inps[-1],
    }
    loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
    return loss
  def output_item(self, sess):
    item_embs = sess.run(self.mid_embeddings_var)
    return item_embs
  def output_user(self, sess, inps):
    user_embs = sess.run(
        self.uid_batch_embedded,
        feed_dict={self.mid_his_batch_ph: inps[0], self.mask: inps[1]},
    )
    return user_embs
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
