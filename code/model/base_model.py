import os

import tensorflow as tf


class BaseModel(object):

    def __init__(self, args):
        self.args = args
        self.sess = None
        self.saver = None

    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)

    def add_train_op(self, lr_method, adam_beta1, adam_beta2, lr, loss, clip=-1):
        """
        Defines self.train_op that performs an update on a batch
        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient.(< 0: no clipping)
        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.variable_scope("train_step"):
          # CREATE OPTIMIZER
            if _lr_m == "adam":
                optimizer = tf.train.AdamOptimizer(lr, adam_beta1, adam_beta2)
            elif _lr_m == "adagrad":
                optimizer = tf.train.AdagradOptimizer(lr)
            elif _lr_m == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

          # GRADIENT CLIPPING
            if clip > 0: # gradient clipping, if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)

    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session.")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.max_checkpoints)

    def restore_session(self):
        """restores from checkpoint"""

        if hasattr(self.args, "checkpoint_model_num") and self.args.checkpoint_model_num is not None:
            checkpoint_model_num = self.args.checkpoint_model_num
            checkpoint_path = self.args.checkpoints_folder/"model-{}".format(self.args.checkpoint_model_num)
        else:
            checkpoint_model_num, checkpoint_path = self.my_latest_checkpoint(self.args.checkpoints_folder)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(var_list=self.checkpoint_variables(), max_to_keep=self.args.max_checkpoints)
        self.saver.restore(self.sess, str(checkpoint_path))
        self.init_embeddings()
        print("Checkpoint {} loaded.".format(checkpoint_path))
        return checkpoint_model_num

    def my_latest_checkpoint(self, folder_path):
        files = folder_path.glob("model*meta") # e.g. model-9.meta
        max_epoch = max([int(f.name[len("model-"):-len(".meta")]) for f in files])
        return max_epoch, folder_path/"model-{}".format(max_epoch)

    def save_session(self, eval_cnt):
        """Saves session = weights"""
        if not self.args.checkpoints_folder.exists():
            os.makedirs(self.args.checkpoints_folder)
        print("Saving session checkpoint.")
        checkpoint_prefix = self.args.checkpoints_folder/"model"
        save_path = self.saver.save(self.sess, checkpoint_prefix, global_step=eval_cnt)
        print("Checkpoint saved in file: %s" % save_path)

    def close_session(self):
        """Closes the session"""
        self.sess.close()

    def checkpoint_variables(self):
        """
           omit entity embeddings from being stored
           in checkpoint in order to save disk space
        """
        omit_variables = []
        if not self.args.train_ent_vecs:
            omit_variables.append("entities/_entity_embeddings:0")
        variables = [n for n in tf.global_variables() if n.name not in omit_variables]
        #print("Checkpoint variables to restore:", variables)
        return variables

    def find_variable_handler_by_name(self, var_name):
        for n in tf.global_variables():
            if n.name == var_name:
                return n
