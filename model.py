from tensorflow.python.framework.ops import reset_default_graph
import tensorflow as tf
from utils import read_test_images, check_and_create_dir, tf_ms_ssim, input_pipeline
import os
import numpy as np
import cv2
from scipy.misc import imsave

class AELikeModel:
    """AE-like Model with Pooling as a Size-changing Factor"""
    def __init__(self, image_size, alpha, trained_model=None):
        reset_default_graph()
        self.image_size = image_size
        self.alpha = alpha
        self.X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        self.Y_clear = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1])
        X_tensor = tf.reshape(self.X, [-1, self.image_size, self.image_size, 1])

        n_filters = [16, 32, 64]
        filter_sizes = [5, 5, 5]

        current_input = X_tensor
        n_input = 1

        Ws = []
        shapes = []

        for layer_i, n_output in enumerate(n_filters):
            with tf.variable_scope("encoder/layer/{}".format(layer_i)):
                shapes.append(current_input.get_shape().as_list())
                W = tf.get_variable(
                    name='W',
                    shape=[
                        filter_sizes[layer_i],
                        filter_sizes[layer_i],
                        n_input,
                        n_output],
                    initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
                h = tf.nn.conv2d(current_input, W,
                    strides=[1, 1, 1, 1], padding='SAME')
                conv = tf.nn.relu(h)
                current_input = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], padding='SAME')
                Ws.append(W)
                n_input = n_output
        Ws.reverse()
        shapes.reverse()
        n_filters.reverse()
        n_filters = n_filters[1:] + [1]

        for layer_i, shape in enumerate(shapes):
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                W = Ws[layer_i]
                h = tf.nn.conv2d_transpose(current_input, W,
                    tf.stack([tf.shape(self.X)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, 2, 2, 1], padding='SAME')
                current_input = tf.nn.relu(h)

        self.Y = current_input

        # MSE
        self.cost_2 = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(self.Y_clear, self.Y), 1))
        self.cost = 1 - tf_ms_ssim(self.Y_clear, self.Y)

        # Using Adam for optimizer
        self.learning_rate = tf.Variable(initial_value=1e-2, trainable=False, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.alpha*self.cost + (1 - self.alpha)*self.cost_2)
        self.batch_size = tf.Variable(initial_value=64, trainable=False, dtype=tf.int32)
        self.trained_model = trained_model

    def calculate_loss_on_test(self, sess):
        """
        Calculate loss value on test images with current model
        """
        loss = []
        loss_2 = []
        for j in range(1, self.test_X_images.shape[0], 1):
            [a, b] = sess.run([self.cost, self.cost_2], feed_dict={self.X: self.test_X_images[(j-1):j], self.Y_clear: self.test_Y_images[(j-1):j]})
            loss += [a]
            loss_2 += [b]
        current_loss = np.mean(loss)
        current_loss_2 = np.mean(loss_2)
        current_loss_val = self.alpha*current_loss + (1 - self.alpha)*current_loss_2
        return (current_loss, current_loss_2, current_loss_val)

    def init_session(self):
        """
        Init session
        """
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if not self.trained_model is None:
            saver.restore(sess, self.trained_model)
        return (sess,saver)

    def train(self, x_path_pattern, y_path_pattern, queue_capacity, capacity, min_after_dequeue, num_threads, test_X_path_dir, test_Y_path_dir, epochs, train_steps, learning_rate, epochs_to_reduce_lr, reduce_lr, output_dir, b_size):
        """
        Train data
        """

        # Load data
        x_filenames = tf.train.match_filenames_once(x_path_pattern)
        y_filenames = tf.train.match_filenames_once(y_path_pattern)
        batch = input_pipeline(x_filenames, y_filenames, self.batch_size, self.image_size, queue_capacity, capacity, min_after_dequeue, num_threads)

        # Load test data
        self.test_X_images = np.array(read_test_images(test_X_path_dir, self.image_size))
        self.test_Y_images = np.array(read_test_images(test_Y_path_dir, self.image_size))

        sess, saver = self.init_session()

        check_and_create_dir(output_dir)

        l_rate = learning_rate
        session_path = 'session_0_{}_{}'.format(epochs, l_rate)

        # Clear epoch and best weights info
        epoch_path = 'epoch_info.txt'
        best_weight_path = 'best_weight.txt'
        with open(os.path.join(output_dir, epoch_path), 'w') as f:
            f.write('')
        with open(os.path.join(output_dir, best_weight_path), 'w') as f:
            f.write('')

        best_loss = None
        best_loss_2 = None
        try:
            for epoch_i in range(epochs):
                if ((epoch_i + 1) % epochs_to_reduce_lr) == 0:
                    l_rate = l_rate * (1 - reduce_lr)
                session_path = 'session_{}_{}_{}'.format(epoch_i, epochs, l_rate)
                print("------------ Epoch : ",epoch_i+1)
                print("Current learning rate {}".format(l_rate))
                current_loss = 0
                current_loss_2 = 0
                current_loss_val = 0

                # Traing steps
                for i in range(train_steps):
                    [x_batch, y_batch] = sess.run(batch, feed_dict={self.batch_size: b_size})

                    sess.run(self.optimizer, feed_dict={ self.X: x_batch, self.Y_clear: y_batch, self.learning_rate: l_rate, self.batch_size: b_size })
                    current_loss, current_loss_2, current_loss_val = self.calculate_loss_on_test(sess)
                    print('{}/{} - loss MS-SSIM: {} - loss MSE: {} - mixed loss: {}'.format(i+1, train_steps, current_loss, current_loss_2, current_loss_val))

                # Calculate loss value
                best_loss_val = None
                if not best_loss is None:
                    best_loss_val = self.alpha*best_loss + (1 - self.alpha)*best_loss_2
                print("Mixed loss value {}".format(current_loss_val))

                # Update best weight
                if best_loss_val is None or current_loss_val < best_loss_val:
                    print("Improve loss value from {} to {}".format(best_loss_val, current_loss_val))
                    best_loss = current_loss_val
                    best_loss_2 = current_loss_2
                    try:
                        with open(os.path.join(output_dir, best_weight_path), "a") as best_epoch_file:
                            best_epoch_file.write(str(epoch_i)+'\n')
                    except:
                        print("Unexpected error:", sys.exc_info()[0])

                # Update epoch info
                epoch_info = '{} {}\n'.format(epoch_i + 1,[current_loss, current_loss_2])
                try:
                    with open(os.path.join(output_dir, epoch_path), "a") as epoch_file:
                        epoch_file.write(epoch_info)
                except:
                    print("Unexpected error:", sys.exc_info()[0])

                print(epoch_info)
                saver.save(sess, os.path.join(output_dir, session_path))
        except KeyboardInterrupt:
            saver.save(sess, os.path.join(output_dir, session_path))

    def test(self, input_dir, output_dir, need_invert=False):
        """
        Test images
        """
        check_and_create_dir(output_dir)
        sess, saver = self.init_session()
        for dirName, subdirList, fileList in os.walk(input_dir):
            for filename in fileList:
                image_path = os.path.join(dirName,filename)
                ds = None
                try:
                    ds = cv2.imread(image_path)
                    ds = cv2.cvtColor(ds, cv2.COLOR_BGR2GRAY)
                    if need_invert:
                        ds = cv2.bitwise_not(ds)
                    ds = cv2.resize(ds,(self.image_size, self.image_size))
                    ds = np.reshape(ds, (1, self.image_size, self.image_size, 1))
                except:
                    print("Cannot test image {}".format(image_path))
                    continue
                encoded_image = sess.run(self.Y, feed_dict={self.X: ds})
                encoded_image = encoded_image.reshape((self.image_size, self.image_size))
                imsave(os.path.join(output_dir, 'encoded_'+ filename), encoded_image)
                print("Done encode image {}".format(filename))
