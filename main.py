import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from moviepy.editor import VideoFileClip
import scipy
import numpy as np

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
  '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
  warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
  print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_boolean('is_training', True, 'Whether the model is training.')
flags.DEFINE_boolean('video_inference', False, 'If to do video inference.')
flags.DEFINE_string('summaries_dir', './logs', 'Where to save summaries.')
flags.DEFINE_string('models_dir', './models/', 'Where to save models.')

FLAGS = flags.FLAGS


def load_vgg(sess, vgg_path):
  """
  Load Pretrained VGG Model into TensorFlow.
  :param sess: TensorFlow Session
  :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
  :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
  """
  # TODO: Implement function
  #   Use tf.saved_model.loader.load to load the model and weights
  vgg_tag = 'vgg16'

  tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

  vgg_input_tensor_name = 'image_input:0'
  vgg_keep_prob_tensor_name = 'keep_prob:0'
  vgg_layer3_out_tensor_name = 'layer3_out:0'
  vgg_layer4_out_tensor_name = 'layer4_out:0'
  vgg_layer7_out_tensor_name = 'layer7_out:0'

  graph = tf.get_default_graph()
  input = graph.get_tensor_by_name(vgg_input_tensor_name)
  keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
  layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
  layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
  layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

  return input, keep_prob, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)


def project_1x1(input_tensor, num_outs, scale=None,
                init_fn=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer_fn=tf.contrib.layers.l2_regularizer(1e-3)):
  if scale:
    input_tensor = input_tensor * scale

  return tf.layers.conv2d(inputs=input_tensor,
                          filters=num_outs,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='same',
                          kernel_initializer=init_fn,
                          kernel_regularizer=regularizer_fn)


def skip_module(net, up, num_classes,
                init_fn=tf.truncated_normal_initializer(stddev=1e-3),
                regularizer_fn=tf.contrib.layers.l2_regularizer(1e-3)):
  # upsample head
  net = tf.layers.conv2d_transpose(inputs=net,
                                   filters=num_classes,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   padding='same',
                                   kernel_initializer=init_fn,
                                   kernel_regularizer=regularizer_fn)

  # add skip connection with upstream features
  return net + up


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, scale8=0.01, scale16=0.1):
  """
  Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
  :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
  :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
  :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
  :param num_classes: Number of classes to classify
  :return: The Tensor for the last layer of output
  """

  ''' apply 1x1 projections '''

  proj_1x1_layer3 = project_1x1(vgg_layer3_out, num_classes, scale8)
  proj_1x1_layer4 = project_1x1(vgg_layer4_out, num_classes, scale16)
  proj_1x1_layer7 = project_1x1(vgg_layer7_out, num_classes)

  '''skip connections for hourglass architecture'''
  net = proj_1x1_layer7
  net = skip_module(net, proj_1x1_layer4, num_classes)
  net = skip_module(net, proj_1x1_layer3, num_classes)

  # scale up by x8 to segment full size
  net = tf.layers.conv2d_transpose(inputs=net,
                                   filters=num_classes,
                                   kernel_size=(16, 16),
                                   strides=(8, 8),
                                   padding='same',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=1e-2),
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

  return net


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes, global_step=None):
  """
  Build the TensorFLow loss and optimizer operations.
  :param nn_last_layer: TF Tensor of the last layer in the neural network
  :param correct_label: TF Placeholder for the correct label image
  :param learning_rate: TF Placeholder for the learning rate
  :param num_classes: Number of classes to classify
  :return: Tuple of (logits, train_op, cross_entropy_loss)
  """
  # TODO: Implement function
  logits = tf.reshape(nn_last_layer, (-1, num_classes))
  correct_label = tf.reshape(correct_label, (-1, num_classes))

  cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                              labels=correct_label))
  # gather the regularization loss
  regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  regularization_loss = 0 if not regularization_losses else tf.add_n(regularization_losses)

  # compute the total loss
  total_loss = tf.add(cross_entropy_loss, regularization_loss)
  # total_loss = cross_entropy_loss

  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=global_step)

  return logits, optimizer, total_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate,
             augm_input_ops=None,
             extra_info=None,
             summary_op=None,
             summary_writer=None):
  """
  Train neural network and print out the loss during training.
  :param sess: TF Session
  :param epochs: Number of epochs
  :param batch_size: Batch size
  :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
  :param train_op: TF Operation to train the neural network
  :param cross_entropy_loss: TF Tensor for the amount of loss
  :param input_image: TF Placeholder for input images
  :param correct_label: TF Placeholder for label images
  :param keep_prob: TF Placeholder for dropout keep probability
  :param learning_rate: TF Placeholder for learning rate
  """
  # TODO: Implement function

  if augm_input_ops:
    (augm_image_op, augm_label_op, input_preprocess) = augm_input_ops

  if extra_info:
    (learning_rate, global_step) = extra_info

  for epoch in range(epochs):
    for batch, (image, label) in enumerate(get_batches_fn(batch_size)):

      if augm_input_ops:
        image, label = sess.run([augm_image_op, augm_label_op], feed_dict={
          input_preprocess: image,
          correct_label: label
        })

      print_buf=''
      if extra_info:
        step = tf.train.global_step(sess, global_step)
        lr = sess.run(learning_rate)
        print_buf = 'Step {}. LR {}. '.format(step, lr)


      feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5}

      if batch == 0 and summary_op is not None:
        print("Recording summary for epoch {}".format(epoch))
        summary, _, loss = sess.run([summary_op, train_op, cross_entropy_loss],
                                    feed_dict=feed_dict)
        summary_writer.add_summary(summary, step)
      else:
        _, loss = sess.run([train_op, cross_entropy_loss],
                           feed_dict=feed_dict)

      print('{}Epoch {}. Batch {}. Loss {}'.format(print_buf, epoch, batch, loss))



tests.test_train_nn(train_nn)


class Config(object):
  '''
    base configuration class
  '''
  BATCH_SIZE = 1


class TrainConfig(Config):
  EPOCHS = 30
  LEARNING_RATE = 1e-3
  BATCH_SIZE = 16
  LR_DECAY = 0.1
  DECAY_STEPS = 250


class EvalConfig(Config):
  BATCH_SIZE = 1


def add_preprocessing(image_input, label_input, is_training):
  def _maybe_flip(input_tensor, mirror_cond, scope):
    return tf.cond(
      mirror_cond,
      lambda: tf.image.flip_left_right(input_tensor),
      lambda: input_tensor,
      name=scope)

  def _preprocess_train(image, label):
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)

    image = _maybe_flip(image, mirror_cond, scope="random_flip_image")
    label = _maybe_flip(label, mirror_cond, scope="random_flip_label")

    return image, label

  def _preprocess_test(image, label):
    return image, label

  def _map(fn, arrays, dtypes):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtypes)
    return out

  mapper = \
    lambda img, label: tf.cond(
      tf.equal(
        is_training,
        tf.constant(True)),
      lambda: _preprocess_train(img, label),
      lambda: _preprocess_test(img, label))

  image_input, label_input = _map(mapper, [image_input, label_input],
                                  dtypes=(image_input.dtype, label_input.dtype))

  return image_input, label_input


def add_summaries(sess, total_loss, learning_rate):

  with tf.name_scope("summaries"):
    loss_summary_op = tf.summary.scalar('total_loss', total_loss)
    lr_summary_op = tf.summary.scalar('learning_rate', learning_rate)
    writer = tf.summary.FileWriter(FLAGS.summaries_dir, sess.graph)
    merged = tf.summary.merge_all()

    return merged, writer


def run():
  num_classes = 2
  image_shape = (160, 576)
  data_dir = './data'
  runs_dir = './runs'
  tests.test_for_kitti_dataset(data_dir)

  # Download pretrained vgg model
  helper.maybe_download_pretrained_vgg(data_dir)

  # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
  # You'll need a GPU with at least 10 teraFLOPS to train on.
  #  https://www.cityscapes-dataset.com/

  with tf.Session() as sess:
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')
    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # TODO: Build NN using load_vgg, layers, and optimize function

    correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
    input_image = tf.placeholder(tf.float32, [None, None, None, 3])

    input_image_vgg16, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
    net_output = layers(layer3_out, layer4_out, layer7_out, num_classes)

    # add augmentation (random flip) as TF ops
    augm_image_op, augm_label_op = add_preprocessing(input_image, correct_label, is_training=True)

    if FLAGS.is_training:
      config = TrainConfig()

      # global step
      global_step = tf.train.create_global_step()
      learning_rate = tf.train.exponential_decay(config.LEARNING_RATE, global_step,
                                                 config.DECAY_STEPS, config.LR_DECAY, staircase=True)

      logits, optimizer, total_loss = optimize(net_output, correct_label, learning_rate,
                                             num_classes, global_step=global_step)

    # TODO: Train NN using the train_nn function

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    # OPTIONAL: Apply the trained model to a video
    if FLAGS.video_inference:
      saver.restore(sess, './models/model')
      video_file = './videos/demo.mp4'
      out_file = './videos/segm.mp4'

      logits = tf.reshape(net_output, (-1, num_classes))

      def _process_frame(frame):
        ''' functional closure to access session and tensors '''
        image = scipy.misc.imresize(frame, image_shape)
        im_softmax = sess.run(
          [tf.nn.softmax(logits)],
          {keep_prob: 1.0, input_image_vgg16: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        return np.array(street_im)


      clip = VideoFileClip(video_file)
      video_clip = clip.fl_image(_process_frame)
      video_clip.write_videofile(out_file, audio=False)

      return

    summary_op = None
    if FLAGS.summaries_dir:
      if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)

      summary_op, summary_writer = add_summaries(sess, total_loss, learning_rate)


    if FLAGS.is_training:
      train_nn(sess, config.EPOCHS, config.BATCH_SIZE, get_batches_fn, optimizer, total_loss, input_image_vgg16,
               correct_label, keep_prob, learning_rate, (augm_image_op, augm_label_op, input_image),
               extra_info=(learning_rate, global_step), summary_op=summary_op, summary_writer=summary_writer)

      # TODO: Save inference data using helper.save_inference_samples
      helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image_vgg16)
      saver.save(sess, os.path.join(FLAGS.models_dir, 'model'))




if __name__ == '__main__':
  run()
