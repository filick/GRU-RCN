import tensorflow as tf
from small_model import RcnVgg16
from input import VideoInput
import numpy as np

SEQ_LEN = 5
BATCH_SIZE = 50

video_input = VideoInput("/home/filick/workspace/VideoClassification/UCF-101")
video_input.select_sub_collection(10)
video_input.grouping(0.7, 0.15, 0.15)
video_input.save("backup/data.txt")
train_data = video_input.get_data("train", "all", SEQ_LEN, secondes=1, size=(224, 224))
validation_data = video_input.get_data("validataion", "all", SEQ_LEN, secondes=1, size=(224, 224))

def next_batch(dataset, batch=BATCH_SIZE):
    data, seq, y = dataset
    choices = np.random.choice(range(y.size), batch, False)
    return data[:, choices, :], seq[choices], y[choices]

saver = tf.train.Saver()

with tf.Session() as sess:

    video_data = tf.placeholder(tf.float32, [SEQ_LEN, None, 224, 224, 3], name="input_data")
    seq_len = tf.placeholder(tf.int32, [None,], name="input_seqlen")
    label = tf.placeholder(tf.int32, [None,], name="ground_truth")
    train_mode = tf.placeholder(tf.bool, name="train_mode")

    rcn_vgg16 = RcnVgg16(video_data, seq_len, label, train_mode)

    accuracy = rcn_vgg16.accuracy
    train_step = tf.train.AdamOptimizer(1e-4).minimize(rcn_vgg16.error)

    sess.run(tf.global_variables_initializer())
    # predict = sess.run(rcn_vgg16.pool4, feed_dict={video_data: data, seq_len:seq, label: y, train_mode: True})
    # print(predict.shape)
    i = 0

    while True:
        train, train_seq, train_y = next_batch(train_data, BATCH_SIZE)

        if i%10 == 0:
            train_accuracy = accuracy.eval(feed_dict={video_data: train, seq_len:train_seq, label: train_y, train_mode: False})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            valid, valid_seq, valid_y = next_batch(validation_data, 50)
            valid_accuracy = accuracy.eval(feed_dict={video_data: valid, seq_len:valid_seq, label: valid_y, train_mode: False})
            print("step %d, validation accuracy %g"%(i, valid_accuracy))

            saver.save(sess, "backup/model.ckpt")
            print("saved!")

        train_step.run(feed_dict={video_data: data_batch, seq_len:seq_batch, label: y_batch, train_mode: True})

