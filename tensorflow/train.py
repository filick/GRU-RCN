import tensorflow as tf
from small_model import RcnVgg16
from input import VideoInput
import numpy as np

SEQ_LEN = 10
BATCH_SIZE = 20

video_input = VideoInput("/home/filick/workspace/VideoClassification/UCF-101")
# video_input.select_sub_collection(10)
# video_input.grouping(0.7, 0.15, 0.15)
# video_input.save("backup/data2.txt")
video_input.load("backup/data2.txt")
print(video_input.selected_classes, len(video_input.group["train"]))

def dataset(group, buffer, epos):
    while True:
        data, seq, y = video_input.get_data(group, buffer, SEQ_LEN, random_mode=True, size=(224, 224))
        for i in range(int(epos * buffer / BATCH_SIZE)):
            choices = np.random.choice(range(y.size), BATCH_SIZE, False)
            yield data[:, choices, :], seq[choices], y[choices]

train_data = dataset("train", 1000, 2)
validation_data = dataset("validation", 100, 20)

video_data = tf.placeholder(tf.float32, [SEQ_LEN, None, 224, 224, 3], name="input_data")
seq_len = tf.placeholder(tf.int32, [None,], name="input_seqlen")
label = tf.placeholder(tf.int32, [None,], name="ground_truth")
train_mode = tf.placeholder(tf.bool, name="train_mode")

rcn_vgg16 = RcnVgg16(video_data, seq_len, label, train_mode)

accuracy = rcn_vgg16.accuracy
train_step = tf.train.AdamOptimizer(1e-4).minimize(rcn_vgg16.error)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    i = 0

    while True:
        train, train_seq, train_y = next(train_data)

        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={video_data: train, seq_len:train_seq, label: train_y, train_mode: False})
            print("step %d, training accuracy %g"%(i, train_accuracy))

            valid, valid_seq, valid_y = next(validation_data)
            valid_accuracy = accuracy.eval(feed_dict={video_data: valid, seq_len:valid_seq, label: valid_y, train_mode: False})
            print("step %d, validation accuracy %g"%(i, valid_accuracy))

            saver.save(sess, "backup/small_model2.ckpt")
            print("saved!")

        train_step.run(feed_dict={video_data: train, seq_len:train_seq, label: train_y, train_mode: True})

        i = i + 1
