'''
方案调研：
    视频处理（用pims等直接从视频中读取相应的帧，速度会不会慢？好处是不用预先确定截取哪些帧，更灵活）
        http://scikit-image.org/docs/dev/user_guide/video.html
        http://soft-matter.github.io/pims/v0.4/video.html
    先将视频转成图片，然后用tf.data.Dataset (用这种，因为好像目前的光流法啥的程序也是拿图片输入的，如flownet2.0；另一方面，直接处理视频效率未知啊)
        https://github.com/kratzert/finetune_alexnet_with_tensorflow
        https://zhuanlan.zhihu.com/p/30751039
        https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
    denseflow是直接拿视频做输入的，输出是rgb图像，flow x, flow y
        https://github.com/bryanyzhu/two-stream-pytorch
    
目前方案:
    懒得转tfrecord了，直接pims + tf.data.dataset (写出来和PyTorch DataLoader+dataset的风格差不多)
        https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/finetune.py
        https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/datagenerator.py
        https://www.tensorflow.org/api_docs/python/tf/data/Dataset （并行）
    tf读取数据总结
        https://zhuanlan.zhihu.com/p/30751039
'''


import tensorflow as tf
import pims 
import pandas as pd
import sys
import sklearn.utils as sku
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor
import numpy as np

class TrainTestSplit(object):
    def __init__(self, path = '../../data/ucfTrainTestlist/', split_id = None):
        self.path = path
        self.split_id = split_id
        
    def readlist(self):
        list1 = pd.read_csv(self.path + 'trainlist' + self.split_id + '.txt',
                            sep = " ",
                            header = None)
        list1.columns = ["dir_file", "label"]
        
        list2 = pd.read_csv(self.path + 'testlist' + self.split_id + '.txt',
                            sep = " ",
                            header = None)
        list2.columns = ["dir_file"]
        
        classInd = pd.read_csv(self.path + 'classInd.txt',
                            sep = " ",
                            header = None)
        classInd.columns = ["label", "dir"]
        
        list2[['dir', 'file']] = list2['dir_file'].str.split('/', expand = True) 
        list2 = pd.merge(list2, classInd, how = 'left', left_on = 'dir', right_on = 'dir')
        list2.drop(['dir', 'file'], inplace = True, axis = 1)
        
        list1['label'] -= 1
        list2['label'] -= 1
        print('read train/test split')
        return list1, list2
        
        
class DataLoader(object):
    """Wrapper for tf.data.dataset, behaves like PyTorch DataLoader + dataset in spirit.
    
        return:
            tf.data.dataset, where
                image_sequence: 5D or 6D Tensor
                    in shape "batch x TTA_zoom x num_frames x height x weight x channel".
                    E.g. TTA_zoom = 10 if ten_crop, = 2 if flip, default value is 1 (5D).
    """
    
    def __init__(self, root= '../../data/UCF-101/', videolist=None, batch_size=2, num_frames=3, 
                 transforms=None,
                 shuffle=False, buffer_size=100, 
                 num_parallel_calls=8):
        """
        Args:
            videolist: list1 or list2 returned by TrainTestSplit.readlist. 
            num_frames: Number of frames extracted per video in the videolist.
            transforms: Name of transforms.
            
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            
            num_parallel_calls: The number of videos to process in parallel.
                
        Note:
            Now we assume num_frames is the same for all the videos, and image sequence
                is uniformly distributed along the time dimesnion.
            For many epochs, we can use repeat(), or we can use a reinitializable iterator to save space, 
                e.g. https://github.com/kratzert/finetune_alexnet_with_tensorflow/blob/master/finetune.py
        """
        self.root = root
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.transforms = transforms
        self.num_parallel_calls = num_parallel_calls
   
        # initial shuffling of the videos and labels 
        if self.shuffle:
            self.videolist = sku.shuffle(videolist)
            self.videolist.reset_index(drop = True, inplace = True)
        else:
            self.videolist = videolist
            
        self.video_paths = (root + self.videolist['dir_file']).tolist()
        self.labels = self.videolist['label'].tolist()
        self.video_paths = convert_to_tensor(self.video_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)
        
        self.data = self.get_data()
    
    def get_data(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.video_paths, self.labels))  
        
        # pims do not support tensor string input, so use tf.py_func here
        dataset = dataset.map(lambda video_path, label: tuple(tf.py_func(
                self._parse_video, [video_path, label], [tf.uint8, label.dtype]
                )),
                num_parallel_calls=self.num_parallel_calls)
        dataset = dataset.map(self._transform_images, num_parallel_calls=self.num_parallel_calls)
        
#        if self.shuffle:
#            dataset = dataset.shuffle(buffer_size = self.buffer_size) # seems to be slow compared with sku.shuffle
        
        dataset = dataset.batch(self.batch_size)
        return dataset
        
    def _parse_video(self, video_path, label):
        """Extract image sequence from video.
        
        return:
            images_per_video: 4D ndarray
              in shape "num_frames x height x weight x channel".
    
        Note:
            Now we assume num_frames is the same for all the videos, and image sequence
                is uniformly distributed along the time dimesnion.
        """
        
#        print(type(video_path))  <class 'bytes'>
#        print(video_path) b'../../data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c04.avi'
        v = pims.Video(video_path.decode()) 
        length = len(v)
        step = np.floor(length / self.num_frames)
        self.sample_index =  [np.random.randint(i*step, (i+1)*step) for i in range(self.num_frames)]
        samples = [v[i] for i in self.sample_index]
        images_per_video = np.stack(samples)
                
        return images_per_video, label
    
    def _transform_images(self, images_per_video, label):
        """
        Interface for various transforms:
            New insight: 表观的色彩/纹理/光照: https://zhuanlan.zhihu.com/p/32443212.
            Two stream: random cropping and horizontal flipping.
            TSN: corner cropping and scale jittering.
        
        Args:
            images_per_video: 4D Tensor
              in shape "num_frames x height x weight x channel".
    
        Note:
            Some tf ops support 4D image, however, seems that slim preprocessing only supports 3D. So
                we will implement transforms for 4D Tensor based on tf ops.
        """
        
        if self.transforms is None:
            pass
        else:
            """Call whatever transform functions here"""
            IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)    

            imgs_resized = tf.image.resize_images(images_per_video, [224, 224])
            images_per_video = tf.subtract(imgs_resized, IMAGENET_MEAN)
        
        return images_per_video, label
    

    
if __name__ == "__main__":

    v = pims.Video('../../data/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
    print(sys.getsizeof(v)) #bytes
    
    split_obj = TrainTestSplit(split_id = '01')
    trainlist, testlist = split_obj.readlist()
    
    with tf.device('/cpu:0'):
        tr_data = DataLoader(videolist = trainlist, batch_size = 2, num_frames = 3, shuffle = True)
        
        iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
        next_batch = iterator.get_next()
    
    training_init_op = iterator.make_initializer(tr_data.data)
    
    with tf.Session() as sess:
        sess.run(training_init_op)
        img_batch, label_batch = sess.run(next_batch)
    
#    image_sequence: 5D or 6D Tensor
#    in shape "batch x TTA_zoom x num_frames x height x weight x channel".
#    E.g. TTA_zoom = 10 if ten_crop, = 2 if flip, default value is 1 (5D).
    print(type(img_batch))
    print(img_batch.shape)
    print(label_batch)
    
    import matplotlib.pyplot as plt
    
    def imshow(inp, title=None):
        #inp = inp.numpy().transpose((1, 2, 0))
        #inp = np.clip(inp, 0, 1)
        plt.figure()
        plt.imshow(inp)
    
    # Make a grid from batch
    
    imshow(img_batch[0,0,:,:,:])
    imshow(img_batch[0,1,:,:,:])
    imshow(img_batch[0,2,:,:,:])
    
    imshow(img_batch[1,0,:,:,:])
    imshow(img_batch[1,1,:,:,:])
    imshow(img_batch[1,2,:,:,:])

        
    
    