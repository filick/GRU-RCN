import os
import numpy as np
from moviepy.video.io.ffmpeg_reader import *


def _list_delete(a, b):
    c = []
    for item in a:
        if item not in b:
            c.append(item)
    return c


class VideoInput:

    VIDEO_HEIGHT = 240
    VIDEO_WIDTH = 320

    def __init__(self, path):
        self.root = path
        self.classes = os.listdir(path)
        self.files = []
        self.sep = [0]
        self.group = None

        for c in self.classes:
            sub_path = os.path.join(path, c)
            files = os.listdir(sub_path)
            self.files += files
            self.sep.append(self.sep[-1] + len(files))

    def grouping(self, train_rate, validation_rate, test_rate):
        self.group = {'train': [], 'validation': [], 'test':[]}
        for i in range(len(self.classes)):
            files = list(range(self.sep[i], self.sep[i+1]))
            train_size = int(len(files) * train_rate)
            validation_size = int(len(files) * validation_rate)
            train = list(np.random.choice(files, train_size, False))
            temp = _list_delete(files, train)
            validation = list(np.random.choice(temp, validation_size, False))
            test = _list_delete(temp, validation)
            if test_rate == 0 and len(test) != 0:
                for item in test:
                    train.append(item) if np.random.randint(2) else validation.append(item)
                test = []

            self.group['train'] += train
            self.group['validation'] += validation
            self.group['test'] += test

        return (len(self.group['train']), len(self.group['validation']), len(self.group['test']))

    def _max_seq_length(self, selected_files):
        pass

    def _read_video(self, video_path, start_frame, gap, length):
        pass

    def get_data(self, group, batch, seq_length=None, sampling_method=None):
        '''Get a batch of video data.

        Parameters:
        ---------------------
        group: str
               choice from "train", "validation" or "test".
        batch: int
               batch size.
        seq_length: int
               length of video frames.
               if not set (None), return all the frames in a video file.
        sampling_method: str
               only valid if `seq_length` is not None. choice from "continuous", "throughout".
               "continuous": randomly select continuous video fragment with `seq_length` frames.
               "throughout": selecte `seq_length` frames uniformly throughout the time range.
               if not set (None), randomely apply one of the two methods above. 

        Return:
        ---------------------
        data: 5D ndarray
              in shape "batch x max_seq_length x height x weight x channel"
        seq_len: 1D ndarray whith size = batch
              the length of video frames of each sample.
        label: 1D ndarray whith size = batch
              the class label of each sample.
        '''

        selected_files = np.ramdom.choice(self.group[group], batch, False)
        label = np.zeros(batch, dtype=np.unit8)
        for sep in self.sep[1:-1]:
            label[selected_files >= sep] += 1
        if seq_length == None:
            seq_length = self._max_seq_length(selected_files)
        data = np.zeros(shape=(batch, seq_length, self.VIDEO_HEIGHT, self.VIDEO_WIDTH, 3), dtype=np.uint8)
        for i in range(batch):
            video_file = selected_files[i]
            video_class = label[i]
            file_path = os.path.join(self.root, self.classes[video_class], self.files[video_file])
            
        data=ndarray([batch,seq_length,240,320,3],dtype=uint8)
        nogro=random.randint(len(self.dirnames),size=batch)
        flength=zeros(batch,dtype=uint8)
        j=0
        result=[]
        for n in nogro:
            arr=array(self.gro[group][n])
            item=random.choice(arr)
            while(item in result):
                item=random.choice(arr)
            result.append(item)
            filepath=self.path+'\\'+self.dirnames[n]+'\\'+self.filenames[item]
            cap=cv2.VideoCapture(filepath)
            npFrames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            if(npFrames<seq_length):
                length=npFrames
            else:
                length=seq_length
            flength[j]=length
            for f in range(length-1):
                _, dat = cap.read()
                dat = array(dat)
                data[j][f]=dat
            j+=1
            cap.release()
        return (data, flength, nogro)

''' test
if __name__ == "__main__":
    video_input = VideoInput("/mnt/CloudSat/Others/UCF-101/")
    result = video_input.grouping(0.7, 0.3, 0)
    print(result, sum(result), video_input.sep[-1])
    total = video_input.group['train'] + video_input.group['validation'] + video_input.group['test']
    total = np.array(sorted(total))
    check = total - np.arange(0, video_input.sep[-1])
    print((check != 0).sum())
'''