import os
import os.path
import random
from numpy import *
import cv2

class VideoInput:
    path=None          #根目录
    dirnames=[]        #每个子文件夹的名称
    filenames=[]       #每个文件的名称
    sep=[]
    gro={}
    def __init__(self, path):
        self.path=path
        num=-1
        folder=os.listdir(path)
        self.sep.append(num)
        for f in folder:
           add=os.path.join(path,f)
           self.dirnames.append(f)
           files=os.listdir(add)
           num+=len(files)
           self.sep.append(num)
           for file in files:
             self.filenames.append(file)
        print(self.dirnames,self.filenames,self.sep)

    def group(self, train_rate, validation_rate, test_rate):
        self.gro={0:{},1:{},2:{}}
        for r in range(len(self.sep)-1):
            result=[]
            counter=[0,0,0]
            size = self.sep[r+1]-self.sep[r]
            Count = [int(size * train_rate), int(size * validation_rate), int(size * test_rate)]
            remain = size - sum(Count)
            if remain > 0:
                for i in range(3):
                    if Count[i] > 0:
                        Count[i] += remain
                        break
            for i in range(3):
                no=[]
                self.gro[i].setdefault(r)
                while(counter[i]<Count[i]):
                    tempInt=random.randint(self.sep[r],self.sep[r+1])
                    if(tempInt not in result):
                        result.append(tempInt)
                        no.append(tempInt)
                        counter[i]+=1
                self.gro[i][r]=no
        print(self.gro)

    def get_data(self, group, batch, seq_length=None, sampling_method=None):
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
        return (data, flength, nogro)    #data是五维ndarry，第二维帧数，第三维宽，第四维高，第五维按BGR，flength是每个文件读取的帧数，nogro是每个文件属于哪个类