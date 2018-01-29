import torch
import torch.utils.data as data
from torchvision import transforms

import pims 
import pandas as pd
import numpy as np

class TrainTestSplit(object):
    def __init__(self, path = 'data/ucfTrainTestlist/', split_id = None):
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
        
        
class VideoDataset(data.Dataset):
    """
        _getitem__: 4D Tensor
            in shape "num_frames x channel x height x weight "
    """
    
    def __init__(self, root= 'data/UCF-101/', 
                 videolist=None,  
                 num_frames=3, 
                 transform=None
                 ):
        """
        Args:
            videolist: list1 or list2 returned by TrainTestSplit.readlist. 
            num_frames: Number of frames extracted per video in the videolist.
            transform: Name of transforms.

        Note:
            Now we assume num_frames is the same for all the videos, and image sequence
                is uniformly distributed along the time dimesnion.
        """
        self.root = root
        self.videolist = videolist
        self.num_frames = num_frames
        self.transform = transform
        
        self.height = 240
        self.width = 320
            
        self.video_paths = (root + self.videolist['dir_file'])
        self.labels = self.videolist['label']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self._parse_video(self.video_paths[idx]), self.labels[idx]
        
    def _parse_video(self, video_path):
        """Extract image sequence from video.
        
        return:
            images_per_video: 4D ndarray
              in shape "num_frames x height x weight x channel",
              or 4D tensor in shape "num_frames x channel x height x weight ".
    
        Note:
            Now we assume num_frames is the same for all the videos, and image sequence
                is uniformly distributed along the time dimesnion.
        """
        
        v = pims.Video(video_path) 
        length = len(v)
        step = np.floor(length / self.num_frames)
        self.sample_index =  [np.random.randint(i*step, (i+1)*step) for i in range(self.num_frames)]
        
        
        if self.transform:
            samples = [self.transform(v[i]) for i in self.sample_index]
        else:
            samples = [v[i] for i in self.sample_index]
        
        images_per_video = torch.stack(samples)
                
        return images_per_video
    

    
if __name__ == "__main__":
    
    split_obj = TrainTestSplit(split_id = '01')
    trainlist, testlist = split_obj.readlist()
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transformed_dataset = VideoDataset(videolist = trainlist, num_frames=4, transform=data_transforms)           
    dataloader = data.DataLoader(transformed_dataset, batch_size=2, shuffle=False, num_workers=2)    
    print(len(dataloader))
    
    
    img_batch, labels = next(iter(dataloader))
    print(type(img_batch))
    print(img_batch.shape)
    print(labels.shape)

        
    
    