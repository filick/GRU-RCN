import torch
from torch.utils.data import DataLoader
from rcn.vgg import VGGGRU
from torchvision.models.vgg import *
import torchvision.transforms as transforms
from data import UCF101Folder, FixedFrameSelector


# gloabl setting
use_gpu = torch.cuda.is_available()
mode = 'train'
batch_size = 1
seq_len = 5

# model
base_model = vgg11_bn(pretrained=False)
modify_layers = [1, 4, 7]
model = VGGGRU(base_model, modify_layers, 101)

# data loader
selector = FixedFrameSelector(seq_len)
data_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
dataset = UCF101Folder('/Users/filick/projects/video/data/UCF-101',
                       '/Users/filick/projects/video/data/ucfTrainTestlist',
                       mode, selector, transform=data_trans)
dataloader = DataLoader(dataset, batch_size, True, num_workers=2, pin_memory=use_gpu)

