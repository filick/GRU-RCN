import rcn
import torch
import torch.nn as nn
import torchvision


__all__ = ['Gru_ResNet', 'gru_resnet']


class Gru_ResNet(nn.Module):

    def __init__(self, base_model=None, ConvGRULayers=None, BottleneckGRULayers=None, num_classes=None):
        super(Gru_ResNet, self).__init__()
        self.base_model = base_model
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.base_model.fc=None # useless now, might save some memory
        
        self.add_ConvGRURCNCell(ConvGRULayers)
        self.add_BottleneckGRURCNCell(BottleneckGRULayers)
        
        self.ConvGRULayers = ConvGRULayers
        self.BottleneckGRULayers = BottleneckGRULayers

    def forward(self, inputs):
        """
        inputs: num_frames x batch_size x channel x height x weight
        """
        
        for i in range(inputs.size(0)):
            x = self.base_model.conv1(inputs[i,:,:,:,:])
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
    
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
    
            x = self.base_model.avgpool(x)
                        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        self.reset_count()
        return x
    
    def reset_count(self):
        if isinstance(self.ConvGRULayers,dict):
            for key in self.ConvGRULayers.keys():
                item = self.ConvGRULayers[key]
                
                if key is 'none':
                    self.base_model._modules['conv1'].reset()
                else:
                    for inner_list in item:
                        if inner_list[1] is 'downsample':
                            self.base_model._modules[key]._modules[inner_list[0]] \
                                ._modules[inner_list[1]]._modules['0'].reset()
                        else:
                            self.base_model._modules[key]._modules[inner_list[0]] \
                                ._modules[inner_list[1]].reset()                            
        
        if isinstance(self.BottleneckGRULayers,dict):            
            for key in self.BottleneckGRULayers.keys():
                item = self.BottleneckGRULayers[key]
            
                for inner in item:
                    self.base_model._modules[key]._modules[inner].reset()
    

    def add_ConvGRURCNCell(self, layers):
        if isinstance(layers,dict):
            for key in layers.keys():
                item = layers[key]
                if key != 'none' and len(item[0]) != 2:
                    raise ValueError("Check GRULayers setting")
                
                if key is 'none':
                    current = self.base_model._modules['conv1']
                    self.base_model._modules['conv1'] = rcn.ConvGRURCNCell(
                                            channels = current.out_channels, 
                                            kernel_size = current.kernel_size, # may be any number you like
                                            x_channels = current.in_channels,
                                            x_kernel_size = current.kernel_size, 
                                            x_stride = current.stride, 
                                            x_padding = current.padding)
                else:
                    for inner_list in item:
                        if inner_list[1] is 'downsample':
                            current = self.base_model._modules[key]._modules[inner_list[0]] \
                                    ._modules[inner_list[1]]._modules['0']
                            self.base_model._modules[key]._modules[inner_list[0]] \
                                ._modules[inner_list[1]]._modules['0'] = rcn.ConvGRURCNCell(
                                                channels = current.out_channels, 
                                                kernel_size = current.kernel_size, # may be any number you like
                                                x_channels = current.in_channels,
                                                x_kernel_size = current.kernel_size, 
                                                x_stride = current.stride, 
                                                x_padding = current.padding)                            
                        else:
                            current = self.base_model._modules[key]._modules[inner_list[0]] \
                                    ._modules[inner_list[1]]
                            self.base_model._modules[key]._modules[inner_list[0]] \
                                ._modules[inner_list[1]] = rcn.ConvGRURCNCell(
                                                channels = current.out_channels, 
                                                kernel_size = current.kernel_size, # may be any number you like
                                                x_channels = current.in_channels,
                                                x_kernel_size = current.kernel_size, 
                                                x_stride = current.stride, 
                                                x_padding = current.padding)
            
    def add_BottleneckGRURCNCell(self, layers):
        if isinstance(layers,dict):
            for key in layers.keys():
                item = layers[key]
                for inner in item:
                    current = self.base_model._modules[key]._modules[inner]
                    self.base_model._modules[key]._modules[inner] = rcn.BottleneckGRURCNCell(
                                        channels = current._modules['conv3'].out_channels, 
                                        x_channels = current._modules['conv1'].in_channels,
                                        x_stride = current._modules['conv2'].stride)

def gru_resnet(base_model, ConvGRULayers, BottleneckGRULayers, num_classes):
    
    model = Gru_ResNet(base_model, ConvGRULayers, BottleneckGRULayers, num_classes)
    return model



if __name__ == "__main__":
    
#    gru_resnet50 = gru_resnet(torchvision.models.resnet.resnet50(pretrained=True), ConvGRULayers=None, BottleneckGRULayers=None)
#    model = gru_resnet50.base_model
#        
#    for name, module in model.named_children():
#        print(name, '->', module)
#    print(model.layer4[1].conv1)
    
    
    ConvGRULayers={'none': ['conv1'], 
                   'layer1': [
                           ['0','conv1'], ['2','conv2'], ['0', 'downsample']
                           ]}
    BottleneckGRULayers={'layer2': ['1','2'],
                         'layer4': ['0','2']
                        }
    gru_resnet50 = gru_resnet(torchvision.models.resnet.resnet50(pretrained=True), 
                              ConvGRULayers, 
                              BottleneckGRULayers, 
                              num_classes = 101)
    
    print(gru_resnet50)
    
    from torch.autograd import Variable
    x = Variable(torch.rand(4, 2, 3, 224, 224))
    y = gru_resnet50(x)
    print(y.size())