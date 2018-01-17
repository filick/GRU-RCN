import torch.nn as nn
from torchvision.models import VGG
from .rcn import *


def vgg_gru_cell(vgg_model, layer_idx, new_cells, keep_classifier=True):
    if not isinstance(vgg_model, VGG):
        raise ValueError("vgg_model is not an instance of Class VGG")
    if len(layer_idx) != len(new_cells):
        raise ValueError("the legth of layer_idx does not match the lenth of new_cells")

    cell = None
    if keep_classifier:
        cell = ModifiedRCNCell(vgg_model)
    else:
        cell = ModifiedRCNCell(vgg_model.features)
    
    i = 0
    for key in vgg_model.features.keys():
        layer = vgg_model.features._modules[key]
        if isinstance(layer, nn.Conv2d):
            if i in layer_idx:
                new_cell = new_cells[layer_idx.index(i)]
                if new_cell == 'conv':
                    new_cell = ConvGRURCNCell(layer.out_channels, layer.kernel_size, layer.in_channels)
                elif new_cell == 'bottleneck':
                    new_cell = BottleneckGRURCNCell(layer.out_channels, layer.in_channels)
                if not isinstance(new_cell, RCNCell):
                    raise ValueError("not an RCNCell")

                if keep_classifier:
                    cell.modify(('features', key), new_cell)
                else:
                    cell.modify(key, new_cell)
            i += 1
    return cell