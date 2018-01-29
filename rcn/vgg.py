import torch.nn as nn
from torchvision.models import VGG
from .rcn import *
from itertools import chain


__all__ = ['vgg_gru_cell', 'VGGGRU']


def vgg_gru_cell(vgg_model, layer_idx, new_cells, keep_classifier=True):
    if not isinstance(vgg_model, VGG):
        raise ValueError("vgg_model is not an instance of Class VGG")
    if len(layer_idx) != len(new_cells):
        raise ValueError("the legth of layer_idx does not match the lenth of new_cells")

    i, j = 0, 0
    sequence = []
    while j < len(vgg_model.features):
        layer = vgg_model.features[j]
        if isinstance(layer, nn.Conv2d) and i in layer_idx:
            new_cell = new_cells[layer_idx.index(i)]
            if new_cell == 'conv':
                new_cell = ConvGRURCNCell(layer.out_channels, layer.kernel_size, layer.in_channels)
            elif new_cell == 'bottleneck':
                new_cell = BottleneckGRURCNCell(layer.out_channels, layer.in_channels, batch_norm=False)
            elif new_cell == 'vanilla':
                new_cell = BottleneckRCNCell(layer.out_channels, layer.in_channels, batch_norm=False)
            if not isinstance(new_cell, RCNCell):
                raise ValueError("not an RCNCell")
            sequence.append(new_cell)
            j += 1
            if isinstance(vgg_model.features[j], nn.BatchNorm2d):
                j += 1
        else:
            sequence.append(layer)
        if isinstance(layer, nn.Conv2d):
            i += 1
        j += 1

    cell = None
    if keep_classifier:
        cell = StackedRCNCell(sequence + [vgg_model.classifier,])
    else:
        cell = StackedRCNCell(sequence)
    return cell


class VGGGRU(nn.Module):

    def __init__(self, model, modify_layers, n_classes, only_last=False, dropout=0):
        super(VGGGRU, self).__init__()
        self._n_modified = len(modify_layers)
        self._outputmode = 'last' if only_last else 'average'
        self._cell = vgg_gru_cell(model, modify_layers, ['vanilla'] * self._n_modified, 
                                  keep_classifier=False)
        self._rcn = RCN(self._cell)
        self.dropout = None
        if dropout != 0:
            self._dropout = nn.Dropout(p=dropout)
        
        self._classifier = model.classifier
        in_features = self._classifier._modules['6'].in_features
        self._fc_changed = False
        if self._classifier._modules['6'].out_features != n_classes:
            self._classifier._modules['6'] = nn.Linear(in_features, n_classes)
            self._fc_changed = True

    def modified_parameters(self):
        params = []
        for layer in self._cell._modules.values():
            if isinstance(layer, RCNCell):
                params.append(layer.parameters())
        if self._fc_changed:
            params.append(self._classifier._modules['6'].parameters())
        return chain(*params)

    def forward(self, x):
        h0 = [None,] * self._n_modified
        out, _ = self._rcn(x, h0, output=self._outputmode)
        out = out.view(out.size(0), -1)
        if self.dropout:
            out = self._dropout(out)
        out = self._classifier(out)
        return out