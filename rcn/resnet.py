import torch.nn as nn
import torchvision.models.resnet as res
from .rcn import *


__all__ = ['resnet_gru_cell', 'ResnetGRU']


def resnet_gru_cell(res_model, layer_idx, new_cells):
    if not isinstance(res_model, res.ResNet):
        raise ValueError("vgg_model is not an instance of ResNet")
    if len(layer_idx) != len(new_cells):
        raise ValueError("the legth of layer_idx does not match the lenth of new_cells")

    cell = ModifiedRCNCell(res_model)
    for ilayer, new_cell in zip(layer_idx, new_cells):
        if ilayer == 0:
            layer = res_model.conv1
            if new_cell == 'conv':
                new_cell = ConvGRURCNCell(layer.out_channels, 3, layer.in_channels, 
                                          layer.kernel_size, layer.stride, layer.padding)
            elif new_cell == 'bottleneck':
                new_cell = BottleneckGRURCNCell(layer.out_channels, layer.in_channels,
                                                x_stride=layer.stride)
            if not isinstance(new_cell, RCNCell):
                raise ValueError("not an RCNCell")
            cell.modify('conv1', new_cell)
        else:
            id1, id2 = ilayer
            if id1 < 1 or id1 > 4:
                raise ValueError("no such sub-module")
            layer = res_model._modules['layer%d'%(id1)]._modules[str(id2)]
            if new_cell == 'bottleneck':
                if isinstance(layer, res.Bottleneck):
                    new_cell = BottleneckGRURCNCell(layer.conv3.out_channels, layer.conv1.in_channels,
                                                    x_stride=layer.conv2.stride, residual=True, downsample=layer.downsample)
                else:
                    new_cell = BottleneckGRURCNCell(layer.conv2.out_channels, layer.conv1.in_channels,
                                                    x_stride=layer.conv1.stride, residual=True, downsample=layer.downsample)
            if not isinstance(new_cell, RCNCell):
                raise ValueError("not an RCNCell")
            cell.modify(['layer%d'%(id1), str(id2)], new_cell)
    return cell


class ResnetGRU(nn.Module):

    def __init__(self, model, modify_layers, n_classes):
        super(ResnetGRU, self).__init__()
        if n_classes != model.fc.out_features:
            model.fc = nn.Linear(model.fc.in_features, n_classes)
        self._n_modified = len(modify_layers)
        self._rcn = RCN(resnet_gru_cell(model, modify_layers, ['bottleneck'] * self._n_modified))

    def forward(self, x):
        h0 = [None,] * self._n_modified
        out, _ = self._rcn(x, h0, seq_output=False)
        return out


# test 
if __name__ == "__main__":
    from torchvision.models import resnet50
    base = resnet50()
    resgru = ResnetGRU(base, [(1, 0),[1,2],[4,0]], 101)