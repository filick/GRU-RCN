import torch
import torch.nn as nn
from torch.autograd import Variable

class GRURCNCellBase(nn.Module):

    def __init__(self, xz, hz, xr, hr, xh, rh):
        super(GRURCNCellBase, self).__init__()
        self._xz = xz
        self._hz = hz
        self._xr = xr
        self._hr = hr
        self._xh = xh
        self._rh = rh
        
        self._count = 0

    def forward(self, x):
        if self._count == 0:
            temp = self._xz(x)
            self.hidden = Variable(torch.zeros(temp.size()))
            self._count = 1
        
        #print(self.hidden.size())
        z = torch.sigmoid(self._xz(x) + self._hz(self.hidden))
        r = torch.sigmoid(self._xr(x) + self._hr(self.hidden))
        h_ = torch.tanh(self._xh(x) + self._rh(r * self.hidden))
        self.hidden = (1 - z) * self.hidden + z * h_
        return self.hidden
    
    def reset(self):
        self._count = 0


class ConvGRURCNCell(nn.Module):
    r"""GRU-RCN cell using the conventional covolution as the inner operation.

    The convolutions on the hidden states must keep the size unchanged, so 
    we do not provide the arguments to set the stride or padding of these 
    convolutions, but fix stride=1 and padding=(kernel_size - 1) / 2).

    The convolutions on the inputs must make the outputs has the same size as
    hidden states. If the sizes of inputs and hidden states are not the same,
    users must provide the arguments (x_channels, x_kernel_size, x_stride, 
    x_padding) to insure the outputs after the convolutions on inputs to be the 
    same as the hidden states.

    Args:
        channels: the number of channels of hidden states. If x_channels is None,
            it means the channels of inputs and hidden states are the same.
        kernel_size: size of the convolving kernel.
        x_channels, x_kernel_size, x_stride, x_padding:
            parameters of the convolution operation on inputs.
            If None, inputs and hidden states have the same sizes, so the
            convolutions on them have no difference.

    Inputs: x (input), hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden** (batch, channel, height, width): tensor containing the hidden
            state for the precious time instance.

    Outputs: h
        - **h**: (batch, hidden_size): tensor containing the next hidden state
    """

    def __init__(self, channels, kernel_size, x_channels=None,
                 x_kernel_size=None, x_stride=None, x_padding=None):
        super(ConvGRURCNCell, self).__init__()

        x_channels = x_channels or channels
        x_kernel_size = x_kernel_size or kernel_size
        x_stride = x_stride or 1
        x_padding = x_padding or ((kernel_size - 1) // 2)

        hz = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] - 1) // 2)
        hr = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] - 1) // 2)
        rh = nn.Conv2d(channels, channels, kernel_size, padding=(kernel_size[0] - 1) // 2)

        xz = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xr = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xh = nn.Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)
                
    def forward(self, x):

        hidden = self._cell(x)
        return hidden
    
    def reset(self):
        self._cell.reset()


class BottleneckGRURCNCell(nn.Module):
    r"""GRU-RCN cell using a bottleneck block as the inner operation.

    If the sizes of input and hidden state are not the same, users must specify 
    the for channel number and stride for the input-to-hidden convolution, to 
    make output has the same size as the hidden state.

    Args:
        channels: the number of channels of hidden states. If x_channels is None,
            it means the channels of inputs and hidden states are the same.
        x_channels, x_stride:
            parameters of the inputs-to-hidden convolutions.
            If None, inputs and hidden states have the same sizes, so the
            convolutions on them have no difference.

    Inputs: x (input), hidden
        - **x** (batch, channel, height, width): tensor containing input features.
        - **hidden** (batch, channel, height, width): tensor containing the hidden
            state for the precious time instance.

    Outputs: h
        - **h**: (batch, hidden_size): tensor containing the next hidden state
    """

    def __init__(self, channels, x_channels=None, x_stride=None):
        super(BottleneckGRURCNCell, self).__init__()

        x_channels = x_channels or channels
        x_stride = x_stride or 1

        hz = Bottleneck(channels, channels)
        hr = Bottleneck(channels, channels)
        rh = Bottleneck(channels, channels)

        xz = Bottleneck(x_channels, channels, stride=x_stride)
        xr = Bottleneck(x_channels, channels, stride=x_stride)
        xh = Bottleneck(x_channels, channels, stride=x_stride)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)

    def forward(self, x):
        return self._cell(x)
    
    def reset(self):
        self._cell.reset()

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super(Bottleneck, self).__init__()
        planes = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        return out


# test
if __name__ == '__main__':
    gru_rcn = BottleneckGRURCNCell(64)
    print(gru_rcn)

    from torch.autograd import Variable
    x = Variable(torch.rand(10, 64, 32, 32))
    h = Variable(torch.rand(10, 64, 32, 32))
    gru_rcn(x, h)

    gru_rcn = BottleneckGRURCNCell(64, 32, 2)
    print(gru_rcn)

    from torch.autograd import Variable
    x = Variable(torch.rand(10, 32, 64, 64))
    h = Variable(torch.rand(10, 64, 32, 32))
    gru_rcn(x, h)