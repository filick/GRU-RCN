import torch
from torch.nn import Module, Conv2d


class GRURCNCellBase(Module):

    def __init__(self, xz, hz, xr, hr, xh, rh):
        super(GRURCNCellBase, self).__init__()
        self._xz = xz
        self._hz = hz
        self._xr = xr
        self._hr = hr
        self._xh = xh
        self._rh = rh

    def forward(self, x, hidden):
        z = torch.sigmoid(self._xz(x) + self._hz(hidden))
        r = torch.sigmoid(self._xr(x) + self._hr(hidden))
        h_ = torch.tanh(self._xh(x) + self._rh(r * hidden))
        h = (1 - z) * hidden + z * h_
        return h


class ConvGRURCNCell(Module):
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

        hz = Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        hr = Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        rh = Conv2d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)

        x_channels = x_channels or channels
        x_kernel_size = x_kernel_size or kernel_size
        x_stride = x_stride or 1
        x_padding = x_padding or ((kernel_size - 1) // 2)
        xz = Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xr = Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)
        xh = Conv2d(x_channels, channels, x_kernel_size, x_stride, x_padding)

        self._cell = GRURCNCellBase(xz, hz, xr, hr, xh, rh)

    def forward(self, x, hidden):
        return self._cell(x, hidden)


# test
if __name__ == '__main__':
    gru_rcn = ConvGRURCNCell(64, 7)
    print(gru_rcn)