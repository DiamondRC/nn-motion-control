import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):
    """
    Helper to crop the 'future' padding from causal convolutions
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """
    An implementation of a Temporal Convolutional Network layer.

    Args:
    - n_inputs: Number of input channels
    - n_outputs: Number of output channels
    - kernel_size: Size of the convolutional kernel
    - stride: Stride of the convolution
    - dilation: Dilation factor for the convolution
    - padding: Amount of padding to apply
    - dropout: Dropout rate
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net1 = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.net2 = nn.Sequential(
            weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net1(x)
        out = self.net2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
