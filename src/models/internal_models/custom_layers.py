
# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride, bias=False, padding=(1,1)):
        super(LocallyConnected2d, self).__init__()

        # Save for later
        self.input_size = _pair(input_size)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        # Compute the output size
        output_size = self.compute_output_size()
        
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size[0]*kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, output_size[0], output_size[1]))
        else:
            self.register_parameter('bias', None)


    def compute_output_size(self):

        # Unpack everything to make it easier to use
        ih, iw = self.input_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        def compute(i, k, s, p):
            out = i
            out += (2*p)
            out -= (k-1)
            out -= 1
            out //= s

            return int(out+1)

        oh = compute(ih, kh, sh, ph)
        ow = compute(iw, kw, sw, pw)

        return _pair((oh, ow))

        
    def forward(self, x):

        # Unpack everything to make it easier to use
        kh, kw = self.kernel_size
        dh, dw = self.stride
        ph, pw = self.padding

        # # Do the padding to make sure that we get the same output size
        x = F.pad(x,[ph, ph, pw, pw],value=0)

        # Unfold and flatten the last dim
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*(x.shape[:-2]), -1)

        # Multiply by the weights
        out = (x.unsqueeze(1) * self.weight)

        # Sum in in_channel and kernel_size dims
        out = out.sum([2, -1])

        # Add the bias if we have one
        if(self.bias is not None):
            out += self.bias

        return out

