def conv_out(h_in, kernel_size, padding, stride, dilation):
    h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    print(int(h_out))

def transpose_conv_out(h_in, kernel_size, padding, stride, dilation, output_padding):
    h_out = (
        (h_in - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )
    print(int(h_out))

import torch
from torch import nn

aa = torch.rand(size=(8, 10, 200))
bb = torch.zeros(size=(8, 10, 200), dtype=torch.float)


"""
bb[:,:,15] = 1.

aa[:,:,15] += 100000.

soft = nn.Softmax(dim=2)

aa = soft(aa)

ce = nn.CrossEntropyLoss(reduction='sum')

res = ce(aa.permute(0,2,1),bb.permute(0,2,1))

print(res)
"""