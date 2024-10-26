import torch
import torch.nn as nn

from nets.eff import EFF


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=True):
        super(Siamese, self).__init__()
        self.eff = EFF(pretrained)#这里注意 上面写了false
        del self.eff.avgpool
        del self.eff.classifier
        #RuntimeError: mat1 and mat2  shapes cannot  be multiplied(32 x20480 and 4500 x500)
        #之前的错误是mat1 and mat2 shapes cannot be multiplied (32x20480 and 4608x512)
        #可以发现 512*9 的倍数
        #发现512是没有问题的
        #可以发现 只要把后面的改成20480*512 就可以了 也就是把flat_shape 改成20480
        flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(62720, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.eff.features(x1)
        x2 = self.eff.features(x2)
        #-------------------------#
        #   相减取绝对值，取l1距离
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)
        return x
