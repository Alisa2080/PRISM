"""
1. `positionalencoding1d`
   
   - 作用 ：生成一维的位置编码。
   - 原理 ：它使用正弦（sin）和余弦（cos）函数来为序列中的每个位置创建一个独特的向量。这种方法的优点是模型可以学习到位置之间的相对关系。
   - 参数 ：
     - d_model ：编码向量的维度。
     - length ：序列的长度。
2. `positionalencoding2d`
   
   - 作用 ：生成二维的位置编码，非常适用于图像数据（可以看作是图像块的网格）。
   - 原理 ：这个函数很巧妙，它将二维问题分解为两个一维问题。它将 d_model 维度一分为二，分别调用 `positionalencoding1d` 来计算高度（height）和宽度（width）上的一维位置编码，然后将这两个编码拼接起来，从而为网格中的每一个 (x, y) 位置都创建了一个独一无二的编码。
   - 参数 ：
     - d_model ：编码向量的维度。
     - height , width ：二维网格的高度和宽度。
### if __name__ == '__main__': 测试代码块
这个部分是用来测试和演示上述函数功能的。它模拟了项目中多尺度特征提取的场景：

- x20 = positionalencoding2d(512, 8, 8) ：为 20x 放大倍率下的 8x8=64 个图像块生成位置编码。
- x10 = positionalencoding2d(512, 4, 4, 2) ：为 10x 放大倍率下的 4x4=16 个图像块生成位置编码。注意这里的 ratio=2 参数，它会调整编码的频率，帮助模型区分不同尺度的位置信息。
- x5 = positionalencoding2d(512, 2, 2, 4) ：为 5x 放大倍率下的 2x2=4 个图像块生成位置编码， ratio=4 。
最后，它计算了不同尺度位置编码之间的 余弦相似度 ，这可以用来验证不同尺度的位置编码既有区别又有联系。

总而言之，这个文件为 ROAM 模型提供了关键的位置感知能力，使其能够理解多尺度图像块的空间布局。
"""

import math
import torch


def positionalencoding1d(d_model, length, ratio=1):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: (length+1)*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length+1, d_model)
    position = torch.arange(0, length+1).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))*ratio
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def positionalencoding2d(d_model, height, width, ratio=1):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(height*width+1, d_model)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    
    height_pe = positionalencoding1d(d_model, height, ratio)
    width_pe = positionalencoding1d(d_model, width, ratio)

    #print(height_pe.shape, width_pe.shape)

    pe[0, :d_model] = height_pe[0]
    pe[0, d_model:] = width_pe[0]

    for i in range(height):
        for j in range(width):
            pe[i*width+j+1, :d_model] = height_pe[i+1]
            pe[i*width+j+1, d_model:] = width_pe[j+1]

    return pe

   
if __name__ == '__main__':
    x20 = positionalencoding2d(512, 8, 8)
    x10 = positionalencoding2d(512, 4, 4, 2)
    x5 = positionalencoding2d(512, 2, 2, 4)
    print(x20, x10, x5)
    cos = torch.nn.CosineSimilarity()
    print(cos(x10[1:2], x20)[1:].reshape((8,8)))


