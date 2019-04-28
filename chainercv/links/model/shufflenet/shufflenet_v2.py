from __future__ import division

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links import Conv2DBNActiv
from chainercv.links import PickableSequentialChain
from chainercv import utils


# RGB order
# This is channel wise mean of mean image distributed at
# https://github.com/KaimingHe/deep-residual-networks
_imagenet_mean = np.array(
    [123.15163084, 115.90288257, 103.0626238],
    dtype=np.float32)[:, np.newaxis, np.newaxis]


class BasicUnit(chainer.Chain):

    def __init__(self, channels, initialW=None):
        super(BasicUnit, self).__init__()

        ch = channels // 2

        with self.init_scope():
            self.conv1 = Conv2DBNActiv(ch, ch, 1,
                                       initialW=initialW, nobias=True)
            self.conv2 = Conv2DBNActiv(ch, ch, 3, pad=1, groups=ch, activ=None,
                                       initialW=initialW, nobias=True)
            self.conv3 = Conv2DBNActiv(ch, ch, 1,
                                       initialW=initialW, nobias=True)

    def forward(self, x):
        l, h = F.split_axis(x, 2, axis=1)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = F.concat((l, h), axis=1)
        return h


class DownSampleUnit(chainer.Chain):

    def __init__(self, in_channels, out_channels, initialW=None):
        super(DownSampleUnit, self).__init__()

        ch = out_channels // 2

        with self.init_scope():
            self.lconv1 = Conv2DBNActiv(in_channels, in_channels, 3,
                                        stride=2, pad=1, groups=in_channels,
                                        activ=None,
                                        initialW=initialW, nobias=True)
            self.lconv2 = Conv2DBNActiv(in_channels, ch, 1,
                                        initialW=initialW, nobias=True)

            self.rconv1 = Conv2DBNActiv(in_channels, ch, 1,
                                        initialW=initialW, nobias=True)
            self.rconv2 = Conv2DBNActiv(ch, ch, 3,
                                        stride=2, pad=1, groups=ch, activ=None,
                                        initialW=initialW, nobias=True)
            self.rconv3 = Conv2DBNActiv(ch, ch, 1,
                                        initialW=initialW, nobias=True)

    def forward(self, x):
        l = r = x
        l = self.lconv1(l)
        l = self.lconv2(l)
        r = self.rconv1(r)
        r = self.rconv2(r)
        r = self.rconv3(r)
        h = F.concat((l, r), axis=1)
        return h


class Block(chainer.ChainList):

    def __init__(self, num_layers, in_channels, out_channels, initialW=None):
        super(Block, self).__init__()
        self.add_link(DownSampleUnit(in_channels, out_channels,
                                     initialW=initialW))
        for i in range(num_layers - 1):
            self.add_link(BasicUnit(out_channels, initialW=initialW))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x


class ShuffleNetV2(PickableSequentialChain):

    def __init__(self, n_class=None,
                 pretrained_model=None,
                 mean=None,
                 scale_factor=1):
        super(ShuffleNetV2, self).__init__()

        out_channel_map = {
            0.25: (24, 48, 96, 512),
            0.33: (32, 64, 128, 512),
            0.5: (48, 96, 192, 1024),
            1.0: (116, 232, 464, 1024),
            1.5: (176, 352, 704, 1024),
            2.0: (244, 488, 976, 2048),
        }
        assert scale_factor in out_channel_map, \
            'Unknown scale_factor: %s' % scale_factor
        out_channels = out_channel_map[scale_factor]

        self.mean = _imagenet_mean

        initialW = chainer.initializers.HeNormal()
        kwargs = {'initialW': initialW}

        with self.init_scope():
            self.conv1 = Conv2DBNActiv(3, 24, 3, stride=2, pad=1,
                                       initialW=initialW, nobias=True)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.stage2 = Block(4, 24, out_channels[0], **kwargs)
            self.stage3 = Block(8, out_channels[0], out_channels[1], **kwargs)
            self.stage4 = Block(4, out_channels[1], out_channels[2], **kwargs)
            self.conv5 = Conv2DBNActiv(out_channels[2], out_channels[3], 1,
                                       initialW=initialW, nobias=True)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(out_channels[3], n_class)
            self.prob = F.softmax


def main():
    import numpy as np
    import onnx_chainer

    #
    # model = BasicUnit(24)
    # print(model(np.random.rand(1, 24, 56, 56)).shape)
    # model = DownSampleUnit(24, 12)
    # print(model(np.random.rand(1, 24, 56, 56)).shape)
    #
    # model = ShuffleNetV2(1.0)
    # print(model(np.random.rand(1, 3, 224, 224).astype(np.float32), np.array([3])))
    model = ShuffleNetV2(2.0)
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    onnx_chainer.export_testcase(model, [x], 'shufflenet_v2_x2.0')


if __name__ == '__main__':
    main()
