from __future__ import division

import time

import numpy as np

import chainer
from chainer.backends import cuda

from chainercv import transforms


class FasterRCNN(chainer.Chain):
    """Base class of Feature Pyramid Networks.

    This is a base class of Feature Pyramid Networks [#]_.

    .. [#] Tsung-Yi Lin et al.
       Feature Pyramid Networks for Object Detection. CVPR 2017

    Args:
        extractor (Link): A link that extracts feature maps.
            This link must have :obj:`scales`, :obj:`mean` and
            :meth:`__call__`.
        rpn (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.RPN`.
            Please refer to the documentation found there.
        head (Link): A link that has the same interface as
            :class:`~chainercv.links.model.fpn.Head`.
            Please refer to the documentation found there.

    Parameters:
        nms_thresh (float): The threshold value
            for :func:`~chainercv.utils.non_maximum_suppression`.
            The default value is :obj:`0.45`.
            This value can be changed directly or by using :meth:`use_preset`.
        score_thresh (float): The threshold value for confidence score.
            If a bounding box whose confidence score is lower than this value,
            the bounding box will be suppressed.
            The default value is :obj:`0.6`.
            This value can be changed directly or by using :meth:`use_preset`.

    """

    _min_size = 800
    _max_size = 1333
    _stride = 32

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.use_preset('visualize')
        self.xcvm1 = None
        self.xcvm2 = None

    def use_preset(self, preset):
        """Use the given preset during prediction.

        This method changes values of :obj:`nms_thresh` and
        :obj:`score_thresh`. These values are a threshold value
        used for non maximum suppression and a threshold value
        to discard low confidence proposals in :meth:`predict`,
        respectively.

        If the attributes need to be changed to something
        other than the values provided in the presets, please modify
        them by directly accessing the public attributes.

        Args:
            preset ({'visualize', 'evaluate'}): A string to determine the
                preset to use.
        """

        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def __call__(self, *args):
        if not hasattr(self, '_export_step'):
            return self.call_orig(*args)
        if self._export_step == 0:
            return self.step0(*args)
        if self._export_step == 1:
            return self.step1(*args)
        if self._export_step == 2:
            return self.step2(*args)
        assert False

    def call_orig(self, x):
        hs, rpn_locs, rpn_confs, anchors = self.step0(x)

        st = time.time()
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        elapsed = (time.time() - st) * 1000
        print('Elapsed (rpn.decode): %s msec' % elapsed)

        # print('uuu', type(rois[0]), type(roi_indices[0]))
        st = time.time()
        rois, roi_indices = self.head.distribute(rois, roi_indices)
        elapsed = (time.time() - st) * 1000
        print('Elapsed (head.distribute): %s msec' % elapsed)

        #head_locs, head_confs = self.head(hs, rois, roi_indices)
        #args = hs + rois + roi_indices
        args = [h.array for h in hs] + rois + roi_indices
        head_locs, head_confs = self.step2(*args)

        print(type(rois), type(roi_indices), type(head_locs), type(head_confs))
        print(len(rois), len(roi_indices), type(head_locs), type(head_confs))
        print(type(rois[0]), type(roi_indices[0]))
        return rois, roi_indices, head_locs, head_confs
        #return head_locs, head_confs
        #return tuple(rois) + tuple(roi_indices) + (head_locs, head_confs)

    def step0(self, x):
        assert(not chainer.config.train)
        if self.xcvm1 is None:
            hs = self.extractor(x)
            rpn_locs, rpn_confs = self.rpn(hs)
        else:
            import chainerx as chx
            x = chx.array(x, copy=False)
            x = self.ccc.value(x)
            self.inputs1['Input_0'] = x
            kwargs = {}
            if self.trace:
                kwargs = {'trace': True, 'chrome_tracing': 'step1.json'}
            results = self.xcvm1.run(self.inputs1, **kwargs)
            results = [
                results[k] for k
                in ('Identity_4', 'Identity_3', 'Identity_2', 'Identity_0', 'Identity_1',
                    'Reshape_9', 'Reshape_7', 'Reshape_5', 'Reshape_1', 'Reshape_3',
                    'Reshape_8', 'Reshape_6', 'Reshape_4', 'Reshape_0', 'Reshape_2',
                )]
            assert len(results) % 3 == 0
            results = [chainer.Variable(chx.to_numpy(r.array(), copy=False))
                       for r in results]
            l = len(results) // 3
            hs = results[:l]
            rpn_locs = results[l:l*2]
            rpn_confs = results[l*2:]
        print('uuu', type(rpn_locs), type(rpn_confs))
        st = time.time()
        anchors = self.rpn.anchors(h.shape[2:] for h in hs)
        elapsed = (time.time() - st) * 1000
        print('Elapsed (rpn.anchor): %s msec' % elapsed)
        return hs, rpn_locs, rpn_confs, anchors

    def step1(self, x):
        hs, rpn_locs, rpn_confs, anchors = self.step0(x)
        return tuple(hs) + tuple(rpn_locs) + tuple(rpn_confs)

    def step2(self, *args):
        assert len(args) % 3 == 0
        l = len(args) // 3
        #hs = [h.array for h in args[:l]]
        hs = args[:l]
        rois = args[l:l*2]
        roi_indices = args[l*2:]
        assert len(rois) == len(roi_indices)
        # print('uuu', len(hs), type(rois[0]), type(roi_indices[0]))

        if self.xcvm2 is None:
            head_locs, head_confs = self.head(hs, rois, roi_indices)
        else:
            import chainerx as chx
            for i in range(len(hs)):
                h = self.ccc.value(chx.array(hs[i], copy=False))
                roi = self.ccc.value(chx.array(rois[i], copy=False))
                roii= self.ccc.value(chx.array(roi_indices[i], copy=False))
                j = (len(hs) - 1 - i) * 3
                self.inputs2['Input_%d' % (j + 0)] = h
                self.inputs2['Input_%d' % (j + 1)] = roi
                self.inputs2['Input_%d' % (j + 2)] = roii

            kwargs = {}
            if self.trace:
                kwargs = {'trace': True, 'chrome_tracing': 'step2.json'}
            results = self.xcvm2.run(self.inputs2, **kwargs)
            results = [results[k] for k in ('Reshape_1', 'Gemm_3')]
            results = [chainer.Variable(chx.to_numpy(r.array(), copy=False))
                       for r in results]
            head_locs, head_confs = results

        # print(type(rois), type(roi_indices), type(head_locs), type(head_confs))
        # print(len(rois), len(roi_indices), type(head_locs), type(head_confs))
        # print(type(rois[0]), type(roi_indices[0]))
        # print('wwwww', type(head_locs), type(head_confs))
        return head_locs, head_confs

    def set_trace(self, t):
        self.trace = t

    def use_chainer_compiler(self):
        import chainer_compiler_core
        g1 = chainer_compiler_core.load('1_faster_rcnn_fpn_resnet50/model.onnx')
        g2 = chainer_compiler_core.load('2_faster_rcnn_fpn_resnet50/model.onnx')
        self.ccc = chainer_compiler_core
        self.inputs1 = dict(g1.params())
        self.inputs2 = dict(g2.params())
        self.xcvm1 = g1.compile(use_ngraph=True,
                                fuse_operations=True,
                                compiler_log=True,
                                dump_after_scheduling=True)
        self.xcvm2 = g2.compile(reset_shape=True,
                                compiler_log=True,
                                dump_after_scheduling=True)

    def predict(self, imgs):
        """Detect objects from images.

        This method predicts objects for each image.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
           tuple of lists:
           This method returns a tuple of three lists,
           :obj:`(bboxes, labels, scores)`.

           * **bboxes**: A list of float arrays of shape :math:`(R, 4)`, \
               where :math:`R` is the number of bounding boxes in a image. \
               Each bounding box is organized by \
               :math:`(y_{min}, x_{min}, y_{max}, x_{max})` \
               in the second axis.
           * **labels** : A list of integer arrays of shape :math:`(R,)`. \
               Each value indicates the class of the bounding box. \
               Values are in range :math:`[0, L - 1]`, where :math:`L` is the \
               number of the foreground classes.
           * **scores** : A list of float arrays of shape :math:`(R,)`. \
               Each value indicates how confident the prediction is.

        """
        if hasattr(self, '_export_step'):
            del self._export_step

        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            rois, roi_indices, head_locs, head_confs = self(x)

        st = time.time()
        bboxes, labels, scores = self.head.decode(
            rois, roi_indices, head_locs, head_confs,
            scales, sizes, self.nms_thresh, self.score_thresh)
        elapsed = (time.time() - st) * 1000
        print('Elapsed (head.decode): %s msec' % elapsed)

        bboxes = [cuda.to_cpu(bbox) for bbox in bboxes]
        labels = [cuda.to_cpu(label) for label in labels]
        scores = [cuda.to_cpu(score) for score in scores]
        return bboxes, labels, scores

    def export(self, imgs, name, **kwargs):
        import onnx_chainer
        from onnx_chainer import onnx_helper

        def convert_roi_average_align_2d(params):
            output_shape = (params.func.outh, params.func.outw)
            return onnx_helper.make_node(
                'ChainerROIAverageAlign2D',
                params.input_names,
                len(params.output_names),
                output_shape=output_shape,
                spatial_scale=params.func.spatial_scale,
                sampling_ratio=params.func.sampling_ratio
            ),

        chainer.config.train = False
        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)
        print('www', type(x))

        self._export_step = 0
        hs, rpn_locs, rpn_confs, anchors = self(x)
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois, roi_indices = self.head.distribute(rois, roi_indices)

        self._export_step = 1
        onnx_chainer.export_testcase(self, [x], '1_' + name, **kwargs)

        self._export_step = 2
        print('hs', type(hs[0]))
        print('rois', type(rois[0]))
        print('roi_indices', type(roi_indices[0]))
        args = [h.array for h in hs] + rois + roi_indices
        onnx_chainer.export_testcase(
            self, args, '2_' + name,
            external_converters={'ROIAverageAlign2D': convert_roi_average_align_2d},
            **kwargs)

    def run(self, imgs):
        sizes = [img.shape[1:] for img in imgs]
        x, scales = self.prepare(imgs)
        print('www', type(x))
        self(x)

    def prepare(self, imgs):
        """Preprocess images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their value is :math:`[0, 255]`.

        Returns:
            Two arrays: preprocessed images and \
            scales that were caluclated in prepocessing.

        """

        scales = []
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            scales.append(scale)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self.extractor.mean
            resized_imgs.append(img)

        size = np.array([im.shape[1:] for im in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        x = self.xp.array(x)
        return x, scales
