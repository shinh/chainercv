import argparse
import sys
import time

import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_semantic_segmentation_label_colors
from chainercv.datasets import voc_semantic_segmentation_label_names
from chainercv.links import DeepLabV3plusXception65
from chainercv import utils
from chainercv.visualizations import vis_image
from chainercv.visualizations import vis_semantic_segmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='cityscapes')
    parser.add_argument('--min-input-size', type=int, default=None)
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--chainer-compiler', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('image')
    args = parser.parse_args()

    model = DeepLabV3plusXception65(
        pretrained_model=args.pretrained_model,
        min_input_size=args.min_input_size)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)

    MODEL_NAME = 'deeplab_v3_%s' % args.pretrained_model

    if args.export:
        import onnx_chainer
        from onnx_chainer import onnx_helper

        def convert_resize_images(params):
            output_shape = (params.func.out_H, params.func.out_W)
            return onnx_helper.make_node('ChainerResizeImages',
                                         params.input_names,
                                         len(params.output_names),
                                         output_shape=output_shape),

        def export(self, *xs):
            onnx_chainer.export_testcase(
                model, xs, MODEL_NAME,
                external_converters={'ResizeImages': convert_resize_images})
            sys.exit()

        model.run_model = export

    if args.chainer_compiler:
        import chainerx as chx
        import chainer_compiler_core as ccc

        g = ccc.load('%s/model.onnx' % MODEL_NAME)
        input_names = g.input_names()
        output_names = g.output_names()
        inputs = g.params()
        xcvm = g.compile(use_ngraph=True,
                         fuse_operations=True,
                         compiler_log=True,
                         dump_after_scheduling=True)

        def run(self, *xs):
            kwargs = {}
            if self.trace:
                kwargs = {'trace': True,
                          'chrome_tracing': '%s.json' % args.model}
            assert len(xs) == len(input_names)
            for n, x in zip(input_names, xs):
                inputs[n] = ccc.value(chx.array(x, copy=False))
            outputs = xcvm.run(inputs, **kwargs)
            outputs = [outputs[name].array() for name in output_names]
            outputs = [chainer.Variable(chx.to_numpy(o)) for o in outputs]
            return tuple(outputs)

        model.run_model = run
        model.trace = True

    labels = model.predict([img])
    label = labels[0]

    if args.iterations > 1:
        model.trace = False
        ni = args.iterations - 1
        st = time.time()
        for i in range(ni):
            model.predict([img])
        elapsed = (time.time() - st) * 1000 / ni
        print('Elapsed: %s msec' % elapsed)
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        vis_image(img, ax=ax1)
        ax2 = fig.add_subplot(1, 2, 2)
        # Do not overlay the label image on the color image
        vis_semantic_segmentation(
            None, label, voc_semantic_segmentation_label_names,
            voc_semantic_segmentation_label_colors, ax=ax2)
        plt.show()


if __name__ == '__main__':
    main()
