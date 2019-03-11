import argparse
import sys
import time

import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.links import SSD512
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512'), default='ssd300')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='voc0712')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--chainer-compiler', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)

    if args.export:
        import onnx_chainer

        def export(self, *xs):
            onnx_chainer.export_testcase(model, xs, args.model)
            sys.exit()

        model.run_model = export

    if args.chainer_compiler:
        import chainerx as chx
        import chainer_compiler_core as ccc

        g = ccc.load('%s/model.onnx' % args.model)
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

    bboxes, labels, scores = model.predict([img])
    bbox, label, score = bboxes[0], labels[0], scores[0]

    if args.iterations > 1:
        model.trace = False
        ni = args.iterations - 1
        st = time.time()
        for i in range(ni):
            model.predict([img])
        elapsed = (time.time() - st) * 1000 / ni
        print('Elapsed: %s msec' % elapsed)
    else:
        vis_bbox(
            img, bbox, label, score, label_names=voc_bbox_label_names)
        plt.show()


if __name__ == '__main__':
    main()
