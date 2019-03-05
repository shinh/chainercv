import argparse
import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import voc_bbox_label_names
from chainercv.experimental.links import YOLOv2Tiny
from chainercv.links import YOLOv2
from chainercv.links import YOLOv3
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('yolo_v2', 'yolo_v2_tiny', 'yolo_v3'),
        default='yolo_v2')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='voc0712')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--chainer_compiler', action='store_true')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'yolo_v2':
        model = YOLOv2(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'yolo_v2_tiny':
        model = YOLOv2Tiny(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)
    elif args.model == 'yolo_v3':
        model = YOLOv3(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image, color=True)

    if args.export:
        import onnx_chainer
        x = model.xp.stack([img])
        onnx_chainer.export_testcase(model, x, args.model)
        return

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

    vis_bbox(
        img, bbox, label, score, label_names=voc_bbox_label_names)
    plt.show()


if __name__ == '__main__':
    main()
