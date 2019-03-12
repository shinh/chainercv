import argparse
import time

import matplotlib.pyplot as plt

import chainer

from chainercv.datasets import coco_bbox_label_names
from chainercv.links import FasterRCNNFPNResNet101
from chainercv.links import FasterRCNNFPNResNet50
from chainercv import utils
from chainercv.visualizations import vis_bbox


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=('faster_rcnn_fpn_resnet50', 'faster_rcnn_fpn_resnet101'),
        default='faster_rcnn_fpn_resnet50')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained-model', default='coco')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--chainer-compiler', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--iterations', '-I', type=int, default=1)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('image')
    args = parser.parse_args()

    if args.model == 'faster_rcnn_fpn_resnet50':
        if args.small:
            FasterRCNNFPNResNet50._min_size = 512
            model = FasterRCNNFPNResNet50(
                n_fg_class=len(coco_bbox_label_names))
        else:
            model = FasterRCNNFPNResNet50(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model=args.pretrained_model)
    elif args.model == 'faster_rcnn_fpn_resnet101':
        if args.small:
            FasterRCNNFPNResNet101._min_size = 256
            model = FasterRCNNFPNResNet101(
                n_fg_class=len(coco_bbox_label_names))
        else:
            model = FasterRCNNFPNResNet101(
                n_fg_class=len(coco_bbox_label_names),
                pretrained_model=args.pretrained_model)

    model.set_trace(args.trace)

    if args.chainer_compiler:
        model.use_chainer_compiler()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    img = utils.read_image(args.image)
    # from chainer_computational_cost import ComputationalCostHook
    # with chainer.no_backprop_mode(), chainer.using_config('train', False):
    #    with ComputationalCostHook(fma_1flop=True) as cch:
    #        y = model.run([img])
    #        cch.show_report(unit='G', mode='md')
    if args.export:
        #model.export([img], args.model + '.onnx',
        #             experimental_onnx_chainer2=True)
        model.export([img], args.model)
        #onnx_chainer.export(model, [img], args.model + '.onnx')

    bboxes, labels, scores = model.predict([img])

    if args.iterations > 1:
        model.set_trace(False)
        st = time.time()
        ni = args.iterations - 1
        for i in range(ni):
            model.predict([img])
        elapsed = ((time.time() - st) * 1000) / ni
        print('Elapsed: %s msec' % elapsed)

    if args.show:
        bbox = bboxes[0]
        label = labels[0]
        score = scores[0]

        vis_bbox(
            img, bbox, label, score, label_names=coco_bbox_label_names)
        plt.show()


if __name__ == '__main__':
    main()
