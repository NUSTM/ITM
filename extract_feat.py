#!/usr/bin/env python

"""Generate bottom-up attention features as a npz file for each image. Can use multiple gpus"""

# Example:
#./tools/extract_feat.py --gpu 0,1,2,3 --cfg experiments/cfgs/faster_rcnn_end2end_resnet_vg.yml --def models/vg/ResNet-101/faster_rcnn_end2end/test.prototxt --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --img_dir /path/to/images/ --out_dir /path/to/outfeat/ --num_bboxes=10,100 --feat_name=pool5_flat

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect,_get_blobs
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import random
import caffe
import argparse
import pprint
import time, os, sys
import base64
import numpy as np
import cv2
from multiprocessing import Process
import random
import json


# load image list in a folder recursively
def load_imagelist(root_dir):
    img_list = []
    g = os.walk(root_dir)
    for path, d, filelist in g:  
        for img in filelist:    
            img_list.append(os.path.join(path, img))
    return img_list

def get_feats_from_im(net, im, min_max_bboxes, feat_name, conf_thresh=0.2):

    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # Keep the original boxes, don't worry about the regresssion bbox outputs
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    blobs, im_scales = _get_blobs(im, None)

    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs['cls_prob'].data
    attr_prob = net.blobs['attr_prob'].data
    feat = net.blobs[feat_name].data # extract feature from layer with name 'feat_name'

    # Keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1,cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        #print keep.shape
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < min_max_bboxes[0]:
        keep_boxes = np.argsort(max_conf)[::-1][:min_max_bboxes[0]]
    elif len(keep_boxes) > min_max_bboxes[1]:
        keep_boxes = np.argsort(max_conf)[::-1][:min_max_bboxes[1]]
   
    image_h = np.size(im, 0)
    image_w = np.size(im, 1)
    return feat[keep_boxes], cls_boxes[keep_boxes], image_h, image_w, len(keep_boxes)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id(s) to use',
                        default='0', type=str)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to use',
                        default=None, type=str)
    parser.add_argument('--out_dir', dest='outpath',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--num_bbox', dest='num_bbox',
                        help='[min,max] number of bboxes to be extracted. Default is [10,100]', default='10,100', type=str)
    parser.add_argument('--feat_name', dest='feat_name',
                        help='layer name to extract features, pool5_flat(default) for ResNet-101, fc7_flat for VGG-16', default='pool5_flat', type=str)
    parser.add_argument('--img_dir', dest='img_dir',
                        help='image folder',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
                    
def generate_npz(gpu_id, prototxt, weights, image_paths, output_dir, min_max_bboxes, feat_name):
    
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
    net = caffe.Net(prototxt, caffe.TEST, weights=weights)
    
    _t = {'misc' : Timer()}
    count = 0
    total = len(image_paths)
    for image_path in image_paths:
        _, image_id = os.path.split(image_path) # split the image path and obtain the image file name
        output_file_path = os.path.join(output_dir, image_id + '.npz')
        
        # skip the proceed images
        if os.path.exists(output_file_path):
            continue

        _t['misc'].tic()
        #print input_img_path
        im = cv2.imread(image_path)
        if im is None:
            print(image_path, "is illegal!")
            continue
        features, bbox, image_h, image_w, num_bbox = get_feats_from_im(net, im, min_max_bboxes, feat_name)
        x = np.transpose(features)
        _t['misc'].toc()

        np.savez_compressed(output_file_path, x=x, bbox=bbox, num_bbox=num_bbox, image_h=image_h, image_w=image_w)        
        
        if (count % 100) == 0:
            print ('GPU {}, proceed{}/{} images'.format(gpu_id, count, total))
        count += 1

     
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    gpu_id = args.gpu_id
    gpu_list = gpu_id.split(',')
    gpus = [int(i) for i in gpu_list]

    num_bbox = args.num_bbox.split(',')
    min_max_bboxes = [int(i) for i in num_bbox]
    if len(min_max_bboxes) != 2 or min_max_bboxes[0] >= min_max_bboxes[1]:
        print('illegal num_bbox format!')

    feat_name = args.feat_name


    print('Using config:')
    pprint.pprint(cfg)
    assert cfg.TEST.HAS_RPN

    image_ids = load_imagelist(args.img_dir)
    random.seed(10)
    random.shuffle(image_ids)
    # Split image ids between gpus
    image_ids = [image_ids[i::len(gpus)] for i in range(len(gpus))]
    
    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))
    procs = []    
    output_dir = args.outpath
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,gpu_id in enumerate(gpus):
        #generate_npz(gpu_id, args.prototxt, args.caffemodel, image_ids[i], args.img_dir, outpath)
        p = Process(target=generate_npz,
                    args=(gpu_id, args.prototxt, args.caffemodel, image_ids[i], output_dir, min_max_bboxes, feat_name))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()            
                  