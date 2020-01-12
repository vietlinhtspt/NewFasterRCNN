
import json
import numpy as np
import os
import tqdm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid

class display(imdb):
    def __init__(self, image_set):
        """
        Args:
            base_dir (str): root of the dataset which contains the subdirectories for each image_set and annotations
            image_set (str): the name of the image_set, e.g. "train2017".
                The image_set has to match an annotation file in "annotations/" and a directory of images.

        Examples:
            For a directory of this structure:
            data/ 
                display/
                annotations/
                    instances_XX.txt
                    instances_YY.txt
                XX/
                YY/

            use `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
        """
        imdb.__init__(self, 'display')
        # COCO specific config options
        self.config = {'use_salt': True,
                    'cleanup': True}
        # name, paths
        self._year = 2020
        self._image_set = image_set
        self._data_path = osp.join(cfg.DATA_DIR, 'display')
        self._classes = tuple(['__background__'] + ['LabelID0', 'LabelID1'])
        self._classes_to_ind = dict(list(zip(self.classes, list(range(2)))))
        self._classes_to_coco_cat_id = dict(list(zip(['LabelID0', 'LabelID1'], list(range(2)))))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('gt')
        self.competition_mode(False)
        self._view_map = {'train': 'train', 'val':'val'}
        display_name = image_set
        self._data_name = (self._view_map[display_name])
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('train', 'val')
    
    def _get_ann_file(self):
        return osp.join(self._data_path, 'annotations', 'instances_', self._image_set, '.txt')

    def _load_image_set_index(self):
        """
        Load image ids.
        """
        annotation_file = os.path.join(
            self._data_path, 'annotations/instances_{}.txt'.format(self._image_set))
        assert os.path.isfile(annotation_file), annotation_file

        txt_annotations = open(annotation_file, 'r')
        annotations = txt_annotations.readlines()

        self.ids = []
        for i in range(0, len(annotations), 3):
            temp = annotations[i].split(',')
            # file name image
            fname = temp[0]
            # Create id = image name
            fid = temp[0].split(".")[0]
            # Create path
            fname = os.path.join(self.imgdir, fname)
            roidb["image_id"] = int(fid)
            self.ids.append(roidb)
        return self.ids

    def _get_widths(self):
        # annotations
        widths = [624 for i in range(0, len(self.ids))]
        return widths

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        file_name = (str(index) + '.png')
        image_path = osp.join(self._data_path, self._data_name, file_name)
        assert osp.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path 

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_coco_annotation(index) for index in self._image_index]

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_display_annotation(self, index):
        """
        Loads Display bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """

        annotation_file = os.path.join(
            self._data_path, 'annotations/instances_{}.txt'.format(self._image_set))
        assert os.path.isfile(annotation_file), annotation_file

        txt_annotations = open(annotation_file, 'r')
        annotations = txt_annotations.readlines()

        num_objs = 3
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        for i in range(0, len(annotations), 3):
            if i != index:
                continue
            temp = annotations[i].split(',')

            # data in ground truth file has 3 line for each img
            for j in range(0, 3):
                temp = annotations[i + j].split(',')
                x1 = int(temp[1]) 
                y1 = int(temp[2])  
                x2 = int(temp[3]) 
                y2 = int(temp[4]) 
                box = [x1, y1, x2, y2] 
                boxes[j, :] = box
                gt_classes[j] = int(temp[5][0]) + 1
                seg_areas[j] = (x2 - x1) * (y2 - y1)
                overlaps[j, int(temp[5][0]) + 1] = 1.0

        return {'width': x2 - x1,
            'height': y2 - y1,
            'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

   def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
        boxes = self.roidb[i]['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = widths[i] - oldx2 - 1
        boxes[:, 2] = widths[i] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self.roidb[i]['gt_classes'],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}

        self.roidb.append(entry)
    self._image_index = self._image_index * 2

    def _get_box_file(self, index):
        raise NotImplementedError
    
    def _print_detection_eval_metrics(self, coco_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(coco_eval, thr):
        ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) & (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
        iou_thr = coco_eval.params.iouThrs[ind]
        assert np.isclose(iou_thr, thr)
        return ind

    ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
        coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
        if cls == '__background__':
            continue
        # minus 1 because of __background__
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
        ap = np.mean(precision[precision > -1])
        print('{:.1f}'.format(100 * ap))

    print('~~~~ Summary metrics ~~~~')
    coco_eval.summarize()

    def _do_detetion_eval(self, res_file, output_dir)
