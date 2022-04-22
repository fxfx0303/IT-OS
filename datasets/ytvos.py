"""
YoutubeVIS data loader
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval
import datasets.transforms as T
from pycocotools import mask as coco_mask
import os
from PIL import Image
from random import randint
import cv2
import random
import numpy as np
import json

class YTVOSDataset:
    def __init__(self, img_folder, ann_file, transforms, return_masks, num_frames):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.num_frames = num_frames
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.vid_ids = self.ytvos.getVidIds()
        self.vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info['filenames'] = info['file_names']
            self.vid_infos.append(info)
        self.img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info['filenames'])):
                self.img_ids.append((idx, frame_id))

        self.root_path = "video_audio/VidVRD-II-main/vidor-dataset/"
        self.video_path = self.root_path+"video_total/"
        self.box_path = self.root_path+"annotation_total/"
        self.anno_path = self.root_path + "VidSTG-Dataset-master/annotations/"
        self.anno_file_path = self.anno_path + "train_annotations.json"
        self.index_file_path = self.anno_path + "train_files.json"
        self.anno_list = json.load(open(self.anno_file_path, "r"))

    def __len__(self):
        return min(len(self.img_ids), len(self.anno_list))

    def __getitem__(self, idx):
        vid,  frame_id = self.img_ids[idx]
        vid_id = self.vid_infos[vid]['id']
        img = []
        vid_len = len(self.vid_infos[vid]['file_names'])
        inds = list(range(self.num_frames))
        inds = [i%vid_len for i in inds][::-1]
        # # if random
        # # random.shuffle(inds)
        # for j in range(self.num_frames):
        #     img_path = os.path.join(str(self.img_folder), self.vid_infos[vid]['file_names'][frame_id-inds[j]])
        #     img.append(Image.open(img_path).convert('RGB'))

        anno = self.anno_list[idx]
        index_str = anno['vid']
        img, temporal_gt_array, target_box_array, frame_labels_array, valid = self._load_video_temGT_box(index_str, anno)


        ann_ids = self.ytvos.getAnnIds(vidIds=[vid_id])
        target = self.ytvos.loadAnns(ann_ids)
        target = {'image_id': idx, 'video_id': vid, 'frame_id': frame_id, 'annotations': target}
        target = self.prepare(img[0], target, inds, self.num_frames)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # print("shape: ")
        # print(len(img))
        # print(img[0].shape)
        # exit()

        return torch.cat(img,dim=0), target

    def _load_video_temGT_box(self, index_str, anno):
        # img_array = None
        img_list = []
        temporal_gt = [999999, -1]
        target_box_list = []
        frame_labels_list = []

        # 读取box框的json文件
        target_box_path = self.box_path + index_str + '.json'
        target_box_total = json.load(open(target_box_path, 'r'))['trajectories']
        target_id = anno['captions'][0]['target_id']

        # 读取mp4文件
        video_path = self.video_path + index_str + '.mp4'
        videoCapture = cv2.VideoCapture(video_path)
        fps_src = videoCapture.get(cv2.CAP_PROP_FPS)
        fpf = round(fps_src)
        success, frame = videoCapture.read()
        frame_shape = frame.shape

        blank_frame = [0, 0, 0, 0]

        i = 0
        j = 0
        valid = []
        while success:
            if i % (3 * fpf) == 0 and \
                    (i >= anno['used_segment']['begin_fid'] and i <= anno['used_segment']['end_fid']):

                target_box_item = blank_frame
                for target_box_item_total in target_box_total[i]:
                    if target_box_item_total['tid'] == target_id:
                        xmin = target_box_item_total['bbox']['xmin']
                        ymin = target_box_item_total['bbox']['ymin']
                        xmax = target_box_item_total['bbox']['xmax']
                        ymax = target_box_item_total['bbox']['ymax']

                        target_box_item[0] = xmin  # (xmin+xmax)//2
                        target_box_item[1] = ymin  # (ymin+ymax)//2
                        target_box_item[2] = xmax  # xmax-xmin
                        target_box_item[3] = ymax  # ymax-ymin

                        # target_box_item[0] = (xmin+xmax)//2
                        # target_box_item[1] = (ymin+ymax)//2
                        # target_box_item[2] = xmax-xmin
                        # target_box_item[3] = ymax-ymin

                target_box_list.append(target_box_item)
                if target_box_item[0] == -1:
                    frame_labels_list.append(0)
                    valid.append(0)
                else:
                    frame_labels_list.append(1)
                    valid.append(1)

                if temporal_gt[0] == 999999 and i >= anno['temporal_gt']['begin_fid']:
                    temporal_gt[0] = j
                if temporal_gt[1] <= j and i <= anno['temporal_gt']['end_fid']:
                    temporal_gt[1] = j
                # 保存帧
                # frame = frame.transpose(2, 0, 1)
                # expand_frame = np.expand_dims(frame, axis=0)
                # if img_array is None:
                #     img_array = expand_frame
                # else:
                #     img_array = np.concatenate((img_array, expand_frame), axis=0)

                frame_PIL = Image.fromarray(frame)
                img_list.append(frame_PIL)
                j += 1
                if j >= self.num_frames:
                    print("超过帧数上限")
                    # self.count+=1
                    break
            i += 1
            success, frame = videoCapture.read()

        # target_box_list = target_box_list[temporal_gt[0]:temporal_gt[1] + 1]

        for add_frame_index in range(len(target_box_list), self.num_frames):
            add_frame = np.uint8(np.zeros(frame_shape))

            # print(add_frame.shape)

            frame_PIL = Image.fromarray(add_frame)
            img_list.append(frame_PIL)

            target_box_item = blank_frame
            target_box_list.append(target_box_item)
            frame_labels_list.append(0)
            valid.append(0)

        # print("target_box_list: ")
        # print(len(target_box_list))

        # ##################
        # ## test
        # ##################
        #         img_list = []
        #         temporal_gt = [0, self.args.num_frames]
        #         target_box_list = []
        #         frame_labels_list = []
        #         valid = []
        #         for add_frame_index in range(0, self.args.num_frames):
        #             add_frame = np.uint8(np.zeros(frame_shape))
        #             # print(add_frame.shape)
        #             frame_PIL = Image.fromarray(add_frame)
        #             img_list.append(frame_PIL)
        #             target_box_item = blank_frame
        #             target_box_list.append(target_box_item)
        #             frame_labels_list.append(0)
        #             valid.append(1)

        # img_array = img_array[anno['used_segment']['begin_fid']:
        #                       anno['used_segment']['end_fid']+1]
        temporal_gt_array = np.array(temporal_gt)
        target_box_array = np.array(target_box_list)
        frame_labels_array = np.array(frame_labels_list)

        return img_list, temporal_gt_array, target_box_array, frame_labels_array, valid


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        if not polygons:
            mask = torch.zeros((height,width), dtype=torch.uint8)
        else:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = torch.as_tensor(mask, dtype=torch.uint8)
            mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, inds, num_frames):
        w, h = image.size
        image_id = target["image_id"]
        frame_id = target['frame_id']
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        boxes = []
        classes = []
        segmentations = []
        area = []
        iscrowd = []
        valid = []
        # add valid flag for bboxes
        for i, ann in enumerate(anno):
            for j in range(num_frames):
                bbox = ann['bboxes'][frame_id-inds[j]]
                areas = ann['areas'][frame_id-inds[j]]
                segm = ann['segmentations'][frame_id-inds[j]]
                clas = ann["category_id"]
                # for empty boxes
                if bbox is None:
                    bbox = [0,0,0,0]
                    areas = 0
                    valid.append(0)
                    clas = 0
                else:
                    valid.append(1)
                crowd = ann["iscrowd"] if "iscrowd" in ann else 0
                boxes.append(bbox)
                area.append(areas)
                segmentations.append(segm)
                classes.append(clas)
                iscrowd.append(crowd)
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        classes = torch.tensor(classes, dtype=torch.int64)
        if self.return_masks:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor(area) 
        iscrowd = torch.tensor(iscrowd)
        target["valid"] = torch.tensor(valid)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return  target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            # T.RandomHorizontalFlip(),
            # T.RandomResize(scales, max_size=800),
            # T.PhotometricDistort(),
            # T.Compose([
            #          T.RandomResize([400, 500, 600]),
            #          T.RandomSizeCrop(384, 600),
            #          # To suit the GPU memory the scale might be different
            #          T.RandomResize([300], max_size=540),#for r50
            #          #T.RandomResize([280], max_size=504),#for r101
            # ]),
            T.RandomResize([300], max_size=540),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.ytvos_path)
    assert root.exists(), f'provided YTVOS path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train/JPEGImages", root / "annotations" / f'{mode}_train_sub.json'),
        "val": (root / "valid/JPEGImages", root / "annotations" / f'{mode}_val_sub.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = YTVOSDataset(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, num_frames = args.num_frames)
    return dataset
