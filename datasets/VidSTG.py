import torch
import torch.utils.data
import torchvision
import json
import cv2
import numpy as np
import nltk
import datasets.transforms as T
from PIL import Image
import copy
import math

def tokenize(caption, word2vec):
    punctuations = ['.', '?', ',', '', '(', ')','_']
    raw_text = caption.lower()
    words = nltk.word_tokenize(raw_text)
    words = [word for word in words if word not in punctuations]
    return [word for word in words if word in word2vec]

def first_NN(caption):
    for word, pos in nltk.pos_tag(nltk.word_tokenize(caption)):
        if pos.startswith('NN'):
            # if word in word2vec:
            #     return word2vec[word]
            return word

def make_coco_transforms(image_set, cautious):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=800),
            T.PhotometricDistort(),
            T.Compose([
                     T.RandomResize([400, 500, 600]),
                     T.RandomSizeCrop(384, 600),
                     # To suit the GPU memory the scale might be different
                     # T.RandomResize([300], max_size=540),#for r50
                     T.RandomResize([280], max_size=504),#for r101
            ]),
            # T.RandomResize([300], max_size=540),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
######################
#记得加载进来word2vec
######################
class VidSTGDataset(torch.utils.data.Dataset):
    def __init__(self, type, args):
        self.word2vec = None
        self.args = args

        self._transforms = make_coco_transforms(type, False)
        self.root_path = "video_audio/VidVRD-II-main/vidor-dataset/"
        self.video_path = self.root_path+"video_total/"
        self.box_path = self.root_path+"annotation_total/"
        self.anno_path = self.root_path + "VidSTG-Dataset-master/annotations/"
        self.type = type

        # # type = params['type']
        # if type == "train":
        #     self.anno_file_path = self.anno_path+"train_annotations.json"
        #     self.index_file_path = self.anno_path+"train_files.json"
        # elif type == "val":
        #     self.anno_file_path = self.anno_path+"val_annotations.json"
        #     self.index_file_path = self.anno_path+"val_files.json"
        # elif type == "test":
        #     self.anno_file_path = self.anno_path+"test_annotations.json"
        #     self.index_file_path = self.anno_path+"test_files.json"

        if type == "train":
            self.anno_file_path = self.anno_path + "train_annotations_caption.json"
            self.index_file_path = self.anno_path + "train_files.json"
        elif type == "val":
            self.anno_file_path = self.anno_path + "val_annotations_caption.json"
            self.index_file_path = self.anno_path + "val_files.json"
        elif type == "test":
            self.anno_file_path = self.anno_path + "test_annotations_caption.json"
            self.index_file_path = self.anno_path + "test_files.json"

        self.anno_list = json.load(open(self.anno_file_path, "r"))


    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, index):
        #读取captions和target_object
        anno = self.anno_list[index]
        index_str = anno['vid']
        # object_array, captions_array, \
        caption, object, labels_array = self._load_captions_labels(anno)

        # # 读取video的目标框
        # target_box_list = self._load_box(index_str)

        #读取video的frames，temporal_gt对应的帧，并读取目标框
        img_list, temporal_gt_array, target_box_array, frame_labels_array, valid = self._load_video_temGT_box(index_str, anno)


        index_vid = int(index_str[-3:])%len(valid)
        valid[index_vid] = 1


        target = {}
        target["boxes"] = torch.from_numpy(target_box_array)

        target["labels"] = torch.from_numpy(frame_labels_array)
        #文本如何输入再看看
        target['caption'] = caption
        target['video_id'] = torch.from_numpy(np.array([int(anno['vid'])]))
        target['orig_size'] = torch.from_numpy(np.array([anno['height'], anno['width']]))
        target['size'] = torch.from_numpy(np.array([480, 640]))
        target['object'] = object
        target['valid'] = torch.from_numpy(np.array(valid))
        target['temporal_gt'] = torch.from_numpy(temporal_gt_array)
        target['type'] = torch.tensor(0)

        if self._transforms is not None:
            img_tensor, target = self._transforms(img_list, target)

        return img_tensor, target



    def _load_captions_labels(self, anno):
        captions_list = []
        object_list = []
        labels_list = []
        capions = anno['captions']

        descr = "None"
        object = None
        for caption_dict in capions:
            #description
            descr = caption_dict['description']

            #找到句子中第一个名词并保存
            object = first_NN(descr)
            # object_list.append(object)

            type = caption_dict['type']

            #获取物体框的编号
            target_id = caption_dict['target_id']
            labels_list.append(target_id)

            break

        return descr, object, np.array(labels_list) #np.array(object_list), np.array(captions_list),

    #anno_list 转 anno_dict
    def _anno_list_to_dict(self):
        anno_dict = {}
        for anno in self.anno_list:
            anno_dict[anno['vid']] = anno
        return anno_dict


    def _load_video_temGT_box(self, index_str, anno):
        # img_array = None
        img_list = []
        temporal_gt = [999999, -1]
        target_box_list = []
        frame_labels_list = []

        #读取box框的json文件
        target_box_path = self.box_path+index_str+'.json'
        target_box_total = json.load(open(target_box_path, 'r'))['trajectories']
        target_id = anno['captions'][0]['target_id']

        #读取mp4文件
        video_path = self.video_path+index_str+'.mp4'
        videoCapture = cv2.VideoCapture(video_path)
        fps_src = videoCapture.get(cv2.CAP_PROP_FPS)
        fpf = round(fps_src)
        success, frame = videoCapture.read()
        frame_shape = frame.shape

        blank_frame = [0, 0, 0, 0]

        my_fp = math.ceil((anno['temporal_gt']['end_fid'] - anno['temporal_gt']['begin_fid'])/self.args.num_frames)

        i = 0
        j = 0
        valid = []
        while success:
            if i % (my_fp) == 0 and \
                (i>=anno['temporal_gt']['begin_fid'] and i<=anno['temporal_gt']['end_fid']):

                j += 1
                if j > self.args.num_frames:
                    print("超过帧数上限")
                    break

                target_box_item = copy.deepcopy(blank_frame)
                for target_box_item_total in target_box_total[i]:
                    if target_box_item_total['tid'] == target_id:
                        xmin = target_box_item_total['bbox']['xmin']
                        ymin = target_box_item_total['bbox']['ymin']
                        xmax = target_box_item_total['bbox']['xmax']
                        ymax = target_box_item_total['bbox']['ymax']

                        target_box_item[0] = xmin#(xmin+xmax)//2
                        target_box_item[1] = ymin#(ymin+ymax)//2
                        target_box_item[2] = xmax#xmax-xmin
                        target_box_item[3] = ymax#ymax-ymin


                target_box_list.append(target_box_item)
                frame_labels_list.append(0)
                valid.append(0)
                if target_box_item != blank_frame:
                    frame_labels_list[-1] = 1
                    valid[-1] = 0#1

                if temporal_gt[0] == 999999 and i >= anno['temporal_gt']['begin_fid']:
                    temporal_gt[0] = j
                if temporal_gt[1] <= j and i <= anno['temporal_gt']['end_fid']:
                    temporal_gt[1] = j

                frame_PIL = Image.fromarray(frame)
                img_list.append(frame_PIL)
            i += 1
            success, frame = videoCapture.read()


        for add_frame_index in range(len(target_box_list), self.args.num_frames):
            add_frame = np.uint8(np.zeros(frame_shape))

            frame_PIL = Image.fromarray(add_frame)
            img_list.append(frame_PIL)

            target_box_item = blank_frame
            target_box_list.append(target_box_item)
            frame_labels_list.append(0)
            valid.append(0)

        temporal_gt_array = np.array(temporal_gt)
        target_box_array = np.array(target_box_list)
        frame_labels_array = np.array(frame_labels_list)

        return img_list, temporal_gt_array, target_box_array, frame_labels_array, valid

if __name__ == "__main__":
    vidSTGDataset = VidSTGDataset("train", 5, 5, 5, 5)
    vidSTGDataset._load_video("10001787725")

