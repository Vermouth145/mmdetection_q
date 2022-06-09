# -*- coding:utf-8 -*-
import io
import os
import cv2
import mmcv
import json
import argparse
import numpy as np
import pycocotools.mask as maskUtils
from mmdet.apis import init_detector, inference_detector, show_result_pyplot

all_categories = [
    {
        "name": "person",
        "id": 1
    },
    {
        "name": "bicycle",
        "id": 2
    },
    {
        "name": "car",
        "id": 3
    },
    {
        "name": "motorcycle",
        "id": 4
    },
    {
        "name": "airplane",
        "id": 5
    },
    {
        "name": "bus",
        "id": 6
    },
    {
        "name": "train",
        "id": 7
    },
    {
        "name": "truck",
        "id": 8
    },
    {
        "name": "boat",
        "id": 9
    },
    {
        "name": "traffic light",
        "id": 10
    },
    {
        "name": "fire hydrant",
        "id": 11
    },
    {
        "name": "stop sign",
        "id": 12
    },
    {
        "name": "parking meter",
        "id": 13
    },
    {
        "name": "bench",
        "id": 14
    },
    {
        "name": "bird",
        "id": 15
    },
    {
        "name": "cat",
        "id": 16
    },
    {
        "name": "dog",
        "id": 17
    },
    {
        "name": "horse",
        "id": 18
    },
    {
        "name": "sheep",
        "id": 19
    },
    {
        "name": "cow",
        "id": 20
    },
    {
        "name": "elephant",
        "id": 21
    },
    {
        "name": "bear",
        "id": 22
    },
    {
        "name": "zebra",
        "id": 23
    },
    {
        "name": "giraffe",
        "id": 24
    },
    {
        "name": "backpack",
        "id": 25
    },
    {
        "name": "umbrella",
        "id": 26
    },
    {
        "name": "handbag",
        "id": 27
    },
    {
        "name": "tie",
        "id": 28
    },
    {
        "name": "suitcase",
        "id": 29
    },
    {
        "name": "frisbee",
        "id": 30
    },
    {
        "name": "skis",
        "id": 31
    },
    {
        "name": "snowboard",
        "id": 32
    },
    {
        "name": "sports ball",
        "id": 33
    },
    {
        "name": "kite",
        "id": 34
    },
    {
        "name": "baseball bat",
        "id": 35
    },
    {
        "name": "baseball glove",
        "id": 36
    },
    {
        "name": "skateboard",
        "id": 37
    },
    {
        "name": "surfboard",
        "id": 38
    },
    {
        "name": "tennis racket",
        "id": 39
    },
    {
        "name": "bottle",
        "id": 40
    },
    {
        "name": "wine glass",
        "id": 41
    },
    {
        "name": "cup",
        "id": 42
    },
    {
        "name": "fork",
        "id": 43
    },
    {
        "name": "knife",
        "id": 44
    },
    {
        "name": "spoon",
        "id": 45
    },
    {
        "name": "bowl",
        "id": 46
    },
]


def reference_labelme_json():
    ref_json_path = './reference_labelme.json'
    data = json.load(open(ref_json_path))
    return data


import colorsys
import random


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)


def searchDirFile(rootDir, path_list, img_end):
    for dir_or_file in os.listdir(rootDir):
        filePath = os.path.join(rootDir, dir_or_file)
        # 判断是否为文件
        if os.path.isfile(filePath):
            # 如果是文件再判断是否以.jpg结尾，不是则跳过本次循环
            if os.path.basename(filePath).endswith(img_end):
                subname = filePath.split('/')[-1]
                path_list.append(subname)
            else:
                continue
        # 如果是个dir，则再次调用此函数，传入当前目录，递归处理。
        elif os.path.isdir(filePath):
            searchDirFile(filePath, path_list, img_end)
        else:
            print('not file and dir ' + os.path.basename(filePath))
            exit()


def main(args):
    # rgb--
    cnum = 80
    self_color = ncolors(cnum)
    colorbar_vis = np.zeros((cnum * 30, 100, 3), dtype=np.uint8)
    for ind, colo in enumerate(self_color):
        k_tm = np.ones((30, 100, 3), dtype=np.uint8) * np.array([colo[-1], colo[-2], colo[-3]])
        colorbar_vis[ind * 30:(ind + 1) * 30, 0:100] = k_tm
    cv2.imwrite('./colorbar_vis.png', colorbar_vis)

    data_ref = reference_labelme_json()

    mkdir_os(args.output_folder)
    mkdir_os(args.output_vis)

    score_thr = 0.3
    model = init_detector(args.input_config_file, args.input_checkpoint_file, device='cuda:0')

    trainimg = []
    searchDirFile(args.input_folder, trainimg, '.jpg')
    for ind, val in enumerate(trainimg):

        print(ind, '/', len(trainimg))
        subname = trainimg[ind]
        suffix = subname.split('.')[1]
        name = os.path.join(args.input_folder, subname)

        result = inference_detector(model, name)

        ori_img = mmcv.imread(name)
        img = ori_img.copy()
        height, width = img.shape[:2]

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None

        # 这里注意inference_detector的结果顺序在mmdetectionrc1.0中
        # 与训练时候的categories顺序相同,并不是categories的id顺序,所以训练时候注意json文件
        data_labelme = {}
        data_labelme['version'] = data_ref['version']
        data_labelme['flags'] = data_ref['flags']
        # data_labelme['lineColor'] = data_ref['lineColor']
        # data_labelme['fillColor'] = data_ref['fillColor']
        data_labelme['imagePath'] = subname
        data_labelme['imageData'] = None
        data_labelme['imageHeight'] = height
        data_labelme['imageWidth'] = width

        shapes = []
        thickness = 2
        for label in range(len(bbox_result)):
            bbox = bbox_result[label]
            for i in range(bbox.shape[0]):
                shape = {}
                if bbox[i][4] > score_thr:
                    # 颜色---rgb2bgr---imwrite
                    # self_color[0]是ignore的专属
                    # 其他颜色和categories中id对应
                    id = label + 1
                    cur_color = self_color[id][::-1]

                    label_name = 'error'
                    for m_id in all_categories:
                        if m_id['id'] == id:
                            label_name = m_id['name']
                    if label_name == 'error':
                        print("categories中id与检测网络不对应")
                        print(subname)
                        exit()

                    shape['label'] = label_name
                    # shape['line_color'] = data_ref['shapes'][0]['line_color']
                    # shape['fill_color'] = data_ref['shapes'][0]['fill_color']
                    shape['points'] = []
                    shape['shape_type'] = "rectangle"
                    shape['flags'] = data_ref['shapes'][0]['flags']
                    # labelme是x1y1x2y2
                    shape['points'].append([int(bbox[i][0]), int(bbox[i][1])])
                    shape['points'].append([int(bbox[i][2]), int(bbox[i][3])])
                    shapes.append(shape)

                    cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])),
                                  (cur_color[0], cur_color[1], cur_color[2]),
                                  thickness)
                    cv2.putText(img, label_name, (int(bbox[i][0]), int(bbox[i][1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (cur_color[0], cur_color[1], cur_color[2]), 2)

        data_labelme['shapes'] = shapes
        with io.open(os.path.join(args.output_folder, subname.replace(suffix, 'json')), 'w',
                     encoding="utf-8") as outfile:
            my_json_str = json.dumps(data_labelme, ensure_ascii=False, indent=1)
            outfile.write(my_json_str)

        cv2.imwrite(os.path.join(args.output_vis, subname), img)

        # 分割
        # 参考
        # /home/boyun/deepglint/environment/mmdetection-1.0rc0/mmdet/core/evaluation/coco_utils.py
        # bboxes = np.vstack(bbox_result)
        #
        # if segm_result is not None:
        #     segms = mmcv.concat_list(segm_result)
        #     inds = np.where(bboxes[:, -1] > score_thr)[0]
        #     np.random.seed(42)
        #     color_masks = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        #     for i in inds:
        #         #i = int(i)
        #         mask = maskUtils.decode(segms[i]).astype(np.bool)
        #         img[mask] = img[mask] * 0.5 + color_masks * 0.5


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "mmdetection_inference_result2labelme-json")
    parser.add_argument('-icf',
                        "--input_config_file",
                        default='./nut5_fine_faster_rcnn_r50_fpn_1x.py',
                        help="set input folder1")
    parser.add_argument('-jcf',
                        "--input_checkpoint_file",
                        default='./epoch_100.pth',
                        help="set input folder2")
    parser.add_argument('-if',
                        "--input_folder",
                        default='',
                        help="set input folder2")
    parser.add_argument('-of',
                        "--output_folder",
                        default='',
                        help="set output folder")
    parser.add_argument('-ov',
                        "--output_vis",
                        default='./vis_mmcv/',
                        help="set output folder")
    args = parser.parse_args()

    if args.input_config_file is None:
        parser.print_help()
        exit()

    main(args)