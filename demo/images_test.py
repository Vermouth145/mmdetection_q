import asyncio
from argparse import ArgumentParser
import os
# from skimage import io
import cv2

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('imgs_file_path', help='Images file Path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    model = init_detector(args.config, args.checkpoint, device=args.device)
    path_list = os.listdir(args.imgs_file_path)
    path_list.sort()
    save_path = 'test_results/low'
    for filename in path_list:
        img_path = args.imgs_file_path + '/' + filename
        result = inference_detector(model, img_path)
        out_file = os.path.join(save_path, filename)
        show_result_pyplot(model, img_path, result, out_file, score_thr=args.score_thr)




if __name__ == '__main__':
    args = parse_args()
    main(args)
