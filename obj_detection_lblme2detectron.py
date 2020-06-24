import argparse
import json
import os

import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm

from utils import parse_annotation, make_dir_if_needed

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Users/UnicornKing/PyCharmProjects/screenshotprocessing/data/screenshots/2')
parser.add_argument('--output_image_dir', default='data/detectron/letters/images')
parser.add_argument('--output_label_dir', default='data/detectron/letters/annotations')
parser.add_argument('--image_path_prefix', default='/home/ihahanov/Projects/ObjectDetectionDetectron')
parser.add_argument('--label_path', default='data/detectron/letters/label.txt')
args = parser.parse_args()

if __name__ == '__main__':
    make_dir_if_needed(args.output_image_dir)
    make_dir_if_needed(args.output_label_dir)
    with open(args.label_path) as f:
        labels = [i.strip() for i in f.readlines()]
        label_to_idx = {l: i for i, l in enumerate(labels)}
    annotations = [(file, parse_annotation(os.path.join(args.data_dir, file))) for file in
                   sorted(os.listdir(args.data_dir))
                   if os.path.splitext(file)[-1] == '.json']
    dirs = set(),
    for file, annot in tqdm(annotations):
        file_name, ext = os.path.splitext(file)
        image_name = file_name + '.png'
        frame = cv2.imread(os.path.join(args.data_dir, image_name))
        frame_path = os.path.join(args.output_image_dir, image_name)

        rec = {
            'file_name': os.path.join(args.image_path_prefix, frame_path),
            'height': frame.shape[0],
            'width': frame.shape[1],
            'image_id': file_name,
        }

        for i, shape in enumerate(annot['shapes']):
            label = shape['label']
            if label not in labels:
                continue
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            if x1 == x2 or y1 == y2:
                continue

            rec.setdefault('annotations', [])
            x1 = min(x1, x2)
            x2 = max(x1, x2)
            y1 = min(y1, y2)
            y2 = max(y1, y2)

            cv2.imwrite(frame_path, frame)

            rec['annotations'].append({
                'bbox': [x1, y1, x2, y2],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': label_to_idx[label]
            })

        if 'annotations' in rec:
            with open(os.path.join(args.output_label_dir, file_name + '.json'), 'w') as f:
                str_annot = json.dumps(rec, indent=4)
                f.write(str_annot)
