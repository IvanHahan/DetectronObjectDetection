import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import dataset_registry
from tqdm import tqdm

config_file = 'nandos_retina'
output_path = 'output.avi'
label_path = 'data/detectron/nandos/label.txt'

with open(label_path) as f:
    labels = [i.strip() for i in f.readlines()]

cfg = get_cfg()
cfg.merge_from_file(f"configs/{config_file}.yaml")
cfg.DATALOADER.NUM_WORKERS = 1

dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
meta.thing_classes = labels

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
predictor = DefaultPredictor(cfg)

cap = cv2.VideoCapture('data/nandos_1.mp4')
ret, frame = cap.read()
video_writer = cv2.VideoWriter(output_path, 0, 10, (frame.shape[1], frame.shape[0]))

max_frames = 25000
take_frame = 4
skip_frames = 6000
i = 0
pbar = tqdm(total=max_frames)

while cap.isOpened():
    i += 1
    ret, frame = cap.read()
    pbar.update(1)
    if i >= max_frames:
        break
    elif i < skip_frames or i % take_frame != 0:
        continue
    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1],
                   metadata=meta,
                   scale=1  # remove the colors of unsegmented pixels
                   )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    rendered = v.get_image()[:, :, ::-1]

    video_writer.write(rendered)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
video_writer.release()
cv2.destroyAllWindows()
