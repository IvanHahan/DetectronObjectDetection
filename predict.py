import os
import random

import cv2
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import dataset_registry

config_file = 'letters_retina'
label_path = 'data/detectron/letters/label.txt'

with open(label_path) as f:
    labels = [i.strip() for i in f.readlines()]

cfg = get_cfg()
cfg.merge_from_file(f"configs/{config_file}.yaml")
cfg.DATALOADER.NUM_WORKERS = 1

dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
meta.thing_classes = labels

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.DATASETS.TEST = ("nandos_dataset", )
predictor = DefaultPredictor(cfg)

for d in random.sample(dataset_dicts, 1):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=meta,
                   scale=0.8  # remove the colors of unsegmented pixels
                   )
    # vis = v.draw_dataset_dict(d)
    # plt.imshow(vis.get_image()[:, :, ::-1])
    # plt.show()

    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imshow('frame', v.get_image()[:, :, ::-1])
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()
