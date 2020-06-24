import argparse
import json
import os

import cv2
from detectron2.structures import BoxMode
from tqdm import tqdm
from detectron2.config import get_cfg
import os
from detectron2.engine import DefaultPredictor
import dataset_registry
from detectron2.utils.visualizer import ColorMode
import random
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import matplotlib.pyplot as plt

config_file = 'letters_dataset'

cfg = get_cfg()
cfg.merge_from_file(f"configs/{config_file}.yaml")
cfg.DATALOADER.NUM_WORKERS = 1

dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.DATASETS.TEST = ("nandos_dataset", )
predictor = DefaultPredictor(cfg)

for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   # metadata=None,
                   scale=0.8  # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.show()
