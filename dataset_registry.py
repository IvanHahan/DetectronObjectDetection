import glob
import json
import os

from detectron2.data import DatasetCatalog


def nandos_dataset():
    annot_dir = 'data/detectron/nandos/annotations'
    annot_paths = list(glob.glob(os.path.join(annot_dir, '*.json')))
    annots = []
    for path in annot_paths:
        with open(path) as f:
            annot = json.loads(f.read())
            annots.append(annot)
    return annots


def letters_dataset():
    annot_dir = 'data/detectron/letters/annotations'
    annot_paths = list(glob.glob(os.path.join(annot_dir, '*.json')))
    annots = []
    for path in annot_paths:
        with open(path) as f:
            annot = json.loads(f.read())
            annots.append(annot)
    return annots


DatasetCatalog.register("nandos_dataset", nandos_dataset)
DatasetCatalog.register("letters_dataset", letters_dataset)
