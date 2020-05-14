import json
import logging
import os
import random

import pandas as pd
import torch
from detectron2.config import get_cfg
from detectron2.data import (MetadataCatalog, DatasetCatalog)
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from sklearn.model_selection import train_test_split

from path_utils import root_path

logging.basicConfig(level='DEBUG')

data_path = root_path() / 'data' / 'wheet'
data_path.mkdir(exist_ok=True)

images_path = data_path / 'train'
csv_path = data_path / 'train.csv'

df = pd.read_csv(csv_path)

class_names = df.source.unique().tolist()
classes = {class_name: idx for idx, class_name in enumerate(class_names)}
LABEL_NAMES = classes
print(LABEL_NAMES)

print(class_names)

print(df.image_id.unique().shape)


def create_dataset(df):
    dataset_dicts = []
    for image_id, img_name in enumerate(df.image_id.unique()):
        record = {}
        image_df = df[df.image_id == img_name]
        file_path = f'{images_path}/{img_name}'
        record['file_name'] = file_path
        record['image_id'] = image_id
        record['height'] = int(image_df.iloc[0].height)
        record['width'] = int(image_df.iloc[0].width)
        objs = []
        for _, row in image_df.iterrows():
            bbox_raw = json.loads(row.bbox)
            bbox = [int(bbox_raw[0]), int(bbox_raw[1]),
                    int(bbox_raw[0] + bbox_raw[2]), int(bbox_raw[1] + bbox_raw[3])]
            obj = dict(bbox=bbox,
                       bbox_mode=BoxMode.XYXY_ABS,
                       segmentation=[],
                       category_id=classes.get(row.source, 0),
                       iscrowd=0)
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts


dataset = create_dataset(df)

print(dataset[1000: 1200])

random.shuffle(dataset)

print(dataset[1000: 1200])

len(dataset)

train_dt, test_dt = train_test_split(dataset, test_size=0.2, random_state=2020, stratify=None)

print(len(train_dt), len(test_dt))

TRAIN_VAL = ['train', 'val']


def _register_if_not(dataset_name: str, data_func: callable):
    """
    Register data if it is not already registered
    Args:
        dataset_name: data-set name
        data_func: data initialization function

    Returns:
        data catalog with registered dataset
    """
    if dataset_name in DatasetCatalog.list():
        print(f'Data-set {dataset_name} is already registered')
    else:
        DatasetCatalog.register(dataset_name, data_func)
        MetadataCatalog.get(dataset_name).set(thing_classes=class_names)
        print(f'Registration of the data-set {dataset_name} is done with labels {LABEL_NAMES}')

    return DatasetCatalog


def register_data_types(name: str, train_dicts: list, test_dicts: list) -> dict:
    """
    Register data types for training
    Args:
        name: name of data
        train_dicts: training data
        test_dicts: validation / test data

    Returns:
        data_catalogs: registered data catalogs
    """
    data_catalogs = dict()
    for d in TRAIN_VAL:
        data_catalog = _register_if_not(f'{name}_{d}', lambda: train_dicts if d == 'train' else test_dicts)
        data_catalogs[d] = data_catalog

    return data_catalogs


register_data_types('wheets', train_dt, test_dt)

cfg_path = root_path() / 'configs' / 'faster_rcnn_X_101_32x8d_FPN_3x.yaml'

outputs_path = root_path() / 'output'
outputs_path.mkdir(exist_ok=True)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

setup_logger()

weighs_dir = root_path() / 'weights'
weighs_dir.mkdir(exist_ok=True)

weights_path = weighs_dir / 'model_final_68b088.pkl'

cfg = get_cfg()
# Initialize model architecture
cfg.merge_from_file(str(cfg_path))
# Thresholds
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Configure model device bindings
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of prediction
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABEL_NAMES)
# Set weights
cfg.MODEL.WEIGHTS = str(weights_path)

cfg.DATASETS.TRAIN = ('wheets_train',)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.DATALOADER.NUM_WORKERS_PB = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.NUM_GPUS = 1
# Learning rate configuration
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 5000
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.SOLVER.CHECKPOINT_PERIOD = 500
# Test configuration`
cfg.TEST.EVAL_PERIOD = 500
# Output directory
cfg.OUTPUT_DIR = str(outputs_path)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
