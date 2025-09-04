import os
import shutil
import random
import cv2
import glob
import yaml
import tempfile
import torch
import ultralytics
import numpy as np

from collections import defaultdict

from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# this motherfuckers must be loaded once
# at a time.
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
torch.serialization.add_safe_globals([ultralytics.nn.modules.Conv])
torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d])
torch.serialization.add_safe_globals([torch.nn.modules.batchnorm.BatchNorm2d])
torch.serialization.add_safe_globals([torch.nn.modules.activation.SiLU])
torch.serialization.add_safe_globals([ultralytics.nn.modules.C2f])
torch.serialization.add_safe_globals([torch.nn.modules.container.ModuleList])
torch.serialization.add_safe_globals([ultralytics.nn.modules.Bottleneck])
torch.serialization.add_safe_globals([ultralytics.nn.modules.SPPF])
torch.serialization.add_safe_globals([torch.nn.modules.pooling.MaxPool2d])
torch.serialization.add_safe_globals([torch.nn.modules.upsampling.Upsample])
torch.serialization.add_safe_globals([ultralytics.nn.modules.Concat])
torch.serialization.add_safe_globals([ultralytics.nn.modules.Detect])
torch.serialization.add_safe_globals([ultralytics.nn.modules.DFL])
torch.serialization.add_safe_globals([ultralytics.yolo.utils.IterableSimpleNamespace])
torch.serialization.add_safe_globals([np._core.multiarray.scalar])
torch.serialization.add_safe_globals([np.dtype])
torch.serialization.add_safe_globals([np.dtypes.Float64DType])

ldir = os.listdir()

if 'invoice' in ldir:
    images_dir = os.path.join(os.getcwd(), "invoice", "test")
    labels_dir = os.path.join(os.getcwd(), "invoice", "test-labels")
else:
    images_dir = os.path.join(os.getcwd(), "..", "invoice", "test")
    labels_dir = os.path.join(os.getcwd(), "..", "invoice", "test-labels")

data_dict = defaultdict(list)

image_base_file = os.path.basename(images_dir)
base_model = GroundedSAM(ontology=CaptionOntology({"tree": str(image_base_file)}))

try:
    base_model.label(input_folder=images_dir, output_folder=labels_dir)
except:
    pass

if 'models' in ldir:
    target_model_dir = os.path.join(os.getcwd(), "models", "yolov8n.pt")
else:
    target_model_dir = os.path.join(os.getcwd(), "..", "models", "yolov8n.pt")

target_model = YOLOv8(target_model_dir)
target_model.train(os.path.join(labels_dir, "data.yaml"), epochs=200)

# run inference on the new model
pred = target_model.predict(os.path.join(labels_dir, "valid", "images", "test-inv29.jpg"), confidence=0.9)
print(pred)
