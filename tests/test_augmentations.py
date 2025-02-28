import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import cv2
import numpy as np
import pytest
from utils.augmentation import save_yolo_labels, load_yolo_labels, transform

def test_yolo_bbox_conversion(tmp_path):
    img_path = tmp_path / "test.jpg"
    label_path = tmp_path / "test.txt"

    dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.imwrite(str(img_path), dummy_img)

    bboxes = [[0.5, 0.5, 0.5, 0.5]]
    labels = [0]
    save_yolo_labels(str(label_path), bboxes, labels)

    loaded_bboxes, loaded_labels = load_yolo_labels(str(label_path))
    assert loaded_labels == [0]
    assert np.allclose(loaded_bboxes[0], [0.5, 0.5, 0.5, 0.5])

def test_augmentation_output(tmp_path):
    dummy_img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    bboxes = [[0.5, 0.5, 0.4, 0.4]]
    labels = [1]

    result = transform(image=dummy_img, bboxes=bboxes, class_labels=labels)
    assert 'image' in result and 'bboxes' in result and 'class_labels' in result
    assert isinstance(result['image'], np.ndarray)
    assert len(result['bboxes']) == len(result['class_labels'])
