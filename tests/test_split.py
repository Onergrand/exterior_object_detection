import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import pytest
from utils.split import split_yolo_dataset

def create_dummy_yolo_dataset(base_path, num_files=10):
    os.makedirs(base_path / "images", exist_ok=True)
    os.makedirs(base_path / "labels", exist_ok=True)

    for i in range(num_files):
        name = f"img_{i}"
        (base_path / "images" / f"{name}.jpg").write_bytes(b"fake image content")
        (base_path / "labels" / f"{name}.txt").write_text(f"0 0.5 0.5 0.5 0.5\n")

    (base_path / "classes.txt").write_text("class0\n")
    (base_path / "notes.json").write_text('{"info": "test"}')

def test_split_yolo_dataset(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"

    create_dummy_yolo_dataset(source, num_files=10)

    split_yolo_dataset(str(source), str(target), test_ratio=0.3, seed=123)

    for sub in ["train", "val"]:
        assert (target / "images" / sub).exists()
        assert (target / "labels" / sub).exists()

    train_imgs = list((target / "images" / "train").glob("*.jpg"))
    val_imgs = list((target / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) == 7
    assert len(val_imgs) == 3

    assert (target / "classes.txt").exists()
    assert (target / "notes.json").exists()
