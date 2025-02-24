from ultralytics import YOLO
import pandas as pd
import os
import shutil

weights = ['yolo8n.pt', 'yolo8s.pt', 'yolo8m.pt', 'yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt']

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    for weight in weights:
        weight_path = os.path.join('weights', weight)
        if not os.path.exists(weight_path):
            print(f"[Ошибка] Файл весов не найден: {weight_path}")
            continue  # пропустить и не пытаться скачать


        model_name = weight.replace('.pt', '')
        model = YOLO(weight_path)

        results = model.train(
            data='data.yaml',
            epochs=50,
            batch=8,
            imgsz=640,
            scale=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.1,
            device="0",
            name=f'{model_name}_train'
        )
