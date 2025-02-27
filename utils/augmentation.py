import os
import cv2
import albumentations as A
import glob
import shutil

IMG_DIR = 'raw_data/images'
LBL_DIR = 'raw_data/labels'
OUT_IMG_DIR = 'augmented/images_aug'
OUT_LBL_DIR = 'augmented/labels_aug'

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Аугментации
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(
    translate_percent={"x": 0.05, "y": 0.05},
    scale=(0.9, 1.1),                          
    rotate=(-10, 10),                          
    shear={"x": (-5, 5), "y": (-5, 5)},       
    fit_output=False,
    p=0.5,
    border_mode=cv2.BORDER_CONSTANT
)

],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

def load_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')
        labels = []
        classes = []
        for line in lines:
            if line:
                parts = line.strip().split()
                cls = int(parts[0])
                bbox = list(map(float, parts[1:]))  # cx, cy, w, h
                labels.append(bbox)
                classes.append(cls)
        return labels, classes

def save_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, cls in zip(bboxes, class_labels):
            f.write(f"{cls} {' '.join(f'{x:.6f}' for x in bbox)}\n")

for img_path in glob.glob(os.path.join(IMG_DIR, '*.*')):
    for i in range(2):
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        label_path = os.path.join(LBL_DIR, f"{name}.txt")

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        bboxes, class_labels = load_yolo_labels(label_path)

        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        except Exception as e:
            print(f"Ошибка на {filename}: {e}")
            continue

        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_classes = augmented['class_labels']

        out_img_path = os.path.join(OUT_IMG_DIR, f"{name}_aug{i}.jpg")
        out_lbl_path = os.path.join(OUT_LBL_DIR, f"{name}_aug{i}.txt")
        cv2.imwrite(out_img_path, aug_img)
        save_yolo_labels(out_lbl_path, aug_bboxes, aug_classes)

        print(f"Сохранено: {out_img_path}")


try:
    shutil.copy('raw_data/classes.txt', 'augmented')
    shutil.copy('raw_data/notes.json', 'augmented')
    print("classes.txt и notes.json скопированы")
except Exception as e:
    print(f"Не удалось скопировать метаданные: {e}")


print("Аугментация завершена")
