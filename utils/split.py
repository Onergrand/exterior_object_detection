
import os
import shutil
import random

def split_yolo_dataset(source_dir, target_dir, test_ratio=0.2, seed=42):
    random.seed(seed)
    
    images_dir = os.path.join(source_dir, "images")
    labels_dir = os.path.join(source_dir, "labels")
    classes_file = os.path.join(source_dir, "classes.txt")
    notes_file = os.path.join(source_dir, "notes.json")
    
    target_images_train = os.path.join(target_dir, "images", "train")
    target_images_test = os.path.join(target_dir, "images", "val")
    target_labels_train = os.path.join(target_dir, "labels", "train")
    target_labels_test = os.path.join(target_dir, "labels", "val")

    os.makedirs(target_images_train, exist_ok=True)
    os.makedirs(target_images_test, exist_ok=True)
    os.makedirs(target_labels_train, exist_ok=True)
    os.makedirs(target_labels_test, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)

    split_idx = int(len(image_files) * (1 - test_ratio))
    train_images = image_files[:split_idx]
    test_images = image_files[split_idx:]

    def copy_files(image_list, img_dest, lbl_dest):
        for image_file in image_list:
            base_name = os.path.splitext(image_file)[0]
            label_file = f"{base_name}.txt"

            shutil.copy(os.path.join(images_dir, image_file), os.path.join(img_dest, image_file))

            label_src_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_src_path):
                shutil.copy(label_src_path, os.path.join(lbl_dest, label_file))

    copy_files(train_images, target_images_train, target_labels_train)
    copy_files(test_images, target_images_test, target_labels_test)

    if os.path.exists(classes_file):
        shutil.copy(classes_file, os.path.join(target_dir, "classes.txt"))
    if os.path.exists(notes_file):
        shutil.copy(notes_file, os.path.join(target_dir, "notes.json"))

    print(f"Разделение завершено: {len(train_images)} train / {len(test_images)} val")


split_yolo_dataset("raw_data", "dataset")