from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('../../runs/detect/train4/weights/best.pt')

    model.predict(source='../test-image.jpg', save=True)

# metrics = model.val()  # по умолчанию — на val, указанном в my_dataset.yaml
# print(metrics)
