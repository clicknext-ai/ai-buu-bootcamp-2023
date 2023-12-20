from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r"{YOUR_FILE_PATH_best.pt}")

# Run inference on 'bus.jpg' with arguments
model.track(
    source=r"{FOLDER_CONTAIN_JPG_IMAGE_FILE}/*.jpg",
    save=True,
    imgsz=640,
    conf=0.5,
    show=False,
)
