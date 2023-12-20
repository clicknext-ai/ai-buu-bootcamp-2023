from ultralytics import YOLO

# Load a model
model = YOLO('yolov8m.yaml').load('yolov8m.pt')  # build from YAML and transfer weights

# Train the model
custom_yaml = r"{YOUR_PATH_data.yaml}"
results = model.train(data=custom_yaml, epochs=1000, imgsz=640, batch=8, workers=1, plots=True, device=0, patience=50, 
                      flipud=0.5, mixup=0.7,
                      name=r"{NAME_OF_FOLDER_TO_SAVE_MODEL}")