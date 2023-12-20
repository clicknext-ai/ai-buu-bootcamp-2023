from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np
import cv2


class YOLOEngine:
    def __init__(self, model_file_path: str = "yolov8n.pt") -> None:
        # Load YOLO model
        self.model = YOLO(model_file_path)

    def detect(self, input_frame, conf=0.5):
        """Detect object from image frame"""
        input_frame = np.array(input_frame)

        # Detect object from image frame
        # https://docs.ultralytics.com/modes/predict/#inference-sources
        detected = self.model(input_frame, conf=conf)

        boxes = []
        for obj in detected:
            for box in obj.boxes:
                result = {
                    "class_id": int(box.cls),
                    "class_name": self.model.names[int(box.cls)],
                    "confidence": box.conf.to("cpu").numpy().astype(float)[0],
                    "coordinator": box.xyxy.to("cpu").numpy().astype(int)[0],
                }
                boxes.append(result)

        debug_image = self.draw_boxes(input_frame, boxes)

        return boxes, debug_image

    def draw_boxes(self, frame, boxes):
        """Draw detected bounding boxes on image frame"""

        # Create annotator object
        annotator = Annotator(frame)
        for box in boxes:
            class_id = box["class_id"]
            class_name = box["class_name"]
            coordinator = box["coordinator"]
            confidence = box["confidence"]

            # Draw bounding box
            annotator.box_label(
                box=coordinator, label=class_name, color=colors(class_id, True)
            )

        return annotator.result()


if __name__ == "__main__":
    image = cv2.imread("image/test.png")
    yolo_engine = YOLOEngine("yolov8n.pt")
    boxes, debug_image = yolo_engine.detect(image)

    cv2.imwrite("result.jpg", debug_image)
    # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    # cv2.imshow("result", debug_image)
    # cv2.waitKey(0)
