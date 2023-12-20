from yolo_engine import YOLOEngine
from utils import base642image, image2base64
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn


# Create FastAPI app
app = FastAPI()

# Create Yolo detector engine
yolo = YOLOEngine("yolov8n.pt")


# Define request data model
class RequestImage(BaseModel):
    base64str: str


# Define response data model
class ResponseResult(BaseModel):
    boxes: List[dict]
    result_image_base64: str


@app.post("/detect", response_model=ResponseResult)
async def detect(request: RequestImage):
    """Detect object from image base64 string"""

    # Remove prefix from base64 string if exists
    if len(request.base64str.split(",")) > 1:
        request.base64str = request.base64str.split(",")[1]

    # Convert base64 to PIL image
    pil_image = base642image(request.base64str)

    # Call Yolo detector engine
    boxes, result_image = yolo.detect(pil_image)

    # Convert numpy array to list of string
    for b in boxes:
        b["coordinator"] = b["coordinator"].tolist()

    # Return result to client
    return ResponseResult(
        boxes=boxes,
        result_image_base64=image2base64(result_image),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
