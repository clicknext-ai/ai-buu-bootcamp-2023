from PIL import Image
import numpy as np
import base64
import io


def base642image(base64str: str) -> Image:
    """Convert base64 string to PIL image"""
    image_bytes = base64.b64decode(base64str)
    image = Image.open(io.BytesIO(image_bytes))
    return image


def image2base64(numpy_image: np.array) -> str:
    """Convert numpy array to base64 string"""
    image = Image.fromarray(numpy_image)
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64str = base64.b64encode(buffered.getvalue())
    return base64str.decode("utf-8")
