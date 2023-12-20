import gradio as gr
import base64
import requests
import io

from PIL import Image


def image(image):
    with open(image, "rb") as f:
        img_to_base64 = base64.b64encode(f.read()).decode("utf-8")

    request = requests.post(
        "http://192.168.82.175:8000/detect", json={"base64str": str(img_to_base64)}
    )
    content = request.json()

    base64_to_img = base64.decodebytes(bytes(content["result_image_base64"], "utf-8"))
    result = Image.open(io.BytesIO(base64_to_img))

    detail = content["boxes"]

    return result, detail


with gr.Blocks() as demo:
    gr.Markdown(
        """
        ![logo](/file=./image/clicknext_logo2x.png) 
        # ClickNext x BUU bootcamp 2023 🥳
        """
    )

    with gr.Column():
        gr.Markdown("## Image object detection")
        gr.Interface(
            image,
            inputs=gr.Image(type="filepath"),
            outputs=["image", "json"],
            allow_flagging="never",
        )

demo.launch()
