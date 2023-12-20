import gradio as gr


def hello(name):
    return "Hello " + name + "!"


with gr.Blocks() as demo:
    with gr.Column():
        gr.Interface(hello, inputs="textbox", outputs="text")

demo.launch()
