import os

import gradio as gr
from fastapi import FastAPI

from RAG_Bot import demo

app = gr.mount_gradio_app(FastAPI(), demo, path="/api")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
    )
